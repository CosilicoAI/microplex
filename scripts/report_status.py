"""Live build-status reporter: chain log -> Supabase -> populace.dev.

Parses the chain runner's log into a small sanitized status JSON (step,
stage, gate verdicts, last log line) and appends it to the PolicyEngine
populace Supabase project's ``build_events`` table every ~45s while the
chain runs; the observatory's live-build strip reads the latest row with
the public (read-only, RLS-enforced) key. Publishes only derived status
fields, never raw paths or environment detail beyond the log lines the
build itself prints. Rows are append-only — the build timeline is queryable
history, not a mutable blob.
"""

import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import populace_telemetry as telemetry

LOG = Path("/tmp/populace_chain.log")
INTERVAL_S = 45

STEP_RE = re.compile(r"=== CHAIN (step (\d): )?([A-Za-z -]+?) ===")
STAGE_RE = re.compile(r"\[build\] (stage [A-Z0-9]+[a-z]?: .+)")

STEP_NAMES = {
    "1": "full build",
    "2": "extract target surface",
    "3": "calibrate + artifact",
    "4": "enrich (sim-dependent layers)",
    "5": "gates (nonzero + parity + smoke)",
}


CHAIN_PIDFILE = Path("/tmp/populace_chain.pid")


def chain_alive() -> bool:
    """Pidfile first (exact); pattern fallback for chains launched before
    the pidfile existed. The pattern matches the script's repo-relative
    path, which survives relative *and* absolute launches."""
    if CHAIN_PIDFILE.exists():
        try:
            pid = int(CHAIN_PIDFILE.read_text().strip())
            return (
                subprocess.run(
                    ["kill", "-0", str(pid)], capture_output=True
                ).returncode
                == 0
            )
        except ValueError:
            pass
    return (
        subprocess.run(
            ["pgrep", "-f", "scripts/run_chain.sh"], capture_output=True
        ).returncode
        == 0
    )


def parse_status(text: str, alive: bool) -> dict:
    step = "1"
    complete = "CHAIN COMPLETE" in text
    for match in STEP_RE.finditer(text):
        if match.group(2):
            step = match.group(2)
    stages = STAGE_RE.findall(text)
    failed = (not alive) and (not complete) and "Traceback" in text
    gates = {}
    nonzero = re.search(r"exported_nonzero: passed=(\w+) \((\d+) stored", text)
    if nonzero:
        gates["exported_nonzero"] = {
            "passed": nonzero.group(1) == "True",
            "columns": int(nonzero.group(2)),
        }
    parity = re.search(r"parity: passed=(\w+) gaps=(\d+)", text)
    if parity:
        gates["parity"] = {
            "passed": parity.group(1) == "True",
            "gaps": int(parity.group(2)),
        }
    calib = re.search(
        r"loss ([\d.]+)->([\d.]+) \| within10 ([\d.]+)%", text
    )
    smoke = re.search(r"smoke: (\{[^}]+\})", text, re.S)
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and "Warning" not in line and "warn" not in line
    ]
    state = (
        "complete"
        if complete
        else ("failed" if failed else ("running" if alive else "stale"))
    )
    status = {
        "run": "us-2024 (eCPS-free rebuild)",
        "state": state,
        "chain_step": int(step),
        "step_name": STEP_NAMES.get(step, "?"),
        "build_stage": stages[-1] if stages else None,
        "last_line": lines[-1][:200] if lines else None,
        "gates": gates,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if calib:
        status["calibration"] = {
            "loss_from": float(calib.group(1)),
            "loss_to": float(calib.group(2)),
            "within10_pct": float(calib.group(3)),
        }
    if smoke:
        try:
            status["smoke"] = json.loads(smoke.group(1))
        except json.JSONDecodeError:
            pass
    return status


def push(status: dict, run_id: str | None) -> bool:
    return telemetry.insert(
        "build_events", {"run_id": run_id, "status": status}
    )


def main() -> int:
    import signal

    def _on_term(signum, frame):  # noqa: ARG001 - signal signature
        run_id = telemetry.current_run_id()
        if run_id is not None:
            telemetry.finish_run(run_id, "stale")
        sys.exit(143)

    signal.signal(signal.SIGTERM, _on_term)
    git_sha = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent),
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or None
    run_id = telemetry.ensure_run(
        country="us",
        year=2024,
        label="us-2024 (eCPS-free rebuild)",
        git_sha=git_sha,
    )
    while True:
        alive = chain_alive()
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        status = parse_status(text, alive)
        if push(status, run_id):
            print(
                f"pushed: step {status['chain_step']} {status['state']} "
                f"@ {status['updated_at']}"
            )
        if not alive:
            if run_id is not None and status["state"] in (
                "complete", "failed", "stale",
            ):
                telemetry.finish_run(run_id, status["state"])
            return 0
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
