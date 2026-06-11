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

SUPABASE_URL = "https://pgrhxxhiyqgngoffwden.supabase.co"
LOG = Path("/tmp/populace_chain.log")
INTERVAL_S = 45


def _secret_key() -> str:
    for cmd in (
        ["agent-secret", "get", "POPULACE_SUPABASE_SECRET_KEY"],
        [
            str(Path.home() / ".claude" / "manage-secret.sh"),
            "get",
            "POPULACE_SUPABASE_SECRET_KEY",
        ],
    ):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout.strip()
            if out:
                return out
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError(
        "POPULACE_SUPABASE_SECRET_KEY not retrievable; refusing to report "
        "without authenticated writes."
    )


_KEY = None

STEP_RE = re.compile(r"=== CHAIN (step (\d): )?([A-Za-z -]+?) ===")
STAGE_RE = re.compile(r"\[build\] (stage [A-Z0-9]+[a-z]?: .+)")

STEP_NAMES = {
    "1": "full build",
    "2": "extract target surface",
    "3": "calibrate + artifact",
    "4": "enrich (sim-dependent layers)",
    "5": "gates (nonzero + parity + smoke)",
}


def chain_alive() -> bool:
    return (
        subprocess.run(
            ["pgrep", "-f", "run_chain.sh"], capture_output=True
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


def push(status: dict) -> None:
    global _KEY
    if _KEY is None:
        _KEY = _secret_key()
    import urllib.request

    request = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/build_events",
        data=json.dumps({"run": status["run"], "status": status}).encode(),
        headers={
            "apikey": _KEY,
            "Authorization": f"Bearer {_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status not in (200, 201, 204):
            raise RuntimeError(f"supabase write HTTP {response.status}")


def main() -> int:
    while True:
        alive = chain_alive()
        text = LOG.read_text(errors="replace") if LOG.exists() else ""
        status = parse_status(text, alive)
        try:
            push(status)
            print(
                f"pushed: step {status['chain_step']} {status['state']} "
                f"@ {status['updated_at']}"
            )
        except Exception as error:  # noqa: BLE001 - keep reporting
            print(f"supabase push failed: {error}", file=sys.stderr)
        if not alive:
            return 0
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
