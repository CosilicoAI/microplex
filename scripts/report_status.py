"""Live build-status reporter: chain log -> gist -> populace.dev.

Parses the chain runner's log into a small sanitized status JSON (step,
stage, gate verdicts, last log line) and PATCHes it to a public gist every
~45s while the chain runs; the observatory's live-build strip polls the
gist's raw URL. Publishes only derived status fields, never raw paths or
environment detail beyond the log lines the build itself prints.
"""

import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

GIST_ID = "c245589ec19ec00b02756995a4af7b48"
GIST_FILE = "populace_build_status.json"
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
    payload = json.dumps(
        {"files": {GIST_FILE: {"content": json.dumps(status, indent=1)}}}
    )
    subprocess.run(
        ["gh", "api", f"gists/{GIST_ID}", "-X", "PATCH", "--input", "-"],
        input=payload.encode(),
        capture_output=True,
        check=True,
    )


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
        except subprocess.CalledProcessError as error:
            print(f"gist push failed: {error}", file=sys.stderr)
        if not alive:
            return 0
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
