"""Shared telemetry client: build scripts -> populace Supabase.

One canonical copy of the write path (the reporter, the calibration step,
and the gate step all import this). Telemetry is OBSERVABILITY, not a gate:
every public function is fail-soft — a Supabase hiccup logs a warning and
never fails a build. Data-integrity loudness lives in the gates themselves.

Writes use the secret key from the keychain; the public site reads through
RLS SELECT-only policies with the publishable key.
"""

import json
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

SUPABASE_URL = "https://pgrhxxhiyqgngoffwden.supabase.co"
RUN_ID_FILE = Path("/tmp/populace_run_id")

_KEY: str | None = None


def _secret_key() -> str:
    global _KEY
    if _KEY is not None:
        return _KEY
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
                _KEY = out
                return out
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("POPULACE_SUPABASE_SECRET_KEY not retrievable")


def _request(
    path: str, payload, method: str = "POST", prefer: str = "return=minimal"
):
    key = _secret_key()
    request = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/{path}",
        data=json.dumps(payload).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        return json.loads(body) if body else None


def ensure_run(
    *, country: str, year: int, label: str, git_sha: str | None = None
) -> str | None:
    """Create (or reuse, via /tmp marker) the current run row; return its id.

    Fail-soft: returns None if Supabase is unreachable.
    """
    try:
        if RUN_ID_FILE.exists():
            run_id = RUN_ID_FILE.read_text().strip()
            if run_id:
                return run_id
        # A crashed earlier chain can leave its row 'running'; the partial
        # unique index (one live build per country+year) would then reject
        # the new row. Sweep leftovers to 'stale' first.
        query = urllib.parse.urlencode(
            {
                "country": f"eq.{country}",
                "year": f"eq.{year}",
                "state": "eq.running",
            }
        )
        _request(
            f"runs?{query}",
            {
                "state": "stale",
                "finished_at": datetime.now(UTC).isoformat(),
            },
            method="PATCH",
        )
        rows = _request(
            "runs",
            {
                "country": country,
                "year": year,
                "label": label,
                "git_sha": git_sha,
            },
            prefer="return=representation",
        )
        run_id = rows[0]["id"]
        RUN_ID_FILE.write_text(run_id)
        return run_id
    except Exception as error:  # noqa: BLE001 - telemetry never fails a build
        print(f"telemetry: ensure_run failed: {error}", file=sys.stderr)
        return None


def current_run_id() -> str | None:
    if RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip()
        return run_id or None
    return None


def finish_run(run_id: str, state: str) -> None:
    """Mark the run complete/failed/stale and retire the marker. Fail-soft.

    The marker MUST go even if the PATCH fails: a finished chain's id must
    never leak into the next chain's events (the one-run-one-id invariant).
    """
    try:
        _request(
            f"runs?id=eq.{run_id}",
            {
                "state": state,
                "finished_at": datetime.now(UTC).isoformat(),
            },
            method="PATCH",
        )
    except Exception as error:  # noqa: BLE001
        print(f"telemetry: finish_run failed: {error}", file=sys.stderr)
    finally:
        RUN_ID_FILE.unlink(missing_ok=True)


def insert(table: str, rows: list[dict] | dict) -> bool:
    """Append row(s). Fail-soft; returns whether the write landed."""
    try:
        _request(table, rows)
        return True
    except Exception as error:  # noqa: BLE001
        print(f"telemetry: insert into {table} failed: {error}", file=sys.stderr)
        return False


def push_gate_result(gate_result, run_id: str | None = None) -> bool:
    """Record one populace.build GateResult row. Fail-soft."""
    run_id = run_id or current_run_id()
    if run_id is None:
        return False
    failures = list(gate_result.failures)
    if len(failures) > 200:
        failures = failures[:200] + [
            f"... truncated {len(gate_result.failures) - 200} further "
            "failures (full list in the build log)"
        ]
    return insert(
        "gate_results",
        {
            "run_id": run_id,
            "gate": gate_result.name,
            "passed": gate_result.passed,
            "failures": failures,
            "details": dict(gate_result.details),
        },
    )


def push_target_diagnostics(diagnostics, run_id: str | None = None) -> bool:
    """Record per-target calibration diagnostics (bulk). Fail-soft.

    Accepts populace.calibrate TargetDiagnostic objects (name, target,
    final_estimate, relative_error); family parses from the name prefix.
    """
    run_id = run_id or current_run_id()
    if run_id is None:
        return False
    rows = [
        {
            "run_id": run_id,
            "name": d.name,
            "family": (
                d.name.split("/", 1)[0] if "/" in d.name else "unspecified"
            ),
            "target": float(d.target),
            "achieved": float(d.final_estimate),
            "rel_error": float(d.relative_error),
        }
        for d in diagnostics
    ]
    ok = True
    for start in range(0, len(rows), 1000):
        ok = insert("target_diagnostics", rows[start : start + 1000]) and ok
    return ok
