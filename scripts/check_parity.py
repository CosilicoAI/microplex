"""Parity + smoke gate for the built artifact, via populace.build gates.

Parity is judged at SIMULATION level over the reference's stored input
layers: for every variable the reference (enhanced CPS) stores and populates,
the candidate's simulation must produce a non-zero layer too (stored or
computed by engine formulas — dropping a formula-masking zero column and
letting the engine compute is parity, not a gap).

Smoke prints the headline aggregates (population, SNAP, net worth, net STCG,
investment interest) for the publish decision.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ART = Path.home() / ".claude-worktrees" / "microplex-spec-build" / "artifacts"
CANDIDATE = ART / "populace_us_2024.h5"
REFERENCE = Path.home() / "populace-score-work" / "enhanced_cps_2024_hf_main.h5"
OUT = ART / "parity_gate.json"
YEAR = 2024


def stored_layers(path: Path) -> dict[str, float]:
    """entity-unprefixed variable -> nonzero share among stored columns."""
    shares: dict[str, float] = {}
    with h5py.File(path) as f:
        def visit(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            var = name.split("/")[-2] if name.endswith(str(YEAR)) else name.split("/")[-1]
            # layouts: "entity/var" (single-year) or "var/2024" (timeperiod)
            parts = name.split("/")
            if len(parts) == 2 and parts[1] == str(YEAR):
                var = parts[0]
            elif len(parts) == 2:
                var = parts[1]
            else:
                var = parts[-1]
            values = node[:]
            if values.dtype.kind in ("S", "O", "U"):
                return
            shares[var] = float((np.asarray(values, dtype=np.float64) != 0).mean())
        f.visititems(visit)
    return shares


def candidate_stored_shares() -> dict[str, float]:
    """entity.column -> nonzero share over every stored candidate column."""
    from policyengine_us.data import USSingleYearDataset

    ds = USSingleYearDataset(file_path=str(CANDIDATE))
    shares: dict[str, float] = {}
    for ent in ("person", "household", "tax_unit", "spm_unit", "family",
                "marital_unit"):
        tbl = getattr(ds, ent)
        structural = {f"{ent}_id", f"{ent}_weight"} | {
            c for c in tbl.columns if c.startswith("person_")
        }
        for c in tbl.columns:
            if c in structural:
                continue
            vals = tbl[c].to_numpy()
            if vals.dtype.kind not in "fiub":
                continue
            shares[f"{ent}.{c}"] = float(
                (np.asarray(vals, dtype=np.float64) != 0).mean()
            )
    return shares


def main() -> int:
    from policyengine_us import Microsimulation
    from populace.build import exported_nonzero_gate, parity_gate

    # gate 0: every stored column carries signal (populate it or drop it)
    nonzero = exported_nonzero_gate(candidate_stored_shares())
    print(
        f"exported_nonzero: passed={nonzero.passed} "
        f"({nonzero.details['columns_checked']} stored columns)"
    )
    for line in nonzero.failures:
        print(f"  ZERO {line}")

    ref_layers = stored_layers(REFERENCE)
    print(f"reference stored layers: {len(ref_layers)}")

    sim = Microsimulation(dataset=str(CANDIDATE))
    tbs = sim.tax_benefit_system
    candidate_shares: dict[str, float] = {}
    reference_shares: dict[str, float] = {}
    skipped: list[str] = []
    for var, ref_share in sorted(ref_layers.items()):
        if var not in tbs.variables:
            skipped.append(var)
            continue
        if tbs.variables[var].definition_period not in ("year",):
            skipped.append(var)
            continue
        try:
            values = np.asarray(
                sim.calculate(var, YEAR).values, dtype=np.float64
            )
        except Exception as error:  # noqa: BLE001 - report, never mask
            candidate_shares[var] = 0.0
            reference_shares[var] = ref_share
            print(f"  calc failed {var}: {type(error).__name__} {error}")
            continue
        candidate_shares[var] = float((values != 0).mean())
        reference_shares[var] = ref_share

    # weights are structural, not layers
    for structural in ("household_weight", "person_weight"):
        candidate_shares.pop(structural, None)
        reference_shares.pop(structural, None)

    result = parity_gate(candidate_shares, reference_shares)
    print(
        f"parity: passed={result.passed} gaps={result.details['gaps']} "
        f"(checked {result.details['reference_populated_layers']} populated "
        f"reference layers, skipped {len(skipped)} non-annual/unknown)"
    )
    for line in result.failures:
        print(f"  GAP {line}")

    # ---- smoke aggregates -------------------------------------------------
    def total(var: str) -> float:
        return float(sim.calculate(var, YEAR).sum())

    smoke = {
        # weighted person count via microdf's weighted .count()
        "people_m": float(sim.calculate("age", YEAR).count()) / 1e6,
        "snap_b": total("snap") / 1e9,
        "net_worth_t": total("net_worth") / 1e12,
        "net_stcg_b": total("short_term_capital_gains") / 1e9,
        "investment_interest_expense_b": total("investment_interest_expense")
        / 1e9,
        "tips_b": total("tip_income") / 1e9,
        "pre_subsidy_rent_b": total("pre_subsidy_rent") / 1e9,
    }
    print("smoke:", json.dumps(smoke, indent=1))

    # Telemetry (fail-soft): gate verdicts as queryable rows.
    try:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import populace_telemetry as _telemetry

        _telemetry.push_gate_result(nonzero)
        _telemetry.push_gate_result(result)
    except Exception as _err:  # noqa: BLE001
        print(f"telemetry skipped: {_err}")

    def _gate_dict(gate):
        return {
            "passed": gate.passed,
            "failures": list(gate.failures),
            "details": dict(gate.details),
        }

    OUT.write_text(
        json.dumps(
            {
                "exported_nonzero": _gate_dict(nonzero),
                "parity": _gate_dict(result),
                "smoke": smoke,
                "skipped_layers": skipped,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")
    return 0 if (result.passed and nonzero.passed) else 1


if __name__ == "__main__":
    sys.exit(main())
