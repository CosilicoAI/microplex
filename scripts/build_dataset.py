"""Build the publishable populace-US v2 dataset: calibrate the v2 pool's
household weights to its raw PE-native surface (ratio-50 bound), write them
into the USSingleYearDataset copy and the timeperiod export, verify."""

import shutil
import sys
import time

import h5py
import numpy as np
import pandas as pd

from populace.calibrate import Target, TargetSet, calibrate
from populace.frame import EntitySchema, Frame, WeightKind, Weights

ART = "/Users/maxghenis/.claude-worktrees/microplex-spec-build/artifacts"
POOL = f"{ART}/spec_candidate_full_2024/candidate_policyengine_us.h5"
TP = f"{ART}/spec_candidate_full_2024/candidate_timeperiod.h5"
SURFACE = f"{ART}/target_surface_raw.npz"
OUT = f"{ART}/populace_us_2024.h5"
OUT_TP = f"{ART}/populace_us_2024_timeperiod.h5"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    surf = np.load(SURFACE, allow_pickle=True)
    A = surf["A"].astype(np.float64)
    b = surf["b"].astype(np.float64)
    w0 = surf["w0"].astype(np.float64)
    names = [str(x) for x in surf["names"]]
    n_hh, n_t = A.shape
    log(f"{n_t} targets x {n_hh} households; w0 sum {w0.sum()/1e6:.1f}M")

    household = pd.DataFrame({"household_id": np.arange(n_hh, dtype=np.int64)})
    person = pd.DataFrame(
        {
            "person_id": np.arange(n_hh, dtype=np.int64),
            "person_household_id": np.arange(n_hh, dtype=np.int64),
        }
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=w0.copy(), kind=WeightKind.DESIGN)},
    )
    targets = TargetSet(
        tuple(
            Target(
                name=names[t],
                entity="household",
                aggregation="sum",
                value=float(b[t]),
                measure=(lambda _f, col=A[:, t].copy(): col),
            )
            for t in range(n_t)
        )
    )
    # Signed heavy-tail targets (architecture review): calibration was free
    # to amplify loss-heavy records in dimensions absent from the surface.
    # Net short-term capital gains is anchored to the PUF donor's own
    # weighted, uprated total — primary-source, computed, never hand-typed.
    import h5py as _h5py

    _puf = pd.read_csv(
        "/Users/maxghenis/.cache/microplex/puf_2015.csv",
        usecols=["P22250", "S006"],
    )
    _stcg_value = float(
        (
            pd.to_numeric(_puf["P22250"], errors="coerce").fillna(0)
            * pd.to_numeric(_puf["S006"], errors="coerce").fillna(0)
            / 100.0
        ).sum()
        * 1.8  # microplex puf.py uprating factor for short_term_capital_gains
    )
    with _h5py.File(TP) as _f:
        _stcg_p = _f["short_term_capital_gains"]["2024"][:].astype(np.float64)
        _phh = _f["person_household_id"]["2024"][:]
        _hid = _f["household_id"]["2024"][:]
    _hidx = {h: i for i, h in enumerate(_hid.tolist())}
    _stcg_hh = np.zeros(n_hh, dtype=np.float64)
    np.add.at(_stcg_hh, np.fromiter((_hidx[h] for h in _phh.tolist()), dtype=np.int64), _stcg_p)
    targets = TargetSet(
        tuple(targets)
        + (
            Target(
                name="puf/net_short_term_capital_gains",
                entity="household",
                aggregation="sum",
                value=_stcg_value,
                measure=(lambda _f2, col=_stcg_hh.copy(): col),
            ),
        )
    )
    log(f"signed STCG target appended: ${_stcg_value/1e9:.1f}B (PUF weighted, uprated)")

    log("calibrating (ratio-50 bound)...")
    t0 = time.time()
    result = calibrate(
        frame,
        targets,
        weight_entity="household",
        epochs=3000,
        learning_rate=0.15,
        mass="free",
        max_weight_ratio=50.0,
        seed=0,
    )
    cw = result.frame.resolve_weights("household").values.astype(np.float64)
    # Telemetry (fail-soft): per-target diagnostics power the observatory's
    # live fit tables and cross-run regression checks.
    try:
        import pathlib as _pathlib
        import sys as _sys

        _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
        import populace_telemetry as _telemetry

        _telemetry.push_target_diagnostics(result.diagnostics)
    except Exception as _err:  # noqa: BLE001 - telemetry never fails a build
        log(f"telemetry skipped: {_err}")

    log(
        f"done {time.time()-t0:.0f}s | loss {result.initial_loss:.3f}->"
        f"{result.final_loss:.4f} | within10 "
        f"{result.fraction_within_10pct*100:.2f}% | max {cw.max():,.0f} | "
        f">500k {(cw>5e5).sum()}"
    )

    from policyengine_us.data import USSingleYearDataset

    shutil.copy(POOL, OUT)
    ds = USSingleYearDataset(file_path=OUT)
    assert len(ds.household) == n_hh
    ds.household["household_weight"] = cw
    for ent in ("person", "household"):
        tbl = getattr(ds, ent)
        if "year" in tbl.columns:
            del tbl["year"]
            log(f"dropped year from {ent}")
    ds.save(OUT)
    # No all-zero stored layers, period (the exported_nonzero gate's
    # invariant): an all-zero column either masks a PE formula (a stored
    # input supersedes computation) or is dead scaffolding shadowing the
    # engine's own default. Drop every all-zero numeric/bool column whose
    # PE default is itself zero/False — the artifact then says exactly what
    # it knows and nothing else. Structural id/weight columns are kept.
    from policyengine_us.system import system as _pe

    dropped_masks = []
    kept_zero = []
    for ent in ("person", "household", "tax_unit", "spm_unit", "family",
                "marital_unit"):
        tbl = getattr(ds, ent)
        structural = {f"{ent}_id", f"{ent}_weight"} | {
            c for c in tbl.columns if c.startswith("person_")
        }
        for c in list(tbl.columns):
            if c in structural:
                continue
            vals = tbl[c].to_numpy()
            if vals.dtype.kind not in "fiub" or np.any(vals):
                continue
            var = _pe.variables.get(c)
            default = getattr(var, "default_value", 0) if var is not None else 0
            default_is_zero = (
                default in (0, 0.0, False) or default is None
            )
            if default_is_zero:
                del tbl[c]
                dropped_masks.append(f"{ent}.{c}")
            else:
                # Dropping would CHANGE semantics (engine default != 0):
                # the stored zeros are a real statement. Keep + report.
                kept_zero.append(f"{ent}.{c} (default {default!r})")
    if dropped_masks:
        log(f"dropped {len(dropped_masks)} all-zero columns: {dropped_masks}")
        ds.save(OUT)
    if kept_zero:
        log(f"kept all-zero columns with nonzero engine defaults: {kept_zero}")

    # other_health_insurance_premiums: usdata's decomposition — reported
    # non-Medicare premiums minus the baseline-computed CHIP/marketplace/
    # Medicaid premiums, floored at zero (mirrors derive_other_health_
    # insurance_premiums; runs a baseline sim on the artifact itself).
    from policyengine_us import Microsimulation as _Msim

    _sim = _Msim(dataset=USSingleYearDataset(file_path=OUT))
    # All terms mapped to person grain explicitly — the premium variables
    # live at different entities (person vs tax unit).
    _reported = np.asarray(
        _sim.calculate(
            "health_insurance_premiums_without_medicare_part_b",
            2024,
            map_to="person",
        ).values,
        dtype=np.float64,
    )
    _modeled = sum(
        np.asarray(
            _sim.calculate(_v, 2024, map_to="person").values, dtype=np.float64
        )
        for _v in ("chip_premium", "marketplace_net_premium", "medicaid_premium")
    )
    _other = np.maximum(_reported - _modeled, 0.0)
    _ds2 = USSingleYearDataset(file_path=OUT)
    _ds2.person["other_health_insurance_premiums"] = _other
    _ds2.save(OUT)
    log(
        f"other_health_insurance_premiums decomposed: nz {(_other>0).mean()*100:.1f}%"
    )

    chk = USSingleYearDataset(file_path=OUT)
    assert np.array_equal(
        np.asarray(chk.household["household_weight"], dtype=np.float64), cw
    )
    # No misplaced-entity columns expected in v2 (fixed at export).
    from collections import Counter

    tables = {
        e: getattr(chk, e)
        for e in ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
    }
    cnt = Counter(c for t in tables.values() for c in t.columns)
    dups = {c: n for c, n in cnt.items() if n > 1}
    assert not dups, f"duplicate columns across entities: {dups}"
    log("USSingleYearDataset verified (weights byte-identical, no dup columns)")

    shutil.copy(TP, OUT_TP)
    with h5py.File(OUT_TP, "r+") as f:
        assert f["household_weight/2024"].shape == cw.shape
        f["household_weight/2024"][...] = cw
    with h5py.File(OUT_TP) as f:
        assert np.array_equal(f["household_weight/2024"][:], cw)
    log("timeperiod export verified")

    np.savez_compressed(
        f"{ART}/populace_us_2024_calibration.npz",
        calibrated_weights=cw,
        max_weight_ratio=50.0,
        final_loss=result.final_loss,
        within_10pct=result.fraction_within_10pct,
        epochs=3000,
        learning_rate=0.15,
        seed=0,
    )
    log("BUILD COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
