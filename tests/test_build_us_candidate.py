from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]


def _load_build_driver():
    spec = importlib.util.spec_from_file_location(
        "build_us_candidate",
        ROOT / "scripts" / "build_us_candidate.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attach_filing_status_inputs_sets_policyengine_status_controls() -> None:
    driver = _load_build_driver()
    person = pd.DataFrame(
        {
            "new_tax_unit_id": [10, 10, 20, 30],
            "tax_unit_role_input": ["HEAD", "DEPENDENT", "HEAD", "HEAD"],
            "is_separated": [False, False, False, True],
        }
    )
    spine = pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 30],
            "filing_status_input": ["SURVIVING_SPOUSE", "SEPARATE", "JOINT"],
        }
    )

    result = driver._attach_filing_status_inputs(person, spine)

    assert result["is_surviving_spouse"].tolist() == [True, False, False, False]
    assert result["is_separated"].tolist() == [False, False, True, True]


def test_attach_filing_status_inputs_requires_status_surface() -> None:
    driver = _load_build_driver()
    person = pd.DataFrame(
        {
            "new_tax_unit_id": [10],
            "tax_unit_role_input": ["HEAD"],
        }
    )

    try:
        driver._attach_filing_status_inputs(
            person,
            pd.DataFrame({"tax_unit_id": [10]}),
        )
    except ValueError as exc:
        assert "filing_status_input" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing filing_status_input should fail")
