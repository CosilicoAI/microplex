from __future__ import annotations

import json
from pathlib import Path

from microplex.content_package import run_content_package_checks
from microplex.spec import CalibrationMethod, load_spec

ROOT = Path(__file__).parents[1]
US_PACK_ROOT = ROOT / "packs" / "us"
US_SPEC = US_PACK_ROOT / "specs" / "us-2024.yaml"
US_CONTRACT = US_PACK_ROOT / "manifests" / "ecps_export_contract.json"
EXPECTED_US_PACK_FILES = {
    "README.md",
    "manifests/ecps_export_contract.json",
    "manifests/frozen_production_ecps_2024_benchmark_manifest.json",
    "manifests/pe_source_impute_blocks.json",
    "manifests/puf.json",
    "specs/us-2024.yaml",
}


def test_us_pack_file_set_is_explicit() -> None:
    actual_files = {
        path.relative_to(US_PACK_ROOT).as_posix()
        for path in US_PACK_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual_files == EXPECTED_US_PACK_FILES


def test_us_pack_contains_only_content_files() -> None:
    python_files = sorted(
        path.relative_to(US_PACK_ROOT).as_posix()
        for path in US_PACK_ROOT.rglob("*.py")
        if path.is_file()
    )

    assert python_files == []


def test_us_pack_manifests_are_json_objects() -> None:
    for manifest in sorted((US_PACK_ROOT / "manifests").glob("*.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), manifest


def test_us_pack_passes_generic_content_package_checks() -> None:
    result = run_content_package_checks(
        root=US_PACK_ROOT,
        spec_resource="specs/us-2024.yaml",
        contract_resource="manifests/ecps_export_contract.json",
    )

    assert result.ok
    assert result.root == str(US_PACK_ROOT.resolve())
    assert result.runtime_python_files == []
    assert result.manifest_diff.required_contract_count > 0
    assert result.manifest_diff.variable_manifest_count >= (
        result.manifest_diff.required_contract_count
    )


def test_us_pack_calibration_stays_apg_without_l0_target_records() -> None:
    spec = load_spec(US_SPEC)

    assert spec.calibrate is not None
    assert spec.calibrate.method is CalibrationMethod.APG
    assert spec.calibrate.target_records is None


def test_us_pack_frozen_benchmark_disables_l0() -> None:
    benchmark_manifest = json.loads(
        (
            US_PACK_ROOT
            / "manifests"
            / "frozen_production_ecps_2024_benchmark_manifest.json"
        ).read_text(encoding="utf-8")
    )

    refit_config = benchmark_manifest["scoring_config"]["refit_config"]
    assert refit_config["lambda_l0"] == 0.0
    assert refit_config["use_gates"] is False
