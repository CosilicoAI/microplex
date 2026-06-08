from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from microplex.content_package import (
    compute_spec_variable_manifest_diff,
    find_runtime_python_files,
)


def _env_with_path(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(path) if not existing else f"{path}:{existing}"
    return env


def _spec_text() -> str:
    return """
meta:
  country: us
  model_year: 2024
sources:
  cps_asec: { dataset: cps, role: spine }
  puf: { dataset: puf, role: donor }
spine:
  base: cps_asec
  method: support_spine
  support: { seed: 42 }
  halves:
    - { name: cps_keep, keep: all }
    - { name: synthetic_puf, strip_to: [demographics] }
imputation:
  - onto: synthetic_puf
    from: puf
    vars:
      - employment_income
      - "quoted_income"  # keep inline comment handling
      - 'single_quoted_income'
    condition_on: [demographics]
  - onto: cps_keep
    from: puf
    vars: [inline_income, "inline_quoted_income"]  # inline list form
variables:
  required_export:
    entity: person
    role: passthrough
    mp_spec: { method: passthrough, operation: { kind: passthrough } }
  employment_income:
    entity: person
    role: impute
    mp_spec: { method: impute, operation: { kind: impute } }
  quoted_income:
    entity: person
    role: impute
    mp_spec: { method: impute, operation: { kind: impute } }
  single_quoted_income:
    entity: person
    role: impute
    mp_spec: { method: impute, operation: { kind: impute } }
  inline_income:
    entity: person
    role: impute
    mp_spec: { method: impute, operation: { kind: impute } }
  inline_quoted_income:
    entity: person
    role: impute
    mp_spec: { method: impute, operation: { kind: impute } }
"""


def test_spec_variable_manifest_diff_covers_required_and_imputed_vars() -> None:
    diff = compute_spec_variable_manifest_diff(
        spec_text=_spec_text(),
        contract={"required": ["required_export"]},
    )

    assert diff.ok
    assert diff.required_contract_count == 1
    assert diff.declared_imputation_count == 5
    assert diff.variable_manifest_count == 6


def test_spec_variable_manifest_diff_uses_loaded_yaml_model() -> None:
    spec_text = """
meta:
    country: us
    model_year: 2024
sources:
    cps_asec: { dataset: cps, role: spine }
    puf: { dataset: puf, role: donor }
spine:
    base: cps_asec
    method: support_spine
    support: { seed: 42 }
    halves:
        - { name: cps_keep, keep: all }
        - { name: synthetic_puf, strip_to: [demographics] }
imputation:
    - onto: synthetic_puf
      from: puf
      vars:
          - employment_income
      condition_on: [demographics]
variables:
    required_export:
        entity: person
        role: passthrough
        mp_spec: { method: passthrough, operation: { kind: passthrough } }
    employment_income:
        entity: person
        role: impute
        mp_spec: { method: impute, operation: { kind: impute } }
"""

    diff = compute_spec_variable_manifest_diff(
        spec_text=spec_text,
        contract={"required": ["required_export"]},
    )

    assert diff.ok
    assert diff.declared_imputation_count == 1
    assert diff.variable_manifest_count == 2


def test_spec_variable_manifest_diff_reports_missing_and_extra_vars() -> None:
    diff = compute_spec_variable_manifest_diff(
        spec_text=_spec_text().replace("  inline_income:\n", "  extra_income:\n"),
        contract={"required": ["required_export"]},
    )

    assert not diff.ok
    assert diff.missing_declared_imputation == ["inline_income"]
    assert diff.extra_variables == ["extra_income"]


def test_spec_variable_manifest_diff_rejects_forbidden_export_variables() -> None:
    diff = compute_spec_variable_manifest_diff(
        spec_text=_spec_text(),
        contract={"required": ["required_export"], "forbidden": ["employment_income"]},
    )

    assert not diff.ok
    assert diff.forbidden_contract_count == 1
    assert diff.forbidden_without_non_export_role == ["employment_income"]


def test_spec_variable_manifest_diff_allows_internal_forbidden_variables() -> None:
    spec_text = _spec_text().replace(
        "  employment_income:\n    entity: person\n    role: impute\n",
        "  employment_income:\n    entity: person\n    role: puf_imputed_non_export\n",
    )
    diff = compute_spec_variable_manifest_diff(
        spec_text=spec_text,
        contract={"required": ["required_export"], "forbidden": ["employment_income"]},
    )

    assert diff.ok
    assert diff.forbidden_contract_count == 1
    assert diff.forbidden_without_non_export_role == []


def test_find_runtime_python_files_reports_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "src" / "country_pack"
    (root / "specs").mkdir(parents=True)
    (root / "specs" / "us-2024.yaml").write_text("meta: {}\n")
    (root / "helpers.py").write_text("print('runtime code')\n")
    (root / "nested").mkdir()
    (root / "nested" / "__init__.py").write_text("")

    assert find_runtime_python_files(root) == ["helpers.py", "nested/__init__.py"]


def test_find_runtime_python_files_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Runtime source root"):
        find_runtime_python_files(tmp_path / "missing")


def test_cli_checks_namespace_package_resources(tmp_path: Path) -> None:
    package_root = tmp_path / "sample_pack"
    (package_root / "specs").mkdir(parents=True)
    (package_root / "manifests").mkdir()
    (package_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (package_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex.content_package",
            "--package",
            "sample_pack",
            "--spec",
            "specs/us-2024.yaml",
            "--contract",
            "manifests/contract.json",
            "--src-root",
            str(package_root),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_env_with_path(tmp_path),
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["manifest_diff"]["variable_manifest_count"] == 6


def test_package_cli_requires_src_root_for_python_scan(tmp_path: Path) -> None:
    package_root = tmp_path / "sample_pack"
    (package_root / "specs").mkdir(parents=True)
    (package_root / "manifests").mkdir()
    (package_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (package_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex.content_package",
            "--package",
            "sample_pack",
            "--spec",
            "specs/us-2024.yaml",
            "--contract",
            "manifests/contract.json",
        ],
        capture_output=True,
        text=True,
        env=_env_with_path(tmp_path),
    )

    assert result.returncode == 1
    assert "Package checks require --src-root" in result.stderr


def test_cli_checks_filesystem_content_root(tmp_path: Path) -> None:
    pack_root = tmp_path / "packs" / "us"
    (pack_root / "specs").mkdir(parents=True)
    (pack_root / "manifests").mkdir()
    (pack_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (pack_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex.content_package",
            "--root",
            str(pack_root),
            "--spec",
            "specs/us-2024.yaml",
            "--contract",
            "manifests/contract.json",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_env_with_path(Path(__file__).parents[1] / "src"),
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["root"] == str(pack_root.resolve())
    assert payload["runtime_python_files"] == []


def test_top_level_import_does_not_eagerly_import_heavy_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, microplex; "
                "print('microplex.calibration' in sys.modules); "
                "print('microplex.synthesizer' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_env_with_path(Path(__file__).parents[1] / "src"),
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_cli_fails_when_content_package_has_runtime_python(tmp_path: Path) -> None:
    package_root = tmp_path / "sample_pack"
    (package_root / "specs").mkdir(parents=True)
    (package_root / "manifests").mkdir()
    (package_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (package_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )
    (package_root / "runtime.py").write_text("print('nope')\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex.content_package",
            "--package",
            "sample_pack",
            "--spec",
            "specs/us-2024.yaml",
            "--contract",
            "manifests/contract.json",
            "--src-root",
            str(package_root),
        ],
        capture_output=True,
        text=True,
        env=_env_with_path(tmp_path),
    )

    assert result.returncode == 1
    assert "runtime.py" in result.stderr


def test_root_cli_fails_when_content_package_has_python(tmp_path: Path) -> None:
    pack_root = tmp_path / "packs" / "us"
    (pack_root / "specs").mkdir(parents=True)
    (pack_root / "manifests").mkdir()
    (pack_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (pack_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )
    (pack_root / "runtime.py").write_text("print('nope')\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex.content_package",
            "--root",
            str(pack_root),
            "--spec",
            "specs/us-2024.yaml",
            "--contract",
            "manifests/contract.json",
        ],
        capture_output=True,
        text=True,
        env=_env_with_path(Path(__file__).parents[1] / "src"),
    )

    assert result.returncode == 1
    assert "runtime.py" in result.stderr


def test_root_cli_rejects_resource_paths_outside_content_root(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "packs" / "us"
    (pack_root / "specs").mkdir(parents=True)
    (pack_root / "manifests").mkdir()
    (pack_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (pack_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )

    for bad_spec in (
        str(pack_root / "specs" / "us-2024.yaml"),
        "specs/../specs/us-2024.yaml",
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "microplex.content_package",
                "--root",
                str(pack_root),
                "--spec",
                bad_spec,
                "--contract",
                "manifests/contract.json",
            ],
            capture_output=True,
            text=True,
            env=_env_with_path(Path(__file__).parents[1] / "src"),
        )

        assert result.returncode == 1
        assert "Resource path must stay inside the content root" in result.stderr


def test_root_cli_fails_when_explicit_src_root_is_missing(tmp_path: Path) -> None:
    pack_root = tmp_path / "packs" / "us"
    (pack_root / "specs").mkdir(parents=True)
    (pack_root / "manifests").mkdir()
    (pack_root / "specs" / "us-2024.yaml").write_text(_spec_text())
    (pack_root / "manifests" / "contract.json").write_text(
        json.dumps({"required": ["required_export"], "forbidden": []})
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "microplex.content_package",
            "--root",
            str(pack_root),
            "--spec",
            "specs/us-2024.yaml",
            "--contract",
            "manifests/contract.json",
            "--src-root",
            str(pack_root / "missing"),
        ],
        capture_output=True,
        text=True,
        env=_env_with_path(Path(__file__).parents[1] / "src"),
    )

    assert result.returncode == 1
    assert "Runtime source root must be a directory" in result.stderr


def test_spec_variable_manifest_diff_requires_variables_section() -> None:
    with pytest.raises(ValueError, match="missing a variables mapping"):
        compute_spec_variable_manifest_diff(
            spec_text=_spec_text().split("variables:")[0],
            contract={"required": ["required_export"]},
        )
