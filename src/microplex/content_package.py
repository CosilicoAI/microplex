"""Generic checks for declarative Microplex content packages.

Country packages are allowed to ship specs, manifests, fixtures, and docs.
Executable behavior belongs in Microplex, microimpute, and microcalibrate. This
module provides the small generic inspection surface that a content package can
use in CI without defining its own Python modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from microplex.spec import load_spec

__all__ = [
    "SpecVariableManifestDiff",
    "ContentPackageCheckResult",
    "compute_spec_variable_manifest_diff",
    "find_runtime_python_files",
    "load_json_resource",
    "load_text_resource",
    "run_content_package_checks",
    "main",
]


@dataclass(frozen=True)
class SpecVariableManifestDiff:
    """Coverage diff between spec variables and the release contract."""

    spec: str
    contract: str
    required_contract_count: int
    declared_imputation_count: int
    variable_manifest_count: int
    missing_required: list[str]
    missing_declared_imputation: list[str]
    extra_variables: list[str]

    @property
    def ok(self) -> bool:
        """Whether ``variables`` exactly covers required + imputed variables."""
        return not (
            self.missing_required
            or self.missing_declared_imputation
            or self.extra_variables
        )


@dataclass(frozen=True)
class ContentPackageCheckResult:
    """Complete generic content-package check result."""

    package: str
    spec_resource: str
    contract_resource: str
    spec_loads: bool
    manifest_diff: SpecVariableManifestDiff
    runtime_python_files: list[str]

    @property
    def ok(self) -> bool:
        """Whether every content-package gate passed."""
        return (
            self.spec_loads
            and self.manifest_diff.ok
            and not self.runtime_python_files
        )


def load_text_resource(package: str, resource: str) -> str:
    """Load a UTF-8 text resource from a package or namespace package."""
    return files(package).joinpath(resource).read_text(encoding="utf-8")


def load_json_resource(package: str, resource: str) -> dict[str, Any]:
    """Load a JSON object resource from a package or namespace package."""
    payload = json.loads(load_text_resource(package, resource))
    if not isinstance(payload, dict):
        raise ValueError(f"{package}:{resource} must contain a JSON object.")
    return payload


def compute_spec_variable_manifest_diff(
    *,
    spec_text: str,
    contract: dict[str, Any],
    spec_label: str = "<spec>",
    contract_label: str = "<contract>",
) -> SpecVariableManifestDiff:
    """Compare ``spec.variables`` with required exports and declared imputations."""
    spec = load_spec_from_text(spec_text, label=spec_label)
    variables = set(spec.variables)
    if not variables:
        raise ValueError(f"Spec {spec_label} is missing a variables mapping.")

    required = {str(column) for column in contract["required"]}
    declared_imputation = {var for step in spec.imputation for var in step.vars}
    expected = required | declared_imputation
    return SpecVariableManifestDiff(
        spec=spec_label,
        contract=contract_label,
        required_contract_count=len(required),
        declared_imputation_count=len(declared_imputation),
        variable_manifest_count=len(variables),
        missing_required=sorted(required - variables),
        missing_declared_imputation=sorted(declared_imputation - variables),
        extra_variables=sorted(variables - expected),
    )


def find_runtime_python_files(root: Path) -> list[str]:
    """Return Python files under a content package's runtime ``src`` tree."""
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.is_file()
    )


def run_content_package_checks(
    *,
    package: str,
    spec_resource: str,
    contract_resource: str,
    src_root: Path | None = None,
) -> ContentPackageCheckResult:
    """Run generic checks for a declarative content package."""
    spec_text = load_text_resource(package, spec_resource)
    contract = load_json_resource(package, contract_resource)
    manifest_diff = compute_spec_variable_manifest_diff(
        spec_text=spec_text,
        contract=contract,
        spec_label=f"{package}:{spec_resource}",
        contract_label=f"{package}:{contract_resource}",
    )
    runtime_python_files = (
        [] if src_root is None else find_runtime_python_files(src_root)
    )
    return ContentPackageCheckResult(
        package=package,
        spec_resource=spec_resource,
        contract_resource=contract_resource,
        spec_loads=True,
        manifest_diff=manifest_diff,
        runtime_python_files=runtime_python_files,
    )


def load_spec_from_text(spec_text: str, *, label: str):
    """Load a spec from text without requiring a country package Python module."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        delete=False,
    ) as handle:
        handle.write(spec_text)
        path = Path(handle.name)
    try:
        return load_spec(path)
    finally:
        path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a declarative Microplex content package."
    )
    parser.add_argument("--package", required=True, help="Import package name.")
    parser.add_argument("--spec", required=True, help="Spec resource path.")
    parser.add_argument("--contract", required=True, help="Contract JSON resource path.")
    parser.add_argument(
        "--src-root",
        type=Path,
        help="Optional runtime source root that must contain no Python files.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for content-package checks."""
    args = _build_parser().parse_args(argv)
    result = run_content_package_checks(
        package=args.package,
        spec_resource=args.spec,
        contract_resource=args.contract,
        src_root=args.src_root,
    )
    payload = asdict(result)
    payload["ok"] = result.ok
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif result.ok:
        print(
            f"{args.package}: content package checks passed "
            f"({result.manifest_diff.variable_manifest_count} variables)"
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
