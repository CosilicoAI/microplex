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
    "load_json_content",
    "load_text_content",
    "run_content_package_checks",
    "main",
]


@dataclass(frozen=True)
class SpecVariableManifestDiff:
    """Coverage diff between spec variables and the release contract."""

    spec: str
    contract: str
    required_contract_count: int
    forbidden_contract_count: int
    declared_imputation_count: int
    variable_manifest_count: int
    missing_required: list[str]
    missing_declared_imputation: list[str]
    extra_variables: list[str]
    forbidden_required: list[str]
    forbidden_without_non_export_role: list[str]

    @property
    def ok(self) -> bool:
        """Whether ``variables`` exactly covers required + imputed variables."""
        return not (
            self.missing_required
            or self.missing_declared_imputation
            or self.extra_variables
            or self.forbidden_required
            or self.forbidden_without_non_export_role
        )


@dataclass(frozen=True)
class ContentPackageCheckResult:
    """Complete generic content-package check result."""

    package: str
    root: str | None
    spec_resource: str
    contract_resource: str
    spec_loads: bool
    manifest_diff: SpecVariableManifestDiff
    runtime_python_files: list[str]

    @property
    def ok(self) -> bool:
        """Whether every content-package gate passed."""
        return (
            self.spec_loads and self.manifest_diff.ok and not self.runtime_python_files
        )


def load_text_resource(package: str, resource: str) -> str:
    """Load a UTF-8 text resource from a package or namespace package."""
    _validate_relative_resource(resource)
    return files(package).joinpath(resource).read_text(encoding="utf-8")


def load_json_resource(package: str, resource: str) -> dict[str, Any]:
    """Load a JSON object resource from a package or namespace package."""
    payload = json.loads(load_text_resource(package, resource))
    if not isinstance(payload, dict):
        raise ValueError(f"{package}:{resource} must contain a JSON object.")
    return payload


def load_text_content(
    *,
    package: str | None,
    root: Path | None,
    resource: str,
) -> str:
    """Load a UTF-8 text resource from a package or content root."""
    if root is not None:
        return _resolve_content_resource(root, resource).read_text(encoding="utf-8")
    if package is None:
        raise ValueError("Either package or root is required.")
    return load_text_resource(package, resource)


def load_json_content(
    *,
    package: str | None,
    root: Path | None,
    resource: str,
) -> dict[str, Any]:
    """Load a JSON object from a package or content root."""
    payload = json.loads(
        load_text_content(package=package, root=root, resource=resource)
    )
    if not isinstance(payload, dict):
        label = f"{root}:{resource}" if root is not None else f"{package}:{resource}"
        raise ValueError(f"{label} must contain a JSON object.")
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
    forbidden = {str(column) for column in contract.get("forbidden", [])}
    declared_imputation = {var for step in spec.imputation for var in step.vars}
    expected = required | declared_imputation
    forbidden_declared = forbidden & variables
    forbidden_without_non_export_role = {
        variable
        for variable in forbidden_declared
        if "non_export" not in (spec.variables[variable].role or "")
    }
    return SpecVariableManifestDiff(
        spec=spec_label,
        contract=contract_label,
        required_contract_count=len(required),
        forbidden_contract_count=len(forbidden),
        declared_imputation_count=len(declared_imputation),
        variable_manifest_count=len(variables),
        missing_required=sorted(required - variables),
        missing_declared_imputation=sorted(declared_imputation - variables),
        extra_variables=sorted(variables - expected),
        forbidden_required=sorted(required & forbidden),
        forbidden_without_non_export_role=sorted(forbidden_without_non_export_role),
    )


def find_runtime_python_files(root: Path) -> list[str]:
    """Return Python files under a content package's runtime ``src`` tree."""
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Runtime source root must be a directory: {root}")
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*.py") if path.is_file()
    )


def run_content_package_checks(
    *,
    spec_resource: str,
    contract_resource: str,
    package: str | None = None,
    root: Path | None = None,
    src_root: Path | None = None,
) -> ContentPackageCheckResult:
    """Run generic checks for a declarative content package."""
    if package is None and root is None:
        raise ValueError("Either package or root is required.")
    if package is not None and root is not None:
        raise ValueError("Only one of package or root may be supplied.")
    if package is not None and src_root is None:
        raise ValueError("Package checks require --src-root to scan for Python files.")
    root = root.resolve() if root is not None else None
    if root is not None and not root.is_dir():
        raise ValueError(f"Content root must be a directory: {root}")
    package_label = package or str(root)
    spec_text = load_text_content(
        package=package,
        root=root,
        resource=spec_resource,
    )
    contract = load_json_content(
        package=package,
        root=root,
        resource=contract_resource,
    )
    manifest_diff = compute_spec_variable_manifest_diff(
        spec_text=spec_text,
        contract=contract,
        spec_label=f"{package_label}:{spec_resource}",
        contract_label=f"{package_label}:{contract_resource}",
    )
    if root is not None and src_root is None:
        src_root = root
    if src_root is not None:
        src_root = src_root.resolve()
    runtime_python_files = (
        [] if src_root is None else find_runtime_python_files(src_root)
    )
    return ContentPackageCheckResult(
        package=package_label,
        root=str(root) if root is not None else None,
        spec_resource=spec_resource,
        contract_resource=contract_resource,
        spec_loads=True,
        manifest_diff=manifest_diff,
        runtime_python_files=runtime_python_files,
    )


def _validate_relative_resource(resource: str) -> None:
    """Reject resource paths that can escape a package/content root."""
    path = Path(resource)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Resource path must stay inside the content root: {resource}")


def _resolve_content_resource(root: Path, resource: str) -> Path:
    """Resolve a resource under ``root`` and fail if it escapes."""
    _validate_relative_resource(resource)
    path = (root / resource).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Resource path must stay inside the content root: {resource}")
    return path


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", help="Import package name.")
    source.add_argument("--root", type=Path, help="Filesystem content-pack root.")
    parser.add_argument("--spec", required=True, help="Spec resource path.")
    parser.add_argument(
        "--contract", required=True, help="Contract JSON resource path."
    )
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
        root=args.root,
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
            f"{result.package}: content package checks passed "
            f"({result.manifest_diff.variable_manifest_count} variables)"
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
