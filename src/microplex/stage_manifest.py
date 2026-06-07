"""Strict stage artifact manifests for resumable spec-driven builds."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

SCHEMA_VERSION = "microplex-stage-manifest/v1"

__all__ = [
    "SCHEMA_VERSION",
    "StageArtifact",
    "StageManifest",
    "assert_stage_manifest",
    "build_stage_manifest",
    "file_sha256",
    "load_stage_manifest",
    "validate_stage_manifest",
    "write_stage_manifest",
]


@dataclass(frozen=True)
class StageArtifact:
    """One file produced by a build stage."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageArtifact:
        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
        )


@dataclass(frozen=True)
class StageManifest:
    """Manifest for a single completed stage.

    The manifest is deliberately stage-local. Pipeline-level manifests can
    compose these later, but each stage must be self-verifying before resume.
    """

    stage_id: str
    artifacts: dict[str, StageArtifact]
    seeds: dict[str, int] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "created_at": self.created_at,
            "seeds": dict(self.seeds),
            "parameters": self.parameters,
            "metadata": self.metadata,
            "artifacts": {
                name: artifact.to_dict()
                for name, artifact in sorted(self.artifacts.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageManifest:
        artifacts = {
            str(name): StageArtifact.from_dict(value)
            for name, value in dict(payload["artifacts"]).items()
        }
        return cls(
            schema_version=str(payload["schema_version"]),
            stage_id=str(payload["stage_id"]),
            created_at=str(payload["created_at"]),
            seeds={
                str(name): int(value) for name, value in dict(payload["seeds"]).items()
            },
            parameters=dict(payload.get("parameters", {})),
            metadata=dict(payload.get("metadata", {})),
            artifacts=artifacts,
        )


def build_stage_manifest(
    *,
    stage_id: str,
    root: str | Path,
    artifacts: Mapping[str, str | Path],
    seeds: Mapping[str, int] | None = None,
    parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> StageManifest:
    """Build a manifest from existing artifact files under ``root``."""
    if not stage_id:
        raise ValueError("stage_id must be non-empty.")
    root_path = Path(root)
    resolved_artifacts: dict[str, StageArtifact] = {}
    for name, artifact_path in artifacts.items():
        if not name:
            raise ValueError("artifact names must be non-empty.")
        relative_path = _normalize_relative_path(artifact_path)
        full_path = root_path / relative_path
        if not full_path.is_file():
            raise FileNotFoundError(f"artifact {name!r} does not exist: {full_path}")
        stat = full_path.stat()
        resolved_artifacts[str(name)] = StageArtifact(
            path=relative_path,
            sha256=file_sha256(full_path),
            size_bytes=int(stat.st_size),
        )

    kwargs: dict[str, Any] = {}
    if created_at is not None:
        kwargs["created_at"] = created_at
    return StageManifest(
        stage_id=stage_id,
        artifacts=resolved_artifacts,
        seeds={str(name): int(value) for name, value in dict(seeds or {}).items()},
        parameters=dict(parameters or {}),
        metadata=dict(metadata or {}),
        **kwargs,
    )


def write_stage_manifest(path: str | Path, manifest: StageManifest) -> None:
    """Write a stage manifest atomically as stable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n"
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def load_stage_manifest(path: str | Path) -> StageManifest:
    """Load and validate a saved stage manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = StageManifest.from_dict(payload)
    assert_stage_manifest(manifest)
    return manifest


def validate_stage_manifest(
    manifest: StageManifest | Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> list[str]:
    """Return validation errors for schema and optional artifact checks."""
    if isinstance(manifest, Mapping):
        try:
            manifest = StageManifest.from_dict(manifest)
        except Exception as exc:
            return [f"invalid_manifest_payload:{exc}"]

    errors: list[str] = []
    if manifest.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported_schema_version:{manifest.schema_version!r}")
    if not manifest.stage_id:
        errors.append("missing_stage_id")
    if not manifest.artifacts:
        errors.append("missing_artifacts")

    seen_paths: set[str] = set()
    for artifact_name, artifact in manifest.artifacts.items():
        if not artifact_name:
            errors.append("empty_artifact_name")
        try:
            relative_path = _normalize_relative_path(artifact.path)
        except ValueError as exc:
            errors.append(f"{artifact_name}:invalid_path:{exc}")
            continue
        if relative_path in seen_paths:
            errors.append(f"{artifact_name}:duplicate_path:{relative_path}")
        seen_paths.add(relative_path)
        if artifact.sha256 == "":
            errors.append(f"{artifact_name}:missing_sha256")
        if artifact.size_bytes < 0:
            errors.append(f"{artifact_name}:negative_size")
        if root is None:
            continue
        full_path = Path(root) / relative_path
        if not full_path.is_file():
            errors.append(f"{artifact_name}:missing_file:{relative_path}")
            continue
        stat = full_path.stat()
        if stat.st_size != artifact.size_bytes:
            errors.append(
                f"{artifact_name}:size_mismatch:{stat.st_size}!={artifact.size_bytes}"
            )
        actual_sha = file_sha256(full_path)
        if actual_sha != artifact.sha256:
            errors.append(f"{artifact_name}:sha256_mismatch")
    return errors


def assert_stage_manifest(
    manifest: StageManifest | Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> None:
    """Raise if a stage manifest is not valid."""
    errors = validate_stage_manifest(manifest, root=root)
    if errors:
        raise ValueError("Invalid stage manifest: " + "; ".join(errors))


def file_sha256(path: str | Path) -> str:
    """Return a file's SHA-256 hex digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative_path(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("artifact paths must be relative to the stage root.")
    if any(part == ".." for part in candidate.parts):
        raise ValueError("artifact paths may not escape the stage root.")
    if str(candidate) in {"", "."}:
        raise ValueError("artifact paths must name a file.")
    return candidate.as_posix()
