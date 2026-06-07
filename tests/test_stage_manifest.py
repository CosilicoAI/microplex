from __future__ import annotations

import json

import pytest

from microplex.stage_manifest import (
    SCHEMA_VERSION,
    StageArtifact,
    StageManifest,
    assert_stage_manifest,
    build_stage_manifest,
    file_sha256,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)


def test_build_stage_manifest_records_artifact_hash_size_and_seeds(tmp_path):
    artifact = tmp_path / "stage" / "checkpoint.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"stage-output")

    manifest = build_stage_manifest(
        stage_id="05_donor_integration",
        root=tmp_path,
        artifacts={"checkpoint": "stage/checkpoint.parquet"},
        seeds={"spine": 42},
        parameters={"backend": "regime_aware"},
        metadata={"commit": "abc123"},
        created_at="2026-06-07T00:00:00+00:00",
    )

    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.stage_id == "05_donor_integration"
    assert manifest.seeds == {"spine": 42}
    assert manifest.parameters == {"backend": "regime_aware"}
    assert manifest.metadata == {"commit": "abc123"}
    recorded = manifest.artifacts["checkpoint"]
    assert recorded.path == "stage/checkpoint.parquet"
    assert recorded.size_bytes == len(b"stage-output")
    assert recorded.sha256 == file_sha256(artifact)
    assert validate_stage_manifest(manifest, root=tmp_path) == []


def test_write_and_load_stage_manifest_roundtrip(tmp_path):
    artifact = tmp_path / "data.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    manifest = build_stage_manifest(
        stage_id="source_loading",
        root=tmp_path,
        artifacts={"data": "data.csv"},
        created_at="2026-06-07T00:00:00+00:00",
    )
    manifest_path = tmp_path / "manifest.json"

    write_stage_manifest(manifest_path, manifest)
    loaded = load_stage_manifest(manifest_path)

    assert loaded == manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION


def test_load_stage_manifest_rejects_invalid_schema(tmp_path):
    artifact = tmp_path / "data.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    manifest = build_stage_manifest(
        stage_id="source_loading",
        root=tmp_path,
        artifacts={"data": "data.csv"},
        created_at="2026-06-07T00:00:00+00:00",
    )
    payload = manifest.to_dict()
    payload["schema_version"] = "old"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported_schema_version"):
        load_stage_manifest(manifest_path)


def test_assert_stage_manifest_fails_on_missing_artifact(tmp_path):
    manifest = StageManifest(
        stage_id="post_imputation",
        artifacts={
            "checkpoint": StageArtifact(
                path="missing.parquet",
                sha256="abc",
                size_bytes=123,
            )
        },
        created_at="2026-06-07T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="missing_file"):
        assert_stage_manifest(manifest, root=tmp_path)


def test_assert_stage_manifest_fails_on_changed_artifact(tmp_path):
    artifact = tmp_path / "checkpoint.parquet"
    artifact.write_bytes(b"old")
    manifest = build_stage_manifest(
        stage_id="post_imputation",
        root=tmp_path,
        artifacts={"checkpoint": "checkpoint.parquet"},
        created_at="2026-06-07T00:00:00+00:00",
    )
    artifact.write_bytes(b"new-but-longer")

    errors = validate_stage_manifest(manifest, root=tmp_path)

    assert "checkpoint:sha256_mismatch" in errors
    assert any(error.startswith("checkpoint:size_mismatch:") for error in errors)


def test_rejects_absolute_and_escaping_artifact_paths(tmp_path):
    artifact = tmp_path / "ok.txt"
    artifact.write_text("ok", encoding="utf-8")

    with pytest.raises(ValueError, match="relative"):
        build_stage_manifest(
            stage_id="x",
            root=tmp_path,
            artifacts={"bad": artifact.resolve()},
        )

    with pytest.raises(ValueError, match="escape"):
        build_stage_manifest(
            stage_id="x",
            root=tmp_path,
            artifacts={"bad": "../outside.txt"},
        )


def test_validate_stage_manifest_reports_schema_errors():
    manifest = StageManifest(
        schema_version="old",
        stage_id="",
        artifacts={},
        created_at="2026-06-07T00:00:00+00:00",
    )

    errors = validate_stage_manifest(manifest)

    assert "unsupported_schema_version:'old'" in errors
    assert "missing_stage_id" in errors
    assert "missing_artifacts" in errors
