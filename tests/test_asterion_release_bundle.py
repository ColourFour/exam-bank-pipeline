from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank import asterion_release_bundle as bundle
from exam_bank.topic_routing_artifact import TopicRoutingArtifactError


def _write(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(payload, encoding="utf-8")


def _validation_report() -> dict:
    return {
        "ok": True,
        "warnings": ["report-only warning"],
        "record_counts": {
            "catalog": 3,
            "runtime": 1,
            "content_lab_candidates": 2,
        },
        "runtime_course_counts": {"p3": 1},
        "safety_level_counts": {
            "advisory_topic_filter_ok": {"p1": 1, "p3": 1},
            "learning_runtime_safe": {"p3": 1},
        },
    }


def _sidecar_report() -> dict:
    return {
        "paths": {
            "durable_sidecar": "data/topic_routing/question_bank.topic_routing.v1.json",
            "local_sidecar": "output/json/question_bank.topic_routing.v1.json",
        },
        "sha256": {
            "durable_sidecar": "durable-sha",
            "local_sidecar": "durable-sha",
            "local_matches_durable": True,
        },
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "catalog": tmp_path / "asterion_exam_bank_catalog_v1.json",
        "runtime": tmp_path / "asterion_question_bank_v1.json",
        "content_lab": tmp_path / "asterion_content_lab_candidates_v1.json",
        "validation": tmp_path / "validation.json",
        "expected": tmp_path / "expected.json",
        "output": tmp_path / "manifest.json",
    }


def _write_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = _paths(tmp_path)
    _write(paths["catalog"], {"name": "catalog"})
    _write(paths["runtime"], {"name": "runtime"})
    _write(paths["content_lab"], {"name": "content"})
    _write(paths["validation"], _validation_report())
    monkeypatch.setattr(bundle, "verify_topic_routing_artifact", lambda: _sidecar_report())
    return paths


def test_asterion_release_manifest_creation_includes_all_exports_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)

    manifest = bundle.build_asterion_release_manifest(
        output_path=paths["output"],
        validation_report_path=paths["validation"],
        expected_provenance_path=None,
        catalog_path=paths["catalog"],
        runtime_path=paths["runtime"],
        content_lab_path=paths["content_lab"],
        generated_at="2026-06-12T00:00:00+00:00",
    )

    assert paths["output"].exists()
    assert set(manifest["export_artifacts"]) == {"catalog", "student_runtime", "content_lab_candidates"}
    assert manifest["counts"]["catalog_records"] == 3
    assert manifest["counts"]["student_runtime_records"] == 1
    assert manifest["counts"]["p3_runtime_records"] == 1
    assert manifest["counts"]["non_p3_runtime_records"] == 0
    assert manifest["validation"]["ok"] is True


def test_asterion_release_manifest_detects_expected_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)
    _write(
        paths["expected"],
        {
            "durable_sidecar": {"sha256": "durable-sha"},
            "export_artifacts": {
                "catalog": {"sha256": "not-the-current-catalog-sha"},
            },
        },
    )

    with pytest.raises(bundle.AsterionReleaseBundleError, match="catalog SHA-256"):
        bundle.build_asterion_release_manifest(
            output_path=paths["output"],
            validation_report_path=paths["validation"],
            expected_provenance_path=paths["expected"],
            catalog_path=paths["catalog"],
            runtime_path=paths["runtime"],
            content_lab_path=paths["content_lab"],
        )


def test_asterion_release_manifest_refresh_mode_allows_expected_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)
    _write(
        paths["expected"],
        {
            "durable_sidecar": {"sha256": "durable-sha"},
            "export_artifacts": {
                "catalog": {"sha256": "not-the-current-catalog-sha"},
            },
        },
    )

    manifest = bundle.build_asterion_release_manifest(
        output_path=paths["output"],
        validation_report_path=paths["validation"],
        expected_provenance_path=paths["expected"],
        catalog_path=paths["catalog"],
        runtime_path=paths["runtime"],
        content_lab_path=paths["content_lab"],
        refresh_expected=True,
    )

    assert manifest["export_artifacts"]["catalog"]["sha256"] != "not-the-current-catalog-sha"


def test_asterion_release_manifest_fails_on_missing_export_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)
    paths["runtime"].unlink()

    with pytest.raises(bundle.AsterionReleaseBundleError, match="Missing release artifact"):
        bundle.build_asterion_release_manifest(
            output_path=paths["output"],
            validation_report_path=paths["validation"],
            expected_provenance_path=None,
            catalog_path=paths["catalog"],
            runtime_path=paths["runtime"],
            content_lab_path=paths["content_lab"],
        )


def test_asterion_release_manifest_fails_on_sidecar_provenance_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)

    def fail_sidecar() -> dict:
        raise TopicRoutingArtifactError("sidecar mismatch")

    monkeypatch.setattr(bundle, "verify_topic_routing_artifact", fail_sidecar)

    with pytest.raises(TopicRoutingArtifactError, match="sidecar mismatch"):
        bundle.build_asterion_release_manifest(
            output_path=paths["output"],
            validation_report_path=paths["validation"],
            expected_provenance_path=None,
            catalog_path=paths["catalog"],
            runtime_path=paths["runtime"],
            content_lab_path=paths["content_lab"],
        )


def test_asterion_release_manifest_verifies_export_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)
    manifest = bundle.build_asterion_release_manifest(
        output_path=paths["output"],
        validation_report_path=paths["validation"],
        expected_provenance_path=None,
        catalog_path=paths["catalog"],
        runtime_path=paths["runtime"],
        content_lab_path=paths["content_lab"],
    )

    report = bundle.verify_asterion_release_manifest(
        manifest_path=paths["output"],
        exports_dir=tmp_path,
    )

    assert report["ok"] is True
    assert set(report["verified_artifacts"]) == set(manifest["export_artifacts"])


def test_asterion_release_manifest_verify_fails_for_missing_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path, monkeypatch)
    bundle.build_asterion_release_manifest(
        output_path=paths["output"],
        validation_report_path=paths["validation"],
        expected_provenance_path=None,
        catalog_path=paths["catalog"],
        runtime_path=paths["runtime"],
        content_lab_path=paths["content_lab"],
    )
    paths["content_lab"].unlink()

    with pytest.raises(bundle.AsterionReleaseBundleError, match="Missing export artifact"):
        bundle.verify_asterion_release_manifest(
            manifest_path=paths["output"],
            exports_dir=tmp_path,
        )
