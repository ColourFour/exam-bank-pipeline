from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank import topic_routing_artifact as artifact


HASH = "a" * 64


def _question_bank(question_ids: list[str]) -> dict:
    return {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": len(question_ids),
        "questions": [{"question_id": question_id} for question_id in question_ids],
    }


def _route(*, review_required: bool = False, evidence_hash: str = HASH) -> dict:
    return {
        "primary_topic_id": None if review_required else "9709_p1_topic_series",
        "topic_distribution": [] if review_required else [{"topic_id": "9709_p1_topic_series", "fit_percent": 100}],
        "confidence": "low" if review_required else "high",
        "review_required": review_required,
        "review_reasons": ["needs review"] if review_required else [],
        "evidence_used": ["question_text"],
        "routing_source": "test",
        "paper_family": "p1",
        "evidence_packet_hash": evidence_hash,
    }


def _sidecar(records: dict[str, dict]) -> dict:
    return {
        "schema_name": "exam_bank.topic_routing_sidecar",
        "schema_version": 1,
        "record_count": len(records),
        "records": records,
        "metadata": {"run_summary": {"safe_for_strict_filters": True}},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sha(path: Path, payload_path: Path) -> str:
    digest = artifact.file_sha256(payload_path)
    path.write_text(f"{digest}  {payload_path.name}\n", encoding="utf-8")
    return digest


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return (
        tmp_path / "question_bank.json",
        tmp_path / "local" / "question_bank.topic_routing.v1.json",
        tmp_path / "data" / "question_bank.topic_routing.v1.json",
        tmp_path / "data" / "question_bank.topic_routing.v1.sha256",
    )


def _release_manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifests" / "question_bank_release_manifest.v1.json"


def _write_valid_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _paths(tmp_path)
    payload = _sidecar(
        {
            "q1": _route(),
            "q2": _route(),
            "q3": _route(review_required=True),
        }
    )
    _write_json(question_bank_path, _question_bank(["q1", "q2", "q3"]))
    _write_json(durable_sidecar_path, payload)
    _write_json(local_sidecar_path, payload)
    _write_sha(durable_sha_path, durable_sidecar_path)
    artifact.build_topic_routing_release_manifest(
        question_bank_path=question_bank_path,
        durable_sidecar_path=durable_sidecar_path,
        release_manifest_path=_release_manifest_path(tmp_path),
        base_dir=tmp_path,
    )
    return question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path


def test_topic_routing_artifact_verifies_matching_checksum_and_counts(tmp_path: Path) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _write_valid_artifacts(tmp_path)

    report = artifact.verify_topic_routing_artifact(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha_path,
        release_manifest_path=_release_manifest_path(tmp_path),
    )

    assert report["ok"] is True
    assert report["sha256"]["local_matches_durable"] is True
    assert report["counts"]["records"] == 3
    assert report["counts"]["review_required"] == 1
    assert report["counts"]["strict_filter_candidates"] == 2
    assert report["id_coverage"]["missing_count"] == 0


def test_topic_routing_artifact_detects_local_durable_mismatch(tmp_path: Path) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _write_valid_artifacts(tmp_path)
    local_payload = _sidecar(
        {
            "q1": _route(),
            "q2": _route(evidence_hash="b" * 64),
            "q3": _route(review_required=True),
        }
    )
    _write_json(local_sidecar_path, local_payload)

    report = artifact.build_topic_routing_artifact_report(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha_path,
        release_manifest_path=_release_manifest_path(tmp_path),
    )

    assert report["ok"] is False
    assert report["sha256"]["local_matches_durable"] is False
    assert "Local output/json topic-routing sidecar does not match the durable sidecar artifact." in report["errors"]


def test_verify_command_ignores_stale_optional_local_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _write_valid_artifacts(tmp_path)
    local_payload = _sidecar(
        {
            "q1": _route(),
            "q2": _route(evidence_hash="b" * 64),
            "q3": _route(review_required=True),
        }
    )
    _write_json(local_sidecar_path, local_payload)

    exit_code = artifact.main(
        [
            "verify",
            "--question-bank",
            str(question_bank_path),
            "--local-sidecar",
            str(local_sidecar_path),
            "--durable-sidecar",
            str(durable_sidecar_path),
            "--durable-sha256",
            str(durable_sha_path),
            "--release-manifest",
            str(_release_manifest_path(tmp_path)),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["ok"] is True
    assert report["sha256"]["local_matches_durable"] is False


def test_topic_routing_artifact_allows_local_cache_to_be_missing(tmp_path: Path) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _write_valid_artifacts(tmp_path)
    local_sidecar_path.unlink()

    report = artifact.verify_topic_routing_artifact(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha_path,
        release_manifest_path=_release_manifest_path(tmp_path),
    )

    assert report["ok"] is True
    assert report["sha256"]["local_sidecar"] is None


def test_topic_routing_artifact_detects_missing_question_bank_ids(tmp_path: Path) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _write_valid_artifacts(tmp_path)
    payload = _sidecar(
        {
            "q1": _route(),
            "q2": _route(),
            "extra": _route(review_required=True),
        }
    )
    _write_json(durable_sidecar_path, payload)
    _write_json(local_sidecar_path, payload)
    _write_sha(durable_sha_path, durable_sidecar_path)

    report = artifact.build_topic_routing_artifact_report(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha_path,
        release_manifest_path=_release_manifest_path(tmp_path),
    )

    assert report["ok"] is False
    assert report["id_coverage"]["missing_ids"] == ["q3"]
    assert report["id_coverage"]["extra_ids"] == ["extra"]


def test_restore_copies_durable_sidecar_then_verifies(tmp_path: Path) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, durable_sha_path = _write_valid_artifacts(tmp_path)
    local_sidecar_path.unlink()

    report = artifact.restore_topic_routing_sidecar(
        question_bank_path=question_bank_path,
        local_sidecar_path=local_sidecar_path,
        durable_sidecar_path=durable_sidecar_path,
        durable_sha256_path=durable_sha_path,
        release_manifest_path=_release_manifest_path(tmp_path),
    )

    assert report["ok"] is True
    assert local_sidecar_path.read_bytes() == durable_sidecar_path.read_bytes()


def test_production_sidecar_path_requires_provenance_guard() -> None:
    assert artifact.should_enforce_production_topic_routing_provenance(
        artifact.DEFAULT_LOCAL_SIDECAR_PATH
    )
    assert not artifact.should_enforce_production_topic_routing_provenance(
        Path("/tmp/question_bank.topic_routing.v1.json")
    )


def test_resolver_uses_manifest_bound_durable_sidecar(tmp_path: Path) -> None:
    question_bank_path, _, durable_sidecar_path, _ = _write_valid_artifacts(tmp_path)

    resolved = artifact.resolve_topic_routing_sidecar(
        question_bank_path=question_bank_path,
        release_manifest_path=_release_manifest_path(tmp_path),
    )

    assert resolved == durable_sidecar_path


def test_resolver_rejects_production_path_without_release_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    question_bank_path, local_sidecar_path, durable_sidecar_path, _ = _write_valid_artifacts(tmp_path)
    monkeypatch.setattr(artifact, "DEFAULT_QUESTION_BANK_PATH", question_bank_path)
    monkeypatch.setattr(artifact, "DEFAULT_LOCAL_SIDECAR_PATH", local_sidecar_path)
    monkeypatch.setattr(artifact, "DEFAULT_DURABLE_SIDECAR_PATH", durable_sidecar_path)

    with pytest.raises(artifact.TopicRoutingArtifactError, match="Missing release manifest"):
        artifact.resolve_topic_routing_sidecar(
            question_bank_path=question_bank_path,
            requested_path=local_sidecar_path,
            release_manifest_path=tmp_path / "missing-release.json",
        )
