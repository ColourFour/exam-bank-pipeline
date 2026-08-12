from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from exam_bank.publication_safety import PublicationReadBlockedError
from exam_bank.release_manifest import (
    ASTERION_CATALOG_ROLE,
    ASTERION_CONTENT_LAB_ROLE,
    ASTERION_RUNTIME_ROLE,
    ASSET_MANIFEST_ROLE,
    CORPUS_MANIFEST_ROLE,
    DIFFICULTY_INDEX_ROLE,
    MARK_EVENTS_ROLE,
    PROMOTION_DECISIONS_ROLE,
    QUESTION_BANK_ROLE,
    QUESTION_ID_COVERAGE_EXACT,
    QUESTION_ID_COVERAGE_EXACT_SET,
    QUESTION_ID_COVERAGE_SUBSET,
    TOPIC_ROUTING_ROLE,
    ReleaseManifestError,
    build_release_manifest,
    file_sha256,
    main,
    resolve_release_artifact,
    verify_release_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_release_id(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        manifest["artifacts"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    manifest["release_id"] = hashlib.sha256(encoded).hexdigest()[:16]
    _write_json(path, manifest)


def _question_bank(ids: list[str], *, corpus_manifest_sha256: str | None = None) -> dict:
    payload = {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": len(ids),
        "questions": [{"question_id": question_id} for question_id in ids],
    }
    if corpus_manifest_sha256 is not None:
        payload["run_manifest"] = {"corpus_manifest_sha256": corpus_manifest_sha256}
    return payload


def _topic_routes(ids: list[str]) -> dict:
    return {
        "schema_name": "exam_bank.topic_routing_sidecar",
        "schema_version": 1,
        "record_count": len(ids),
        "records": {question_id: {"question_id": question_id} for question_id in ids},
    }


def test_release_manifest_binds_hashes_counts_and_exact_question_ids(tmp_path: Path) -> None:
    bank = tmp_path / "output/json/question_bank.json"
    routes = tmp_path / "data/topic_routing/question_bank.topic_routing.v1.json"
    manifest_path = tmp_path / "manifests/releases/question_bank_release_manifest.v1.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    _write_json(routes, _topic_routes(["q1", "q2"]))

    manifest = build_release_manifest(
        question_bank_path=bank,
        artifacts={TOPIC_ROUTING_ROLE: routes},
        output_path=manifest_path,
        base_dir=tmp_path,
        generated_at="2026-08-09T00:00:00+00:00",
    )
    report = verify_release_manifest(
        manifest_path,
        required_roles=(QUESTION_BANK_ROLE, TOPIC_ROUTING_ROLE),
    )

    assert report["ok"] is True
    assert manifest["artifacts"][TOPIC_ROUTING_ROLE]["bound_question_bank_sha256"] == manifest["artifacts"][QUESTION_BANK_ROLE]["sha256"]
    assert manifest["artifacts"][TOPIC_ROUTING_ROLE]["question_id_coverage"] == "exact"
    assert resolve_release_artifact(TOPIC_ROUTING_ROLE, manifest_path=manifest_path) == routes


def test_release_manifest_binds_question_bank_to_exact_corpus_manifest(tmp_path: Path) -> None:
    bank = tmp_path / "output/json/question_bank.json"
    corpus = tmp_path / "manifests/corpora/active.json"
    manifest_path = tmp_path / "manifests/releases/release.json"
    _write_json(
        corpus,
        {
            "schema_name": "exam_bank.corpus_manifest",
            "schema_version": 1,
            "record_count": 0,
            "documents": [],
        },
    )
    _write_json(bank, _question_bank(["q1"], corpus_manifest_sha256=file_sha256(corpus)))

    manifest = build_release_manifest(
        question_bank_path=bank,
        artifacts={CORPUS_MANIFEST_ROLE: corpus},
        output_path=manifest_path,
        base_dir=tmp_path,
    )
    report = verify_release_manifest(
        manifest_path,
        required_roles=(QUESTION_BANK_ROLE, CORPUS_MANIFEST_ROLE),
    )

    assert report["provenance_ok"] is True
    assert (
        manifest["artifacts"][QUESTION_BANK_ROLE]["declared_corpus_manifest_sha256"]
        == manifest["artifacts"][CORPUS_MANIFEST_ROLE]["sha256"]
    )


def test_release_manifest_rejects_corpus_manifest_not_declared_by_bank(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    corpus = tmp_path / "corpus.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(corpus, {"schema_name": "exam_bank.corpus_manifest", "schema_version": 1, "documents": []})

    with pytest.raises(ReleaseManifestError, match="must declare"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={CORPUS_MANIFEST_ROLE: corpus},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_rejects_different_declared_corpus_hash(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    corpus = tmp_path / "corpus.json"
    _write_json(bank, _question_bank(["q1"], corpus_manifest_sha256="0" * 64))
    _write_json(corpus, {"schema_name": "exam_bank.corpus_manifest", "schema_version": 1, "documents": []})

    with pytest.raises(ReleaseManifestError, match="does not match"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={CORPUS_MANIFEST_ROLE: corpus},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_fails_closed_during_incomplete_question_bank_publication(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "output/json/question_bank.json"
    routes = tmp_path / "routes.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    (tmp_path / ".output.rollback-crashed").mkdir()

    with pytest.raises(PublicationReadBlockedError, match="awaiting recovery"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={TOPIC_ROUTING_ROLE: routes},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_rejects_sidecar_membership_drift(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "topic.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    _write_json(routes, _topic_routes(["q1", "extra"]))

    with pytest.raises(ReleaseManifestError, match="question-ID coverage"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={TOPIC_ROUTING_ROLE: routes},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_detects_artifact_changed_after_binding(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "topic.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    build_release_manifest(
        question_bank_path=bank,
        artifacts={TOPIC_ROUTING_ROLE: routes},
        output_path=manifest_path,
        base_dir=tmp_path,
    )
    _write_json(routes, _topic_routes(["q1", "q2"]))

    with pytest.raises(ReleaseManifestError, match="SHA-256 mismatch"):
        verify_release_manifest(manifest_path)


def test_release_manifest_binds_exact_subset_and_metadata_artifacts(tmp_path: Path) -> None:
    bank = tmp_path / "output/json/question_bank.json"
    manifest_path = tmp_path / "manifests/releases/release.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    bank_sha = file_sha256(bank)

    paths = {
        TOPIC_ROUTING_ROLE: tmp_path / "topic.json",
        ASSET_MANIFEST_ROLE: tmp_path / "assets.json",
        MARK_EVENTS_ROLE: tmp_path / "mark_events.json",
        DIFFICULTY_INDEX_ROLE: tmp_path / "difficulty.json",
        ASTERION_CATALOG_ROLE: tmp_path / "catalog.json",
        ASTERION_RUNTIME_ROLE: tmp_path / "runtime.json",
        ASTERION_CONTENT_LAB_ROLE: tmp_path / "content_lab.json",
        PROMOTION_DECISIONS_ROLE: tmp_path / "promotions.json",
        "asterion_package": tmp_path / "asterion_package.json",
    }
    _write_json(paths[TOPIC_ROUTING_ROLE], _topic_routes(["q1", "q2"]))
    _write_json(
        paths[ASSET_MANIFEST_ROLE],
        {
            "schema_name": "exam_bank.asset_manifest",
            "schema_version": 1,
            "asset_count": 3,
            "assets": [
                {"asset_id": "question-q1", "question_id": "q1"},
                {"asset_id": "mark-scheme-q1", "question_id": "q1"},
                {"asset_id": "question-q2", "question_id": "q2"},
            ],
        },
    )
    _write_json(
        paths[MARK_EVENTS_ROLE],
        {
            "schema_name": "exam_bank.question_bank.mark_events",
            "schema_version": 1,
            "source_question_bank_sha256": bank_sha,
            "record_count": 2,
            "records": [{"question_id": "q1"}, {"question_id": "q2"}],
        },
    )
    _write_json(
        paths[DIFFICULTY_INDEX_ROLE],
        {
            "schema_name": "exam_bank.difficulty_index",
            "schema_version": 1,
            "source_question_bank_sha256": bank_sha,
            "source_topic_routing_sha256": file_sha256(paths[TOPIC_ROUTING_ROLE]),
            "record_count": 2,
            "records": [{"question_id": "q1"}, {"question_id": "q2"}],
        },
    )
    _write_json(
        paths[ASTERION_CATALOG_ROLE],
        {
            "schema_name": "asterion.exam_bank_catalog",
            "schema_version": 1,
            "record_count": 2,
            "questions": [{"question_id": "q1"}, {"question_id": "q2"}],
        },
    )
    _write_json(
        paths[ASTERION_RUNTIME_ROLE],
        {
            "schema_name": "asterion.question_bank",
            "schema_version": 1,
            "record_count": 1,
            "questions": [{"question_id": "q2"}],
        },
    )
    _write_json(
        paths[ASTERION_CONTENT_LAB_ROLE],
        {
            "schema_name": "asterion.content_lab_candidates",
            "schema_version": 1,
            "record_count": 3,
            "candidates": [
                {"candidate_id": "c1", "question_id": "q1"},
                {"candidate_id": "c2", "question_id": "q1"},
                {"candidate_id": "c3", "question_id": "q2"},
            ],
        },
    )
    _write_json(
        paths[PROMOTION_DECISIONS_ROLE],
        {
            "schema": "asterion.student_runtime_promotion_decisions",
            "schema_version": 1,
            "decision_count": 2,
            "decisions": [
                {"decision_id": "p1", "question_id": "q1"},
                {"decision_id": "p2", "question_id": "q1"},
            ],
        },
    )
    _write_json(
        paths["asterion_package"],
        {
            "schema_name": "exam_bank.asterion_export_release_manifest",
            "schema_version": 1,
            "record_count": 2,
            "release_inputs": {
                "durable_sidecar": {"sha256": file_sha256(paths[TOPIC_ROUTING_ROLE])},
            },
            "export_artifacts": {
                "catalog": {"sha256": file_sha256(paths[ASTERION_CATALOG_ROLE])},
                "student_runtime": {"sha256": file_sha256(paths[ASTERION_RUNTIME_ROLE])},
                "content_lab_candidates": {
                    "sha256": file_sha256(paths[ASTERION_CONTENT_LAB_ROLE]),
                },
            },
        },
    )

    exact_roles = {
        TOPIC_ROUTING_ROLE,
        MARK_EVENTS_ROLE,
        DIFFICULTY_INDEX_ROLE,
        ASTERION_CATALOG_ROLE,
    }
    subset_roles = {
        ASTERION_RUNTIME_ROLE,
        ASTERION_CONTENT_LAB_ROLE,
        PROMOTION_DECISIONS_ROLE,
    }
    manifest = build_release_manifest(
        question_bank_path=bank,
        artifacts=paths,
        output_path=manifest_path,
        base_dir=tmp_path,
        question_id_coverage_by_role={
            **{role: QUESTION_ID_COVERAGE_EXACT for role in exact_roles},
            **{role: QUESTION_ID_COVERAGE_SUBSET for role in subset_roles},
            ASSET_MANIFEST_ROLE: QUESTION_ID_COVERAGE_EXACT_SET,
        },
        dependencies_by_role={
            DIFFICULTY_INDEX_ROLE: (TOPIC_ROUTING_ROLE, MARK_EVENTS_ROLE),
            ASTERION_RUNTIME_ROLE: (ASTERION_CATALOG_ROLE,),
            ASTERION_CONTENT_LAB_ROLE: (ASTERION_CATALOG_ROLE, MARK_EVENTS_ROLE),
            PROMOTION_DECISIONS_ROLE: (ASTERION_CONTENT_LAB_ROLE,),
            "asterion_package": (
                TOPIC_ROUTING_ROLE,
                ASTERION_CATALOG_ROLE,
                ASTERION_RUNTIME_ROLE,
                ASTERION_CONTENT_LAB_ROLE,
            ),
        },
    )
    report = verify_release_manifest(manifest_path, required_roles=paths)

    assert report["ok"] is True
    assert manifest["artifacts"][ASTERION_CONTENT_LAB_ROLE]["record_count"] == 3
    assert manifest["artifacts"][ASTERION_CONTENT_LAB_ROLE]["question_id_count"] == 2
    assert manifest["artifacts"][ASTERION_CONTENT_LAB_ROLE]["question_id_coverage"] == "subset"
    assert manifest["artifacts"]["asterion_package"]["record_collection"] is None
    assert manifest["artifacts"]["asterion_package"]["record_count"] == 2
    assert manifest["artifacts"]["asterion_package"]["bound_artifact_sha256"][ASTERION_RUNTIME_ROLE]
    assert report["artifacts"][PROMOTION_DECISIONS_ROLE]["record_collection"] == "decisions"
    assert report["artifacts"][ASSET_MANIFEST_ROLE]["record_collection"] == "assets"
    assert report["artifacts"][ASSET_MANIFEST_ROLE]["record_count"] == 3
    assert manifest["artifacts"][ASSET_MANIFEST_ROLE]["question_id_coverage"] == "exact_set"


def test_release_manifest_rejects_foreign_id_in_subset_artifact(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    runtime = tmp_path / "runtime.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    _write_json(
        runtime,
        {
            "schema_name": "asterion.question_bank",
            "schema_version": 1,
            "record_count": 1,
            "questions": [{"question_id": "foreign"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="not a subset"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={ASTERION_RUNTIME_ROLE: runtime},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            question_id_coverage_by_role={ASTERION_RUNTIME_ROLE: QUESTION_ID_COVERAGE_SUBSET},
        )


def test_release_manifest_recognizes_source_question_id_in_review_decisions(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "question_bank.json"
    decisions = tmp_path / "reviewed_mark_events.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    _write_json(
        decisions,
        {
            "schema_name": "exam_bank.reviewed_mark_events",
            "schema_version": 1,
            "decision_count": 2,
            "decisions": [
                {"decision_id": "d1", "source_question_id": "q1"},
                {"decision_id": "d2", "source_question_id": "q1"},
            ],
        },
    )

    manifest = build_release_manifest(
        question_bank_path=bank,
        artifacts={"reviewed_mark_events": decisions},
        output_path=manifest_path,
        base_dir=tmp_path,
        question_id_coverage_by_role={
            "reviewed_mark_events": QUESTION_ID_COVERAGE_SUBSET,
        },
    )
    report = verify_release_manifest(manifest_path)

    assert report["ok"] is True
    entry = manifest["artifacts"]["reviewed_mark_events"]
    assert entry["record_count"] == 2
    assert entry["question_id_count"] == 1
    assert entry["question_id_reference_count"] == 2


def test_release_manifest_rejects_stale_declared_source_bank_hash(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    marks = tmp_path / "marks.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        marks,
        {
            "schema_name": "exam_bank.question_bank.mark_events",
            "schema_version": 1,
            "source_question_bank_sha256": "0" * 64,
            "record_count": 1,
            "records": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="different source question-bank SHA-256"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={MARK_EVENTS_ROLE: marks},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            question_id_coverage_by_role={MARK_EVENTS_ROLE: QUESTION_ID_COVERAGE_EXACT},
        )


def test_release_manifest_rejects_missing_dependency_role(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    difficulty = tmp_path / "difficulty.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        difficulty,
        {
            "schema_name": "exam_bank.difficulty_index",
            "schema_version": 1,
            "record_count": 1,
            "records": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="depends on roles missing"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={DIFFICULTY_INDEX_ROLE: difficulty},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            dependencies_by_role={DIFFICULTY_INDEX_ROLE: (MARK_EVENTS_ROLE,)},
        )


def test_release_manifest_rejects_stale_declared_dependency_hash(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "topic.json"
    difficulty = tmp_path / "difficulty.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    _write_json(
        difficulty,
        {
            "schema_name": "exam_bank.difficulty_index",
            "schema_version": 1,
            "source_question_bank_sha256": file_sha256(bank),
            "source_topic_routing_sha256": "0" * 64,
            "record_count": 1,
            "records": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="declares a stale SHA-256"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={
                TOPIC_ROUTING_ROLE: routes,
                DIFFICULTY_INDEX_ROLE: difficulty,
            },
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            dependencies_by_role={DIFFICULTY_INDEX_ROLE: (TOPIC_ROUTING_ROLE,)},
        )


def test_release_manifest_rejects_manifest_output_collision(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    original = _question_bank(["q1"])
    _write_json(bank, original)

    with pytest.raises(ReleaseManifestError, match="would overwrite the question_bank"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={},
            output_path=bank,
            base_dir=tmp_path,
        )

    assert json.loads(bank.read_text(encoding="utf-8")) == original


def test_release_manifest_rejects_one_path_assigned_to_multiple_roles(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))

    with pytest.raises(ReleaseManifestError, match="assigned to multiple roles"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={TOPIC_ROUTING_ROLE: routes, MARK_EVENTS_ROLE: routes},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


@pytest.mark.parametrize("role", ["", " role", "../role", "Role", 1])
def test_release_manifest_rejects_invalid_roles(tmp_path: Path, role: object) -> None:
    bank = tmp_path / "question_bank.json"
    artifact = tmp_path / "artifact.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(artifact, _topic_routes(["q1"]))

    with pytest.raises(ReleaseManifestError, match="Invalid artifact role"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={role: artifact},  # type: ignore[dict-item]
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_cli_rejects_duplicate_role_options(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "build",
            "--artifact",
            "topic_routing=one.json",
            "--artifact",
            "topic_routing=two.json",
        ]
    )

    assert result == 1
    assert "Duplicate --artifact role: topic_routing" in capsys.readouterr().out


def test_release_manifest_rejects_multiple_record_collections(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    ambiguous = tmp_path / "ambiguous.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        ambiguous,
        {
            "schema_name": "exam_bank.ambiguous",
            "schema_version": 1,
            "records": [{"question_id": "q1"}],
            "items": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="multiple record collections"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={"ambiguous": ambiguous},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_subset_coverage_rejects_records_without_question_ids(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    runtime = tmp_path / "runtime.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        runtime,
        {
            "schema_name": "asterion.question_bank",
            "schema_version": 1,
            "record_count": 2,
            "questions": [
                {"question_id": "q1"},
                {"candidate_id": "missing-question-id"},
            ],
        },
    )

    with pytest.raises(ReleaseManifestError, match="non-empty question ID on every record"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={ASTERION_RUNTIME_ROLE: runtime},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            question_id_coverage_by_role={ASTERION_RUNTIME_ROLE: QUESTION_ID_COVERAGE_SUBSET},
        )


def test_exact_set_coverage_rejects_missing_question_membership(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    assets = tmp_path / "assets.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    _write_json(
        assets,
        {
            "schema_name": "exam_bank.asset_manifest",
            "schema_version": 1,
            "asset_count": 2,
            "assets": [
                {"asset_id": "q1-question", "question_id": "q1"},
                {"asset_id": "q1-marks", "question_id": "q1"},
            ],
        },
    )

    with pytest.raises(ReleaseManifestError, match="question-ID set does not match"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={ASSET_MANIFEST_ROLE: assets},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            question_id_coverage_by_role={ASSET_MANIFEST_ROLE: QUESTION_ID_COVERAGE_EXACT_SET},
        )


def test_release_manifest_rejects_conflicting_embedded_dependency_hashes(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    difficulty = tmp_path / "difficulty.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    _write_json(
        difficulty,
        {
            "schema_name": "exam_bank.difficulty_index",
            "schema_version": 1,
            "record_count": 1,
            "source_topic_routing_sha256": file_sha256(routes),
            "source_sidecars": {"topic_routing_sha256": "0" * 64},
            "records": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="Conflicting declared SHA-256"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={TOPIC_ROUTING_ROLE: routes, DIFFICULTY_INDEX_ROLE: difficulty},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_verify_rejects_payload_declared_count_mismatch_even_when_manifest_is_resigned(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    build_release_manifest(
        question_bank_path=bank,
        artifacts={TOPIC_ROUTING_ROLE: routes},
        output_path=manifest_path,
        base_dir=tmp_path,
    )

    changed = _topic_routes(["q1"])
    changed["record_count"] = 99
    _write_json(routes, changed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    route_entry = manifest["artifacts"][TOPIC_ROUTING_ROLE]
    route_entry["sha256"] = file_sha256(routes)
    route_entry["size_bytes"] = routes.stat().st_size
    _write_json(manifest_path, manifest)
    _rewrite_release_id(manifest_path)

    with pytest.raises(ReleaseManifestError, match="Declared record count does not match"):
        verify_release_manifest(manifest_path)


def test_release_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    bank.write_text(
        '{"schema_name":"exam_bank.question_bank","schema_version":2,'
        '"record_count":1,"record_count":1,"questions":[{"question_id":"q1"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ReleaseManifestError, match="duplicate object key 'record_count'"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_id_is_deterministic_across_order_and_generation_time(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    marks = tmp_path / "marks.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    _write_json(
        marks,
        {
            "schema_name": "exam_bank.question_bank.mark_events",
            "schema_version": 1,
            "record_count": 1,
            "records": [{"question_id": "q1"}],
        },
    )

    first = build_release_manifest(
        question_bank_path=bank,
        artifacts={TOPIC_ROUTING_ROLE: routes, MARK_EVENTS_ROLE: marks},
        output_path=tmp_path / "first.json",
        base_dir=tmp_path,
        question_id_coverage_by_role={MARK_EVENTS_ROLE: QUESTION_ID_COVERAGE_EXACT},
        dependencies_by_role={MARK_EVENTS_ROLE: (TOPIC_ROUTING_ROLE,)},
        generated_at="2026-01-01T00:00:00+00:00",
    )
    second = build_release_manifest(
        question_bank_path=bank,
        artifacts={MARK_EVENTS_ROLE: marks, TOPIC_ROUTING_ROLE: routes},
        output_path=tmp_path / "second.json",
        base_dir=tmp_path,
        question_id_coverage_by_role={MARK_EVENTS_ROLE: QUESTION_ID_COVERAGE_EXACT},
        dependencies_by_role={MARK_EVENTS_ROLE: (TOPIC_ROUTING_ROLE,)},
        generated_at="2026-12-31T23:59:59+00:00",
    )

    assert first["release_id"] == second["release_id"]


def test_verify_rejects_artifact_path_that_escapes_declared_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    _write_json(outside, _question_bank(["q1"]))
    manifest_path = tmp_path / "release.json"
    _write_json(
        manifest_path,
        {
            "schema_name": "exam_bank.question_bank_release_manifest",
            "schema_version": 1,
            "artifact_root": "root",
            "release_id": "unused",
            "artifacts": {
                QUESTION_BANK_ROLE: {
                    "path": "../outside.json",
                    "sha256": file_sha256(outside),
                    "size_bytes": outside.stat().st_size,
                    "record_count": 1,
                    "question_id_count": 1,
                    "question_id_reference_count": 1,
                },
            },
        },
    )

    with pytest.raises(ReleaseManifestError, match="escapes artifact_root"):
        verify_release_manifest(manifest_path)


def test_verify_rejects_missing_coverage_even_when_manifest_is_resigned(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    build_release_manifest(
        question_bank_path=bank,
        artifacts={TOPIC_ROUTING_ROLE: routes},
        output_path=manifest_path,
        base_dir=tmp_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][TOPIC_ROUTING_ROLE].pop("question_id_coverage")
    _write_json(manifest_path, manifest)
    _rewrite_release_id(manifest_path)

    with pytest.raises(ReleaseManifestError, match="missing question-ID coverage"):
        verify_release_manifest(manifest_path)


def test_verify_exact_coverage_rejects_an_extra_record_without_question_id(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1", "q2"]))
    _write_json(routes, _topic_routes(["q1", "q2"]))
    build_release_manifest(
        question_bank_path=bank,
        artifacts={TOPIC_ROUTING_ROLE: routes},
        output_path=manifest_path,
        base_dir=tmp_path,
    )

    changed = {
        "schema_name": "exam_bank.topic_routing_sidecar",
        "schema_version": 1,
        "record_count": 3,
        "records": [
            {"question_id": "q1"},
            {"question_id": "q2"},
            {"route_status": "orphan"},
        ],
    }
    _write_json(routes, changed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    route_entry = manifest["artifacts"][TOPIC_ROUTING_ROLE]
    route_entry.update(
        {
            "sha256": file_sha256(routes),
            "size_bytes": routes.stat().st_size,
            "record_count": 3,
            "question_id_count": 2,
            "question_id_reference_count": 2,
        }
    )
    _write_json(manifest_path, manifest)
    _rewrite_release_id(manifest_path)

    with pytest.raises(ReleaseManifestError, match="one unique question ID per record"):
        verify_release_manifest(manifest_path)


def test_required_roles_generator_is_not_consumed_before_enforcement(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1"]))
    build_release_manifest(
        question_bank_path=bank,
        artifacts={},
        output_path=manifest_path,
        base_dir=tmp_path,
    )

    required = (role for role in [TOPIC_ROUTING_ROLE])
    with pytest.raises(ReleaseManifestError, match="missing required roles"):
        verify_release_manifest(manifest_path, required_roles=required)


def test_validation_policy_is_reported_and_can_be_required(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    validation = tmp_path / "validation.json"
    manifest_path = tmp_path / "release.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        validation,
        {
            "schema_name": "exam_bank.example.validation",
            "schema_version": 1,
            "ok": False,
            "error_count": 1,
        },
    )
    build_release_manifest(
        question_bank_path=bank,
        artifacts={"example_validation": validation},
        output_path=manifest_path,
        base_dir=tmp_path,
    )

    report = verify_release_manifest(manifest_path)
    assert report["ok"] is True
    assert report["provenance_ok"] is True
    assert report["policy_ok"] is False
    assert report["validation_statuses"] == {"example_validation": False}
    assert report["blocking_validation_roles"] == ["example_validation"]

    with pytest.raises(ReleaseManifestError, match="not all ok:true"):
        verify_release_manifest(manifest_path, require_validation_ok=True)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "schema_name": "not.question_bank",
                "schema_version": 2,
                "record_count": 1,
                "questions": [{"question_id": "q1"}],
            },
            "Unexpected question_bank schema",
        ),
        (_question_bank([]), "at least one record"),
    ],
)
def test_question_bank_contract_is_fail_closed(
    tmp_path: Path,
    payload: dict,
    message: str,
) -> None:
    bank = tmp_path / "question_bank.json"
    _write_json(bank, payload)

    with pytest.raises(ReleaseManifestError, match=message):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_rejects_invalid_record_collection_type(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    artifact = tmp_path / "artifact.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        artifact,
        {
            "schema_name": "exam_bank.invalid",
            "schema_version": 1,
            "records": "not-a-collection",
        },
    )

    with pytest.raises(ReleaseManifestError, match="must be a list or object"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={"invalid": artifact},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_rejects_conflicting_schema_aliases(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    artifact = tmp_path / "artifact.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(
        artifact,
        {
            "schema_name": "exam_bank.one",
            "schema": "exam_bank.two",
            "schema_version": 1,
            "records": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="Conflicting schema_name and schema"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={"conflicting_schema": artifact},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
        )


def test_release_manifest_rejects_duplicate_dependency_roles(tmp_path: Path) -> None:
    bank = tmp_path / "question_bank.json"
    routes = tmp_path / "routes.json"
    marks = tmp_path / "marks.json"
    _write_json(bank, _question_bank(["q1"]))
    _write_json(routes, _topic_routes(["q1"]))
    _write_json(
        marks,
        {
            "schema_name": "exam_bank.question_bank.mark_events",
            "schema_version": 1,
            "records": [{"question_id": "q1"}],
        },
    )

    with pytest.raises(ReleaseManifestError, match="duplicate dependency roles"):
        build_release_manifest(
            question_bank_path=bank,
            artifacts={TOPIC_ROUTING_ROLE: routes, MARK_EVENTS_ROLE: marks},
            output_path=tmp_path / "release.json",
            base_dir=tmp_path,
            dependencies_by_role={
                MARK_EVENTS_ROLE: (TOPIC_ROUTING_ROLE, TOPIC_ROUTING_ROLE),
            },
        )
