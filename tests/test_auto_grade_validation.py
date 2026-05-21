from __future__ import annotations

import json
from pathlib import Path

from exam_bank.auto_grade.eligibility import build_eligible_items
from exam_bank.auto_grade.validation import validate_eligible_items
from tests.test_auto_grade_eligibility import _write_fixture


def test_validator_accepts_generated_fixture_artifact(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _write_bytes(paths["artifact_root"] / "p1/11summer26/questions/q02.png")
    _write_bytes(paths["artifact_root"] / "p1/11summer26/mark_scheme/q03.png")
    build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
    )

    report = validate_eligible_items(
        eligible_items_path=paths["eligible"],
        question_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
    )

    assert report["ok"] is True


def test_validator_rejects_unsafe_promotion_duplicate_ids_and_empty_block_reasons(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=None,
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )
    payload["items"][0]["eligibility_status"] = "student_ready"
    payload["items"][0]["block_reasons"] = []
    payload["items"][1]["question_id"] = payload["items"][0]["question_id"]
    paths["eligible"].parent.mkdir(parents=True, exist_ok=True)
    paths["eligible"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_eligible_items(
        eligible_items_path=paths["eligible"],
        question_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
    )

    assert report["ok"] is False
    assert any(error.startswith("unsafe_promotion_without_reviewed_rubric") for error in report["errors"])
    assert any(error.startswith("student_safe_without_reviewed_rubric") for error in report["errors"])
    assert any(error.startswith("duplicate_question_id") for error in report["errors"])


def test_validator_rejects_blocked_item_with_no_reasons(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=None,
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )
    payload["items"][0]["eligibility_status"] = "blocked"
    payload["items"][0]["block_reasons"] = []
    paths["eligible"].parent.mkdir(parents=True, exist_ok=True)
    paths["eligible"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_eligible_items(
        eligible_items_path=paths["eligible"],
        question_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
    )

    assert report["ok"] is False
    assert any(error.startswith("blocked_item_without_block_reasons") for error in report["errors"])


def test_validator_rejects_malformed_top_level_and_missing_files(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=None,
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-21T00:00:00Z",
        dry_run=True,
    )
    payload["schema"] = "wrong"
    payload["items"][0]["canonical_question_artifact"] = "missing.png"
    paths["eligible"].parent.mkdir(parents=True, exist_ok=True)
    paths["eligible"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_eligible_items(
        eligible_items_path=paths["eligible"],
        question_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
    )

    assert report["ok"] is False
    assert any(error.startswith("schema_mismatch") for error in report["errors"])
    assert any(error.startswith("artifact_file_missing") for error in report["errors"])


def _write_bytes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
