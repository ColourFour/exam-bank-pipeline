from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from exam_bank.auto_grade.eligibility import build_eligible_items
from exam_bank.auto_grade.registry import promote_reviewed_rubrics_registry
from exam_bank.auto_grade.validation import validate_eligible_items
from tests.test_auto_grade_eligibility import (
    _approved_rubric,
    _ensure_fixture_artifacts,
    _reviewed_rubric,
    _write_fixture,
    _write_reviewed_payload,
)


def test_promotion_copies_only_approved_rubrics_into_live_registry(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    source = tmp_path / "review_batches" / "reviewed_rubrics_draft_0001.v1.json"
    _write_reviewed_payload(
        source,
        _approved_rubric("rubric-1", "11summer26_q01", "q01"),
        _approved_rubric("rubric-2", "11summer26_q02", "q02"),
        _approved_rubric("rubric-3", "11summer26_q03", "q03"),
        _reviewed_rubric(
            "rubric-4",
            "11summer26_q04",
            "q04",
            review_status="needs_human_review",
            safe_for_auto_grade_lab=False,
            safe_for_teacher_beta=False,
        ),
    )
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload["source_review_batch_path"] = "output/auto_grade/review_batches/review_batch_0001.v1.json"
    source.write_text(json.dumps(source_payload, indent=2), encoding="utf-8")

    result = promote_reviewed_rubrics_registry(
        source_reviewed_rubrics_path=source,
        question_bank_path=paths["question_bank"],
        output_path=paths["reviewed_rubrics"],
        mode="replace-approved",
        report_path=tmp_path / "registry_summary.md",
        generated_at="2026-05-22T00:00:00Z",
    )

    assert result["ok"] is True
    registry = json.loads(paths["reviewed_rubrics"].read_text(encoding="utf-8"))
    assert registry["artifact_kind"] == "live_approved_reviewed_rubric_registry"
    assert registry["source_reviewed_rubrics_path"] == str(source)
    assert registry["source_review_batch_path"] == "output/auto_grade/review_batches/review_batch_0001.v1.json"
    assert registry["approved_rubric_count"] == 3
    assert registry["excluded_incomplete_count"] == 1
    assert registry["teacher_beta_safe_count"] == 3
    assert registry["student_self_check_safe_count"] == 0
    assert [rubric["source_question_id"] for rubric in registry["rubrics"]] == [
        "11summer26_q01",
        "11summer26_q02",
        "11summer26_q03",
    ]
    assert all(rubric["review_status"] == "approved" for rubric in registry["rubrics"])


def test_invalid_approved_rubric_blocks_registry_write(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    source = tmp_path / "reviewed_draft.json"
    invalid = _approved_rubric("rubric-1", "11summer26_q01", "q01")
    invalid["events"][0]["mark_code"] = "unknown"
    _write_reviewed_payload(source, invalid)

    result = promote_reviewed_rubrics_registry(
        source_reviewed_rubrics_path=source,
        question_bank_path=paths["question_bank"],
        output_path=paths["reviewed_rubrics"],
        report_path=None,
    )

    assert result["ok"] is False
    assert not paths["reviewed_rubrics"].exists()
    assert any("approved_event_unknown_mark_code" in error for error in result["errors"])


def test_live_registry_promotes_exactly_three_teacher_beta_items(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _ensure_fixture_artifacts(paths["artifact_root"], "q02", "q03")
    source = tmp_path / "reviewed_draft.json"
    _write_reviewed_payload(
        source,
        _approved_rubric("rubric-1", "11summer26_q01", "q01"),
        _approved_rubric("rubric-2", "11summer26_q02", "q02"),
        _approved_rubric("rubric-3", "11summer26_q03", "q03"),
        _reviewed_rubric(
            "rubric-4",
            "11summer26_q04",
            "q04",
            review_status="needs_human_review",
            safe_for_auto_grade_lab=False,
            safe_for_teacher_beta=False,
        ),
    )
    originals = {
        name: path.read_text(encoding="utf-8")
        for name, path in {
            "question_bank": paths["question_bank"],
            "mark_events": paths["mark_events"],
            "topic_routing": paths["topic_routing"],
        }.items()
    }
    result = promote_reviewed_rubrics_registry(
        source_reviewed_rubrics_path=source,
        question_bank_path=paths["question_bank"],
        output_path=paths["reviewed_rubrics"],
        report_path=None,
    )
    assert result["ok"] is True

    payload = build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-22T00:00:00Z",
    )

    promoted = sorted(item["question_id"] for item in payload["items"] if item["eligibility_status"] == "teacher_beta")
    assert promoted == ["11summer26_q01", "11summer26_q02", "11summer26_q03"]
    assert payload["summary"]["teacher_beta_count"] == 3
    assert payload["summary"]["student_self_check_beta_count"] == 0
    assert payload["summary"]["student_ready_count"] == 0
    assert payload["source_sidecars"]["reviewed_rubrics_path"] == str(paths["reviewed_rubrics"])
    assert payload["source_sidecars"]["reviewed_rubrics_sha256"]
    for name, path in {
        "question_bank": paths["question_bank"],
        "mark_events": paths["mark_events"],
        "topic_routing": paths["topic_routing"],
    }.items():
        assert path.read_text(encoding="utf-8") == originals[name]


def test_eligibility_validation_detects_empty_or_mismatched_reviewed_rubrics_source(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _ensure_fixture_artifacts(paths["artifact_root"], "q02", "q03")
    source = tmp_path / "reviewed_draft.json"
    _write_reviewed_payload(
        source,
        _approved_rubric("rubric-1", "11summer26_q01", "q01"),
        _approved_rubric("rubric-2", "11summer26_q02", "q02"),
        _approved_rubric("rubric-3", "11summer26_q03", "q03"),
    )
    promote_reviewed_rubrics_registry(
        source_reviewed_rubrics_path=source,
        question_bank_path=paths["question_bank"],
        output_path=paths["reviewed_rubrics"],
        report_path=None,
    )
    build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-22T00:00:00Z",
    )
    empty = tmp_path / "empty_reviewed_rubrics.v1.json"
    _write_reviewed_payload(empty)

    report = validate_eligible_items(
        eligible_items_path=paths["eligible"],
        question_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=empty,
    )

    assert report["ok"] is False
    assert any(error.startswith("reviewed_rubrics_path_mismatch") for error in report["errors"])
    assert any(error.startswith("teacher_beta_promotions_against_empty_reviewed_rubrics") for error in report["errors"])
    assert any(error.startswith("unsafe_promotion_without_reviewed_rubric") for error in report["errors"])


def test_eligibility_validation_detects_missing_recorded_reviewed_rubrics_source(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _ensure_fixture_artifacts(paths["artifact_root"], "q02", "q03")
    source = tmp_path / "reviewed_draft.json"
    _write_reviewed_payload(source, _approved_rubric("rubric-1", "11summer26_q01", "q01"))
    promote_reviewed_rubrics_registry(
        source_reviewed_rubrics_path=source,
        question_bank_path=paths["question_bank"],
        output_path=paths["reviewed_rubrics"],
        report_path=None,
    )
    build_eligible_items(
        question_bank_path=paths["question_bank"],
        output_path=paths["eligible"],
        artifact_root=paths["artifact_root"],
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        mark_events_path=paths["mark_events"],
        topic_routing_path=paths["topic_routing"],
        generated_at="2026-05-22T00:00:00Z",
    )
    paths["reviewed_rubrics"].unlink()

    report = validate_eligible_items(
        eligible_items_path=paths["eligible"],
        question_bank_path=paths["question_bank"],
        artifact_root=paths["artifact_root"],
    )

    assert report["ok"] is False
    assert any(error.startswith("reviewed_rubrics_source_missing") for error in report["errors"])
    assert any(error.startswith("teacher_beta_promotions_without_loaded_reviewed_rubrics") for error in report["errors"])


def test_promotion_cli_help_exits_successfully() -> None:
    env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    result = subprocess.run(
        [sys.executable, "scripts/promote_auto_grade_reviewed_rubrics.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
