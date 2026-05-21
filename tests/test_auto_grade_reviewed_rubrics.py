from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.auto_grade.reviewed_rubrics import validate_reviewed_rubrics
from tests.test_auto_grade_eligibility import _write_fixture


def test_valid_reviewed_rubric_fixture_passes(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _write_reviewed(paths["reviewed_rubrics"], _valid_rubric())

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is True
    assert report["approved_question_ids"] == ["11summer26_q01"]


def test_total_mismatch_fails(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["total_marks"] = 5
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is False
    assert any("event_total_mismatch" in error for error in report["errors"])


def test_unknown_mark_code_in_approved_rubric_fails(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["events"][0]["mark_code"] = "unknown"
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is False
    assert any("approved_event_unknown_mark_code" in error for error in report["errors"])


def test_missing_review_metadata_fails(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["reviewed_by"] = ""
    rubric["reviewed_at"] = ""
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is False
    assert any("missing_reviewer_identity" in error for error in report["errors"])
    assert any("missing_reviewed_at" in error for error in report["errors"])


def test_dependency_and_follow_through_policy_requirements_fail(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["events"][0]["mark_code"] = "DM"
    rubric["events"][0]["dependency"] = ""
    rubric["events"][1]["mark_code"] = "FT"
    rubric["events"][1]["follow_through_policy"] = ""
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is False
    assert any("dependent_mark_missing_dependency_policy" in error for error in report["errors"])
    assert any("follow_through_mark_missing_follow_through_policy" in error for error in report["errors"])


def test_student_safe_and_teacher_beta_safety_rules_fail(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["safe_for_auto_grade_lab"] = False
    rubric["safe_for_teacher_beta"] = True
    rubric["safe_for_student_self_check"] = True
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is False
    assert any("student_safe_flag_forbidden_in_phase_2a" in error for error in report["errors"])
    assert any("teacher_beta_without_auto_grade_lab_safety" in error for error in report["errors"])


def test_advisory_only_promotion_without_review_metadata_fails(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["review_status"] = "candidate"
    rubric["reviewed_by"] = ""
    rubric["reviewed_at"] = ""
    rubric["rubric_total_verified"] = False
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
    )

    assert report["ok"] is False
    assert any("approved_scope_without_approved_review_status" in error for error in report["errors"])
    assert any("missing_reviewer_identity" in error for error in report["errors"])
    assert any("approved_rubric_total_not_verified" in error for error in report["errors"])


def test_allow_missing_reports_zero_approved(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=tmp_path / "missing.json",
        question_bank_path=paths["question_bank"],
        allow_missing=True,
        report_path=None,
    )

    assert report["ok"] is True
    assert report["approved_rubric_count"] == 0


def test_reviewed_rubric_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_auto_grade_reviewed_rubrics.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def _write_reviewed(path: Path, *rubrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "exam_bank.auto_grade.reviewed_rubrics",
                "schema_version": 1,
                "generated_at": "2026-05-21T00:00:00Z",
                "source_question_bank_path": "fixture/question_bank.json",
                "source_mark_events_path": "fixture/question_bank.mark_events.v1.json",
                "rubric_count": len(rubrics),
                "event_count": sum(len(rubric["events"]) for rubric in rubrics),
                "summary": {},
                "rubrics": list(rubrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _valid_rubric() -> dict[str, object]:
    return {
        "rubric_id": "fixture-rubric-1",
        "source_question_id": "11summer26_q01",
        "source_mark_scheme_image_path": "p1/11summer26/mark_scheme/q01.png",
        "source_mark_events_record_id": "11summer26_q01",
        "paper": "11summer26",
        "paper_family": "p1",
        "question_number": "1",
        "part_path": [],
        "total_marks": 4,
        "rubric_total_verified": True,
        "safe_for_auto_grade_lab": True,
        "safe_for_teacher_beta": True,
        "safe_for_student_self_check": False,
        "review_status": "approved",
        "reviewed_by": "fixture reviewer",
        "reviewed_at": "2026-05-21T00:00:00Z",
        "approval_scope": "teacher_beta",
        "events": [
            _event("fixture-rubric-1-e1", "M", 2),
            _event("fixture-rubric-1-e2", "A", 2),
        ],
    }


def _event(event_id: str, mark_code: str, mark_value: int) -> dict[str, object]:
    return {
        "event_id": event_id,
        "source_event_id": f"source-{event_id}",
        "part_path": [],
        "mark_code": mark_code,
        "mark_type": mark_code,
        "mark_value": mark_value,
        "dependency": "independent",
        "follow_through_policy": "none",
        "accepted_evidence": ["canonical mark-scheme image reviewed by human"],
        "common_errors": [],
        "alternative_methods": [],
        "learning_target_ids": ["9709_p1_topic_algebra"],
        "review_status": "approved",
        "review_notes": "fixture only",
    }
