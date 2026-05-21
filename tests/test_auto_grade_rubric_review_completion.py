from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from exam_bank.auto_grade.eligibility import build_eligible_items
from exam_bank.auto_grade.review_batch import check_rubric_review_completion
from tests.test_auto_grade_eligibility import _write_fixture
from tests.test_auto_grade_reviewed_rubrics import _event, _valid_rubric


def test_completion_checker_reports_draft_gaps(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed.json"
    rubric = _valid_rubric()
    rubric["review_status"] = "needs_human_review"
    rubric["reviewed_by"] = None
    rubric["reviewed_at"] = None
    rubric["rubric_total_verified"] = False
    rubric["safe_for_auto_grade_lab"] = False
    rubric["safe_for_teacher_beta"] = False
    rubric["events"][0]["accepted_evidence"] = []
    rubric["events"][0]["learning_target_ids"] = []
    _write_reviewed(reviewed, rubric)

    report = check_rubric_review_completion(
        reviewed_rubrics_path=reviewed,
        report_path=tmp_path / "completion.md",
        generated_at="2026-05-21T00:00:00Z",
    )

    assert report["rubric_count"] == 1
    assert report["approved_count"] == 0
    assert report["missing_reviewer_metadata_count"] == 1
    assert report["missing_accepted_evidence_count"] == 1
    assert report["missing_learning_target_ids_count"] == 1
    assert (tmp_path / "completion.md").exists()


def test_completion_checker_detects_missing_packet_pages(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed.json"
    _write_reviewed(reviewed, _valid_rubric())
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "index.md").write_text("# packet\n", encoding="utf-8")

    report = check_rubric_review_completion(
        reviewed_rubrics_path=reviewed,
        reviewer_packet_dir=packet_dir,
        report_path=None,
        generated_at="2026-05-21T00:00:00Z",
    )

    assert report["reviewer_packet"]["packet_generated"] is True
    assert report["missing_reviewer_packet_page_count"] == 1
    assert "missing_reviewer_packet_page" in report["rubrics"][0]["blockers"]


def test_completion_checker_detects_placeholder_values(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed.json"
    rubric = _valid_rubric()
    rubric["reviewed_by"] = "TODO_REVIEWED_BY"
    rubric["events"][0]["accepted_evidence"] = ["TODO_REWRITE_FROM_IMAGE"]
    _write_reviewed(reviewed, rubric)

    report = check_rubric_review_completion(
        reviewed_rubrics_path=reviewed,
        report_path=None,
        generated_at="2026-05-21T00:00:00Z",
    )

    assert report["placeholder_value_count"] == 1
    assert "placeholder_value" in report["rubrics"][0]["blockers"]


def test_completion_checker_reports_dependency_follow_through_unknown_and_total_gaps(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed.json"
    rubric = _valid_rubric()
    rubric["total_marks"] = 5
    rubric["events"] = [
        _event("fixture-rubric-1-e1", "DM", 1) | {"dependency": "needs_human_review"},
        _event("fixture-rubric-1-e2", "FT", 1) | {"follow_through_policy": "needs_human_review"},
        _event("fixture-rubric-1-e3", "unknown", 1),
    ]
    _write_reviewed(reviewed, rubric)

    report = check_rubric_review_completion(
        reviewed_rubrics_path=reviewed,
        report_path=None,
        generated_at="2026-05-21T00:00:00Z",
    )

    assert report["unresolved_unknown_mark_code_count"] == 1
    assert report["dependency_policy_gap_count"] == 1
    assert report["follow_through_policy_gap_count"] == 1
    assert report["total_mismatch_count"] == 1


def test_completion_checker_reports_teacher_beta_promotion_only(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed.json"
    _write_reviewed(reviewed, _valid_rubric())

    report = check_rubric_review_completion(
        reviewed_rubrics_path=reviewed,
        report_path=None,
        generated_at="2026-05-21T00:00:00Z",
    )

    assert report["eligibility_promotion_candidate_count"] == 1
    assert report["student_self_check_beta_candidate_count"] == 0
    assert report["student_ready_candidate_count"] == 0


def test_explicit_student_safe_fixture_does_not_promote_student_statuses(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["safe_for_student_self_check"] = True
    _write_reviewed(paths["reviewed_rubrics"], rubric)

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

    statuses = {item["eligibility_status"] for item in payload["items"]}
    assert "student_self_check_beta" not in statuses
    assert "student_ready" not in statuses
    assert payload["summary"]["student_self_check_beta_count"] == 0
    assert payload["summary"]["student_ready_count"] == 0


def test_phase_2c_student_safe_reviewed_rubric_fails_validation(tmp_path: Path) -> None:
    from exam_bank.auto_grade.reviewed_rubrics import validate_reviewed_rubrics

    paths = _write_fixture(tmp_path)
    rubric = _valid_rubric()
    rubric["safe_for_student_self_check"] = True
    _write_reviewed(paths["reviewed_rubrics"], rubric)

    report = validate_reviewed_rubrics(
        reviewed_rubrics_path=paths["reviewed_rubrics"],
        question_bank_path=paths["question_bank"],
        report_path=None,
        phase="2C",
    )

    assert report["ok"] is False
    assert any("student_safe_flag_forbidden_in_phase_2c" in error for error in report["errors"])


def test_review_completion_cli_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_auto_grade_rubric_review_completion.py", "--help"],
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
