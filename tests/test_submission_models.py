from pathlib import Path
from datetime import datetime, timezone

import pytest

from exam_bank.submissions.ingest import load_assignment, load_roster, match_student_id_from_filename
from exam_bank.submissions.models import DraftGradingResult, SubmissionExtractionResult


FIXTURES = Path("tests/fixtures/submissions")


def test_load_assignment_json() -> None:
    assignment = load_assignment(FIXTURES / "assignment_p3_vectors_hw1.json")

    assert assignment.assignment_id == "p3_vectors_hw1"
    assert assignment.course_id == "p3"
    assert assignment.class_id == "class_12a"
    assert assignment.accepted_file_types == ["pdf"]
    assert assignment.max_files_per_student == 1
    assert assignment.allow_late is True


def test_load_roster_csv() -> None:
    roster = load_roster(FIXTURES / "roster_class_12a.csv")

    assert [student.student_id for student in roster] == ["S0001", "S0002", "S0003"]
    assert roster[0].display_name == "Student One"
    assert roster[0].email == "student.one@example.invalid"
    assert all(student.active for student in roster)


def test_match_filename_to_student_id() -> None:
    student_ids = {"S0001", "S0002"}

    assert match_student_id_from_filename(Path("S0001.pdf"), student_ids) == "S0001"
    assert match_student_id_from_filename(Path("S0001_vectors_hw1.pdf"), student_ids) == "S0001"
    assert match_student_id_from_filename(Path("p3_vectors_hw1_S0002.pdf"), student_ids, "p3_vectors_hw1") == "S0002"
    assert match_student_id_from_filename(Path("unknown_student.pdf"), student_ids) is None


def test_phase3_extraction_result_enforces_draft_only_flags() -> None:
    result = SubmissionExtractionResult(
        extraction_id="p3_vectors_hw1:S0001:s1:extraction",
        assignment_id="p3_vectors_hw1",
        student_id="S0001",
        submission_id="s1",
        stored_pdf_path="output/submissions/p3_vectors_hw1/accepted_pdfs/S0001.pdf",
        status="extracted",
        page_count=1,
        text_extractable=True,
        extracted_text_preview="synthetic preview",
        extraction_warnings=[],
        created_at=datetime.now(timezone.utc),
    )

    assert result.grading_mode == "draft_auto"
    assert result.student_facing is False
    assert result.teacher_review_required is True


def test_phase3_draft_grading_result_rejects_student_facing_output() -> None:
    with pytest.raises(ValueError, match="student_facing=false"):
        DraftGradingResult(
            draft_grading_id="draft1",
            grading_result_id="manual1",
            assignment_id="p3_vectors_hw1",
            student_id="S0001",
            submission_id="s1",
            grading_mode="draft_auto",
            status="needs_review",
            draft_score=None,
            draft_max_score=None,
            confidence="low",
            confidence_reasons=["missing_reviewed_rubric_mapping"],
            teacher_review_required=True,
            student_facing=True,
            question_results=[],
            overall_notes=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
