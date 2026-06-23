from pathlib import Path

from exam_bank.submissions.review_queue import create_grading_prep_record, create_review_record


def _review():
    return create_review_record(
        {
            "assignment_id": "p3_vectors_hw1",
            "student_id": "S0001",
            "submission_id": "p3_vectors_hw1:S0001.pdf:abc123",
            "stored_pdf_path": "output/submissions/p3_vectors_hw1/accepted_pdfs/S0001.pdf",
            "late": False,
        }
    )


def test_create_manual_grading_prep_placeholder_with_no_score() -> None:
    grading = create_grading_prep_record(_review())

    assert grading.grading_mode == "manual_placeholder"
    assert grading.status == "not_started"
    assert grading.score is None
    assert grading.max_score is None
    assert grading.rubric_id == ""
    assert grading.question_notes == []
    assert grading.teacher_notes == ""
    assert grading.review_required is True
    assert grading.reviewed_by == ""
    assert grading.reviewed_at is None


def test_phase_2_has_no_email_ocr_or_automated_grading_behavior() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/exam_bank/submissions").glob("*.py"))

    for forbidden in [
        "smtplib",
        "imaplib",
        "poplib",
        "outlook",
        "gmail",
        "sendmail",
        "pytesseract",
        "run_question_crop_ocr",
        "automated_grade",
        "auto_grade_submission",
    ]:
        assert forbidden not in source.lower()
