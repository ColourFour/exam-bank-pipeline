from pathlib import Path

import fitz

from exam_bank.submissions.extraction import extract_submission_pdf


FIXTURES = Path("tests/fixtures/submissions")


def _write_native_text_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extract_submission_pdf_stores_short_native_text_preview(tmp_path: Path) -> None:
    pdf_path = tmp_path / "native_text.pdf"
    _write_native_text_pdf(pdf_path, "Synthetic answer text " * 80)

    result = extract_submission_pdf(
        assignment_id="p3_vectors_hw1",
        student_id="S0001",
        submission_id="s1",
        stored_pdf_path=pdf_path,
    )

    assert result.status == "extracted"
    assert result.page_count == 1
    assert result.text_extractable is True
    assert result.extracted_text_preview.startswith("Synthetic answer text")
    assert len(result.extracted_text_preview) <= 320
    assert result.grading_mode == "draft_auto"
    assert result.student_facing is False
    assert result.teacher_review_required is True


def test_extract_submission_pdf_fails_closed_when_native_text_is_empty() -> None:
    result = extract_submission_pdf(
        assignment_id="p3_vectors_hw1",
        student_id="S0001",
        submission_id="s1",
        stored_pdf_path=FIXTURES / "inbox" / "S0001.pdf",
    )

    assert result.status == "partial"
    assert result.page_count == 1
    assert result.text_extractable is False
    assert result.extracted_text_preview == ""
    assert "empty_native_text" in result.extraction_warnings


def test_extract_submission_pdf_records_missing_pdf_failure(tmp_path: Path) -> None:
    result = extract_submission_pdf(
        assignment_id="p3_vectors_hw1",
        student_id="S0001",
        submission_id="s1",
        stored_pdf_path=tmp_path / "missing.pdf",
    )

    assert result.status == "failed"
    assert result.extraction_id.endswith(":extraction")
    assert "missing_stored_pdf" in result.extraction_warnings
