from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import fitz

from exam_bank.mupdf_tools import quiet_mupdf
from exam_bank.submissions.models import SubmissionExtractionResult


quiet_mupdf(fitz)

DEFAULT_PREVIEW_CHARS = 320
SUSPICIOUS_PAGE_COUNT = 50


def extract_submission_pdf(
    *,
    assignment_id: str,
    student_id: str,
    submission_id: str,
    stored_pdf_path: str | Path,
    created_at: datetime | None = None,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> SubmissionExtractionResult:
    timestamp = created_at or datetime.now(timezone.utc)
    path = Path(stored_pdf_path)
    warnings: list[str] = []
    page_count = 0
    preview = ""

    if not path.exists():
        return _result(
            assignment_id=assignment_id,
            student_id=student_id,
            submission_id=submission_id,
            stored_pdf_path=stored_pdf_path,
            status="failed",
            page_count=0,
            text_extractable=False,
            extracted_text_preview="",
            extraction_warnings=["missing_stored_pdf"],
            created_at=timestamp,
        )

    try:
        with fitz.open(path) as doc:
            if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
                return _result(
                    assignment_id=assignment_id,
                    student_id=student_id,
                    submission_id=submission_id,
                    stored_pdf_path=stored_pdf_path,
                    status="failed",
                    page_count=int(getattr(doc, "page_count", 0) or 0),
                    text_extractable=False,
                    extracted_text_preview="",
                    extraction_warnings=["pdf_encrypted"],
                    created_at=timestamp,
                )

            page_count = int(doc.page_count)
            if page_count <= 0:
                warnings.append("pdf_has_no_pages")
            if page_count > SUSPICIOUS_PAGE_COUNT:
                warnings.append("suspicious_page_count")

            text_chunks = [page.get_text("text") for page in doc]
    except Exception as exc:
        return _result(
            assignment_id=assignment_id,
            student_id=student_id,
            submission_id=submission_id,
            stored_pdf_path=stored_pdf_path,
            status="failed",
            page_count=page_count,
            text_extractable=False,
            extracted_text_preview="",
            extraction_warnings=[f"pdf_extraction_failed:{exc.__class__.__name__}"],
            created_at=timestamp,
        )

    normalized_text = _normalize_preview_text("\n".join(text_chunks))
    if not normalized_text:
        warnings.append("empty_native_text")
        return _result(
            assignment_id=assignment_id,
            student_id=student_id,
            submission_id=submission_id,
            stored_pdf_path=stored_pdf_path,
            status="partial",
            page_count=page_count,
            text_extractable=False,
            extracted_text_preview="",
            extraction_warnings=warnings,
            created_at=timestamp,
        )

    preview = normalized_text[: max(0, preview_chars)]
    return _result(
        assignment_id=assignment_id,
        student_id=student_id,
        submission_id=submission_id,
        stored_pdf_path=stored_pdf_path,
        status="extracted",
        page_count=page_count,
        text_extractable=True,
        extracted_text_preview=preview,
        extraction_warnings=warnings,
        created_at=timestamp,
    )


def _result(
    *,
    assignment_id: str,
    student_id: str,
    submission_id: str,
    stored_pdf_path: str | Path,
    status: str,
    page_count: int,
    text_extractable: bool,
    extracted_text_preview: str,
    extraction_warnings: list[str],
    created_at: datetime,
) -> SubmissionExtractionResult:
    return SubmissionExtractionResult(
        extraction_id=f"{assignment_id}:{student_id}:{submission_id}:extraction",
        assignment_id=assignment_id,
        student_id=student_id,
        submission_id=submission_id,
        stored_pdf_path=str(stored_pdf_path),
        status=status,
        page_count=page_count,
        text_extractable=text_extractable,
        extracted_text_preview=extracted_text_preview,
        extraction_warnings=extraction_warnings,
        created_at=created_at,
    )


def _normalize_preview_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())
