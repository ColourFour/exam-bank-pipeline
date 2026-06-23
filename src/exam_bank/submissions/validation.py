from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from exam_bank.mupdf_tools import quiet_mupdf
from exam_bank.submissions.models import Assignment

quiet_mupdf(fitz)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf(path: Path, assignment: Assignment) -> list[str]:
    reasons: list[str] = []

    if not path.exists():
        return ["missing_file"]
    if path.suffix.lower() != ".pdf" or "pdf" not in {item.lower().lstrip(".") for item in assignment.accepted_file_types}:
        reasons.append("not_pdf")

    size = path.stat().st_size
    if size == 0:
        reasons.append("empty_file")
    max_bytes = assignment.max_file_size_mb * 1024 * 1024
    if max_bytes >= 0 and size > max_bytes:
        reasons.append("file_too_large")

    if "not_pdf" in reasons or "empty_file" in reasons:
        return reasons

    try:
        with fitz.open(path) as doc:
            if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
                reasons.append("pdf_encrypted")
            elif doc.page_count <= 0:
                reasons.append("pdf_has_no_pages")
    except Exception:
        reasons.append("pdf_open_failed")

    return reasons
