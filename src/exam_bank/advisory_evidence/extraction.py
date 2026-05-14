from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from exam_bank.advisory_evidence.common import load_json, rel_path, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    EXAMINER_REPORT_INVENTORY,
    EXAMINER_TEXT_DIR,
    EXTRACTED_TEXT_SCHEMA,
    GRADE_THRESHOLD_INVENTORY,
    GRADE_THRESHOLD_TEXT_DIR,
)
from exam_bank.atomic_json import write_atomic_json
from exam_bank.mupdf_tools import quiet_mupdf


def extract_all_advisory_text(
    *,
    inventory_dir: str | Path = "output/advisory_evidence/inventory",
    output_root: str | Path = "output/advisory_evidence",
    dry_run: bool = False,
) -> dict[str, Any]:
    inventory_dir = Path(inventory_dir)
    output_root = Path(output_root)
    specs = [
        (
            inventory_dir / EXAMINER_REPORT_INVENTORY.name,
            output_root / "extracted_text" / "examiner_reports",
        ),
        (
            inventory_dir / GRADE_THRESHOLD_INVENTORY.name,
            output_root / "extracted_text" / "grade_thresholds",
        ),
    ]
    summaries: list[dict[str, Any]] = []
    for inventory_path, output_dir in specs:
        inventory = load_json(inventory_path, default={"documents": [], "warnings": [f"missing_inventory:{inventory_path}"]})
        summaries.append(extract_text_from_inventory(inventory, output_dir=output_dir, dry_run=dry_run))
    return {"dry_run": dry_run, "summaries": summaries}


def extract_text_from_inventory(inventory: dict[str, Any], *, output_dir: str | Path, dry_run: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    outputs: list[str] = []
    failures = 0
    for document in sorted(inventory.get("documents", []), key=lambda item: item.get("source_path", "")):
        output_path = output_dir / str(document.get("output_filename") or "")
        if not output_path.name:
            continue
        payload = extract_native_pdf_text(document)
        outputs.append(rel_path(output_path))
        if payload.get("warnings"):
            failures += int(any(str(warning).startswith("extraction_failed") for warning in payload["warnings"]))
        if not dry_run:
            write_atomic_json(payload, output_path)
    return {
        "schema": EXTRACTED_TEXT_SCHEMA,
        "source_inventory": inventory.get("source_dir", ""),
        "document_type": inventory.get("document_type", ""),
        "output_dir": rel_path(output_dir),
        "document_count": len(outputs),
        "failed_extractions": failures,
        "outputs": outputs,
        "warnings": list(inventory.get("warnings", [])),
    }


def extract_native_pdf_text(document: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(str(document.get("source_path", "")))
    warnings = list(document.get("warnings", []))
    page_text: list[dict[str, Any]] = []
    try:
        import fitz
    except ImportError:
        warnings.append("pymupdf_unavailable")
        return _empty_payload(document, warnings)
    quiet_mupdf(fitz)
    try:
        with fitz.open(source_path) as doc:
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if not text.strip():
                    warnings.append(f"empty_page:{index}")
                page_text.append({"page_number": index, "text": text, "text_length": len(text.strip())})
    except Exception as exc:
        warnings.append(f"extraction_failed:{exc.__class__.__name__}")
        return _empty_payload(document, warnings)

    raw_text = "\n".join(page["text"] for page in page_text)
    text_length = len(raw_text.strip())
    if text_length < 100:
        warnings.append("low_native_text_length")

    return {
        "schema": EXTRACTED_TEXT_SCHEMA,
        "generated_at": utc_now_iso(),
        "source_path": document.get("source_path", ""),
        "syllabus": document.get("syllabus", ""),
        "year": document.get("year", ""),
        "session": document.get("session", ""),
        "session_key": document.get("session_key", ""),
        "session_slug": document.get("session_slug", ""),
        "document_type": document.get("document_type", ""),
        "document_identity": document.get("document_identity", ""),
        "page_count": len(page_text),
        "extraction_method": "native_pdf_text",
        "text_length": text_length,
        "raw_text": raw_text,
        "page_text": page_text,
        "detected_headers": _detected_headers(raw_text),
        "warnings": warnings,
    }


def _empty_payload(document: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "schema": EXTRACTED_TEXT_SCHEMA,
        "generated_at": utc_now_iso(),
        "source_path": document.get("source_path", ""),
        "syllabus": document.get("syllabus", ""),
        "year": document.get("year", ""),
        "session": document.get("session", ""),
        "session_key": document.get("session_key", ""),
        "session_slug": document.get("session_slug", ""),
        "document_type": document.get("document_type", ""),
        "document_identity": document.get("document_identity", ""),
        "page_count": 0,
        "extraction_method": "native_pdf_text",
        "text_length": 0,
        "raw_text": "",
        "page_text": [],
        "detected_headers": [],
        "warnings": warnings,
    }


def _detected_headers(text: str) -> list[str]:
    headers: list[str] = []
    patterns = [
        r"^\s*Paper\s+9709\s*/\s*[1-6][0-9]\b.*$",
        r"^\s*Key messages\s*$",
        r"^\s*General comments\s*$",
        r"^\s*Comments on specific questions\s*$",
        r"^\s*Component\s+[1-6][0-9]\s*$",
        r"^\s*Option\s*$",
    ]
    for line in text.splitlines():
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned and cleaned not in headers:
                headers.append(cleaned)
    return headers

