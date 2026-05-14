from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from exam_bank.advisory_evidence.common import rel_path, session_slug, stable_source_stem, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    EXAMINER_REPORT_INVENTORY,
    GRADE_THRESHOLD_INVENTORY,
    INVENTORY_SCHEMA,
)
from exam_bank.atomic_json import write_atomic_json
from exam_bank.document_metadata import parse_filename_metadata
from exam_bank.mupdf_tools import quiet_mupdf


DOCUMENT_SPECS = {
    "examiner_report": {
        "source_dir": Path("input/examiner_reports"),
        "output": EXAMINER_REPORT_INVENTORY,
        "output_subdir": "examiner_reports",
    },
    "grade_thresholds": {
        "source_dir": Path("input/grade_thresholds"),
        "output": GRADE_THRESHOLD_INVENTORY,
        "output_subdir": "grade_thresholds",
    },
}


def build_inventory_for_dir(source_dir: str | Path, expected_document_type: str) -> dict[str, Any]:
    source_dir = Path(source_dir)
    warnings: list[str] = []
    paths = sorted(source_dir.glob("*.pdf")) if source_dir.exists() else []
    if not source_dir.exists():
        warnings.append(f"missing_input_dir:{source_dir.as_posix()}")

    records = [_document_record(path, expected_document_type) for path in paths]
    identity_counts = Counter(record["document_identity"] for record in records if record["document_identity"])
    for record in records:
        identity = record["document_identity"]
        if identity and identity_counts[identity] > 1:
            record["warnings"].append(f"duplicate_document_identity:{identity}")
            record["duplicate_identity"] = True

    return {
        "schema": INVENTORY_SCHEMA,
        "generated_at": utc_now_iso(),
        "source_dir": rel_path(source_dir),
        "document_type": expected_document_type,
        "document_count": len(records),
        "documents": records,
        "warnings": warnings,
    }


def build_all_inventories(
    *,
    examiner_reports_dir: str | Path = DOCUMENT_SPECS["examiner_report"]["source_dir"],
    grade_thresholds_dir: str | Path = DOCUMENT_SPECS["grade_thresholds"]["source_dir"],
) -> dict[str, dict[str, Any]]:
    return {
        "examiner_report": build_inventory_for_dir(examiner_reports_dir, "examiner_report"),
        "grade_thresholds": build_inventory_for_dir(grade_thresholds_dir, "grade_thresholds"),
    }


def write_all_inventories(
    *,
    output_root: str | Path = "output/advisory_evidence",
    examiner_reports_dir: str | Path = DOCUMENT_SPECS["examiner_report"]["source_dir"],
    grade_thresholds_dir: str | Path = DOCUMENT_SPECS["grade_thresholds"]["source_dir"],
    dry_run: bool = False,
) -> dict[str, Any]:
    inventories = build_all_inventories(
        examiner_reports_dir=examiner_reports_dir,
        grade_thresholds_dir=grade_thresholds_dir,
    )
    output_root = Path(output_root)
    outputs = {
        "examiner_report": output_root / "inventory" / "examiner_report_inventory.json",
        "grade_thresholds": output_root / "inventory" / "grade_threshold_inventory.json",
    }
    if not dry_run:
        for key, payload in inventories.items():
            write_atomic_json(payload, outputs[key])
    return {
        "dry_run": dry_run,
        "outputs": {key: rel_path(path) for key, path in outputs.items()},
        "inventories": inventories,
    }


def _document_record(path: Path, expected_document_type: str) -> dict[str, Any]:
    metadata = parse_filename_metadata(path)
    warnings = list(metadata.warnings)
    if metadata.document_type != expected_document_type:
        warnings.append(f"document_type_mismatch:expected={expected_document_type}:actual={metadata.document_type or 'unknown'}")
    can_open, page_count, text_length, open_warning = _pdf_readiness(path)
    if open_warning:
        warnings.append(open_warning)
    if can_open and text_length <= 0:
        warnings.append("native_text_empty")
    if not metadata.syllabus:
        warnings.append("missing_syllabus")
    if not metadata.year:
        warnings.append("missing_year")
    if not metadata.session:
        warnings.append("missing_session")

    session_key = metadata.session_key
    identity = f"{metadata.syllabus}_{metadata.year}_{metadata.session}_{expected_document_type}" if metadata.syllabus and metadata.year and metadata.session else ""
    output_filename = stable_source_stem(path) + ".json"
    return {
        "source_path": rel_path(path),
        "filename": path.name,
        "syllabus": metadata.syllabus,
        "subject": metadata.subject,
        "year": metadata.year,
        "session": metadata.session,
        "session_key": session_key,
        "session_slug": session_slug(metadata.session, metadata.year),
        "document_type": expected_document_type,
        "document_identity": identity,
        "duplicate_identity": False,
        "page_count": page_count,
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "can_open": can_open,
        "can_extract_native_text": can_open and text_length > 0,
        "native_text_length": text_length,
        "output_filename": output_filename,
        "warnings": warnings,
    }


def _pdf_readiness(path: Path) -> tuple[bool, int, int, str]:
    try:
        import fitz
    except ImportError:
        return False, 0, 0, "pymupdf_unavailable"
    quiet_mupdf(fitz)
    try:
        with fitz.open(path) as doc:
            lengths = [len(page.get_text("text").strip()) for page in doc]
    except Exception as exc:
        return False, 0, 0, f"pdf_open_failed:{exc.__class__.__name__}"
    return True, len(lengths), sum(lengths), ""

