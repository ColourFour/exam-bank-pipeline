from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_NAME = "exam_bank.text_candidate_metadata.coverage"
SCHEMA_VERSION = 1
SIDECAR_SCHEMA_NAME = "exam_bank.text_candidate_metadata.sidecar"
SIDECAR_SCHEMA_VERSION = 1

TARGET_FIELDS = (
    "crop_pixel_dimensions",
    "normalized_crop_area",
    "crop_confidence",
    "raw_native_text_candidate_window",
    "raw_ocr_text_candidate_window",
    "rejected_candidate_reasons",
    "selector_warnings",
    "selector_structural_warnings",
    "text_line_bounding_boxes",
    "candidate_provenance",
)

STRUCTURAL_WARNING_FLAGS = {
    "question_scope_contaminated",
    "question_subparts_incomplete",
    "weak_question_anchor",
    "weak_question_text",
    "ocr_merged_sparse_lower_region",
    "paper_total_focus_candidate",
}


def load_question_bank(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Question bank must be a list or contain a questions list.")
    return records


def build_metadata_coverage_report(
    records: list[dict[str, Any]],
    *,
    artifact_root: Path | None = None,
    include_records: bool = True,
) -> dict[str, Any]:
    artifact_root = artifact_root or Path("output")
    record_reports = [extract_record_metadata(record, artifact_root=artifact_root) for record in records]
    field_counts: dict[str, dict[str, int]] = {}
    populated_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()

    for record in record_reports:
        source_counts[str(record["candidate_provenance"].get("selected_source") or "unknown")] += 1
        for field in TARGET_FIELDS:
            field_report = record["fields"][field]
            if field_report["present"]:
                field_counts.setdefault(field, {"present": 0, "missing": 0})["present"] += 1
            else:
                field_counts.setdefault(field, {"present": 0, "missing": 0})["missing"] += 1
                missing_reason_counts[f"{field}:{field_report['reason']}"] += 1
            if field_report.get("populated"):
                populated_counts[field] += 1

    report: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "sidecar_schema_name": SIDECAR_SCHEMA_NAME,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "record_count": len(record_reports),
        "target_fields": list(TARGET_FIELDS),
        "field_counts": {field: field_counts.get(field, {"present": 0, "missing": 0}) for field in TARGET_FIELDS},
        "populated_counts": {field: populated_counts.get(field, 0) for field in TARGET_FIELDS},
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "inventory_summary": inventory_summary(),
    }
    if include_records:
        report["records"] = record_reports
    return report


def extract_record_metadata(record: dict[str, Any], *, artifact_root: Path | None = None) -> dict[str, Any]:
    artifact_root = artifact_root or Path("output")
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    crop_diagnostics = (
        notes.get("question_crop_diagnostics") if isinstance(notes.get("question_crop_diagnostics"), dict) else {}
    )
    regions = crop_diagnostics.get("regions") if isinstance(crop_diagnostics.get("regions"), list) else []
    image_path = text_value(record.get("question_image_path") or first_value(record.get("question_image_paths")))
    image_dimensions = read_image_dimensions(artifact_root, image_path)
    normalized_area = compute_normalized_crop_area(crop_diagnostics)
    line_bboxes = extract_text_line_bboxes(crop_diagnostics)
    region_text_bbox_count = sum(
        1 for region in regions if isinstance(region, dict) and isinstance(region.get("text_bbox"), dict)
    )
    selected_source = text_value(notes.get("text_candidate_source") or notes.get("text_source_profile"))
    selected_text = text_value(record.get("question_text"))
    ocr_text = text_value(record.get("ocr_text"))
    decision_reasons = list_values(notes.get("text_candidate_decision_reasons"))
    ocr_rejected_reasons = list_values(notes.get("ocr_rejected_reasons"))
    review_flags = list_values(notes.get("review_flags"))
    validation_flags = list_values(notes.get("validation_flags"))
    extraction_quality_flags = list_values(notes.get("extraction_quality_flags"))
    text_fidelity_flags = list_values(notes.get("text_fidelity_flags"))
    structural_warnings = selector_structural_warnings(notes)

    fields = {
        "crop_pixel_dimensions": field_report(
            image_dimensions is not None,
            populated=image_dimensions is not None,
            reason="question image file missing or unreadable",
            value={"width": image_dimensions[0], "height": image_dimensions[1]} if image_dimensions else None,
        ),
        "normalized_crop_area": field_report(
            normalized_area is not None,
            populated=normalized_area is not None,
            reason="crop diagnostics lack source page dimensions",
            value=normalized_area,
        ),
        "crop_confidence": field_report(
            has_key(notes, "question_crop_confidence"),
            populated=bool(text_value(notes.get("question_crop_confidence"))),
            reason="notes.question_crop_confidence missing",
            value=notes.get("question_crop_confidence"),
        ),
        "raw_native_text_candidate_window": field_report(
            selected_source == "native" and bool(selected_text),
            populated=selected_source == "native" and bool(selected_text),
            reason="raw native candidate text is not stored independently of selected text",
            value={"length": len(selected_text)} if selected_source == "native" and selected_text else None,
        ),
        "raw_ocr_text_candidate_window": field_report(
            bool(ocr_text),
            populated=bool(ocr_text),
            reason="record.ocr_text missing or empty",
            value={"length": len(ocr_text)} if ocr_text else None,
        ),
        "rejected_candidate_reasons": field_report(
            has_key(notes, "ocr_rejected_reasons") or has_key(notes, "text_candidate_decision_reasons"),
            populated=bool(ocr_rejected_reasons),
            reason="candidate rejection reason fields missing",
            value={"ocr_rejected_reasons": ocr_rejected_reasons, "decision_reasons": decision_reasons},
        ),
        "selector_warnings": field_report(
            any(
                has_key(notes, key)
                for key in ("review_flags", "validation_flags", "extraction_quality_flags", "text_fidelity_flags")
            ),
            populated=bool(review_flags or validation_flags or extraction_quality_flags or text_fidelity_flags),
            reason="selector warning flag fields missing",
            value={
                "review_flags": review_flags,
                "validation_flags": validation_flags,
                "extraction_quality_flags": extraction_quality_flags,
                "text_fidelity_flags": text_fidelity_flags,
            },
        ),
        "selector_structural_warnings": field_report(
            has_key(notes, "review_flags") or has_key(notes, "text_candidate_decision_reasons"),
            populated=bool(structural_warnings),
            reason="selector structural warning inputs missing",
            value=structural_warnings,
        ),
        "text_line_bounding_boxes": field_report(
            bool(line_bboxes),
            populated=bool(line_bboxes),
            reason="true text-line bounding boxes are not stored; region text bboxes are partial crop diagnostics",
            value={"line_bbox_count": len(line_bboxes), "region_text_bbox_count": region_text_bbox_count},
        ),
        "candidate_provenance": field_report(
            bool(selected_source or notes.get("text_candidate_decision") or notes.get("text_source_profile")),
            populated=bool(selected_source),
            reason="candidate provenance fields missing",
            value={
                "selected_source": selected_source or "unknown",
                "text_candidate_decision": text_value(notes.get("text_candidate_decision")),
                "text_source_profile": text_value(notes.get("text_source_profile")),
                "ocr_engine": text_value(record.get("ocr_engine") or notes.get("ocr_engine")),
            },
        ),
    }

    return {
        "record_id": text_value(record.get("question_id") or record.get("record_id")),
        "paper_id": text_value(record.get("paper") or record.get("paper_id")),
        "paper_family": text_value(record.get("paper_family")),
        "question_number": text_value(record.get("question_number")),
        "canonical_question_artifact": text_value(record.get("canonical_question_artifact")),
        "question_image_path": image_path,
        "mark_scheme_image_path": text_value(record.get("mark_scheme_image_path")),
        "candidate_provenance": fields["candidate_provenance"]["value"],
        "fields": fields,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Text Candidate Metadata Coverage",
        "",
        "This report inventories metadata available for future advisory text gates and candidate replay. It is measurement-only: it does not change selected question text, OCR/native selection, canonical image paths, or Asterion exports.",
        "",
        f"- Schema: `{report['schema_name']}` v{report['schema_version']}",
        f"- Records inspected: {report['record_count']}",
        f"- Sidecar contract: `{report['sidecar_schema_name']}` v{report['sidecar_schema_version']}",
        "",
        "## Field Coverage",
        "",
        "| Field | Present | Missing | Populated | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    summary = report.get("inventory_summary", {})
    for field in TARGET_FIELDS:
        counts = report["field_counts"][field]
        lines.append(
            f"| `{field}` | {counts['present']} | {counts['missing']} | {report['populated_counts'][field]} | {summary.get(field, '')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Provenance",
            "",
            "| Selected source | Count |",
            "| --- | ---: |",
        ]
    )
    for source, count in report.get("candidate_source_counts", {}).items():
        lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "## Missing Metadata Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in report.get("missing_reason_counts", {}).items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- Available now: crop confidence, selected-source provenance, OCR candidate text, selector warning fields, selector decision/rejection reasons, PDF-space crop region diagnostics, and image pixel dimensions when the referenced PNG exists.",
            "- Partially available: native candidate text is only available when native text was selected, because current records do not retain an independent raw native candidate window for OCR-selected records.",
            "- Missing for replay-grade gates: page-normalized crop area, raw candidate windows for both native and OCR with crop positions, true text-line bounding boxes, and rejected candidate windows beyond reason strings.",
            "- Canonical image evidence remains separate: `canonical_question_artifact`, `question_image_path`, and mark-scheme image paths are references only and are not modified by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def compute_normalized_crop_area(crop_diagnostics: dict[str, Any]) -> float | None:
    page_width, page_height = page_dimensions(crop_diagnostics)
    if not page_width or not page_height:
        return None
    regions = crop_diagnostics.get("regions") if isinstance(crop_diagnostics.get("regions"), list) else []
    area = 0.0
    for region in regions:
        if not isinstance(region, dict):
            continue
        bbox = region.get("final_crop_bbox") if isinstance(region.get("final_crop_bbox"), dict) else None
        if bbox:
            area += bbox_area(bbox)
    if area <= 0:
        return None
    return round(area / (page_width * page_height), 6)


def page_dimensions(crop_diagnostics: dict[str, Any]) -> tuple[float | None, float | None]:
    for width_key, height_key in (("page_width", "page_height"), ("source_page_width", "source_page_height")):
        width = number_value(crop_diagnostics.get(width_key))
        height = number_value(crop_diagnostics.get(height_key))
        if width and height:
            return width, height
    page_bbox = crop_diagnostics.get("page_bbox")
    if isinstance(page_bbox, dict):
        width = number_value(page_bbox.get("width"))
        height = number_value(page_bbox.get("height"))
        if width and height:
            return width, height
        if all(key in page_bbox for key in ("x0", "y0", "x1", "y1")):
            return bbox_width(page_bbox), bbox_height(page_bbox)
    return None, None


def extract_text_line_bboxes(crop_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("text_line_bboxes", "line_bboxes", "text_lines")
    candidates: list[Any] = []
    for field in fields:
        value = crop_diagnostics.get(field)
        if isinstance(value, list):
            candidates.extend(value)
    regions = crop_diagnostics.get("regions") if isinstance(crop_diagnostics.get("regions"), list) else []
    for region in regions:
        if not isinstance(region, dict):
            continue
        for field in fields:
            value = region.get(field)
            if isinstance(value, list):
                candidates.extend(value)
    bboxes = []
    for item in candidates:
        bbox = item.get("bbox") if isinstance(item, dict) and isinstance(item.get("bbox"), dict) else item
        if isinstance(bbox, dict) and all(key in bbox for key in ("x0", "y0", "x1", "y1")):
            bboxes.append({key: number_value(bbox.get(key)) for key in ("x0", "y0", "x1", "y1")})
    return bboxes


def selector_structural_warnings(notes: dict[str, Any]) -> list[str]:
    flags = set(list_values(notes.get("review_flags"))) & STRUCTURAL_WARNING_FLAGS
    reasons = {
        reason
        for reason in list_values(notes.get("text_candidate_decision_reasons")) + list_values(notes.get("ocr_rejected_reasons"))
        if any(token in reason for token in ("suspiciously_short", "merged", "sparse", "lost", "truncated"))
    }
    return sorted(flags | reasons)


def read_image_dimensions(root: Path, relative_path: str) -> tuple[int, int] | None:
    if not relative_path:
        return None
    path = Path(relative_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def field_report(present: bool, *, populated: bool, reason: str, value: Any) -> dict[str, Any]:
    return {
        "present": bool(present),
        "populated": bool(populated),
        "reason": "" if present else reason,
        "value": value if present else None,
    }


def inventory_summary() -> dict[str, str]:
    return {
        "crop_pixel_dimensions": "Resolved from existing question PNG files; not embedded in canonical records.",
        "normalized_crop_area": "Requires source page dimensions; current crop diagnostics expose bboxes but not page size.",
        "crop_confidence": "Stored in notes.question_crop_confidence.",
        "raw_native_text_candidate_window": "Only recoverable from selected question_text when native was selected.",
        "raw_ocr_text_candidate_window": "Stored in record.ocr_text.",
        "rejected_candidate_reasons": "Stored as OCR rejection and selector decision reason strings.",
        "selector_warnings": "Stored as review, validation, extraction-quality, and text-fidelity flags.",
        "selector_structural_warnings": "Derived from existing selector flags and structural decision reasons.",
        "text_line_bounding_boxes": "Not stored as line boxes; region-level text bboxes are only partial diagnostics.",
        "candidate_provenance": "Stored in selector source/profile/decision fields.",
    }


def bbox_area(bbox: dict[str, Any]) -> float:
    return max(0.0, bbox_width(bbox) or 0.0) * max(0.0, bbox_height(bbox) or 0.0)


def bbox_width(bbox: dict[str, Any]) -> float | None:
    x0 = number_value(bbox.get("x0"))
    x1 = number_value(bbox.get("x1"))
    if x0 is None or x1 is None:
        return None
    return x1 - x0


def bbox_height(bbox: dict[str, Any]) -> float | None:
    y0 = number_value(bbox.get("y0"))
    y1 = number_value(bbox.get("y1"))
    if y0 is None or y1 is None:
        return None
    return y1 - y0


def text_value(value: Any) -> str:
    return str(value or "").strip()


def list_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def first_value(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return ""


def has_key(mapping: dict[str, Any], key: str) -> bool:
    return key in mapping


def number_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
