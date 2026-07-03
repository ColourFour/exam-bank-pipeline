#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from exam_bank.config import AppConfig
from exam_bank.identifiers import normalize_question_id
from exam_bank.mark_scheme_regeneration import regenerate_mark_scheme_pngs_from_question_bank
from exam_bank.mark_schemes import (
    _detect_formulaic_left_margin_mark_scheme_anchors,
    _extract_mark_scheme_words,
)
from exam_bank.pdf_extract import extract_pdf_layout


DEFAULT_RECORDS = [
    "32winter09_q05",
    "32winter09_q10",
    "33summer10_q08",
    "33summer12_q09",
    "31summer15_q06",
    "32summer15_q10",
    "33summer15_q04",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair formulaic CAIE mark-scheme segmentation failures.")
    parser.add_argument("--question-bank", required=True)
    parser.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    parser.add_argument("--report", required=True, help="Report path prefix, without .json/.md")
    parser.add_argument("--write", action="store_true", help="Write regenerated crop PNGs.")
    args = parser.parse_args()

    if not args.write:
        raise SystemExit("This repair command writes validated crop PNGs; rerun with --write.")

    config = AppConfig()
    question_bank_path = Path(args.question_bank)
    rows = _load_question_rows(question_bank_path)
    by_id = {str(row.get("question_id") or ""): row for row in rows}

    result = regenerate_mark_scheme_pngs_from_question_bank(
        question_bank_path=question_bank_path,
        config=config,
        question_ids=args.records,
    )
    outputs = {str(item.get("question_id") or ""): item for item in result.get("outputs", [])}
    records: list[dict[str, Any]] = []
    for question_id in args.records:
        row = by_id.get(question_id, {})
        output = outputs.get(question_id, {})
        source_pdf = _mark_scheme_source_pdf(row)
        page_numbers = [int(page) for page in output.get("page_numbers") or []]
        image_path = str(output.get("image_path") or "")
        records.append(
            {
                "question_id": question_id,
                "source_mark_scheme_pdf": source_pdf,
                "expected_answer_crop_path": _expected_answer_crop_path(row),
                "current_segmentation_failure_reason": _current_failure_reason(row),
                "status": output.get("status") or "fail",
                "segmentation_strategy": output.get("crop_method") or "manual_review_required",
                "image_path": image_path,
                "crop_dimensions": _image_dimensions(image_path),
                "source_page_numbers": page_numbers,
                "page_span": "multi_page" if len(set(page_numbers)) > 1 else "single_page",
                "nearest_detected_anchors": _nearest_formulaic_anchors(source_pdf, row, config),
                "failure_reason": output.get("failure_reason") or "",
                "review_flags": output.get("review_flags") or [],
            }
        )

    payload = {
        "question_bank": str(question_bank_path),
        "records_attempted": len(args.records),
        "recovered_count": sum(1 for record in records if record["image_path"]),
        "still_failed_count": sum(1 for record in records if not record["image_path"]),
        "regeneration_summary": {
            key: result.get(key)
            for key in ("selected_count", "rendered_count", "failed_count", "skipped_count", "missing_requested_ids", "debug_jsonl")
        },
        "records": records,
    }
    report_prefix = Path(args.report)
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    report_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report_prefix.with_suffix(".md").write_text(_markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["regeneration_summary"], indent=2, sort_keys=True))
    print(f"Wrote {report_prefix.with_suffix('.json')} and {report_prefix.with_suffix('.md')}")
    return 0


def _load_question_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _mark_scheme_source_pdf(row: dict[str, Any]) -> str:
    notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
    return str(notes.get("mark_scheme_source_pdf") or "")


def _expected_answer_crop_path(row: dict[str, Any]) -> str:
    for value in (
        row.get("canonical_mark_scheme_artifact"),
        row.get("mark_scheme_image_path"),
        _nested_canonical_path(row),
    ):
        if value:
            return str(value)
    return ""


def _nested_canonical_path(row: dict[str, Any]) -> str:
    notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
    detected = notes.get("mark_scheme_structure_detected") if isinstance(notes.get("mark_scheme_structure_detected"), dict) else {}
    asset_identity = detected.get("asset_identity") if isinstance(detected.get("asset_identity"), dict) else {}
    return str(asset_identity.get("canonical_path") or "")


def _current_failure_reason(row: dict[str, Any]) -> str:
    notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
    return str(notes.get("missing_mark_scheme_reason") or notes.get("mark_scheme_failure_reason") or "segmentation_failure")


def _image_dimensions(path: str) -> dict[str, int] | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return {"width": image.width, "height": image.height}


def _nearest_formulaic_anchors(source_pdf: str, row: dict[str, Any], config: AppConfig) -> dict[str, str]:
    if not source_pdf:
        return {"before": "", "target": "", "after": ""}
    pdf_path = Path(source_pdf)
    if not pdf_path.exists():
        return {"before": "", "target": "", "after": ""}
    question_number = normalize_question_id(str(row.get("question_number") or ""))
    layouts = extract_pdf_layout(pdf_path, config)
    words = _extract_mark_scheme_words(pdf_path)
    anchors = _detect_formulaic_left_margin_mark_scheme_anchors(layouts, words, config)
    labels = [anchor.question_number for anchor in anchors]
    try:
        index = labels.index(question_number)
    except ValueError:
        return {"before": labels[-1] if labels else "", "target": "", "after": ""}
    return {
        "before": labels[index - 1] if index > 0 else "",
        "target": labels[index],
        "after": labels[index + 1] if index + 1 < len(labels) else "",
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mark Scheme Segmentation Repair",
        "",
        f"- Records attempted: {payload['records_attempted']}",
        f"- Recovered: {payload['recovered_count']}",
        f"- Still failed: {payload['still_failed_count']}",
        "",
        "| Question | Status | Strategy | Pages | Dimensions | Output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in payload["records"]:
        dims = record.get("crop_dimensions") or {}
        dim_text = f"{dims.get('width')}x{dims.get('height')}" if dims else ""
        lines.append(
            "| {question_id} | {status} | {segmentation_strategy} | {pages} | {dims} | {path} |".format(
                question_id=record["question_id"],
                status=record["status"],
                segmentation_strategy=record["segmentation_strategy"],
                pages=",".join(str(page) for page in record["source_page_numbers"]),
                dims=dim_text,
                path=record["image_path"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
