from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .core.paper_identity import IdentityError, PaperIdentity, paper_identity_from_parts
from .identifiers import normalize_question_id
from .image_rendering import render_question_image
from .pdf_extract import extract_pdf_layout
from .question_detection import detect_question_spans


@dataclass(frozen=True)
class _SelectedRecord:
    row: dict[str, Any]
    question_number: str
    identity: PaperIdentity
    question_pdf: Path


def regenerate_question_pngs_from_question_bank(
    *,
    question_bank_path: str | Path,
    config: AppConfig,
    question_ids: Iterable[str] | None = None,
    all_records: bool = False,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    requested = _normalize_requested_ids(question_ids or [])
    rows = _load_question_rows(question_bank_path)
    selected = _select_records(
        rows,
        requested_ids=requested,
        all_records=all_records,
        year_min=year_min,
        year_max=year_max,
        limit=limit,
    )

    grouped: dict[Path, list[_SelectedRecord]] = defaultdict(list)
    skipped: list[dict[str, str]] = []
    for row in selected:
        try:
            record = _selected_record(row)
        except (IdentityError, ValueError) as exc:
            skipped.append({"question_id": str(row.get("question_id") or ""), "reason": str(exc)})
            continue
        grouped[record.question_pdf].append(record)

    outputs: list[dict[str, Any]] = []
    for question_pdf, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        if not question_pdf.exists():
            for record in records:
                skipped.append({"question_id": str(record.row.get("question_id") or ""), "reason": f"missing source_pdf: {question_pdf}"})
            continue
        layouts = extract_pdf_layout(question_pdf, config)
        spans = {normalize_question_id(span.question_number): span for span in detect_question_spans(layouts, question_pdf, config)}
        for record in records:
            span = spans.get(record.question_number)
            if span is None:
                outputs.append(_failed_output(record, "span_not_found"))
                continue
            rendered = render_question_image(question_pdf, span, layouts, config, identity=record.identity)
            outputs.append(
                {
                    "question_id": record.row.get("question_id"),
                    "canonical_question_artifact": record.row.get("canonical_question_artifact"),
                    "source_pdf": str(question_pdf),
                    "question_number": record.question_number,
                    "status": "pass" if rendered.screenshot_path else "fail",
                    "failure_reason": "" if rendered.screenshot_path else "render_failed",
                    "image_path": str(rendered.screenshot_path) if rendered.screenshot_path else "",
                    "review_flags": rendered.review_flags,
                    "crop_uncertain": rendered.crop_uncertain,
                    "crop_regions": rendered.crop_diagnostics.get("regions") if isinstance(rendered.crop_diagnostics, dict) else [],
                }
            )

    selected_keys: set[str] = set()
    for row in selected:
        selected_keys.update(_record_match_keys(row))
    missing_requested = sorted(requested - selected_keys)
    return {
        "selected_count": len(selected),
        "rendered_count": sum(1 for item in outputs if item["image_path"]),
        "failed_count": sum(1 for item in outputs if not item["image_path"]),
        "skipped_count": len(skipped),
        "missing_requested_ids": missing_requested,
        "outputs": sorted(outputs, key=lambda item: str(item.get("canonical_question_artifact") or item.get("question_id") or "")),
        "skipped": skipped,
    }


def _failed_output(record: _SelectedRecord, reason: str) -> dict[str, Any]:
    return {
        "question_id": record.row.get("question_id"),
        "canonical_question_artifact": record.row.get("canonical_question_artifact"),
        "source_pdf": str(record.question_pdf),
        "question_number": record.question_number,
        "status": "fail",
        "failure_reason": reason,
        "image_path": "",
        "review_flags": ["question_image_missing", reason],
        "crop_uncertain": True,
        "crop_regions": [],
    }


def _load_question_rows(question_bank_path: str | Path) -> list[dict[str, Any]]:
    path = Path(question_bank_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Question bank does not contain a questions list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _select_records(
    rows: list[dict[str, Any]],
    *,
    requested_ids: set[str],
    all_records: bool,
    year_min: int | None,
    year_max: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("canonical_question_artifact") and not row.get("question_image_path"):
            continue
        if requested_ids and not (_record_match_keys(row) & requested_ids):
            continue
        if not requested_ids and not all_records:
            continue
        year = _int_or_none(row.get("canonical_year_folder") or row.get("year"))
        if year_min is not None and (year is None or year < year_min):
            continue
        if year_max is not None and (year is None or year > year_max):
            continue
        selected.append(row)
        if limit and len(selected) >= limit:
            break
    return selected


def _selected_record(row: dict[str, Any]) -> _SelectedRecord:
    notes = row.get("notes") if isinstance(row.get("notes"), dict) else {}
    source_pdf = str(notes.get("source_pdf") or row.get("source_pdf") or "")
    if not source_pdf:
        raise ValueError("missing source_pdf")
    question_number = normalize_question_id(str(row.get("question_number") or ""))
    if not question_number:
        raise ValueError("missing question_number")
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family=str(row.get("paper_family") or ""),
        year=str(row.get("canonical_year_folder") or ""),
        session=str(row.get("canonical_session") or ""),
        component=str(notes.get("source_paper_code") or ""),
        question_number=question_number,
        expected_question_id=str(row.get("question_id") or ""),
    )
    return _SelectedRecord(
        row=row,
        question_number=question_number,
        identity=identity,
        question_pdf=Path(source_pdf),
    )


def _normalize_requested_ids(values: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        for part in str(value or "").replace("\n", ",").split(","):
            cleaned = _normalize_lookup_key(part)
            if cleaned:
                normalized.add(cleaned)
    return normalized


def _record_match_keys(row: dict[str, Any]) -> set[str]:
    keys = {_normalize_lookup_key(row.get("question_id"))}
    for field in ("canonical_question_artifact", "question_image_path"):
        value = str(row.get(field) or "")
        if not value:
            continue
        path = Path(value)
        stem = path.stem
        keys.add(_normalize_lookup_key(stem))
        keys.add(_normalize_lookup_key(stem.removesuffix("_question")))
        keys.add(_normalize_lookup_key(path.name))
    return {key for key in keys if key}


def _normalize_lookup_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = Path(text).name if "/" in text else text
    text = text.removesuffix(".png")
    return text.strip()


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
