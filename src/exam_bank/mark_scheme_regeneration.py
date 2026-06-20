from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .config import AppConfig
from .core.paper_identity import IdentityError, PaperIdentity, paper_identity_from_parts
from .identifiers import normalize_question_id
from .mark_schemes import render_mark_scheme_images


@dataclass(frozen=True)
class _SelectedRecord:
    row: dict[str, Any]
    question_number: str
    identity: PaperIdentity
    mark_scheme_pdf: Path


def regenerate_mark_scheme_pngs_from_question_bank(
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
        grouped[record.mark_scheme_pdf].append(record)

    outputs: list[dict[str, Any]] = []
    for mark_scheme_pdf, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        expected_numbers = [record.question_number for record in records]
        result = render_mark_scheme_images(
            mark_scheme_pdf,
            config,
            expected_numbers,
            question_marks={record.question_number: _int_or_none(record.row.get("question_solution_marks")) for record in records},
            question_subparts={record.question_number: _string_list(record.row.get("subparts")) for record in records},
            question_identities={record.question_number: record.identity for record in records},
            clear_stale=False,
        )
        for record in records:
            rendered = result.get(record.question_number)
            outputs.append(
                {
                    "question_id": record.row.get("question_id"),
                    "canonical_mark_scheme_artifact": record.row.get("canonical_mark_scheme_artifact"),
                    "source_pdf": str(mark_scheme_pdf),
                    "question_number": record.question_number,
                    "status": rendered.mapping_status if rendered else "fail",
                    "failure_reason": rendered.failure_reason if rendered else "missing_result",
                    "crop_method": _debug_crop_method(rendered.mapping_method if rendered else ""),
                    "image_path": str(rendered.image_path) if rendered and rendered.image_path else "",
                    "review_flags": rendered.review_flags if rendered else ["markscheme_image_missing"],
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
        "outputs": sorted(outputs, key=lambda item: str(item.get("canonical_mark_scheme_artifact") or item.get("question_id") or "")),
        "skipped": skipped,
        "debug_jsonl": str(config.output.debug_dir / "mark_scheme_crop_debug.jsonl"),
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
        if not row.get("canonical_mark_scheme_artifact") and not row.get("mark_scheme_image_path"):
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
    mark_scheme_source = str(notes.get("mark_scheme_source_pdf") or "")
    if not mark_scheme_source:
        raise ValueError("missing mark_scheme_source_pdf")
    question_number = normalize_question_id(str(row.get("question_number") or ""))
    if not question_number:
        raise ValueError("missing question_number")
    artifact_parts = _identity_parts_from_mark_scheme_artifact(row)
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family=str(row.get("paper_family") or ""),
        year=artifact_parts.get("year") or str(row.get("canonical_year_folder") or ""),
        session=artifact_parts.get("session") or str(row.get("canonical_session") or ""),
        component=artifact_parts.get("component") or str(notes.get("source_paper_code") or ""),
        question_number=question_number,
        expected_question_id=str(row.get("question_id") or ""),
    )
    return _SelectedRecord(
        row=row,
        question_number=question_number,
        identity=identity,
        mark_scheme_pdf=Path(mark_scheme_source),
    )


def _identity_parts_from_mark_scheme_artifact(row: dict[str, Any]) -> dict[str, str]:
    for field in ("canonical_mark_scheme_artifact", "mark_scheme_image_path"):
        value = str(row.get(field) or "")
        if not value:
            continue
        match = re.search(
            r"_(?P<year>\d{4})_(?P<session>[msw]\d{2})_(?P<component>\d{2})_ms_q\d{2}_markscheme(?:_v\d+)?\.png$",
            Path(value).name,
            re.IGNORECASE,
        )
        if match:
            return {key: match.group(key) for key in ("year", "session", "component")}
    return {}


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
    for field in ("canonical_mark_scheme_artifact", "mark_scheme_image_path"):
        value = str(row.get(field) or "")
        if not value:
            continue
        path = Path(value)
        stem = path.stem
        keys.add(_normalize_lookup_key(stem))
        keys.add(_normalize_lookup_key(stem.removesuffix("_markscheme")))
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _debug_crop_method(mapping_method: str) -> str:
    if mapping_method == "table_row_block":
        return "table_grid"
    if mapping_method in {"table_grid", "ocr_left_column"}:
        return mapping_method
    return "fallback"
