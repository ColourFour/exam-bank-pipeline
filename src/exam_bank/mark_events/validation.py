from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.mark_events import (
    MARK_EVENTS_SCHEMA_NAME,
    MARK_EVENTS_SCHEMA_VERSION,
    MARK_EVENTS_VALIDATION_SCHEMA_NAME,
)
from exam_bank.mark_events.parsing import normalize_part_path
from exam_bank.mark_events.sidecar import SERIOUS_REVIEW_FLAGS


VALID_EXTRACTION_STATUSES = {"parsed", "partial", "review", "failed"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def validate_mark_events(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    mark_events_path: str | Path = "output/json/question_bank.mark_events.v1.json",
    artifact_root: str | Path = "output",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    question_bank = _load_json(question_bank_path)
    sidecar = _load_json(mark_events_path)
    question_records = _question_records(question_bank)
    by_question = {str(record.get("question_id")): record for record in question_records if record.get("question_id")}
    errors: list[str] = []
    warnings: list[str] = []

    if sidecar.get("schema_name") != MARK_EVENTS_SCHEMA_NAME:
        errors.append(f"schema_name_mismatch:expected={MARK_EVENTS_SCHEMA_NAME}:actual={sidecar.get('schema_name')}")
    if int(sidecar.get("schema_version") or 0) != MARK_EVENTS_SCHEMA_VERSION:
        errors.append(f"schema_version_mismatch:expected={MARK_EVENTS_SCHEMA_VERSION}:actual={sidecar.get('schema_version')}")
    records = [record for record in sidecar.get("records", []) if isinstance(record, dict)]
    if sidecar.get("record_count") != len(records):
        errors.append(f"record_count_mismatch:declared={sidecar.get('record_count')}:actual={len(records)}")

    for index, record in enumerate(records):
        _validate_record(index, record, by_question, Path(artifact_root), errors, warnings)

    report = {
        "schema_name": MARK_EVENTS_VALIDATION_SCHEMA_NAME,
        "schema_version": MARK_EVENTS_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "notes": [
            "Mark-event evidence is advisory-only.",
            "safe_for_marking_use must remain false for generated records.",
        ],
    }
    if output_path:
        write_atomic_json(report, output_path)
    return report


def _validate_record(
    index: int,
    record: dict[str, Any],
    question_records: dict[str, dict[str, Any]],
    artifact_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    question_id = str(record.get("question_id") or "")
    source_record = question_records.get(question_id)
    if source_record is None:
        errors.append(f"orphan_mark_event_record:{index}:{question_id}")
    if record.get("safe_for_marking_use") is not False:
        errors.append(f"unsafe_marking_claim:{index}:{question_id}")
    if record.get("extraction_status") not in VALID_EXTRACTION_STATUSES:
        errors.append(f"invalid_extraction_status:{index}:{question_id}:{record.get('extraction_status')}")
    if normalize_part_path(record.get("part_path")) != []:
        errors.append(f"invalid_record_part_path:{index}:{question_id}:{record.get('part_path')}")

    source_path = str(record.get("source_mark_scheme_image_path") or "")
    if source_path:
        exists = _path_exists(source_path, artifact_root)
        if not exists and not _source_has_known_missing_companion(source_record):
            errors.append(f"missing_mark_scheme_image_file:{index}:{question_id}:{source_path}")
        if record.get("source_mark_scheme_image_exists") is True and not exists:
            errors.append(f"source_image_exists_flag_wrong:{index}:{question_id}:{source_path}")
    elif not _source_has_known_missing_companion(source_record):
        errors.append(f"missing_mark_scheme_image_path:{index}:{question_id}")

    detected = _int_or_none(record.get("total_marks_detected"))
    expected = _int_or_none(record.get("total_marks_expected"))
    if detected is not None and expected is not None:
        if detected > expected:
            errors.append(f"parsed_total_exceeds_expected:{index}:{question_id}:detected={detected}:expected={expected}")
        elif detected != expected:
            warnings.append(f"parsed_total_mismatch:{index}:{question_id}:detected={detected}:expected={expected}")
        if record.get("total_marks_match") is not (detected == expected):
            errors.append(f"total_match_flag_wrong:{index}:{question_id}")
    if expected is not None and expected <= 0:
        errors.append(f"non_positive_expected_total:{index}:{question_id}:{expected}")
    if detected is not None and detected <= 0:
        errors.append(f"non_positive_detected_total:{index}:{question_id}:{detected}")

    event_ids: set[str] = set()
    for event_index, event in enumerate(record.get("mark_events") or []):
        _validate_event(index, event_index, question_id, event, event_ids, errors, warnings)

    if record.get("safe_for_advisory_use") is True:
        serious = set(record.get("review_flags") or []) & SERIOUS_REVIEW_FLAGS
        if serious:
            errors.append(f"safe_for_advisory_with_serious_flags:{index}:{question_id}:{','.join(sorted(serious))}")
        if not record.get("mark_events"):
            errors.append(f"safe_for_advisory_without_events:{index}:{question_id}")
        if record.get("total_marks_match") is not True:
            errors.append(f"safe_for_advisory_without_total_match:{index}:{question_id}")


def _validate_event(
    record_index: int,
    event_index: int,
    question_id: str,
    event: dict[str, Any],
    event_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    event_id = str(event.get("event_id") or "")
    if not event_id:
        errors.append(f"missing_event_id:{record_index}:{event_index}:{question_id}")
    elif event_id in event_ids:
        errors.append(f"duplicate_event_id:{record_index}:{event_index}:{question_id}:{event_id}")
    event_ids.add(event_id)
    if normalize_part_path(event.get("part_path")) is None:
        errors.append(f"invalid_event_part_path:{record_index}:{event_index}:{question_id}:{event.get('part_path')}")
    value = event.get("mark_value")
    if not isinstance(value, int) or value <= 0:
        errors.append(f"invalid_mark_value:{record_index}:{event_index}:{question_id}:{value}")
    if event.get("confidence") not in VALID_CONFIDENCE:
        errors.append(f"invalid_event_confidence:{record_index}:{event_index}:{question_id}:{event.get('confidence')}")
    if event.get("mark_type") == "unknown" and "unknown_mark_code" not in set(event.get("review_flags") or []):
        warnings.append(f"unknown_mark_code_without_review_flag:{record_index}:{event_index}:{question_id}")
    for dependency in event.get("depends_on_event_ids") or []:
        if dependency not in event_ids:
            warnings.append(f"dependency_not_prior_event:{record_index}:{event_index}:{question_id}:{dependency}")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _source_has_known_missing_companion(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    reason = str(notes.get("mapping_failure_reason") or "").lower()
    path = str(record.get("mark_scheme_image_path") or "").strip()
    return not path and any(token in reason for token in ["missing", "no_row", "no_valid_answer_table", "partial_question_block"])


def _path_exists(path_value: str, artifact_root: Path) -> bool:
    path = Path(path_value)
    if path.is_absolute():
        return path.is_file()
    return (artifact_root / path).is_file() or path.is_file()


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

