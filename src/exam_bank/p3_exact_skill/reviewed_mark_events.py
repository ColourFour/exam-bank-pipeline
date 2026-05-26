from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    DEFAULT_REVIEWED_MARK_EVENTS_PATH,
    P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA,
    P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION,
    P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_VALIDATION_SCHEMA,
)

ALLOWED_MARK_EVENT_STATUSES = {"approved", "reviewed", "rejected", "advisory"}
GENERATION_SATISFYING_MARK_EVENT_STATUSES = {"approved", "reviewed"}

REQUIRED_DECISION_FIELDS = {
    "schema_version",
    "decision_id",
    "event_id",
    "source_question_id",
    "part_path",
    "reviewer",
    "reviewed_at",
    "status",
    "rationale",
}

STATUS_SEMANTICS = {
    "approved": "Explicitly approved for Content Lab generation gating.",
    "reviewed": "Human-reviewed; accepted for generation gating by this conservative validator.",
    "rejected": "Reviewed and must not be used for generation gating.",
    "advisory": "Advisory-only context and must not unlock generation.",
}


def validate_reviewed_mark_events(
    *,
    reviewed_mark_events_path: str | Path = DEFAULT_REVIEWED_MARK_EVENTS_PATH,
    question_bank_path: str | Path = "output/json/question_bank.json",
    mark_events_path: str | Path | None = "output/json/question_bank.mark_events.v1.json",
    base_dir: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(reviewed_mark_events_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        errors.append(f"artifact_missing:{path}")
        payload: Any = {}
    else:
        payload = _load_json(path)

    question_ids = _load_question_ids(question_bank_path, warnings)
    mark_event_ids = _load_mark_event_ids(mark_events_path, warnings) if mark_events_path else set()
    payload_errors, payload_warnings = validate_reviewed_mark_events_payload(
        payload,
        question_ids=question_ids,
        mark_event_ids=mark_event_ids,
        base_dir=base_dir,
        require_event_ids_in_mark_events=bool(mark_event_ids),
    )
    errors.extend(payload_errors)
    warnings.extend(payload_warnings)
    decisions = payload.get("decisions") if isinstance(payload, dict) else []
    report = {
        "schema": P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_VALIDATION_SCHEMA,
        "schema_name": P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_VALIDATION_SCHEMA,
        "schema_version": P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": not errors,
        "decision_count": len(decisions) if isinstance(decisions, list) else 0,
        "generation_satisfying_statuses": sorted(GENERATION_SATISFYING_MARK_EVENT_STATUSES),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    if output_path:
        write_atomic_json(report, output_path, sort_keys=True)
    return report


def validate_reviewed_mark_events_payload(
    payload: Any,
    *,
    question_ids: set[str] | None = None,
    mark_event_ids: set[str] | None = None,
    base_dir: str | Path = ".",
    require_event_ids_in_mark_events: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    question_ids = question_ids or set()
    mark_event_ids = mark_event_ids or set()
    if not isinstance(payload, dict):
        return ["top_level_not_object"], warnings
    if payload.get("schema") != P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA:
        errors.append(
            "schema_mismatch:"
            f"expected={P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA}:actual={payload.get('schema')}"
        )
    if int(payload.get("schema_version") or 0) != P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION:
        errors.append(
            "schema_version_mismatch:"
            f"expected={P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION}:actual={payload.get('schema_version')}"
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return errors + ["decisions_not_list"], warnings
    if payload.get("decision_count") not in (None, len(decisions)):
        errors.append(f"decision_count_mismatch:declared={payload.get('decision_count')}:actual={len(decisions)}")

    seen_decision_ids: set[str] = set()
    seen_event_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"decision_not_object:{index}")
            continue
        _validate_decision(
            index,
            decision,
            question_ids=question_ids,
            mark_event_ids=mark_event_ids,
            require_event_ids_in_mark_events=require_event_ids_in_mark_events,
            base_dir=Path(base_dir),
            seen_decision_ids=seen_decision_ids,
            seen_event_ids=seen_event_ids,
            errors=errors,
            warnings=warnings,
        )
    return errors, warnings


def reviewed_mark_event_status_by_id(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return {}
    result: dict[str, str] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        event_id = _text(decision.get("event_id"))
        status = _text(decision.get("status")).lower()
        if event_id and status in ALLOWED_MARK_EVENT_STATUSES:
            result[event_id] = status
    return result


def mark_event_status_satisfies_generation(status: str | None) -> bool:
    return _text(status).lower() in GENERATION_SATISFYING_MARK_EVENT_STATUSES


def _validate_decision(
    index: int,
    decision: dict[str, Any],
    *,
    question_ids: set[str],
    mark_event_ids: set[str],
    require_event_ids_in_mark_events: bool,
    base_dir: Path,
    seen_decision_ids: set[str],
    seen_event_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    decision_id = _text(decision.get("decision_id"))
    event_id = _text(decision.get("event_id"))
    prefix = f"decision:{index}:{decision_id or 'missing'}:{event_id or 'missing'}"
    for field in sorted(REQUIRED_DECISION_FIELDS):
        if field not in decision:
            errors.append(f"{prefix}:missing_required_field:{field}")
    if "question_image_ref" not in decision and "question_image_path" not in decision:
        errors.append(f"{prefix}:missing_question_image_ref_or_path")
    if "mark_scheme_image_ref" not in decision and "mark_scheme_image_path" not in decision:
        errors.append(f"{prefix}:missing_mark_scheme_image_ref_or_path")

    if not decision_id:
        errors.append(f"{prefix}:missing_decision_id")
    elif decision_id in seen_decision_ids:
        errors.append(f"{prefix}:duplicate_decision_id")
    seen_decision_ids.add(decision_id)

    if not event_id:
        errors.append(f"{prefix}:missing_event_id")
    elif event_id in seen_event_ids:
        errors.append(f"{prefix}:duplicate_event_id")
    seen_event_ids.add(event_id)
    if require_event_ids_in_mark_events and event_id and event_id not in mark_event_ids:
        errors.append(f"{prefix}:event_id_not_found_in_mark_events")

    if int(decision.get("schema_version") or 0) != P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION:
        errors.append(f"{prefix}:decision_schema_version_mismatch:{decision.get('schema_version')}")

    question_id = _text(decision.get("source_question_id"))
    if not question_id:
        errors.append(f"{prefix}:missing_source_question_id")
    elif question_ids and question_id not in question_ids:
        errors.append(f"{prefix}:source_question_id_not_found:{question_id}")

    part_path = decision.get("part_path")
    if not _valid_part_path(part_path):
        errors.append(f"{prefix}:invalid_part_path")

    status = _text(decision.get("status")).lower()
    if status not in ALLOWED_MARK_EVENT_STATUSES:
        errors.append(f"{prefix}:invalid_status:{status or 'missing'}")
    if status in {"approved", "reviewed", "rejected"}:
        if not _text(decision.get("reviewer")):
            errors.append(f"{prefix}:missing_reviewer_for_non_advisory_status")
        if not _valid_iso_timestamp(_text(decision.get("reviewed_at"))):
            errors.append(f"{prefix}:missing_or_invalid_reviewed_at_for_non_advisory_status")
    if status in {"rejected", "advisory"} and decision.get("satisfies_generation_gate") is True:
        errors.append(f"{prefix}:{status}_cannot_satisfy_generation_gate")
    if status in {"approved", "reviewed"} and decision.get("satisfies_generation_gate") is False:
        warnings.append(f"{prefix}:generation_satisfying_status_marked_false")

    if not _has_explanation(decision.get("rationale")):
        errors.append(f"{prefix}:missing_rationale")

    for field in ("question_image_path", "mark_scheme_image_path"):
        path_text = _text(decision.get(field))
        if path_text and not _path_exists(path_text, base_dir=base_dir):
            errors.append(f"{prefix}:{field}_not_found:{path_text}")


def _load_question_ids(path: str | Path, warnings: list[str]) -> set[str]:
    path = Path(path)
    if not path.exists():
        warnings.append(f"question_bank_missing:{path}")
        return set()
    payload = _load_json(path)
    records = payload.get("questions") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        warnings.append(f"question_bank_questions_not_list:{path}")
        return set()
    return {_text(record.get("question_id")) for record in records if isinstance(record, dict) and _text(record.get("question_id"))}


def _load_mark_event_ids(path: str | Path | None, warnings: list[str]) -> set[str]:
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        warnings.append(f"mark_events_missing:{path}")
        return set()
    payload = _load_json(path)
    ids: set[str] = set()
    records = payload.get("records") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        warnings.append(f"mark_events_records_not_list:{path}")
        return ids
    for record in records:
        if not isinstance(record, dict):
            continue
        events = record.get("mark_events") if isinstance(record.get("mark_events"), list) else []
        for event in events:
            if isinstance(event, dict) and _text(event.get("event_id")):
                ids.add(_text(event.get("event_id")))
    return ids


def _valid_part_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(_text(part) for part in value)
    return False


def _path_exists(path_text: str, *, base_dir: Path) -> bool:
    path = Path(path_text)
    if path.is_absolute():
        return path.exists()
    return (base_dir / path).exists() or path.exists()


def _has_explanation(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_text(item) for item in value)
    if isinstance(value, dict):
        return any(_text(value.get(field)) for field in ("summary", "reason", "notes", "rationale"))
    return False


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _valid_iso_timestamp(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
