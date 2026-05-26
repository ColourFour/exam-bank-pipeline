from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    ALLOWED_USE_CASE_KEYS,
    DEFAULT_P3_SKILL_MAP_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
    P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA,
    P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION,
    P3_EXACT_SKILL_REVIEWED_DECISIONS_VALIDATION_SCHEMA,
    ROUTE_STATUSES,
)

REQUIRED_RECORD_FIELDS = {
    "evidence_id",
    "question_id",
    "part_id",
    "subpart_id",
    "paper",
    "session",
    "variant",
    "reviewed_source_skill_ids",
    "reviewed_region",
    "route_status",
    "source_question_asset_refs",
    "source_mark_scheme_asset_refs",
    "mark_event_refs",
    "evidence_basis",
    "blockers",
    "allowed_use_cases",
    "reviewer",
    "provenance",
}

STRICT_CLEAN_USE_CASES = {
    "mastery",
    "guardian",
    "source_backed_examples",
    "candidate_generation",
}


def validate_reviewed_decisions(
    *,
    reviewed_decisions_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    p3_skill_map_path: str | Path = DEFAULT_P3_SKILL_MAP_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(reviewed_decisions_path)
    skill_ids = _load_p3_skill_ids(p3_skill_map_path)
    errors, warnings = validate_reviewed_decisions_payload(payload, p3_skill_ids=skill_ids)
    records = payload.get("records") if isinstance(payload, dict) else []
    report = {
        "schema": P3_EXACT_SKILL_REVIEWED_DECISIONS_VALIDATION_SCHEMA,
        "schema_name": P3_EXACT_SKILL_REVIEWED_DECISIONS_VALIDATION_SCHEMA,
        "schema_version": P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": not errors,
        "record_count": len(records) if isinstance(records, list) else 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    if output_path:
        write_atomic_json(report, output_path, sort_keys=True)
    return report


def validate_reviewed_decisions_payload(
    payload: Any,
    *,
    p3_skill_ids: set[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ["top_level_not_object"], warnings
    if payload.get("schema") != P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA:
        errors.append(
            "schema_mismatch:"
            f"expected={P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA}:actual={payload.get('schema')}"
        )
    if int(payload.get("schema_version") or 0) != P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION:
        errors.append(
            "schema_version_mismatch:"
            f"expected={P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION}:actual={payload.get('schema_version')}"
        )
    if payload.get("artifact_kind") not in (None, "manual_reviewed_decision_input"):
        errors.append(f"artifact_kind_not_reviewed_input:{payload.get('artifact_kind')}")

    records = payload.get("records")
    if not isinstance(records, list):
        return errors + ["records_not_list"], warnings
    if payload.get("record_count") not in (None, len(records)):
        errors.append(f"record_count_mismatch:declared={payload.get('record_count')}:actual={len(records)}")

    seen_evidence_ids: set[str] = set()
    seen_scopes: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record_not_object:{index}")
            continue
        _validate_record(
            index,
            record,
            p3_skill_ids=p3_skill_ids,
            seen_evidence_ids=seen_evidence_ids,
            seen_scopes=seen_scopes,
            errors=errors,
            warnings=warnings,
        )
    return errors, warnings


def _validate_record(
    index: int,
    record: dict[str, Any],
    *,
    p3_skill_ids: set[str],
    seen_evidence_ids: set[str],
    seen_scopes: set[tuple[str, str, str]],
    errors: list[str],
    warnings: list[str],
) -> None:
    evidence_id = _text(record.get("evidence_id"))
    question_id = _text(record.get("question_id"))
    prefix = f"record:{index}:{evidence_id or 'missing'}:{question_id or 'missing'}"

    for field in sorted(REQUIRED_RECORD_FIELDS):
        if field not in record:
            errors.append(f"{prefix}:missing_required_field:{field}")

    if not evidence_id:
        errors.append(f"{prefix}:missing_evidence_id")
    elif evidence_id in seen_evidence_ids:
        errors.append(f"{prefix}:duplicate_evidence_id")
    seen_evidence_ids.add(evidence_id)

    if not question_id:
        errors.append(f"{prefix}:missing_question_id")
    part_id = _text(record.get("part_id"))
    subpart_id = _text(record.get("subpart_id"))
    scope_key = (question_id, part_id, subpart_id)
    if question_id and scope_key in seen_scopes:
        errors.append(f"{prefix}:duplicate_question_scope")
    seen_scopes.add(scope_key)

    for field in ("paper", "session", "variant"):
        if not _text(record.get(field)):
            errors.append(f"{prefix}:missing_{field}")

    route_status = _text(record.get("route_status"))
    if route_status not in ROUTE_STATUSES:
        errors.append(f"{prefix}:invalid_route_status:{route_status or 'missing'}")

    skill_ids = record.get("reviewed_source_skill_ids")
    if not isinstance(skill_ids, list):
        errors.append(f"{prefix}:reviewed_source_skill_ids_not_list")
        skill_ids = []
    skill_ids = [_text(skill_id) for skill_id in skill_ids if _text(skill_id)]
    non_p3_skill_ids = [skill_id for skill_id in skill_ids if not skill_id.startswith("9709_p3_")]
    unknown_p3_skill_ids = [skill_id for skill_id in skill_ids if skill_id.startswith("9709_p3_") and skill_id not in p3_skill_ids]

    use_cases = record.get("allowed_use_cases")
    if not isinstance(use_cases, dict):
        errors.append(f"{prefix}:allowed_use_cases_not_object")
        use_cases = {}
    missing_use_cases = sorted(ALLOWED_USE_CASE_KEYS - set(use_cases))
    if missing_use_cases:
        errors.append(f"{prefix}:missing_allowed_use_cases:{','.join(missing_use_cases)}")
    unknown_use_cases = sorted(set(use_cases) - ALLOWED_USE_CASE_KEYS)
    if unknown_use_cases:
        errors.append(f"{prefix}:unknown_allowed_use_cases:{','.join(unknown_use_cases)}")
    for use_case, allowed in use_cases.items():
        if use_case in ALLOWED_USE_CASE_KEYS and not isinstance(allowed, bool):
            errors.append(f"{prefix}:allowed_use_case_not_bool:{use_case}")

    if non_p3_skill_ids and _is_true(use_cases.get("mastery")):
        errors.append(f"{prefix}:p1_or_support_skill_cannot_be_mastery:{','.join(non_p3_skill_ids)}")

    if route_status != "clean":
        for use_case in sorted(STRICT_CLEAN_USE_CASES):
            if _is_true(use_cases.get(use_case)):
                errors.append(f"{prefix}:{use_case}_true_requires_clean_route")

    if _is_true(use_cases.get("candidate_generation")) and not _has_candidate_generation_basis(record):
        errors.append(f"{prefix}:candidate_generation_true_without_reviewed_generation_basis")

    blockers = record.get("blockers")
    if not isinstance(blockers, list):
        errors.append(f"{prefix}:blockers_not_list")
        blockers = []

    if route_status in {"blocked", "ambiguous", "deferred", "review_needed"} and not blockers and not _has_explanation(record.get("evidence_basis")):
        errors.append(f"{prefix}:{route_status}_without_blockers_or_evidence_basis")

    for field in ("source_question_asset_refs", "source_mark_scheme_asset_refs", "mark_event_refs"):
        refs = record.get(field)
        if not isinstance(refs, list):
            errors.append(f"{prefix}:{field}_not_list")
            continue
        for ref_index, ref in enumerate(refs):
            if not isinstance(ref, dict):
                errors.append(f"{prefix}:{field}_ref_not_object:{ref_index}")
            elif field != "mark_event_refs" and not _text(ref.get("path")):
                errors.append(f"{prefix}:{field}_ref_missing_path:{ref_index}")

    if route_status == "clean":
        _validate_clean_record(
            prefix,
            record,
            skill_ids=skill_ids,
            unknown_p3_skill_ids=unknown_p3_skill_ids,
            errors=errors,
            warnings=warnings,
        )
    elif unknown_p3_skill_ids:
        warnings.append(f"{prefix}:unknown_p3_skill_ids_on_non_clean_record:{','.join(unknown_p3_skill_ids)}")

    if _advisory_only(record) and route_status == "clean":
        errors.append(f"{prefix}:advisory_only_evidence_cannot_be_clean")


def _validate_clean_record(
    prefix: str,
    record: dict[str, Any],
    *,
    skill_ids: list[str],
    unknown_p3_skill_ids: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not skill_ids:
        errors.append(f"{prefix}:clean_without_reviewed_p3_skill")
    if any(not skill_id.startswith("9709_p3_") for skill_id in skill_ids):
        errors.append(f"{prefix}:clean_contains_non_p3_skill")
    if unknown_p3_skill_ids:
        errors.append(f"{prefix}:clean_contains_unknown_p3_skill:{','.join(unknown_p3_skill_ids)}")
    if not isinstance(record.get("reviewed_region"), dict) or not record.get("reviewed_region"):
        errors.append(f"{prefix}:clean_without_reviewed_region")
    if not _has_explanation(record.get("evidence_basis")):
        errors.append(f"{prefix}:clean_without_evidence_basis")
    if not record.get("source_question_asset_refs"):
        errors.append(f"{prefix}:clean_without_question_asset_refs")
    if not record.get("source_mark_scheme_asset_refs"):
        errors.append(f"{prefix}:clean_without_mark_scheme_asset_refs")
    if record.get("blockers"):
        errors.append(f"{prefix}:clean_with_unresolved_blockers")
    unreviewed_mark_event_ids = _unreviewed_mark_event_ids(record)
    if unreviewed_mark_event_ids:
        warnings.append(f"{prefix}:clean_with_unreviewed_mark_event_refs:{','.join(unreviewed_mark_event_ids)}")
    unverified_question_refs = _unverified_asset_ref_paths(record, "source_question_asset_refs")
    if unverified_question_refs:
        errors.append(f"{prefix}:clean_with_unverified_question_asset_refs:{','.join(unverified_question_refs)}")
    unverified_mark_scheme_refs = _unverified_asset_ref_paths(record, "source_mark_scheme_asset_refs")
    if unverified_mark_scheme_refs:
        errors.append(f"{prefix}:clean_with_unverified_mark_scheme_asset_refs:{','.join(unverified_mark_scheme_refs)}")

    reviewer = record.get("reviewer")
    if not isinstance(reviewer, dict):
        errors.append(f"{prefix}:reviewer_not_object")
    else:
        if not _text(reviewer.get("reviewed_by")):
            errors.append(f"{prefix}:missing_reviewer_identity")
        if not _valid_iso_timestamp(_text(reviewer.get("reviewed_at"))):
            errors.append(f"{prefix}:missing_or_invalid_reviewed_at")
        if _text(reviewer.get("review_status")) not in {"reviewed", "approved"}:
            errors.append(f"{prefix}:clean_without_reviewed_status")

    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        errors.append(f"{prefix}:clean_without_provenance")
    elif not _valid_iso_timestamp(_text(provenance.get("timestamp"))):
        errors.append(f"{prefix}:missing_or_invalid_provenance_timestamp")


def _has_candidate_generation_basis(record: dict[str, Any]) -> bool:
    if _text(record.get("route_status")) != "clean":
        return False
    basis = record.get("evidence_basis")
    if not isinstance(basis, dict):
        return False
    if basis.get("candidate_generation_reviewed") is not True:
        return False
    mark_event_refs = record.get("mark_event_refs")
    if not isinstance(mark_event_refs, list) or not mark_event_refs:
        return False
    return all(isinstance(ref, dict) and ref.get("review_status") in {"reviewed", "approved"} for ref in mark_event_refs)


def _unreviewed_mark_event_ids(record: dict[str, Any]) -> list[str]:
    refs = record.get("mark_event_refs")
    if not isinstance(refs, list):
        return []
    event_ids: list[str] = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        if ref.get("review_status") not in {"reviewed", "approved"}:
            event_ids.append(_text(ref.get("event_id")) or f"index_{index}")
    return event_ids


def _unverified_asset_ref_paths(record: dict[str, Any], field: str) -> list[str]:
    refs = record.get(field)
    if not isinstance(refs, list):
        return []
    paths: list[str] = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        if ref.get("verified") is not True:
            paths.append(_text(ref.get("path")) or f"index_{index}")
    return paths


def _advisory_only(record: dict[str, Any]) -> bool:
    basis = record.get("evidence_basis")
    if not isinstance(basis, dict):
        return False
    basis_type = _text(basis.get("basis_type"))
    return basis_type in {"advisory_only", "advisory_mark_events_only", "ocr_only", "native_text_only", "ai_label_only"}


def _has_explanation(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_explanation(item) for item in value)
    if isinstance(value, dict):
        for field in ("basis_type", "review_notes", "summary", "why_not_clean"):
            if _text(value.get(field)):
                return True
        notes = value.get("notes")
        if isinstance(notes, list) and any(_text(note) for note in notes):
            return True
    return False


def _load_p3_skill_ids(path: str | Path) -> set[str]:
    payload = _load_json(path)
    skills = payload.get("skills") if isinstance(payload, dict) else []
    if not isinstance(skills, list):
        return set()
    return {_text(skill.get("skill_id")) for skill in skills if isinstance(skill, dict) and _text(skill.get("skill_id"))}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _is_true(value: Any) -> bool:
    return value is True


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
