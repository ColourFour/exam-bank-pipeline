from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade import AUTO_GRADE_SCHEMA_VERSION
from exam_bank.auto_grade.constants import (
    ALLOWED_RUBRIC_MARK_CODES,
    APPROVED_REVIEW_STATUSES,
    DEFAULT_REVIEWED_RUBRICS_VALIDATION_REPORT_PATH,
    DEFAULT_RUBRIC_REVIEW_QUEUE_REPORT_PATH,
    REVIEWED_RUBRIC_SCHEMA_VERSION,
    REVIEWED_RUBRICS_SCHEMA,
    REVIEWED_RUBRICS_VALIDATION_SCHEMA,
    RUBRIC_REVIEW_QUEUE_SCHEMA,
)


def validate_reviewed_rubrics(
    *,
    reviewed_rubrics_path: str | Path,
    question_bank_path: str | Path,
    allow_missing: bool = False,
    phase: str = "2A",
    output_path: str | Path | None = None,
    report_path: str | Path | None = DEFAULT_REVIEWED_RUBRICS_VALIDATION_REPORT_PATH,
) -> dict[str, Any]:
    reviewed_rubrics_path = Path(reviewed_rubrics_path)
    question_bank_path = Path(question_bank_path)
    if not reviewed_rubrics_path.exists():
        errors = [] if allow_missing else [f"reviewed_rubrics_file_missing:{reviewed_rubrics_path}"]
        report = _validation_report(
            ok=not errors,
            errors=errors,
            warnings=["reviewed_rubrics_missing_allowed_zero_approved"] if allow_missing else [],
            approved_question_ids=[],
            rubric_count=0,
        )
        _write_validation_outputs(report, output_path=output_path, report_path=report_path)
        return report

    payload = _load_json(reviewed_rubrics_path)
    question_bank = _load_json(question_bank_path)
    question_records = _question_records(question_bank)
    question_by_id = {str(record.get("question_id") or ""): record for record in question_records}
    errors, warnings, approved_question_ids = validate_reviewed_rubrics_payload(
        payload,
        question_by_id=question_by_id,
        phase=phase,
    )
    rubrics = payload.get("rubrics") if isinstance(payload, dict) else []
    report = _validation_report(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        approved_question_ids=approved_question_ids,
        rubric_count=len(rubrics) if isinstance(rubrics, list) else 0,
    )
    _write_validation_outputs(report, output_path=output_path, report_path=report_path)
    return report


def validate_reviewed_rubrics_payload(
    payload: Any,
    *,
    question_by_id: dict[str, dict[str, Any]],
    phase: str = "2A",
) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    approved_question_ids: list[str] = []
    if not isinstance(payload, dict):
        return ["top_level_not_object"], warnings, approved_question_ids
    if payload.get("schema") != REVIEWED_RUBRICS_SCHEMA:
        errors.append(f"schema_mismatch:expected={REVIEWED_RUBRICS_SCHEMA}:actual={payload.get('schema')}")
    if int(payload.get("schema_version") or 0) != REVIEWED_RUBRIC_SCHEMA_VERSION:
        errors.append(
            f"schema_version_mismatch:expected={REVIEWED_RUBRIC_SCHEMA_VERSION}:actual={payload.get('schema_version')}"
        )
    rubrics = payload.get("rubrics")
    if not isinstance(rubrics, list):
        return errors + ["rubrics_not_list"], warnings, approved_question_ids
    if payload.get("rubric_count") not in (None, len(rubrics)):
        errors.append(f"rubric_count_mismatch:declared={payload.get('rubric_count')}:actual={len(rubrics)}")

    seen_rubric_ids: set[str] = set()
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            errors.append(f"rubric_not_object:{index}")
            continue
        rubric_errors = _validate_rubric(index, rubric, question_by_id=question_by_id, phase=phase, seen=seen_rubric_ids)
        errors.extend(rubric_errors)
        if not rubric_errors and _is_approved(rubric):
            approved_question_ids.append(str(rubric.get("source_question_id") or ""))
    return errors, warnings, sorted(set(approved_question_ids))


def approved_question_ids_from_reviewed_rubrics(
    payload: Any,
    *,
    question_bank_payload: Any,
) -> set[str]:
    question_by_id = {str(record.get("question_id") or ""): record for record in _question_records(question_bank_payload)}
    errors, _, approved = validate_reviewed_rubrics_payload(payload, question_by_id=question_by_id)
    if errors:
        return set()
    return set(approved)


def build_rubric_review_queue(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    eligible_items_path: str | Path = "output/auto_grade/eligible_items.v1.json",
    mark_events_path: str | Path = "output/json/question_bank.mark_events.v1.json",
    topic_routing_path: str | Path | None = "output/json/question_bank.topic_routing.v1.json",
    output_path: str | Path | None = "output/auto_grade/rubric_review_queue.v1.json",
    report_path: str | Path | None = DEFAULT_RUBRIC_REVIEW_QUEUE_REPORT_PATH,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    eligible_items_path = Path(eligible_items_path)
    mark_events_path = Path(mark_events_path)
    question_bank = _load_json(question_bank_path)
    eligible_items = _records_by_question_id(_load_json(eligible_items_path).get("items", []))
    mark_events = _records_by_question_id(_load_json(mark_events_path).get("records", []))
    topic_routes = _records_by_question_id(_load_optional_json(topic_routing_path).get("records", {}))

    candidates = [
        _candidate_for_question(
            question,
            eligible_item=eligible_items.get(str(question.get("question_id") or "")),
            mark_event=mark_events.get(str(question.get("question_id") or "")),
            topic_route=topic_routes.get(str(question.get("question_id") or "")),
        )
        for question in _question_records(question_bank)
    ]
    candidates.sort(key=lambda item: (-int(item["candidate_priority"]), str(item["question_id"])))
    summary = summarize_rubric_review_queue(candidates)
    payload = {
        "schema": RUBRIC_REVIEW_QUEUE_SCHEMA,
        "schema_version": AUTO_GRADE_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "source_question_bank_path": _rel_path(question_bank_path),
        "source_eligible_items_path": _rel_path(eligible_items_path),
        "source_mark_events_path": _rel_path(mark_events_path),
        "candidate_count": len(candidates),
        "summary": summary,
        "candidates": candidates,
    }
    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    if report_path and not dry_run:
        write_rubric_review_queue_summary(payload, output_path=report_path)
    return payload


def summarize_rubric_review_queue(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    priority_buckets = Counter(_priority_bucket(candidate.get("candidate_priority")) for candidate in candidates)
    blockers = Counter(blocker for candidate in candidates for blocker in candidate.get("candidate_blockers") or [])
    risk_flags = Counter(flag for candidate in candidates for flag in candidate.get("candidate_risk_flags") or [])
    return {
        "candidate_count": len(candidates),
        "priority_buckets": dict(priority_buckets),
        "top_candidate_blockers": dict(blockers.most_common(20)),
        "top_risk_flags": dict(risk_flags.most_common(20)),
        "count_by_paper_family": dict(Counter(str(candidate.get("paper_family") or "missing") for candidate in candidates)),
        "count_by_mark_event_status": dict(
            Counter(str(candidate.get("mark_event_status") or "missing") for candidate in candidates)
        ),
        "unknown_mark_code_count": sum(1 for candidate in candidates if candidate.get("has_unknown_mark_codes")),
        "dependency_complexity_count": sum(1 for candidate in candidates if candidate.get("has_dependency_complexity")),
        "follow_through_complexity_count": sum(
            1 for candidate in candidates if candidate.get("has_follow_through_complexity")
        ),
        "suggested_first_review_batch": [candidate["question_id"] for candidate in candidates[:25]],
    }


def write_rubric_review_queue_summary(
    queue: dict[str, Any],
    *,
    output_path: str | Path = DEFAULT_RUBRIC_REVIEW_QUEUE_REPORT_PATH,
) -> str:
    text = render_rubric_review_queue_summary(queue)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def render_rubric_review_queue_summary(queue: dict[str, Any]) -> str:
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    lines = [
        "# Rubric Review Queue Summary",
        "",
        "This queue is review planning evidence only. It is not approved scoring evidence and must not be used to grade submissions.",
        "",
        "## Summary",
        "",
        f"- Total review candidates: {summary.get('candidate_count', queue.get('candidate_count', 0))}",
        "- Priority buckets:",
        *_counter_lines(summary.get("priority_buckets") or {}),
        "- Count by paper family:",
        *_counter_lines(summary.get("count_by_paper_family") or {}),
        "- Count by mark-event status:",
        *_counter_lines(summary.get("count_by_mark_event_status") or {}),
        f"- Candidates with unknown mark codes: {summary.get('unknown_mark_code_count', 0)}",
        f"- Candidates with dependency complexity: {summary.get('dependency_complexity_count', 0)}",
        f"- Candidates with follow-through complexity: {summary.get('follow_through_complexity_count', 0)}",
        "",
        "## Top Candidate Blockers",
        "",
        *_counter_lines(summary.get("top_candidate_blockers") or {}),
        "",
        "## Top Risk Flags",
        "",
        *_counter_lines(summary.get("top_risk_flags") or {}),
        "",
        "## Suggested First Review Batch",
        "",
        *[f"- `{qid}`" for qid in summary.get("suggested_first_review_batch", [])],
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_reviewed_rubrics_validation(report: dict[str, Any]) -> str:
    lines = [
        "# Reviewed Rubrics Validation",
        "",
        f"- OK: {report.get('ok')}",
        f"- Rubrics checked: {report.get('rubric_count', 0)}",
        f"- Approved rubrics accepted: {report.get('approved_rubric_count', 0)}",
        f"- Error count: {report.get('error_count', 0)}",
        f"- Warning count: {report.get('warning_count', 0)}",
        "",
        "## Errors",
        "",
        *_list_lines(report.get("errors") or []),
        "",
        "## Warnings",
        "",
        *_list_lines(report.get("warnings") or []),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _validate_rubric(
    index: int,
    rubric: dict[str, Any],
    *,
    question_by_id: dict[str, dict[str, Any]],
    phase: str,
    seen: set[str],
) -> list[str]:
    errors: list[str] = []
    rubric_id = str(rubric.get("rubric_id") or "").strip()
    question_id = str(rubric.get("source_question_id") or "").strip()
    prefix = f"rubric:{index}:{rubric_id or 'missing'}:{question_id or 'missing'}"
    if not rubric_id:
        errors.append(f"{prefix}:missing_rubric_id")
    elif rubric_id in seen:
        errors.append(f"{prefix}:duplicate_rubric_id")
    seen.add(rubric_id)
    if not question_id:
        errors.append(f"{prefix}:missing_source_question_id")
    elif question_id not in question_by_id:
        errors.append(f"{prefix}:source_question_id_not_in_question_bank")
    if not str(rubric.get("source_mark_scheme_image_path") or "").strip():
        errors.append(f"{prefix}:missing_source_mark_scheme_image_path")

    review_status = str(rubric.get("review_status") or "").strip()
    lab_safe = _bool(rubric.get("safe_for_auto_grade_lab"))
    teacher_safe = _bool(rubric.get("safe_for_teacher_beta"))
    student_safe = _bool(rubric.get("safe_for_student_self_check"))
    approved = review_status in APPROVED_REVIEW_STATUSES or lab_safe or teacher_safe or student_safe
    if phase == "2A" and student_safe:
        errors.append(f"{prefix}:student_safe_flag_forbidden_in_phase_2a")
    if student_safe and not teacher_safe:
        errors.append(f"{prefix}:student_safe_without_teacher_beta")
    if teacher_safe and not lab_safe:
        errors.append(f"{prefix}:teacher_beta_without_auto_grade_lab_safety")
    if approved:
        if not str(rubric.get("reviewed_by") or "").strip():
            errors.append(f"{prefix}:missing_reviewer_identity")
        if not str(rubric.get("reviewed_at") or "").strip():
            errors.append(f"{prefix}:missing_reviewed_at")
        if review_status not in APPROVED_REVIEW_STATUSES:
            errors.append(f"{prefix}:approved_scope_without_approved_review_status")
        if _bool(rubric.get("rubric_total_verified")) is not True:
            errors.append(f"{prefix}:approved_rubric_total_not_verified")

    total_marks = _int_or_none(rubric.get("total_marks"))
    events = rubric.get("events")
    if not isinstance(events, list):
        errors.append(f"{prefix}:events_not_list")
        events = []
    seen_event_ids: set[str] = set()
    event_total = 0
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"{prefix}:event_not_object:{event_index}")
            continue
        event_errors, mark_value = _validate_event(prefix, event_index, event, approved=approved, seen=seen_event_ids)
        errors.extend(event_errors)
        event_total += mark_value
    if approved and total_marks is not None and event_total != total_marks:
        errors.append(f"{prefix}:event_total_mismatch:events={event_total}:rubric={total_marks}")
    if approved and not events:
        errors.append(f"{prefix}:approved_rubric_without_events")
    return errors


def _validate_event(
    prefix: str,
    event_index: int,
    event: dict[str, Any],
    *,
    approved: bool,
    seen: set[str],
) -> tuple[list[str], int]:
    errors: list[str] = []
    event_id = str(event.get("event_id") or "").strip()
    event_prefix = f"{prefix}:event:{event_index}:{event_id or 'missing'}"
    if not event_id:
        errors.append(f"{event_prefix}:missing_event_id")
    elif event_id in seen:
        errors.append(f"{event_prefix}:duplicate_event_id")
    seen.add(event_id)
    mark_code = str(event.get("mark_code") or "").strip() or "unknown"
    if mark_code not in ALLOWED_RUBRIC_MARK_CODES:
        errors.append(f"{event_prefix}:invalid_mark_code:{mark_code}")
    if approved and mark_code == "unknown":
        errors.append(f"{event_prefix}:approved_event_unknown_mark_code")
    mark_value = _int_or_none(event.get("mark_value")) or 0
    if approved and mark_value <= 0:
        errors.append(f"{event_prefix}:approved_event_invalid_mark_value")
    accepted_evidence = event.get("accepted_evidence")
    if approved and not _has_content(accepted_evidence):
        errors.append(f"{event_prefix}:approved_event_missing_accepted_evidence")
    dependency = event.get("dependency")
    if approved and mark_code in {"DM"} and not _has_content(dependency):
        errors.append(f"{event_prefix}:dependent_mark_missing_dependency_policy")
    if approved and mark_code in {"FT"} and not _has_content(event.get("follow_through_policy")):
        errors.append(f"{event_prefix}:follow_through_mark_missing_follow_through_policy")
    learning_target_ids = event.get("learning_target_ids")
    if approved and not (isinstance(learning_target_ids, list) and learning_target_ids):
        errors.append(f"{event_prefix}:approved_event_missing_learning_target_ids")
    if approved and str(event.get("review_status") or "") not in APPROVED_REVIEW_STATUSES:
        errors.append(f"{event_prefix}:approved_rubric_contains_unapproved_event")
    return errors, mark_value


def _candidate_for_question(
    question: dict[str, Any],
    *,
    eligible_item: dict[str, Any] | None,
    mark_event: dict[str, Any] | None,
    topic_route: dict[str, Any] | None,
) -> dict[str, Any]:
    question_id = str(question.get("question_id") or "")
    mark_events = mark_event.get("mark_events") if isinstance(mark_event, dict) else []
    mark_events = mark_events if isinstance(mark_events, list) else []
    mark_codes = sorted({_mark_code_from_event(event) for event in mark_events if isinstance(event, dict)})
    unknown = not mark_codes or "unknown" in mark_codes
    dependency = any(_bool(event.get("is_dependent")) or event.get("depends_on_event_ids") for event in mark_events if isinstance(event, dict))
    follow_through = any(_bool(event.get("is_follow_through")) for event in mark_events if isinstance(event, dict))
    total_mismatch = bool(mark_event and mark_event.get("total_marks_match") is False)
    canonical_question = str(
        (eligible_item or {}).get("canonical_question_artifact")
        or question.get("canonical_question_artifact")
        or question.get("question_image_path")
        or ""
    )
    canonical_mark_scheme = str(
        (eligible_item or {}).get("canonical_mark_scheme_artifact")
        or question.get("mark_scheme_image_path")
        or ""
    )
    blockers: list[str] = []
    risks: list[str] = []
    if not canonical_question:
        blockers.append("missing_canonical_question_artifact")
    if not canonical_mark_scheme:
        blockers.append("missing_canonical_mark_scheme_artifact")
    if not mark_event:
        blockers.append("missing_mark_events")
    elif mark_event.get("safe_for_advisory_use") is not True:
        risks.append("mark_events_not_advisory_safe")
    if total_mismatch:
        blockers.append("total_mismatch")
    if unknown:
        risks.append("unknown_mark_codes")
    if dependency:
        risks.append("dependency_complexity")
    if follow_through:
        risks.append("follow_through_complexity")
    for flag in (mark_event or {}).get("review_flags") or []:
        risks.append(str(flag))
    learning_ready = "ready"
    if not topic_route:
        learning_ready = "missing"
        risks.append("learning_target_mapping_missing")
    elif topic_route.get("review_required") is True or str(topic_route.get("confidence") or "").lower() != "high":
        learning_ready = "review_required"
        risks.append("learning_target_mapping_review_required")

    priority = 100
    priority -= 30 * len(blockers)
    priority -= 8 * len(set(risks))
    if mark_event and mark_event.get("extraction_status") not in {"parsed", "partial", None}:
        priority -= 10
    if set(mark_codes) - {"M", "A", "B"}:
        priority -= 6
    if not mark_events:
        priority -= 15
    priority = max(0, min(100, priority))
    return {
        "question_id": question_id,
        "paper": question.get("paper"),
        "paper_family": question.get("paper_family"),
        "question_number": question.get("question_number"),
        "total_marks": (eligible_item or {}).get("total_marks") or question.get("question_solution_marks"),
        "canonical_question_artifact": canonical_question,
        "canonical_mark_scheme_artifact": canonical_mark_scheme,
        "mark_event_status": (mark_event or {}).get("extraction_status") or ("missing" if not mark_event else "present"),
        "advisory_safe_for_use": bool(mark_event and mark_event.get("safe_for_advisory_use") is True),
        "candidate_priority": priority,
        "candidate_risk_flags": sorted(set(risks)),
        "candidate_blockers": sorted(set(blockers)),
        "review_recommendation": _review_recommendation(priority, blockers, risks),
        "suggested_review_scope": "full_rubric_review_from_images",
        "event_count": len(mark_events),
        "mark_codes_detected": mark_codes,
        "has_unknown_mark_codes": unknown,
        "has_dependency_complexity": dependency,
        "has_follow_through_complexity": follow_through,
        "has_total_mismatch": total_mismatch,
        "learning_target_readiness": learning_ready,
    }


def _mark_code_from_event(event: dict[str, Any]) -> str:
    raw = str(event.get("mark_code") or event.get("mark_code_raw") or "").strip().upper()
    if raw.startswith("DM"):
        return "DM"
    for code in ("M", "A", "B", "E"):
        if raw.startswith(code):
            return code
    if _bool(event.get("is_follow_through")):
        return "FT"
    return "unknown"


def _review_recommendation(priority: int, blockers: list[str], risks: list[str]) -> str:
    if blockers:
        return "resolve_blockers_before_approval"
    if priority >= 80:
        return "good_first_batch_candidate"
    if risks:
        return "review_with_extra_attention"
    return "standard_review_candidate"


def _priority_bucket(value: Any) -> str:
    value = int(value or 0)
    if value >= 80:
        return "high"
    if value >= 50:
        return "medium"
    return "low"


def _validation_report(
    *,
    ok: bool,
    errors: list[str],
    warnings: list[str],
    approved_question_ids: list[str],
    rubric_count: int,
) -> dict[str, Any]:
    return {
        "schema": REVIEWED_RUBRICS_VALIDATION_SCHEMA,
        "schema_version": REVIEWED_RUBRIC_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": ok,
        "rubric_count": rubric_count,
        "approved_rubric_count": len(approved_question_ids),
        "approved_question_ids": approved_question_ids,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def _write_validation_outputs(
    report: dict[str, Any],
    *,
    output_path: str | Path | None,
    report_path: str | Path | None,
) -> None:
    if output_path:
        write_atomic_json(report, output_path, sort_keys=True)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_reviewed_rubrics_validation(report), encoding="utf-8")


def _is_approved(rubric: dict[str, Any]) -> bool:
    return str(rubric.get("review_status") or "") in APPROVED_REVIEW_STATUSES and _bool(
        rubric.get("safe_for_auto_grade_lab")
    )


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_optional_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return _load_json(path)


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _records_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    if isinstance(records, dict):
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}
    if not isinstance(records, list):
        return {}
    return {str(record.get("question_id")): record for record in records if isinstance(record, dict) and record.get("question_id")}


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    return value not in (None, False)


def _counter_lines(values: dict[str, int]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: {values[key]}" for key in sorted(values)]


def _list_lines(values: list[str]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{value}`" for value in values]


def _rel_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
