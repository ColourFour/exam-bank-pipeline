from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade import AUTO_GRADE_SCHEMA_VERSION
from exam_bank.auto_grade.constants import REVIEWED_RUBRIC_SCHEMA_VERSION, REVIEWED_RUBRICS_SCHEMA

REVIEW_BATCH_SCHEMA = "exam_bank.auto_grade.rubric_review_batch"
DEFAULT_REVIEW_BATCH_PATH = "output/auto_grade/review_batches/review_batch_0001.v1.json"
DEFAULT_REVIEW_BATCH_REPORT_PATH = "reports/auto_grade/review_batch_0001_summary.md"
DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH = "output/auto_grade/review_batches/reviewed_rubrics_draft_0001.v1.json"
DEFAULT_COMPLETION_REPORT_PATH = "reports/auto_grade/review_completion_0001.md"
DEFAULT_EXCLUDED_RISK_FLAGS = ("unknown_mark_codes", "total_mismatch")


def build_rubric_review_batch(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    review_queue_path: str | Path = "output/auto_grade/rubric_review_queue.v1.json",
    mark_events_path: str | Path = "output/json/question_bank.mark_events.v1.json",
    eligible_items_path: str | Path | None = "output/auto_grade/eligible_items.v1.json",
    output_path: str | Path | None = DEFAULT_REVIEW_BATCH_PATH,
    report_path: str | Path | None = DEFAULT_REVIEW_BATCH_REPORT_PATH,
    draft_output_path: str | Path | None = DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH,
    max_rubrics: int = 25,
    max_events: int = 75,
    paper_families: list[str] | tuple[str, ...] = ("p1", "p3"),
    exclude_risk_flags: list[str] | tuple[str, ...] = DEFAULT_EXCLUDED_RISK_FLAGS,
    include_medium_priority: bool = False,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    review_queue_path = Path(review_queue_path)
    mark_events_path = Path(mark_events_path)
    queue = _load_json(review_queue_path)
    question_bank = _load_json(question_bank_path)
    mark_events = _records_by_question_id(_load_json(mark_events_path).get("records", []))
    eligible_items = _records_by_question_id(_load_optional_json(eligible_items_path).get("items", []))
    questions = _records_by_question_id(_question_records(question_bank))

    excluded = {str(flag).strip() for flag in exclude_risk_flags if str(flag).strip()}
    preferred_papers = [family.lower() for family in paper_families if str(family).strip()]
    candidates = queue.get("candidates") if isinstance(queue, dict) else []
    candidates = [candidate for candidate in candidates if isinstance(candidate, dict)]

    selected: list[dict[str, Any]] = []
    selected_events = 0
    excluded_counts: Counter[str] = Counter()
    for queue_candidate in sorted(candidates, key=_batch_sort_key):
        decision = _candidate_exclusion_reason(
            queue_candidate,
            excluded_risk_flags=excluded,
            preferred_paper_families=preferred_papers,
            include_medium_priority=include_medium_priority,
        )
        if decision:
            excluded_counts[decision] += 1
            continue
        question_id = str(queue_candidate.get("question_id") or "")
        mark_event_record = mark_events.get(question_id, {})
        source_events = _mark_events_for_record(mark_event_record)
        event_count = len(source_events) or int(queue_candidate.get("event_count") or 0)
        if not event_count:
            excluded_counts["missing_mark_events"] += 1
            continue
        if "total_mismatch" in excluded and _event_sum_mismatch(queue_candidate, source_events):
            excluded_counts["excluded_risk_flags"] += 1
            continue
        if len(selected) >= max_rubrics:
            excluded_counts["max_rubrics_reached"] += 1
            continue
        if selected_events + event_count > max_events:
            excluded_counts["max_events_reached"] += 1
            continue
        selected.append(
            _batch_candidate(
                queue_candidate,
                question=questions.get(question_id, {}),
                eligible_item=eligible_items.get(question_id, {}),
                mark_event_record=mark_event_record,
            )
        )
        selected_events += event_count
        if len(selected) >= max_rubrics or selected_events >= max_events:
            continue

    summary = summarize_review_batch(selected, excluded_counts=dict(excluded_counts))
    payload = {
        "schema": REVIEW_BATCH_SCHEMA,
        "schema_version": AUTO_GRADE_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "source_question_bank_path": _rel_path(question_bank_path),
        "source_review_queue_path": _rel_path(review_queue_path),
        "source_mark_events_path": _rel_path(mark_events_path),
        "source_eligible_items_path": _rel_path(eligible_items_path) if eligible_items_path else None,
        "selection_policy": {
            "max_rubrics": max_rubrics,
            "max_events": max_events,
            "paper_families": preferred_papers,
            "exclude_risk_flags": sorted(excluded),
            "include_medium_priority": include_medium_priority,
            "review_status_default": "needs_human_review",
        },
        "rubric_count": len(selected),
        "event_count": selected_events,
        "summary": summary,
        "candidates": selected,
    }
    draft = build_reviewed_rubrics_draft_from_batch(payload, generated_at=generated_at)
    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    if draft_output_path and not dry_run:
        write_atomic_json(draft, draft_output_path, sort_keys=True)
    if report_path and not dry_run:
        write_review_batch_summary(payload, output_path=report_path)
    return {"batch": payload, "draft_reviewed_rubrics": draft}


def build_reviewed_rubrics_draft_from_batch(batch: dict[str, Any], *, generated_at: str | None = None) -> dict[str, Any]:
    rubrics = [_draft_rubric(candidate) for candidate in batch.get("candidates") or [] if isinstance(candidate, dict)]
    return {
        "schema": REVIEWED_RUBRICS_SCHEMA,
        "schema_version": REVIEWED_RUBRIC_SCHEMA_VERSION,
        "generated_at": generated_at or batch.get("generated_at") or _utc_now_iso(),
        "fixture_or_demo": False,
        "approval_note": "Draft workspace only. These rubrics are not approved scoring evidence until a human completes review metadata, accepted evidence, total verification, and approval flags.",
        "source_question_bank_path": batch.get("source_question_bank_path"),
        "source_mark_events_path": batch.get("source_mark_events_path"),
        "source_review_batch_path": batch.get("source_review_batch_path") or DEFAULT_REVIEW_BATCH_PATH,
        "rubric_count": len(rubrics),
        "event_count": sum(len(rubric.get("events") or []) for rubric in rubrics),
        "summary": {
            "approved_count": 0,
            "needs_human_review_count": len(rubrics),
            "safe_for_teacher_beta_count": 0,
            "safe_for_student_self_check_count": 0,
            "student_ready_count": 0,
        },
        "rubrics": rubrics,
    }


def check_rubric_review_completion(
    *,
    reviewed_rubrics_path: str | Path = DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH,
    report_path: str | Path | None = DEFAULT_COMPLETION_REPORT_PATH,
    generated_at: str | None = None,
) -> dict[str, Any]:
    reviewed_rubrics_path = Path(reviewed_rubrics_path)
    payload = _load_json(reviewed_rubrics_path)
    rubrics = payload.get("rubrics") if isinstance(payload, dict) else []
    rubrics = [rubric for rubric in rubrics if isinstance(rubric, dict)]
    blocker_counts: Counter[str] = Counter()
    rubric_reports: list[dict[str, Any]] = []
    approved = 0
    promotion_candidates = 0
    student_self_check = 0
    student_ready = 0
    for rubric in rubrics:
        blockers = _completion_blockers(rubric)
        blocker_counts.update(blockers)
        is_approved = str(rubric.get("review_status") or "") == "approved"
        if is_approved:
            approved += 1
        if is_approved and rubric.get("safe_for_auto_grade_lab") is True and rubric.get("safe_for_teacher_beta") is True:
            promotion_candidates += 1
        if rubric.get("safe_for_student_self_check") is True:
            student_self_check += 1
        if "student_ready" in {str(value) for value in rubric.get("approved_for") or []}:
            student_ready += 1
        rubric_reports.append(
            {
                "rubric_id": rubric.get("rubric_id"),
                "source_question_id": rubric.get("source_question_id"),
                "review_status": rubric.get("review_status"),
                "blockers": blockers,
            }
        )
    report = {
        "schema": "exam_bank.auto_grade.rubric_review_completion",
        "schema_version": AUTO_GRADE_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "source_reviewed_rubrics_path": _rel_path(reviewed_rubrics_path),
        "rubric_count": len(rubrics),
        "approved_count": approved,
        "needs_human_review_count": sum(1 for rubric in rubrics if str(rubric.get("review_status") or "") != "approved"),
        "missing_reviewer_metadata_count": blocker_counts.get("missing_reviewer_metadata", 0),
        "missing_accepted_evidence_count": blocker_counts.get("missing_accepted_evidence", 0),
        "unresolved_unknown_mark_code_count": blocker_counts.get("unresolved_unknown_mark_codes", 0),
        "total_mismatch_count": blocker_counts.get("total_mismatch", 0),
        "dependency_policy_gap_count": blocker_counts.get("dependency_policy_gap", 0),
        "follow_through_policy_gap_count": blocker_counts.get("follow_through_policy_gap", 0),
        "missing_learning_target_ids_count": blocker_counts.get("missing_learning_target_ids", 0),
        "eligibility_promotion_candidate_count": promotion_candidates,
        "student_self_check_beta_candidate_count": student_self_check,
        "student_ready_candidate_count": student_ready,
        "blocker_counts": dict(blocker_counts),
        "rubrics": rubric_reports,
    }
    if report_path:
        write_review_completion_report(report, output_path=report_path)
    return report


def summarize_review_batch(candidates: list[dict[str, Any]], *, excluded_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "selected_rubrics": len(candidates),
        "selected_events": sum(int(candidate.get("event_count") or 0) for candidate in candidates),
        "paper_family_distribution": dict(Counter(str(candidate.get("paper_family") or "missing") for candidate in candidates)),
        "mark_code_distribution": dict(
            Counter(code for candidate in candidates for code in candidate.get("mark_codes_detected") or [])
        ),
        "risk_flag_distribution": dict(
            Counter(flag for candidate in candidates for flag in candidate.get("candidate_risk_flags") or [])
        ),
        "excluded_candidate_counts": excluded_counts,
        "first_25_question_ids": [str(candidate.get("question_id")) for candidate in candidates[:25]],
    }


def write_review_batch_summary(batch: dict[str, Any], *, output_path: str | Path = DEFAULT_REVIEW_BATCH_REPORT_PATH) -> str:
    text = render_review_batch_summary(batch)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def render_review_batch_summary(batch: dict[str, Any]) -> str:
    summary = batch.get("summary") if isinstance(batch.get("summary"), dict) else {}
    lines = [
        "# Rubric Review Batch 0001 Summary",
        "",
        "This batch is a human worklist only. It is not approved scoring evidence and must not be used to grade submissions.",
        "",
        "## Summary",
        "",
        f"- Selected rubrics: {summary.get('selected_rubrics', batch.get('rubric_count', 0))}",
        f"- Selected mark events: {summary.get('selected_events', batch.get('event_count', 0))}",
        "- Paper-family distribution:",
        *_counter_lines(summary.get("paper_family_distribution") or {}),
        "- Mark-code distribution:",
        *_counter_lines(summary.get("mark_code_distribution") or {}),
        "- Risk flags in selected batch:",
        *_counter_lines(summary.get("risk_flag_distribution") or {}),
        "- Excluded candidate counts:",
        *_counter_lines(summary.get("excluded_candidate_counts") or {}),
        "",
        "## First 25 Selected Question IDs",
        "",
        *_list_lines(summary.get("first_25_question_ids") or []),
        "",
        "## Reviewer Instructions",
        "",
        "- Open each canonical question image and mark-scheme image before editing the draft reviewed rubric.",
        "- Rewrite advisory event text into accepted evidence; do not copy generated text blindly.",
        "- Verify total marks against both canonical images before setting `rubric_total_verified: true`.",
        "- Leave `review_status: needs_human_review` until reviewer identity, review timestamp, accepted evidence, dependencies, follow-through policy, and learning targets are complete.",
        "- Phase 2B may promote explicitly approved teacher-beta rubrics only; student self-check and student-ready statuses must remain zero.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_review_completion_report(
    report: dict[str, Any],
    *,
    output_path: str | Path = DEFAULT_COMPLETION_REPORT_PATH,
) -> str:
    text = render_review_completion_report(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def render_review_completion_report(report: dict[str, Any]) -> str:
    lines = [
        "# Rubric Review Completion 0001",
        "",
        f"- Rubrics inspected: {report.get('rubric_count', 0)}",
        f"- Approved count: {report.get('approved_count', 0)}",
        f"- Remaining needs-review count: {report.get('needs_human_review_count', 0)}",
        f"- Eligibility promotion candidates for teacher_beta: {report.get('eligibility_promotion_candidate_count', 0)}",
        f"- Student self-check candidates: {report.get('student_self_check_beta_candidate_count', 0)}",
        f"- Student-ready candidates: {report.get('student_ready_candidate_count', 0)}",
        "",
        "Student-safe statuses remain 0 when both student counts above are 0.",
        "",
        "## Blockers To Approval",
        "",
        *_counter_lines(report.get("blocker_counts") or {}),
        "",
        "## Rubric Status",
        "",
        *[
            f"- `{item.get('source_question_id')}` / `{item.get('rubric_id')}`: {', '.join(item.get('blockers') or ['ready_for_validation'])}"
            for item in report.get("rubrics") or []
            if isinstance(item, dict)
        ],
    ]
    return "\n".join(lines).rstrip() + "\n"


def _batch_candidate(
    queue_candidate: dict[str, Any],
    *,
    question: dict[str, Any],
    eligible_item: dict[str, Any],
    mark_event_record: dict[str, Any],
) -> dict[str, Any]:
    question_id = str(queue_candidate.get("question_id") or "")
    events = [_draft_event(event, index=index) for index, event in enumerate(_mark_events_for_record(mark_event_record), start=1)]
    return {
        "question_id": question_id,
        "paper": queue_candidate.get("paper"),
        "paper_family": queue_candidate.get("paper_family"),
        "question_number": queue_candidate.get("question_number"),
        "part_path": mark_event_record.get("part_path") or question.get("part_path") or [],
        "total_marks": queue_candidate.get("total_marks"),
        "canonical_question_artifact": queue_candidate.get("canonical_question_artifact"),
        "canonical_mark_scheme_artifact": queue_candidate.get("canonical_mark_scheme_artifact"),
        "mark_event_source_reference": {
            "record_id": question_id,
            "paper_id": mark_event_record.get("paper_id"),
            "source_mark_scheme_image_path": mark_event_record.get("source_mark_scheme_image_path"),
            "extraction_status": mark_event_record.get("extraction_status"),
            "safe_for_advisory_use": mark_event_record.get("safe_for_advisory_use"),
            "total_marks_detected": mark_event_record.get("total_marks_detected"),
            "total_marks_expected": mark_event_record.get("total_marks_expected"),
            "total_marks_match": mark_event_record.get("total_marks_match"),
        },
        "proposed_rubric_id": f"rr_{question_id}",
        "proposed_events": events,
        "mark_codes_detected": queue_candidate.get("mark_codes_detected") or [],
        "event_count": len(events),
        "dependency_flags": _dedupe(
            event["source_event_id"] for event in events if event.get("is_dependent") or event.get("depends_on_event_ids")
        ),
        "follow_through_flags": _dedupe(event["source_event_id"] for event in events if event.get("is_follow_through")),
        "candidate_risk_flags": queue_candidate.get("candidate_risk_flags") or [],
        "candidate_blockers": queue_candidate.get("candidate_blockers") or [],
        "candidate_priority": queue_candidate.get("candidate_priority"),
        "reviewer_checklist": [
            "canonical_question_image_verified",
            "canonical_mark_scheme_image_verified",
            "total_marks_reconciled",
            "advisory_events_rewritten_as_accepted_evidence",
            "mark_codes_verified",
            "dependencies_documented",
            "follow_through_documented",
            "alternative_methods_documented",
            "learning_target_ids_attached",
            "approval_scope_confirmed_teacher_beta_only",
        ],
        "learning_target_ids_advisory": eligible_item.get("learning_target_ids") or [],
        "review_status": "needs_human_review",
    }


def _draft_rubric(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rubric_id": candidate.get("proposed_rubric_id"),
        "source_question_id": candidate.get("question_id"),
        "source_mark_scheme_image_path": candidate.get("canonical_mark_scheme_artifact"),
        "source_mark_events_record_id": (candidate.get("mark_event_source_reference") or {}).get("record_id"),
        "paper": candidate.get("paper"),
        "paper_family": candidate.get("paper_family"),
        "question_number": candidate.get("question_number"),
        "part_path": candidate.get("part_path") or [],
        "total_marks": candidate.get("total_marks"),
        "rubric_total_verified": False,
        "safe_for_auto_grade_lab": False,
        "safe_for_teacher_beta": False,
        "safe_for_student_self_check": False,
        "review_status": "needs_human_review",
        "reviewed_by": None,
        "reviewed_at": None,
        "approval_scope": "none",
        "events": [
            {
                "event_id": f"{candidate.get('proposed_rubric_id')}_e{index:04d}",
                "source_event_id": event.get("source_event_id"),
                "part_path": event.get("part_path") or [],
                "mark_code": event.get("mark_code"),
                "mark_type": event.get("mark_type"),
                "mark_value": event.get("mark_value"),
                "dependency": event.get("dependency"),
                "follow_through_policy": event.get("follow_through_policy"),
                "accepted_evidence": [],
                "advisory_evidence": event.get("advisory_evidence") or {},
                "common_errors": [],
                "alternative_methods": [],
                "learning_target_ids": [],
                "review_status": "needs_human_review",
                "review_notes": "",
            }
            for index, event in enumerate(candidate.get("proposed_events") or [], start=1)
            if isinstance(event, dict)
        ],
    }


def _draft_event(event: dict[str, Any], *, index: int) -> dict[str, Any]:
    mark_code = _mark_code_from_event(event)
    return {
        "source_event_id": event.get("event_id") or f"source_event_{index:04d}",
        "part_path": event.get("part_path") or [],
        "mark_code": mark_code,
        "mark_code_raw": event.get("mark_code_raw"),
        "mark_type": event.get("mark_type"),
        "mark_value": event.get("mark_value"),
        "is_follow_through": event.get("is_follow_through") is True,
        "is_dependent": event.get("is_dependent") is True,
        "depends_on_event_ids": event.get("depends_on_event_ids") or [],
        "dependency": "needs_human_review" if event.get("is_dependent") or event.get("depends_on_event_ids") else "independent",
        "follow_through_policy": "needs_human_review" if event.get("is_follow_through") is True else "none",
        "alternative_group_id": event.get("alternative_group_id"),
        "review_flags": event.get("review_flags") or [],
        "advisory_evidence": {
            "raw_text": event.get("raw_text"),
            "normalized_text": event.get("normalized_text"),
            "answer_text": event.get("answer_text"),
            "condition_text": event.get("condition_text"),
            "common_error_text": event.get("common_error_text"),
            "confidence": event.get("confidence"),
        },
        "review_status": "needs_human_review",
    }


def _candidate_exclusion_reason(
    candidate: dict[str, Any],
    *,
    excluded_risk_flags: set[str],
    preferred_paper_families: list[str],
    include_medium_priority: bool,
) -> str:
    family = str(candidate.get("paper_family") or "").lower()
    if preferred_paper_families and family not in preferred_paper_families:
        return "paper_family_not_selected"
    flags = set(str(flag) for flag in candidate.get("candidate_risk_flags") or [])
    blockers = set(str(flag) for flag in candidate.get("candidate_blockers") or [])
    normalized = flags | blockers
    if candidate.get("has_unknown_mark_codes") is True:
        normalized.add("unknown_mark_codes")
    if candidate.get("has_total_mismatch") is True:
        normalized.add("total_mismatch")
    if excluded_risk_flags & normalized:
        return "excluded_risk_flags"
    priority = int(candidate.get("candidate_priority") or 0)
    if priority < 80 and not include_medium_priority:
        return "below_high_priority"
    if priority < 50:
        return "below_medium_priority"
    if str(candidate.get("mark_event_status") or "") != "parsed":
        return "mark_event_not_parsed"
    if set(candidate.get("mark_codes_detected") or []) - {"M", "A", "B"}:
        return "mark_codes_not_first_batch_simple"
    if candidate.get("has_dependency_complexity") is True:
        return "dependency_complexity"
    if candidate.get("has_follow_through_complexity") is True:
        return "follow_through_complexity"
    return ""


def _batch_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    family_rank = {"p1": 0, "p3": 1}.get(str(candidate.get("paper_family") or "").lower(), 2)
    simple_codes = 0 if not (set(candidate.get("mark_codes_detected") or []) - {"M", "A", "B"}) else 1
    return (
        family_rank,
        -int(candidate.get("candidate_priority") or 0),
        0 if str(candidate.get("mark_event_status") or "") == "parsed" else 1,
        simple_codes,
        int(candidate.get("event_count") or 0),
        str(candidate.get("question_id") or ""),
    )


def _completion_blockers(rubric: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(rubric.get("review_status") or "") != "approved":
        blockers.append("not_approved")
    if not str(rubric.get("reviewed_by") or "").strip() or not str(rubric.get("reviewed_at") or "").strip():
        blockers.append("missing_reviewer_metadata")
    if rubric.get("rubric_total_verified") is not True:
        blockers.append("total_not_verified")
    total_marks = _int_or_none(rubric.get("total_marks"))
    event_total = 0
    for event in rubric.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_total += _int_or_none(event.get("mark_value")) or 0
        mark_code = str(event.get("mark_code") or "unknown")
        if mark_code == "unknown":
            blockers.append("unresolved_unknown_mark_codes")
        if not _has_content(event.get("accepted_evidence")):
            blockers.append("missing_accepted_evidence")
        if mark_code == "DM" and not _policy_complete(event.get("dependency")):
            blockers.append("dependency_policy_gap")
        if (mark_code == "FT" or event.get("follow_through_policy") == "needs_human_review") and not _policy_complete(
            event.get("follow_through_policy")
        ):
            blockers.append("follow_through_policy_gap")
        if not (isinstance(event.get("learning_target_ids"), list) and event.get("learning_target_ids")):
            blockers.append("missing_learning_target_ids")
    if total_marks is not None and event_total != total_marks:
        blockers.append("total_mismatch")
    if rubric.get("safe_for_teacher_beta") is True and rubric.get("safe_for_auto_grade_lab") is not True:
        blockers.append("teacher_beta_without_lab_safety")
    if rubric.get("safe_for_student_self_check") is True:
        blockers.append("student_self_check_forbidden_in_phase_2b")
    return sorted(set(blockers))


def _mark_events_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("mark_events") if isinstance(record, dict) else []
    return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []


def _event_sum_mismatch(candidate: dict[str, Any], source_events: list[dict[str, Any]]) -> bool:
    total_marks = _int_or_none(candidate.get("total_marks"))
    if total_marks is None or not source_events:
        return False
    event_total = sum(_int_or_none(event.get("mark_value")) or 0 for event in source_events)
    return event_total != total_marks


def _mark_code_from_event(event: dict[str, Any]) -> str:
    raw = str(event.get("mark_code") or event.get("mark_code_raw") or "").strip().upper()
    if raw.startswith("DM"):
        return "DM"
    for code in ("M", "A", "B", "E"):
        if raw.startswith(code):
            return code
    if event.get("is_follow_through") is True:
        return "FT"
    return "unknown"


def _policy_complete(value: Any) -> bool:
    if not _has_content(value):
        return False
    if isinstance(value, str) and value.strip() in {"needs_human_review", "TODO", "todo"}:
        return False
    return True


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    return value not in (None, False)


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


def _dedupe(values: Any) -> list[str]:
    return sorted(dict.fromkeys(str(value) for value in values if str(value)))


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
