from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    DEFAULT_P3_SKILL_MAPPINGS_PATH,
    DEFAULT_P3_TOPIC_ASSIGNMENTS_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
    DEFAULT_REVIEW_QUEUE_REPORT_PATH,
    REVIEW_QUEUE_SCHEMA,
)
from exam_bank.p3_exact_skill.part_decomposition import enrich_queue_items_with_part_decomposition

QUEUE_SCHEMA_VERSION = 1
QUEUE_STATUSES = {
    "clean_candidate",
    "cross_topic_candidate",
    "split_needed_candidate",
    "conflict_candidate",
    "weak_candidate",
    "ambiguous_candidate",
    "blocked_candidate",
    "review_needed",
    "fallback_only",
}

REVIEW_ACTIONS = {
    "review_assets_and_skill",
    "split_by_part",
    "verify_mark_scheme_alignment",
    "reject_p1_prerequisite_only",
    "reject_missing_question_asset",
    "reject_missing_mark_scheme_asset",
    "defer_ambiguous_skill",
    "defer_visual_dependency",
    "already_reviewed",
    "needs_human_math_review",
    "verify_de_vs_implicit_differentiation",
    "verify_parametric_equation_parameter",
}

PARAMETRIC_IMPLICIT_SKILL_ID = "9709_p3_3_4_parametric_implicit_differentiation"
CROSS_TOPIC_REVIEWER_CHECKLIST = [
    "Identify the main skill being assessed.",
    "Identify any supporting skills used in the method.",
    "Decide whether the current whole-question/part scope is safe.",
    "Split by part/subpart if the item tests multiple independent skills.",
    "Avoid promoting broad whole-question evidence when the exact skill belongs only to one part.",
    "Do not use supporting skill context as mastery evidence unless reviewed directly.",
]


def build_p3_exact_skill_review_queue(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    topic_routing_path: str | Path | None = "output/json/question_bank.topic_routing.v1.json",
    asterion_question_bank_path: str | Path | None = "output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json",
    content_lab_candidates_path: str | Path | None = (
        "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json"
    ),
    mark_events_path: str | Path | None = "output/json/question_bank.mark_events.v1.json",
    p3_skill_mappings_path: str | Path = DEFAULT_P3_SKILL_MAPPINGS_PATH,
    p3_topic_assignments_path: str | Path | None = DEFAULT_P3_TOPIC_ASSIGNMENTS_PATH,
    reviewed_decisions_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    output_path: str | Path | None = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    report_path: str | Path | None = DEFAULT_REVIEW_QUEUE_REPORT_PATH,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    p3_skill_mappings_path = Path(p3_skill_mappings_path)
    reviewed_decisions_path = Path(reviewed_decisions_path)

    question_bank = _load_json(question_bank_path)
    skill_mappings = _load_json(p3_skill_mappings_path)
    reviewed_decisions = _load_optional_json(reviewed_decisions_path)
    topic_routes = _records_by_question_id(_load_optional_json(topic_routing_path).get("records", {}))
    asterion_questions = _records_by_question_id(_load_optional_json(asterion_question_bank_path).get("questions", []))
    content_lab_candidates = _content_lab_by_scope(_load_optional_json(content_lab_candidates_path).get("candidates", []))
    mark_events = _records_by_question_id(_load_optional_json(mark_events_path).get("records", []))
    topic_assignments = _topic_assignments_by_scope(_load_optional_json(p3_topic_assignments_path).get("assignments", []))

    question_by_id = _records_by_question_id(_question_records(question_bank))
    reviewed_scope_index, reconciliation = _reviewed_scope_index(reviewed_decisions, question_by_id=question_by_id)
    clean_reviewed_counts = _clean_reviewed_counts(reviewed_decisions)

    items: list[dict[str, Any]] = []
    seen_question_ids_with_mappings: set[str] = set()
    for mapping in _mapping_records(skill_mappings):
        if str(mapping.get("caie_class_or_component") or "").lower() != "paper 3":
            continue
        question_id = _text(mapping.get("question_id"))
        if not question_id:
            continue
        subpart_id = _text(mapping.get("subpart_id")) or f"{question_id}_whole"
        scope = (question_id, subpart_id)
        seen_question_ids_with_mappings.add(question_id)
        question = question_by_id.get(question_id, {})
        item = build_review_queue_item(
            mapping=mapping,
            question=question,
            topic_route=topic_routes.get(question_id, {}),
            asterion_question=asterion_questions.get(question_id, {}),
            content_lab_candidate=content_lab_candidates.get(scope, {}),
            mark_event_record=mark_events.get(question_id, {}),
            topic_assignment=topic_assignments.get(scope, {}),
            reviewed_decision=reviewed_scope_index.get(scope),
            clean_reviewed_counts=clean_reviewed_counts,
        )
        items.append(item)

    for question in _question_records(question_bank):
        question_id = _text(question.get("question_id"))
        if _text(question.get("paper_family")).lower() != "p3" or not question_id:
            continue
        subpart_id = f"{question_id}_whole"
        scope = (question_id, subpart_id)
        if question_id in seen_question_ids_with_mappings:
            continue
        item = build_review_queue_item(
            mapping={"question_id": question_id, "subpart_id": subpart_id, "subpart_label": "whole"},
            question=question,
            topic_route=topic_routes.get(question_id, {}),
            asterion_question=asterion_questions.get(question_id, {}),
            content_lab_candidate=content_lab_candidates.get(scope, {}),
            mark_event_record=mark_events.get(question_id, {}),
            topic_assignment=topic_assignments.get(scope, {}),
            reviewed_decision=reviewed_scope_index.get(scope),
            clean_reviewed_counts=clean_reviewed_counts,
        )
        items.append(item)

    queue_input_scopes = {(item["question_id"], item["subpart_id"]) for item in items}
    reconciliation["reviewed_records_missing_from_queue_inputs"] = sorted(
        f"{question_id}:{subpart_id}"
        for question_id, subpart_id in reconciliation["reviewed_scopes"]
        if (question_id, subpart_id) not in queue_input_scopes
    )
    for item in items:
        scope = (item["question_id"], item["subpart_id"])
        if scope in reconciliation["duplicate_reviewed_scopes"]:
            item["reconciliation_flags"].append("duplicate_reviewed_registry_scope")

    enrich_queue_items_with_part_decomposition(items)
    items.sort(key=lambda item: (-int(item["priority_score"]), item["question_id"], item["subpart_id"]))
    summary = summarize_review_queue(items, reconciliation=reconciliation)
    payload = {
        "schema": REVIEW_QUEUE_SCHEMA,
        "schema_version": QUEUE_SCHEMA_VERSION,
        "artifact_kind": "review_queue_report",
        "generated_at": generated_at or _utc_now_iso(),
        "source_inputs": _source_inputs(
            question_bank_path,
            topic_routing_path,
            asterion_question_bank_path,
            content_lab_candidates_path,
            mark_events_path,
            p3_skill_mappings_path,
            p3_topic_assignments_path,
            reviewed_decisions_path,
        ),
        "summary": summary,
        "reconciliation": _public_reconciliation(reconciliation),
        "items": items,
    }
    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    if report_path and not dry_run:
        write_review_queue_report(payload, output_path=report_path)
    return payload


def build_review_queue_item(
    *,
    mapping: dict[str, Any],
    question: dict[str, Any] | None = None,
    topic_route: dict[str, Any] | None = None,
    asterion_question: dict[str, Any] | None = None,
    content_lab_candidate: dict[str, Any] | None = None,
    mark_event_record: dict[str, Any] | None = None,
    topic_assignment: dict[str, Any] | None = None,
    reviewed_decision: dict[str, Any] | None = None,
    clean_reviewed_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    question = question or {}
    topic_route = topic_route or {}
    asterion_question = asterion_question or {}
    content_lab_candidate = content_lab_candidate or {}
    mark_event_record = mark_event_record or {}
    topic_assignment = topic_assignment or {}
    clean_reviewed_counts = clean_reviewed_counts or {}

    question_id = _text(mapping.get("question_id") or question.get("question_id"))
    subpart_id = _text(mapping.get("subpart_id")) or f"{question_id}_whole"
    subpart_label = _text(mapping.get("subpart_label")) or "whole"
    part_id = "whole" if subpart_label == "whole" else subpart_label
    candidate_p3_ids = _unique_texts(
        list(mapping.get("primary_skill_ids") or [])
        + list(mapping.get("secondary_skill_ids") or [])
        + list(content_lab_candidate.get("source_skill_ids") or [])
    )
    primary_candidate_skill_ids = _unique_texts(
        list(mapping.get("primary_skill_ids") or []) + list(content_lab_candidate.get("source_skill_ids") or [])
    )
    prerequisite_ids = _unique_texts(mapping.get("prerequisite_skill_ids") or [])
    supporting_candidate_skill_ids = _unique_texts(list(mapping.get("secondary_skill_ids") or []) + prerequisite_ids)
    candidate_source_skill_ids = _unique_texts(candidate_p3_ids + prerequisite_ids)
    non_p3_candidate_ids = [skill_id for skill_id in candidate_source_skill_ids if not skill_id.startswith("9709_p3_")]

    question_assets = _question_asset_refs(question, asterion_question, content_lab_candidate)
    mark_scheme_assets = _mark_scheme_asset_refs(question, asterion_question, content_lab_candidate, mark_event_record)
    mark_event_refs = _mark_event_refs(mark_event_record)
    quality_gate = asterion_question.get("quality_gate") if isinstance(asterion_question.get("quality_gate"), dict) else {}
    route_status, blockers, action = classify_review_queue_item(
        has_question_asset=_has_existing_asset(question_assets),
        has_mark_scheme_asset=_has_existing_asset(mark_scheme_assets),
        p3_skill_ids=candidate_p3_ids,
        non_p3_skill_ids=non_p3_candidate_ids,
        topic_route=topic_route,
        mapping=mapping,
        mark_event_record=mark_event_record,
        quality_gate=quality_gate,
    )
    cross_topic = build_cross_topic_context(
        mapping=mapping,
        topic_route=topic_route,
        topic_assignment=topic_assignment,
        primary_candidate_skill_ids=primary_candidate_skill_ids,
        supporting_candidate_skill_ids=supporting_candidate_skill_ids,
        candidate_p3_skill_ids=candidate_p3_ids,
        proposed_blockers=blockers,
    )
    route_status, action = refine_route_status_for_review_triage(
        route_status=route_status,
        action=action,
        cross_topic_status=cross_topic["cross_topic_status"],
        blockers=blockers,
    )
    reviewed_status = "not_reviewed"
    reviewed_route_status = None
    reviewed_evidence_ids: list[str] = []
    reconciliation_flags: list[str] = []
    if reviewed_decision:
        reviewed_status = "already_reviewed"
        reviewed_route_status = _text(reviewed_decision.get("route_status"))
        reviewed_evidence_ids = [_text(reviewed_decision.get("evidence_id"))]
        action = "already_reviewed"
        if blockers:
            reconciliation_flags.append("reviewed_record_still_has_upstream_blockers")
        reviewed_skill_ids = _unique_texts(reviewed_decision.get("reviewed_source_skill_ids") or [])
        if reviewed_skill_ids and not set(reviewed_skill_ids).issubset(set(candidate_p3_ids)):
            reconciliation_flags.append("reviewed_skill_ids_not_in_current_candidate_mapping")

    sparse_skill_ids = [skill_id for skill_id in candidate_p3_ids if clean_reviewed_counts.get(skill_id, 0) == 0]
    item = {
        "queue_id": f"p3_exact_skill_review_queue:v1:{question_id}:{subpart_id}",
        "question_id": question_id,
        "part_id": part_id,
        "subpart_id": subpart_id,
        "subpart_label": subpart_label,
        "paper": _text(mapping.get("paper") or question.get("paper") or asterion_question.get("paper")),
        "session": _text(mapping.get("session") or mark_event_record.get("session")),
        "variant": _text(mapping.get("variant") or mark_event_record.get("variant")),
        "question_number": _text(mapping.get("question_number") or question.get("question_number")),
        "candidate_source_skill_ids": candidate_source_skill_ids,
        "candidate_p3_skill_ids": candidate_p3_ids,
        "primary_candidate_skill_ids": primary_candidate_skill_ids,
        "supporting_candidate_skill_ids": supporting_candidate_skill_ids,
        "candidate_prerequisite_skill_ids": prerequisite_ids,
        "candidate_region_topic": _candidate_region_topic(mapping, topic_assignment, topic_route),
        "topic_routing": _topic_routing_summary(topic_route),
        "cross_topic_status": cross_topic["cross_topic_status"],
        "topic_routing_topic_ids": cross_topic["topic_routing_topic_ids"],
        "topic_routing_alignment": cross_topic["topic_routing_alignment"],
        "cross_topic_notes": cross_topic["cross_topic_notes"],
        "recommended_scope": cross_topic["recommended_scope"],
        "reviewer_cross_topic_checklist": cross_topic["reviewer_cross_topic_checklist"],
        "ambiguity_reason": ambiguity_reason_for_item(
            route_status=route_status,
            cross_topic_status=cross_topic["cross_topic_status"],
            topic_routing_alignment=cross_topic["topic_routing_alignment"],
            blockers=blockers,
            candidate_p3_skill_count=len(candidate_p3_ids),
            subpart_label=subpart_label,
        ),
        "review_priority_group": review_priority_group(route_status),
        "asterion_candidate": _content_lab_summary(content_lab_candidate),
        "source_question_asset_refs": question_assets,
        "source_mark_scheme_asset_refs": mark_scheme_assets,
        "mark_event_refs": mark_event_refs,
        "mark_event_safety": _mark_event_safety(mark_event_record),
        "crop_quality_status": _crop_quality_status(question, quality_gate),
        "text_advisory_status": _text_advisory_status(question, mark_event_record),
        "visual_dependency": _visual_dependency(question, quality_gate),
        "reviewed_decision_status": reviewed_status,
        "reviewed_decision_route_status": reviewed_route_status,
        "reviewed_evidence_ids": reviewed_evidence_ids,
        "reviewed_decision_skill_ids": _unique_texts((reviewed_decision or {}).get("reviewed_source_skill_ids") or []),
        "proposed_route_status": route_status,
        "proposed_blockers": blockers,
        "recommended_review_action": action,
        "reviewer_checklist": reviewer_checklist_for_action(action, blockers),
        "reconciliation_flags": reconciliation_flags,
        "priority_score": _priority_score(
            route_status=route_status,
            action=action,
            has_question_asset=_has_existing_asset(question_assets),
            has_mark_scheme_asset=_has_existing_asset(mark_scheme_assets),
            candidate_p3_skill_count=len(candidate_p3_ids),
            mark_event_count=len(mark_event_refs),
            sparse_skill_count=len(sparse_skill_ids),
            in_content_lab=bool(content_lab_candidate),
            reviewed_status=reviewed_status,
        ),
    }
    return item


def classify_review_queue_item(
    *,
    has_question_asset: bool,
    has_mark_scheme_asset: bool,
    p3_skill_ids: list[str],
    non_p3_skill_ids: list[str],
    topic_route: dict[str, Any] | None = None,
    mapping: dict[str, Any] | None = None,
    mark_event_record: dict[str, Any] | None = None,
    quality_gate: dict[str, Any] | None = None,
) -> tuple[str, list[str], str]:
    topic_route = topic_route or {}
    mapping = mapping or {}
    mark_event_record = mark_event_record or {}
    quality_gate = quality_gate or {}
    blockers: list[str] = []

    if not has_question_asset:
        blockers.append("missing_question_asset")
    if not has_mark_scheme_asset:
        blockers.append("missing_mark_scheme_asset")
    if non_p3_skill_ids and not p3_skill_ids:
        blockers.append("p1_or_support_only_candidate_skill")
    if not p3_skill_ids:
        blockers.append("no_candidate_p3_skill")
    if _topic_ambiguous(topic_route, mapping):
        blockers.append("mixed_or_ambiguous_topic")
    if _mark_events_advisory_only(mark_event_record):
        blockers.append("mark_events_advisory_only")
    if mark_event_record and not mark_event_record.get("safe_for_advisory_use", False):
        blockers.append("mark_events_not_advisory_safe")
    if quality_gate and quality_gate.get("question_crop_ok") is False:
        blockers.append("question_crop_not_high_confidence")
    if quality_gate and quality_gate.get("mark_scheme_crop_ok") is False:
        blockers.append("mark_scheme_crop_not_high_confidence")
    if quality_gate and quality_gate.get("text_only_display_allowed") is False:
        blockers.append("text_or_ocr_not_authoritative")
    if quality_gate and quality_gate.get("visual_required") is True:
        blockers.append("visual_dependency")
    parametric_boundary_blocker = _parametric_implicit_boundary_blocker(
        p3_skill_ids=p3_skill_ids,
        topic_route=topic_route,
        mapping=mapping,
    )
    if parametric_boundary_blocker:
        blockers.append(parametric_boundary_blocker)

    if "missing_question_asset" in blockers:
        return "blocked_candidate", blockers, "reject_missing_question_asset"
    if "missing_mark_scheme_asset" in blockers:
        return "blocked_candidate", blockers, "reject_missing_mark_scheme_asset"
    if "p1_or_support_only_candidate_skill" in blockers:
        return "blocked_candidate", blockers, "reject_p1_prerequisite_only"
    if "no_candidate_p3_skill" in blockers:
        return "blocked_candidate", blockers, "needs_human_math_review"
    if parametric_boundary_blocker:
        if parametric_boundary_blocker == "weak_parametric_equation_evidence_missing_parameter":
            return "conflict_candidate", blockers, "verify_parametric_equation_parameter"
        return "conflict_candidate", blockers, "verify_de_vs_implicit_differentiation"
    if "mixed_or_ambiguous_topic" in blockers or len(p3_skill_ids) > 1:
        return "ambiguous_candidate", blockers, "defer_ambiguous_skill"
    if "visual_dependency" in blockers and "question_crop_not_high_confidence" in blockers:
        return "fallback_only", blockers, "defer_visual_dependency"
    if not _mark_event_refs_present(mark_event_record):
        return "weak_candidate", blockers, "verify_mark_scheme_alignment"
    return "clean_candidate", blockers, "review_assets_and_skill"


def refine_route_status_for_review_triage(
    *,
    route_status: str,
    action: str,
    cross_topic_status: str,
    blockers: list[str],
) -> tuple[str, str]:
    if route_status in {"blocked_candidate", "fallback_only", "conflict_candidate"}:
        return route_status, action
    if cross_topic_status == "conflict_needs_review":
        return "conflict_candidate", action
    if cross_topic_status == "cross_topic_split_needed":
        return "split_needed_candidate", "split_by_part"
    if cross_topic_status == "cross_topic_reviewable":
        return "cross_topic_candidate", action
    if route_status == "weak_candidate":
        return route_status, action
    if route_status == "ambiguous_candidate" and _only_weak_context_blockers(blockers):
        return "weak_candidate", action
    return route_status, action


def _only_weak_context_blockers(blockers: list[str]) -> bool:
    strong_blockers = {
        "missing_question_asset",
        "missing_mark_scheme_asset",
        "p1_or_support_only_candidate_skill",
        "no_candidate_p3_skill",
        "mixed_or_ambiguous_topic",
        "possible_differential_equation_not_parametric_or_implicit",
        "weak_parametric_implicit_evidence_dydx_only",
        "weak_parametric_equation_evidence_missing_parameter",
    }
    return bool(blockers) and not (set(blockers) & strong_blockers)


def review_priority_group(route_status: str) -> str:
    return {
        "clean_candidate": "1_clean_candidate",
        "cross_topic_candidate": "2_cross_topic_candidate",
        "split_needed_candidate": "3_split_needed_candidate",
        "weak_candidate": "4_weak_candidate",
        "conflict_candidate": "5_conflict_candidate",
        "fallback_only": "6_fallback_only",
        "ambiguous_candidate": "7_ambiguous_candidate",
        "blocked_candidate": "8_blocked_candidate",
    }.get(route_status, "9_review_needed")


def ambiguity_reason_for_item(
    *,
    route_status: str,
    cross_topic_status: str,
    topic_routing_alignment: str,
    blockers: list[str],
    candidate_p3_skill_count: int,
    subpart_label: str,
) -> str:
    blocker_set = set(blockers)
    if route_status == "conflict_candidate":
        if "possible_differential_equation_not_parametric_or_implicit" in blocker_set:
            return "de_vs_parametric_implicit_conflict"
        if "weak_parametric_implicit_evidence_dydx_only" in blocker_set:
            return "de_vs_parametric_implicit_conflict"
        if "weak_parametric_equation_evidence_missing_parameter" in blocker_set:
            return "weak_candidate_skill_context"
        return "topic_routing_candidate_mismatch"
    if route_status == "cross_topic_candidate":
        return "topic_routing_candidate_mismatch" if topic_routing_alignment == "supporting_topic" else "cross_topic_reviewable"
    if route_status == "split_needed_candidate":
        if subpart_label == "whole":
            return "broad_whole_question_scope"
        return "multiple_candidate_skills"
    if route_status == "weak_candidate":
        return "fallback_or_low_quality_context" if _has_quality_blockers(blocker_set) else "weak_candidate_skill_context"
    if route_status == "fallback_only":
        return "fallback_or_low_quality_context"
    if route_status == "blocked_candidate":
        return "fallback_or_low_quality_context"
    if route_status == "ambiguous_candidate":
        if candidate_p3_skill_count > 1:
            return "multiple_candidate_skills"
        if cross_topic_status == "unknown":
            return "unknown_ambiguity"
    return "unknown_ambiguity"


def _has_quality_blockers(blockers: set[str]) -> bool:
    return bool(
        blockers
        & {
            "question_crop_not_high_confidence",
            "mark_scheme_crop_not_high_confidence",
            "text_or_ocr_not_authoritative",
            "visual_dependency",
            "mark_events_not_advisory_safe",
        }
    )


def summarize_review_queue(items: list[dict[str, Any]], *, reconciliation: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(_text(item.get("proposed_route_status")) for item in items)
    blockers = Counter(blocker for item in items for blocker in item.get("proposed_blockers") or [])
    actions = Counter(_text(item.get("recommended_review_action")) for item in items)
    cross_topic_statuses = Counter(_text(item.get("cross_topic_status")) for item in items)
    topic_alignments = Counter(_text(item.get("topic_routing_alignment")) for item in items)
    already_reviewed = sum(1 for item in items if item.get("reviewed_decision_status") == "already_reviewed")
    reconciliation_flags = Counter(flag for item in items for flag in item.get("reconciliation_flags") or [])
    return {
        "total_queue_items": len(items),
        "candidate_clean_looking_not_reviewed": sum(
            1
            for item in items
            if item.get("proposed_route_status") == "clean_candidate"
            and item.get("reviewed_decision_status") != "already_reviewed"
        ),
        "thin_candidates": status_counts.get("thin_candidate", 0),
        "ambiguous_candidates": status_counts.get("ambiguous_candidate", 0),
        "cross_topic_candidates": status_counts.get("cross_topic_candidate", 0),
        "split_needed_candidates": status_counts.get("split_needed_candidate", 0),
        "conflict_candidates": status_counts.get("conflict_candidate", 0),
        "weak_candidates": status_counts.get("weak_candidate", 0),
        "blocked_candidates": status_counts.get("blocked_candidate", 0),
        "fallback_only_candidates": status_counts.get("fallback_only", 0),
        "review_needed_candidates": status_counts.get("review_needed", 0),
        "already_reviewed_records": already_reviewed,
        "missing_question_asset_count": blockers.get("missing_question_asset", 0),
        "missing_mark_scheme_asset_count": blockers.get("missing_mark_scheme_asset", 0),
        "no_candidate_skill_count": blockers.get("no_candidate_p3_skill", 0),
        "advisory_only_mark_event_count": blockers.get("mark_events_advisory_only", 0),
        "status_counts": dict(status_counts),
        "cross_topic_status_counts": dict(cross_topic_statuses),
        "topic_routing_alignment_counts": dict(topic_alignments),
        "blocker_counts": dict(blockers.most_common()),
        "recommended_action_counts": dict(actions.most_common()),
        "reconciliation_flag_counts": dict(reconciliation_flags.most_common()),
        "reviewed_registry_duplicate_scope_count": len(reconciliation["duplicate_reviewed_scopes"]),
        "reviewed_records_missing_from_bank_count": len(reconciliation["reviewed_records_without_bank_match"]),
        "reviewed_records_missing_from_queue_input_count": len(reconciliation["reviewed_records_missing_from_queue_inputs"]),
    }


def write_review_queue_report(
    queue: dict[str, Any],
    *,
    output_path: str | Path = DEFAULT_REVIEW_QUEUE_REPORT_PATH,
) -> str:
    text = render_review_queue_report(queue)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def render_review_queue_report(queue: dict[str, Any]) -> str:
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    reconciliation = queue.get("reconciliation") if isinstance(queue.get("reconciliation"), dict) else {}
    items = [item for item in queue.get("items", []) if isinstance(item, dict)]
    priority_order = [
        "clean_candidate",
        "cross_topic_candidate",
        "split_needed_candidate",
        "weak_candidate",
        "conflict_candidate",
        "fallback_only",
        "ambiguous_candidate",
        "blocked_candidate",
    ]
    priority_items = [
        item
        for item in items
        if item.get("reviewed_decision_status") != "already_reviewed"
        and item.get("proposed_route_status")
        in {"clean_candidate", "cross_topic_candidate", "split_needed_candidate", "weak_candidate"}
    ][:40]
    blocked_items = [
        item
        for item in items
        if item.get("proposed_route_status") in {"blocked_candidate", "fallback_only", "conflict_candidate"}
    ]
    lines = [
        "# P3 Exact-Skill Review Queue",
        "",
        "This is a reviewer queue, not the final Asterion sidecar. `clean_candidate` means worth review; it does not mean reviewed clean evidence.",
        "Reduced ambiguity is triage only. New candidate statuses explain review handling; they do not increase trust or create reviewed evidence.",
        "",
        "## Summary",
        "",
        f"- Total queue items: {summary.get('total_queue_items', 0)}",
        f"- Candidate clean-looking but not reviewed: {summary.get('candidate_clean_looking_not_reviewed', 0)}",
        f"- Thin candidates: {summary.get('thin_candidates', 0)}",
        f"- Cross-topic candidates: {summary.get('cross_topic_candidates', 0)}",
        f"- Split-needed candidates: {summary.get('split_needed_candidates', 0)}",
        f"- Conflict candidates: {summary.get('conflict_candidates', 0)}",
        f"- Weak candidates: {summary.get('weak_candidates', 0)}",
        f"- Ambiguous candidates: {summary.get('ambiguous_candidates', 0)}",
        f"- Blocked candidates: {summary.get('blocked_candidates', 0)}",
        f"- Fallback-only candidates: {summary.get('fallback_only_candidates', 0)}",
        f"- Already reviewed records: {summary.get('already_reviewed_records', 0)}",
        f"- Missing question asset count: {summary.get('missing_question_asset_count', 0)}",
        f"- Missing mark-scheme asset count: {summary.get('missing_mark_scheme_asset_count', 0)}",
        f"- No candidate P3 skill count: {summary.get('no_candidate_skill_count', 0)}",
        f"- Advisory-only mark-event count: {summary.get('advisory_only_mark_event_count', 0)}",
        "",
        "## Status Counts",
        "",
        *_ordered_counter_lines(summary.get("status_counts") or {}, priority_order),
        "",
        "## Human-Review Priority Groups",
        "",
        *_priority_group_lines(summary.get("status_counts") or {}, priority_order),
        "",
        "## Cross-Topic Status Counts",
        "",
        *_counter_lines(summary.get("cross_topic_status_counts") or {}),
        "",
        "## Topic-Routing Alignment Counts",
        "",
        *_counter_lines(summary.get("topic_routing_alignment_counts") or {}),
        "",
        "## Priority Review Items",
        "",
        *_item_lines(priority_items),
        "",
        "## Blocked And Fallback Groups",
        "",
        *_blocked_group_lines(blocked_items),
        "",
        "## Top Blockers",
        "",
        *_counter_lines(summary.get("blocker_counts") or {}),
        "",
        "## Existing Reviewed Registry Reconciliation",
        "",
        f"- Reviewed scopes: {len(reconciliation.get('reviewed_scopes', []))}",
        f"- Duplicate reviewed scopes: {len(reconciliation.get('duplicate_reviewed_scopes', []))}",
        f"- Reviewed records with no matching canonical bank item: {len(reconciliation.get('reviewed_records_without_bank_match', []))}",
        f"- Reviewed records missing from queue inputs: {len(reconciliation.get('reviewed_records_missing_from_queue_inputs', []))}",
        "",
        "### Already Reviewed Queue Items",
        "",
        *_item_lines([item for item in items if item.get("reviewed_decision_status") == "already_reviewed"][:40]),
        "",
        "### Reconciliation Flag Counts",
        "",
        *_counter_lines(summary.get("reconciliation_flag_counts") or {}),
        "",
        "### Reconciliation Flags",
        "",
        *_reconciliation_lines(reconciliation),
        "",
        "## Reviewer Checklist",
        "",
        "- Verify canonical question and mark-scheme images directly; OCR/native text is review context only.",
        "- Confirm the candidate P3 skill is exact, not merely a parent topic or prerequisite.",
        "- Split mixed whole-question evidence into part/subpart records before adding clean reviewed decisions.",
        "- Confirm mark-scheme alignment and mark-event refs; advisory mark events alone cannot authorize generation.",
        "- Preserve blockers in the reviewed-decision registry for rejected, ambiguous, deferred, or fallback-only records.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def reviewer_checklist_for_action(action: str, blockers: list[str]) -> list[str]:
    checklist = [
        "Open the canonical question image and mark-scheme image.",
        "Verify the exact P3 skill against the source images.",
        "Record a reviewed-decision route status; do not copy queue status as clean evidence.",
    ]
    if action == "split_by_part" or "mixed_or_ambiguous_topic" in blockers:
        checklist.append("Split the evidence by part or subpart before approving an exact skill.")
    if action == "verify_mark_scheme_alignment":
        checklist.append("Check that mark-event refs align to the proposed part/subpart.")
    if action == "verify_de_vs_implicit_differentiation":
        checklist.append("Check whether the method is separation of variables or an actual parametric/implicit differentiation task.")
        checklist.append("Do not approve parametric/implicit differentiation from dy/dx notation alone.")
    if action == "verify_parametric_equation_parameter":
        checklist.append("Check that parametric evidence has separate x and y relations in a parameter such as t or theta.")
        checklist.append("Do not approve parametric differentiation from a source-topic hint or loose x = / y = text alone.")
    if action == "defer_visual_dependency" or "visual_dependency" in blockers:
        checklist.append("Inspect visual/diagram dependency manually; text is insufficient.")
    if action.startswith("reject_"):
        checklist.append("Keep the rejected scope with blockers if it remains useful review context.")
    return checklist


def build_cross_topic_context(
    *,
    mapping: dict[str, Any],
    topic_route: dict[str, Any],
    topic_assignment: dict[str, Any],
    primary_candidate_skill_ids: list[str],
    supporting_candidate_skill_ids: list[str],
    candidate_p3_skill_ids: list[str],
    proposed_blockers: list[str],
) -> dict[str, Any]:
    assignment_topic_ids = _topic_assignment_topic_ids(topic_assignment)
    routing_topic_ids = _topic_routing_topic_ids(topic_route)
    primary_route_topic_id = _text(topic_route.get("primary_topic_id"))
    source_topic = _text(_nested(mapping, "evidence", "source_topic"))
    subpart_label = _text(mapping.get("subpart_label")) or "whole"
    evidence_granularity = _text(mapping.get("evidence_granularity"))
    blockers = set(proposed_blockers)
    notes: list[str] = []

    if routing_topic_ids and assignment_topic_ids:
        if set(routing_topic_ids) & set(assignment_topic_ids):
            alignment = "aligned"
        elif supporting_candidate_skill_ids or _plausible_cross_topic_source(source_topic):
            alignment = "supporting_topic"
            notes.append("Candidate skill/topic and topic-routing context differ but may describe different stages of one solution.")
        else:
            alignment = "cross_topic_possible"
            notes.append("Candidate topic and topic-routing topic differ; reviewer should decide whether this is support context or a conflict.")
    elif routing_topic_ids or assignment_topic_ids:
        alignment = "unknown"
        notes.append("Only one side of candidate topic or topic-routing context is available.")
    else:
        alignment = "unknown"

    if supporting_candidate_skill_ids:
        notes.append("Supporting candidate skills are review context only, not mastery evidence.")
    if source_topic:
        notes.append(f"Source topic hint: {source_topic}.")

    conflict_blockers = {
        "possible_differential_equation_not_parametric_or_implicit",
        "weak_parametric_implicit_evidence_dydx_only",
        "weak_parametric_equation_evidence_missing_parameter",
    }
    if blockers & conflict_blockers:
        status = "conflict_needs_review"
        alignment = "conflicting"
        recommended_scope = "reviewer_decide"
        notes.append("Method-critical mismatch flagged; do not treat this as ordinary supporting-topic context.")
    elif len(candidate_p3_skill_ids) > 1 and (subpart_label == "whole" or evidence_granularity == "whole_question_only"):
        status = "cross_topic_split_needed"
        recommended_scope = "part_level"
        notes.append("Multiple candidate P3 skills on a broad scope; reviewer should split by part/subpart if possible.")
    elif alignment in {"supporting_topic", "cross_topic_possible"} or supporting_candidate_skill_ids:
        status = "cross_topic_reviewable"
        recommended_scope = "reviewer_decide" if subpart_label == "whole" else "subpart_level"
    elif len(primary_candidate_skill_ids) == 1:
        status = "single_skill_candidate"
        recommended_scope = "whole_question" if subpart_label == "whole" else "subpart_level"
    else:
        status = "unknown"
        recommended_scope = "reviewer_decide"

    return {
        "cross_topic_status": status,
        "primary_candidate_skill_ids": primary_candidate_skill_ids,
        "supporting_candidate_skill_ids": supporting_candidate_skill_ids,
        "topic_routing_topic_ids": routing_topic_ids or ([primary_route_topic_id] if primary_route_topic_id else []),
        "topic_routing_alignment": alignment,
        "cross_topic_notes": _unique_texts(notes),
        "recommended_scope": recommended_scope,
        "reviewer_cross_topic_checklist": CROSS_TOPIC_REVIEWER_CHECKLIST,
    }


def _question_asset_refs(
    question: dict[str, Any],
    asterion_question: dict[str, Any],
    content_lab_candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    integrity = asterion_question.get("artifact_integrity") if isinstance(asterion_question.get("artifact_integrity"), dict) else {}
    refs = integrity.get("question_images") if isinstance(integrity.get("question_images"), list) else []
    if refs:
        return [_asset_ref(ref) for ref in refs if isinstance(ref, dict)]
    path = _text(
        question.get("canonical_question_artifact")
        or question.get("question_image_path")
        or _nested(content_lab_candidate, "source_artifacts", "question_crop_path")
    )
    return [{"path": path, "exists": bool(path)}] if path else []


def _mark_scheme_asset_refs(
    question: dict[str, Any],
    asterion_question: dict[str, Any],
    content_lab_candidate: dict[str, Any],
    mark_event_record: dict[str, Any],
) -> list[dict[str, Any]]:
    integrity = asterion_question.get("artifact_integrity") if isinstance(asterion_question.get("artifact_integrity"), dict) else {}
    refs = integrity.get("mark_scheme_images") if isinstance(integrity.get("mark_scheme_images"), list) else []
    if refs:
        return [_asset_ref(ref) for ref in refs if isinstance(ref, dict)]
    path = _text(
        question.get("canonical_mark_scheme_artifact")
        or question.get("mark_scheme_image_path")
        or _nested(content_lab_candidate, "source_artifacts", "mark_scheme_crop_path")
        or mark_event_record.get("source_mark_scheme_image_path")
    )
    return [{"path": path, "exists": bool(path)}] if path else []


def _asset_ref(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _text(ref.get("path")),
        "sha256": _text(ref.get("sha256")),
        "exists": bool(ref.get("exists")),
    }


def _mark_event_refs(mark_event_record: dict[str, Any]) -> list[dict[str, Any]]:
    events = mark_event_record.get("mark_events") if isinstance(mark_event_record.get("mark_events"), list) else []
    return [
        {
            "event_id": _text(event.get("event_id")),
            "part_path": event.get("part_path") if isinstance(event.get("part_path"), list) else [],
            "mark_code": _text(event.get("mark_code_raw") or event.get("mark_code")),
            "review_status": "advisory",
        }
        for event in events
        if isinstance(event, dict) and _text(event.get("event_id"))
    ]


def _mark_event_safety(mark_event_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "extraction_status": _text(mark_event_record.get("extraction_status")),
        "safe_for_advisory_use": bool(mark_event_record.get("safe_for_advisory_use")),
        "safe_for_marking_use": bool(mark_event_record.get("safe_for_marking_use")),
        "total_marks_match": bool(mark_event_record.get("total_marks_match")),
        "review_flags": mark_event_record.get("review_flags") if isinstance(mark_event_record.get("review_flags"), list) else [],
        "mark_event_count": len(mark_event_record.get("mark_events") or []),
    }


def _crop_quality_status(question: dict[str, Any], quality_gate: dict[str, Any]) -> dict[str, Any]:
    notes = question.get("notes") if isinstance(question.get("notes"), dict) else {}
    return {
        "question_crop_confidence": _text(notes.get("question_crop_confidence")),
        "mark_scheme_crop_confidence": _text(notes.get("mark_scheme_crop_confidence")),
        "question_crop_ok": quality_gate.get("question_crop_ok"),
        "mark_scheme_crop_ok": quality_gate.get("mark_scheme_crop_ok"),
        "quality_gate_reason_codes": quality_gate.get("reason_codes") if isinstance(quality_gate.get("reason_codes"), list) else [],
    }


def _text_advisory_status(question: dict[str, Any], mark_event_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_text_trust": _text(question.get("question_text_trust") or _nested(question, "notes", "question_text_trust")),
        "ocr_text_trust": _text(question.get("ocr_text_trust")),
        "text_only_status": _text(question.get("text_only_status")),
        "visual_curation_status": _text(question.get("visual_curation_status")),
        "mark_event_source_text_kind": _text(mark_event_record.get("source_text_kind")),
    }


def _visual_dependency(question: dict[str, Any], quality_gate: dict[str, Any]) -> dict[str, Any]:
    flags = question.get("visual_reason_flags") or _nested(question, "notes", "visual_reason_flags") or []
    return {
        "visual_required": bool(question.get("visual_required") or quality_gate.get("visual_required")),
        "visual_reason_flags": flags if isinstance(flags, list) else [],
    }


def _candidate_region_topic(
    mapping: dict[str, Any],
    topic_assignment: dict[str, Any],
    topic_route: dict[str, Any],
) -> dict[str, Any]:
    assignments = topic_assignment.get("topic_assignments")
    first_assignment = assignments[0] if isinstance(assignments, list) and assignments and isinstance(assignments[0], dict) else {}
    evidence = mapping.get("evidence") if isinstance(mapping.get("evidence"), dict) else {}
    return {
        "mapping_source_topic": _text(evidence.get("source_topic")),
        "topic_assignment_id": _text(first_assignment.get("topic_id")),
        "topic_assignment_name": _text(first_assignment.get("topic_name")),
        "subtopic_id": _text(first_assignment.get("subtopic_id")),
        "subtopic_name": _text(first_assignment.get("subtopic_name")),
        "topic_routing_primary_topic_id": _text(topic_route.get("primary_topic_id")),
    }


def _topic_routing_summary(topic_route: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_topic_id": _text(topic_route.get("primary_topic_id")),
        "confidence": _text(topic_route.get("confidence")),
        "review_required": bool(topic_route.get("review_required")),
        "review_reasons": topic_route.get("review_reasons") if isinstance(topic_route.get("review_reasons"), list) else [],
        "evidence_used": topic_route.get("evidence_used") if isinstance(topic_route.get("evidence_used"), list) else [],
        "routing_source": _text(topic_route.get("routing_source")),
    }


def _topic_assignment_topic_ids(topic_assignment: dict[str, Any]) -> list[str]:
    assignments = topic_assignment.get("topic_assignments")
    if not isinstance(assignments, list):
        return []
    return _unique_texts(
        [
            assignment.get("topic_id")
            for assignment in assignments
            if isinstance(assignment, dict) and _text(assignment.get("topic_id"))
        ]
    )


def _topic_routing_topic_ids(topic_route: dict[str, Any]) -> list[str]:
    topic_ids = [_text(topic_route.get("primary_topic_id"))]
    distribution = topic_route.get("topic_distribution")
    if isinstance(distribution, list):
        topic_ids.extend(
            _text(entry.get("topic_id"))
            for entry in distribution
            if isinstance(entry, dict) and _text(entry.get("topic_id"))
        )
    return _unique_texts(topic_ids)


def _plausible_cross_topic_source(source_topic: str) -> bool:
    return source_topic in {
        "differentiation",
        "differential_equations",
        "integration",
        "parametric_equations",
        "trigonometry",
        "vectors",
    }


def _content_lab_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    gate = candidate.get("generation_gate") if isinstance(candidate.get("generation_gate"), dict) else {}
    return {
        "candidate_id": _text(candidate.get("candidate_id")),
        "review_status": _text(candidate.get("review_status")),
        "source_mark_event_count": int(candidate.get("source_mark_event_count") or 0),
        "role_statuses": candidate.get("role_statuses") if isinstance(candidate.get("role_statuses"), dict) else {},
        "generation_gate_status": _text(gate.get("status")),
        "generation_gate_blocked": bool(gate.get("blocked")),
        "generation_gate_block_reasons": gate.get("block_reasons") if isinstance(gate.get("block_reasons"), list) else [],
    }


def _topic_ambiguous(topic_route: dict[str, Any], mapping: dict[str, Any]) -> bool:
    evidence = mapping.get("evidence") if isinstance(mapping.get("evidence"), dict) else {}
    return bool(
        topic_route.get("review_required")
        or _text(topic_route.get("confidence")) in {"low", "missing"}
        or evidence.get("topic_uncertain")
        or _text(evidence.get("topic_confidence")) == "low"
    )


def _parametric_implicit_boundary_blocker(
    *,
    p3_skill_ids: list[str],
    topic_route: dict[str, Any],
    mapping: dict[str, Any],
) -> str | None:
    if PARAMETRIC_IMPLICIT_SKILL_ID not in p3_skill_ids:
        return None
    evidence = mapping.get("evidence") if isinstance(mapping.get("evidence"), dict) else {}
    text = _normalised_evidence_text(evidence)
    if _has_separable_de_context(text, topic_route):
        return "possible_differential_equation_not_parametric_or_implicit"
    if (
        _looks_like_parametric_source(evidence)
        and not _has_implicit_context(text)
        and not _has_true_parametric_equation_context(text)
    ):
        return "weak_parametric_equation_evidence_missing_parameter"
    if _has_dydx_signal(text) and not _has_strong_parametric_or_implicit_context(text):
        return "weak_parametric_implicit_evidence_dydx_only"
    return None


def _normalised_evidence_text(evidence: dict[str, Any]) -> str:
    pieces = [
        evidence.get("question_text_snippet"),
        evidence.get("mark_scheme_text_snippet"),
    ]
    return " ".join(_text(piece).lower() for piece in pieces if _text(piece))


def _has_separable_de_context(text: str, topic_route: dict[str, Any]) -> bool:
    routed_to_de = _text(topic_route.get("primary_topic_id")) == "9709_p3_topic_differential_equations"
    method_terms = (
        "separate variables",
        "separated variables",
        "separation of variables",
        "solve the differential equation",
        "differential equation",
        "initial condition",
        "boundary condition",
    )
    return routed_to_de or any(term in text for term in method_terms)


def _has_dydx_signal(text: str) -> bool:
    dydx_terms = ("dy/dx", "d y/d x", "dy)/(dx", "ddyx", "dydx", "dx/dy")
    return any(term in text for term in dydx_terms)


def _has_strong_parametric_or_implicit_context(text: str) -> bool:
    strong_terms = (
        "parametric",
        "parameter",
        "in terms of t",
        "dx/dt",
        "dy/dt",
        "implicit",
        "implicitly",
        "dy/dx must be isolated",
        "isolate dy/dx",
    )
    return any(term in text for term in strong_terms)


def _looks_like_parametric_source(evidence: dict[str, Any]) -> bool:
    if _text(evidence.get("source_topic")) == "parametric_equations":
        return True
    matched_signals = evidence.get("matched_signals") if isinstance(evidence.get("matched_signals"), list) else []
    return any("parametric" in _text(signal) for signal in matched_signals)


def _has_implicit_context(text: str) -> bool:
    implicit_terms = (
        "implicit",
        "implicitly",
        "dy/dx must be isolated",
        "isolate dy/dx",
        "collect terms in dy/dx",
    )
    return any(term in text for term in implicit_terms)


def _has_true_parametric_equation_context(text: str) -> bool:
    has_x_relation = bool(re.search(r"\bx\s*=", text))
    has_y_relation = bool(re.search(r"\by\s*=", text))
    parameter_terms = (
        "in terms of t",
        "where t",
        "at t",
        "t =",
        "dx/dt",
        "dy/dt",
        "d x/d t",
        "d y/d t",
        "theta",
        "θ",
        "\u03b8",
        "dθ",
        "d θ",
    )
    return has_x_relation and has_y_relation and any(term in text for term in parameter_terms)


def _mark_events_advisory_only(mark_event_record: dict[str, Any]) -> bool:
    return bool(mark_event_record.get("mark_events")) and mark_event_record.get("safe_for_marking_use") is not True


def _mark_event_refs_present(mark_event_record: dict[str, Any]) -> bool:
    return bool(mark_event_record.get("mark_events"))


def _has_existing_asset(refs: list[dict[str, Any]]) -> bool:
    return any(_text(ref.get("path")) and ref.get("exists", True) is not False for ref in refs)


def _priority_score(
    *,
    route_status: str,
    action: str,
    has_question_asset: bool,
    has_mark_scheme_asset: bool,
    candidate_p3_skill_count: int,
    mark_event_count: int,
    sparse_skill_count: int,
    in_content_lab: bool,
    reviewed_status: str,
) -> int:
    if reviewed_status == "already_reviewed":
        return 0
    score = 0
    score += {"clean_candidate": 60, "thin_candidate": 35, "ambiguous_candidate": 25, "fallback_only": 10}.get(
        route_status,
        0,
    )
    score += 12 if has_question_asset else -40
    score += 12 if has_mark_scheme_asset else -40
    score += 15 if candidate_p3_skill_count == 1 else 5 if candidate_p3_skill_count > 1 else -20
    score += min(mark_event_count, 4) * 3
    score += min(sparse_skill_count, 2) * 6
    score += 8 if in_content_lab else 0
    score += 5 if action == "review_assets_and_skill" else 0
    return score


def _reviewed_scope_index(
    reviewed_decisions: dict[str, Any],
    *,
    question_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    duplicate_scopes: set[tuple[str, str]] = set()
    no_bank_match: list[str] = []
    reviewed_scopes: set[tuple[str, str]] = set()
    for record in reviewed_decisions.get("records", []) if isinstance(reviewed_decisions.get("records"), list) else []:
        if not isinstance(record, dict):
            continue
        question_id = _text(record.get("question_id"))
        subpart_id = _text(record.get("subpart_id")) or f"{question_id}_whole"
        scope = (question_id, subpart_id)
        reviewed_scopes.add(scope)
        if scope in seen:
            duplicate_scopes.add(scope)
        seen.add(scope)
        index.setdefault(scope, record)
        if question_id not in question_by_id:
            no_bank_match.append(f"{question_id}:{subpart_id}")
    reconciliation = {
        "reviewed_scopes": reviewed_scopes,
        "duplicate_reviewed_scopes": duplicate_scopes,
        "reviewed_records_without_bank_match": sorted(no_bank_match),
        "reviewed_records_missing_from_queue_inputs": [],
    }
    return index, reconciliation


def _clean_reviewed_counts(reviewed_decisions: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in reviewed_decisions.get("records", []) if isinstance(reviewed_decisions.get("records"), list) else []:
        if isinstance(record, dict) and record.get("route_status") == "clean":
            counts.update(_unique_texts(record.get("reviewed_source_skill_ids") or []))
    return dict(counts)


def _public_reconciliation(reconciliation: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewed_scopes": sorted(f"{question_id}:{subpart_id}" for question_id, subpart_id in reconciliation["reviewed_scopes"]),
        "duplicate_reviewed_scopes": sorted(
            f"{question_id}:{subpart_id}" for question_id, subpart_id in reconciliation["duplicate_reviewed_scopes"]
        ),
        "reviewed_records_without_bank_match": reconciliation["reviewed_records_without_bank_match"],
        "reviewed_records_missing_from_queue_inputs": reconciliation["reviewed_records_missing_from_queue_inputs"],
    }


def _source_inputs(*paths: str | Path | None) -> list[dict[str, Any]]:
    source_inputs = []
    for path in paths:
        if path is None:
            continue
        source_inputs.append({"path": str(path), "exists": Path(path).exists()})
    return source_inputs


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _mapping_records(payload: Any) -> list[dict[str, Any]]:
    records = payload.get("mappings") if isinstance(payload, dict) else []
    return [record for record in records if isinstance(record, dict)]


def _records_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    if isinstance(records, dict):
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}
    if not isinstance(records, list):
        return {}
    return {
        _text(record.get("question_id")): record
        for record in records
        if isinstance(record, dict) and _text(record.get("question_id"))
    }


def _content_lab_by_scope(records: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(records, list):
        return {}
    return {
        (_text(record.get("question_id")), _text(record.get("subpart_id"))): record
        for record in records
        if isinstance(record, dict) and _text(record.get("question_id")) and _text(record.get("subpart_id"))
    }


def _topic_assignments_by_scope(records: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(records, list):
        return {}
    return {
        (_text(record.get("question_id")), _text(record.get("subpart_id"))): record
        for record in records
        if isinstance(record, dict) and _text(record.get("question_id")) and _text(record.get("subpart_id"))
    }


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _unique_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _text(value: Any) -> str:
    return str(value or "").strip()


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _counter_lines(counter: dict[str, Any]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- `{key}`: {value}" for key, value in counter.items()]


def _ordered_counter_lines(counter: dict[str, Any], order: list[str]) -> list[str]:
    if not counter:
        return ["- None"]
    lines = [f"- `{key}`: {counter.get(key, 0)}" for key in order if key in counter]
    lines.extend(f"- `{key}`: {value}" for key, value in counter.items() if key not in order)
    return lines or ["- None"]


def _priority_group_lines(counter: dict[str, Any], order: list[str]) -> list[str]:
    if not counter:
        return ["- None"]
    descriptions = {
        "clean_candidate": "clean-looking source-backed review candidates; still not reviewed evidence",
        "cross_topic_candidate": "plausible primary/supporting P3 cross-topic candidates",
        "split_needed_candidate": "scope likely too broad; part/subpart review needed",
        "weak_candidate": "some skill evidence, but weaker context or alignment",
        "conflict_candidate": "known-risk or method-critical conflict; avoid routine batches",
        "fallback_only": "low-quality/visual-dependent fallback review only",
        "ambiguous_candidate": "true unresolved ambiguity after sharper triage",
        "blocked_candidate": "missing or invalid inputs for review",
    }
    return [
        f"- `{status}`: {counter.get(status, 0)} ({descriptions[status]})"
        for status in order
        if status in counter or counter.get(status, 0)
    ]


def _item_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    lines: list[str] = []
    for item in items:
        skills = ", ".join(item.get("candidate_p3_skill_ids") or []) or "none"
        blockers = ", ".join(item.get("proposed_blockers") or []) or "none"
        cross_topic = item.get("cross_topic_status") or "unknown"
        lines.append(
            "- "
            f"`{item.get('queue_id')}` | `{item.get('proposed_route_status')}` | "
            f"cross-topic `{cross_topic}` | action `{item.get('recommended_review_action')}` | "
            f"skills `{skills}` | blockers `{blockers}`"
        )
    return lines


def _blocked_group_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        blockers = item.get("proposed_blockers") or ["no_blocker_recorded"]
        for blocker in blockers:
            grouped[str(blocker)].append(item)
    lines: list[str] = []
    for blocker, blocked_items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        sample_ids = ", ".join(f"`{item['question_id']}:{item['subpart_id']}`" for item in blocked_items[:10])
        lines.append(f"- `{blocker}`: {len(blocked_items)} items. Sample: {sample_ids}")
    return lines


def _reconciliation_lines(reconciliation: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for field in (
        "duplicate_reviewed_scopes",
        "reviewed_records_without_bank_match",
        "reviewed_records_missing_from_queue_inputs",
    ):
        values = reconciliation.get(field) or []
        lines.append(f"- `{field}`: {', '.join(f'`{value}`' for value in values) if values else 'none'}")
    return lines
