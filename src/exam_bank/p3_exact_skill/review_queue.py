from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
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

QUEUE_SCHEMA_VERSION = 1
QUEUE_STATUSES = {
    "clean_candidate",
    "thin_candidate",
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
}


def build_p3_exact_skill_review_queue(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    topic_routing_path: str | Path | None = "output/json/question_bank.topic_routing.v1.json",
    asterion_question_bank_path: str | Path | None = "output/asterion/exports/latest/asterion_question_bank_v1.json",
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
    prerequisite_ids = _unique_texts(mapping.get("prerequisite_skill_ids") or [])
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
        "candidate_prerequisite_skill_ids": prerequisite_ids,
        "candidate_region_topic": _candidate_region_topic(mapping, topic_assignment, topic_route),
        "topic_routing": _topic_routing_summary(topic_route),
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

    if "missing_question_asset" in blockers:
        return "blocked_candidate", blockers, "reject_missing_question_asset"
    if "missing_mark_scheme_asset" in blockers:
        return "blocked_candidate", blockers, "reject_missing_mark_scheme_asset"
    if "p1_or_support_only_candidate_skill" in blockers:
        return "blocked_candidate", blockers, "reject_p1_prerequisite_only"
    if "no_candidate_p3_skill" in blockers:
        return "blocked_candidate", blockers, "needs_human_math_review"
    if "mixed_or_ambiguous_topic" in blockers or len(p3_skill_ids) > 1:
        return "ambiguous_candidate", blockers, "defer_ambiguous_skill"
    if "visual_dependency" in blockers and "question_crop_not_high_confidence" in blockers:
        return "fallback_only", blockers, "defer_visual_dependency"
    if not _mark_event_refs_present(mark_event_record):
        return "thin_candidate", blockers, "verify_mark_scheme_alignment"
    return "clean_candidate", blockers, "review_assets_and_skill"


def summarize_review_queue(items: list[dict[str, Any]], *, reconciliation: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(_text(item.get("proposed_route_status")) for item in items)
    blockers = Counter(blocker for item in items for blocker in item.get("proposed_blockers") or [])
    actions = Counter(_text(item.get("recommended_review_action")) for item in items)
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
        "blocked_candidates": status_counts.get("blocked_candidate", 0),
        "fallback_only_candidates": status_counts.get("fallback_only", 0),
        "review_needed_candidates": status_counts.get("review_needed", 0),
        "already_reviewed_records": already_reviewed,
        "missing_question_asset_count": blockers.get("missing_question_asset", 0),
        "missing_mark_scheme_asset_count": blockers.get("missing_mark_scheme_asset", 0),
        "no_candidate_skill_count": blockers.get("no_candidate_p3_skill", 0),
        "advisory_only_mark_event_count": blockers.get("mark_events_advisory_only", 0),
        "status_counts": dict(status_counts),
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
    priority_items = [
        item
        for item in items
        if item.get("reviewed_decision_status") != "already_reviewed"
        and item.get("proposed_route_status") in {"clean_candidate", "thin_candidate", "ambiguous_candidate"}
    ][:40]
    blocked_items = [item for item in items if item.get("proposed_route_status") in {"blocked_candidate", "fallback_only"}]
    lines = [
        "# P3 Exact-Skill Review Queue",
        "",
        "This is a reviewer queue, not the final Asterion sidecar. `clean_candidate` means worth review; it does not mean reviewed clean evidence.",
        "",
        "## Summary",
        "",
        f"- Total queue items: {summary.get('total_queue_items', 0)}",
        f"- Candidate clean-looking but not reviewed: {summary.get('candidate_clean_looking_not_reviewed', 0)}",
        f"- Thin candidates: {summary.get('thin_candidates', 0)}",
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
        *_counter_lines(summary.get("status_counts") or {}),
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
    if action == "defer_visual_dependency" or "visual_dependency" in blockers:
        checklist.append("Inspect visual/diagram dependency manually; text is insufficient.")
    if action.startswith("reject_"):
        checklist.append("Keep the rejected scope with blockers if it remains useful review context.")
    return checklist


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


def _item_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    lines: list[str] = []
    for item in items:
        skills = ", ".join(item.get("candidate_p3_skill_ids") or []) or "none"
        blockers = ", ".join(item.get("proposed_blockers") or []) or "none"
        lines.append(
            "- "
            f"`{item.get('queue_id')}` | `{item.get('proposed_route_status')}` | "
            f"action `{item.get('recommended_review_action')}` | skills `{skills}` | blockers `{blockers}`"
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
