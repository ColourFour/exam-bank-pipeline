from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    DEFAULT_AMBIGUITY_AUDIT_JSON_PATH,
    DEFAULT_AMBIGUITY_AUDIT_REPORT_PATH,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
)

AUDIT_SCHEMA = "exam_bank.p3_exact_skill.ambiguity_audit"
AUDIT_SCHEMA_VERSION = 1


def build_p3_exact_skill_ambiguity_audit(
    *,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    output_path: str | Path | None = DEFAULT_AMBIGUITY_AUDIT_JSON_PATH,
    report_path: str | Path | None = DEFAULT_AMBIGUITY_AUDIT_REPORT_PATH,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    queue = _load_json(queue_path)
    items = [item for item in queue.get("items", []) if isinstance(item, dict)]
    audit_items = [item for item in items if _is_audit_candidate(item)]
    groups = _audit_groups(audit_items)
    payload = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_kind": "ambiguity_reduction_audit",
        "generated_at": generated_at or _utc_now_iso(),
        "source_queue_path": str(queue_path),
        "warning": (
            "This audit is triage only. It explains candidate review categories and must not be consumed "
            "as reviewed evidence."
        ),
        "summary": {
            "total_queue_items": len(items),
            "audited_item_count": len(audit_items),
            "ambiguous_candidate_count": sum(1 for item in items if item.get("proposed_route_status") == "ambiguous_candidate"),
            "status_counts": dict(Counter(_text(item.get("proposed_route_status")) for item in items)),
            "audit_group_counts": {name: group["count"] for name, group in groups.items()},
            "audit_group_counts_are_not_mutually_exclusive": True,
        },
        "groups": groups,
    }
    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    if report_path and not dry_run:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(render_ambiguity_audit_report(payload), encoding="utf-8")
    return payload


def render_ambiguity_audit_report(audit: dict[str, Any]) -> str:
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    groups = audit.get("groups") if isinstance(audit.get("groups"), dict) else {}
    lines = [
        "# P3 Exact-Skill Ambiguity Audit",
        "",
        "This audit is a triage artifact, not reviewed evidence. It explains where broad ambiguity moved after sharper candidate statuses were applied.",
        "",
        "## Summary",
        "",
        f"- Total queue items: {summary.get('total_queue_items', 0)}",
        f"- Audited item count: {summary.get('audited_item_count', 0)}",
        f"- Remaining ambiguous candidates: {summary.get('ambiguous_candidate_count', 0)}",
        f"- Audit group counts are not mutually exclusive: `{str(summary.get('audit_group_counts_are_not_mutually_exclusive', False)).lower()}`",
        "",
        "## Status Counts",
        "",
        *_counter_lines(summary.get("status_counts") or {}),
        "",
        "## Audit Groups",
        "",
    ]
    for name, group in groups.items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- Count: {group.get('count', 0)}",
                f"- Automatically reclassifiable safely: `{str(group.get('can_reclassify_safely', False)).lower()}`",
                f"- Recommended handling: {group.get('recommended_handling', '')}",
                f"- Representative queue IDs: {', '.join(f'`{queue_id}`' for queue_id in group.get('representative_queue_ids', [])) or 'none'}",
                f"- Candidate skills involved: {', '.join(f'`{skill}`' for skill in group.get('candidate_skills_involved', [])) or 'none'}",
                f"- Topic-routing topics involved: {', '.join(f'`{topic}`' for topic in group.get('topic_routing_topics_involved', [])) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _is_audit_candidate(item: dict[str, Any]) -> bool:
    return _text(item.get("proposed_route_status")) in {
        "cross_topic_candidate",
        "split_needed_candidate",
        "conflict_candidate",
        "weak_candidate",
        "ambiguous_candidate",
        "fallback_only",
        "blocked_candidate",
    }


def _audit_groups(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        for cause in _audit_causes(item):
            grouped[cause].append(item)
    ordered_names = [
        "cross_topic_reviewable",
        "cross_topic_split_needed",
        "topic_routing_candidate_mismatch",
        "de_vs_parametric_implicit_conflict",
        "multiple_candidate_skills",
        "broad_whole_question_scope",
        "advisory_only_mark_events",
        "weak_candidate_skill_context",
        "fallback_or_low_quality_context",
        "reviewed_registry_conflict",
        "unknown_ambiguity",
    ]
    result: dict[str, dict[str, Any]] = {}
    for name in ordered_names:
        group_items = grouped.get(name, [])
        if not group_items:
            continue
        result[name] = _group_summary(name, group_items)
    return result


def _audit_causes(item: dict[str, Any]) -> list[str]:
    causes: list[str] = []
    status = _text(item.get("proposed_route_status"))
    cross_topic_status = _text(item.get("cross_topic_status"))
    blockers = set(item.get("proposed_blockers") or [])
    alignment = _text(item.get("topic_routing_alignment"))
    if status == "cross_topic_candidate" or cross_topic_status == "cross_topic_reviewable":
        causes.append("cross_topic_reviewable")
    if status == "split_needed_candidate" or cross_topic_status == "cross_topic_split_needed":
        causes.append("cross_topic_split_needed")
    if alignment in {"supporting_topic", "cross_topic_possible", "conflicting"}:
        causes.append("topic_routing_candidate_mismatch")
    if blockers & {"possible_differential_equation_not_parametric_or_implicit", "weak_parametric_implicit_evidence_dydx_only"}:
        causes.append("de_vs_parametric_implicit_conflict")
    if len(item.get("candidate_p3_skill_ids") or []) > 1:
        causes.append("multiple_candidate_skills")
    if _text(item.get("subpart_label")) == "whole" and status == "split_needed_candidate":
        causes.append("broad_whole_question_scope")
    if "mark_events_advisory_only" in blockers:
        causes.append("advisory_only_mark_events")
    if status == "weak_candidate" or "weak_parametric_equation_evidence_missing_parameter" in blockers:
        causes.append("weak_candidate_skill_context")
    if blockers & {
        "question_crop_not_high_confidence",
        "mark_scheme_crop_not_high_confidence",
        "text_or_ocr_not_authoritative",
        "visual_dependency",
        "mark_events_not_advisory_safe",
    }:
        causes.append("fallback_or_low_quality_context")
    if item.get("reconciliation_flags"):
        causes.append("reviewed_registry_conflict")
    if status == "ambiguous_candidate" and not causes:
        causes.append("unknown_ambiguity")
    return _unique_texts(causes) or ["unknown_ambiguity"]


def _group_summary(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "representative_queue_ids": [_text(item.get("queue_id")) for item in items[:10]],
        "candidate_skills_involved": sorted(
            {
                _text(skill_id)
                for item in items
                for skill_id in item.get("candidate_p3_skill_ids") or []
                if _text(skill_id)
            }
        ),
        "topic_routing_topics_involved": sorted(
            {
                _text(topic_id)
                for item in items
                for topic_id in item.get("topic_routing_topic_ids") or []
                if _text(topic_id)
            }
        ),
        "recommended_handling": _recommended_handling(name),
        "can_reclassify_safely": name
        in {
            "cross_topic_reviewable",
            "cross_topic_split_needed",
            "de_vs_parametric_implicit_conflict",
            "weak_candidate_skill_context",
            "fallback_or_low_quality_context",
        },
    }


def _recommended_handling(name: str) -> str:
    return {
        "cross_topic_reviewable": "Keep reviewable; ask reviewer to confirm primary skill and supporting context.",
        "cross_topic_split_needed": "Route to part/subpart scope review before any clean decision.",
        "topic_routing_candidate_mismatch": "Show as review cue; reject only when method context is genuinely conflicting.",
        "de_vs_parametric_implicit_conflict": "Treat as known-risk conflict unless canonical images clearly resolve it.",
        "multiple_candidate_skills": "Prefer split review; avoid broad whole-question evidence.",
        "broad_whole_question_scope": "Require scope decision before approving.",
        "advisory_only_mark_events": "Use mark events only as context, not authority.",
        "weak_candidate_skill_context": "Deprioritize for clean batches; review only after stronger items or by targeted pass.",
        "fallback_or_low_quality_context": "Use visual inspection; do not rely on OCR/native text.",
        "reviewed_registry_conflict": "Reconcile against existing reviewed registry before changing any decision.",
        "unknown_ambiguity": "Keep as ambiguous until a human or richer metadata resolves it.",
    }.get(name, "Review manually.")


def _counter_lines(counter: dict[str, Any]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- `{key}`: {value}" for key, value in counter.items()]


def _unique_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
