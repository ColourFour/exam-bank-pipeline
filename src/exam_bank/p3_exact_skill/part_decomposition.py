from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import DEFAULT_REVIEW_QUEUE_JSON_PATH

PART_DECOMPOSITION_SCHEMA = "exam_bank.p3_exact_skill.part_decomposition"
PART_DECOMPOSITION_SCHEMA_VERSION = 1
DEFAULT_PART_DECOMPOSITION_JSON_PATH = "reports/p3_exact_skill_part_decomposition.v1.json"
DEFAULT_PART_DECOMPOSITION_REPORT_PATH = "reports/p3_exact_skill_part_decomposition.md"

DECOMPOSABLE_STATUSES = {"cross_topic_candidate", "split_needed_candidate"}
CONFLICT_STATUSES = {"conflict_candidate"}


def enrich_queue_items_with_part_decomposition(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question = _items_by_question(items)
    for item in items:
        status, candidates, summary, warning = decompose_queue_item(item, question_items=by_question.get(_text(item.get("question_id")), []))
        item["decomposition_status"] = status
        item["proposed_part_level_candidates"] = candidates
        item["part_signal_summary"] = summary
        item["part_scope_warning"] = warning
        if status in {"part_level_candidate", "subpart_level_candidate", "needs_manual_split"}:
            item["recommended_scope"] = "part_level"
    return items


def build_p3_exact_skill_part_decomposition(
    *,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    output_path: str | Path | None = DEFAULT_PART_DECOMPOSITION_JSON_PATH,
    report_path: str | Path | None = DEFAULT_PART_DECOMPOSITION_REPORT_PATH,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    queue = _load_json(queue_path)
    items = [item for item in queue.get("items", []) if isinstance(item, dict)]
    enriched = enrich_queue_items_with_part_decomposition([dict(item) for item in items])
    decomposition_items = [
        candidate
        for item in enriched
        for candidate in item.get("proposed_part_level_candidates") or []
    ]
    status_counts = Counter(_text(item.get("decomposition_status")) for item in enriched)
    summary = {
        "whole_question_cross_topic_count": sum(
            1
            for item in enriched
            if _is_whole_scope(item) and _text(item.get("proposed_route_status")) in DECOMPOSABLE_STATUSES
        ),
        "decomposition_candidate_count": len(decomposition_items),
        "part_level_candidate_count": sum(
            1 for item in decomposition_items if item.get("decomposition_status") == "part_level_candidate"
        ),
        "subpart_level_candidate_count": sum(
            1 for item in decomposition_items if item.get("decomposition_status") == "subpart_level_candidate"
        ),
        "needs_manual_split_count": status_counts.get("needs_manual_split", 0),
        "insufficient_part_signal_count": status_counts.get("insufficient_part_signal", 0),
        "already_part_scoped_count": status_counts.get("already_part_scoped", 0),
        "conflict_needs_review_count": status_counts.get("conflict_needs_review", 0),
        "not_decomposable_count": status_counts.get("not_decomposable", 0),
        "decomposition_status_counts": dict(status_counts),
    }
    payload = {
        "schema": PART_DECOMPOSITION_SCHEMA,
        "schema_version": PART_DECOMPOSITION_SCHEMA_VERSION,
        "artifact_kind": "part_level_decomposition_review_aid",
        "generated_at": generated_at or _utc_now_iso(),
        "source_queue_path": str(queue_path),
        "summary": summary,
        "diagnostics": _diagnostics(enriched),
        "items": decomposition_items,
        "warning": "This is a decomposition candidate report, not reviewed evidence.",
    }
    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    if report_path and not dry_run:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(render_part_decomposition_report(payload), encoding="utf-8")
    return payload


def decompose_queue_item(item: dict[str, Any], *, question_items: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any], str]:
    status = _text(item.get("proposed_route_status"))
    subpart_label = _text(item.get("subpart_label"))
    grouped_events = _mark_events_by_part(item.get("mark_event_refs") or [])
    sibling_parts = sorted({_text(sibling.get("subpart_label")) for sibling in question_items if not _is_whole_scope(sibling)})
    summary = {
        "queue_scope": subpart_label or "whole",
        "is_whole_question_scope": _is_whole_scope(item),
        "queue_part_labels_for_question": sibling_parts,
        "mark_event_part_paths": [".".join(path) for path in sorted(grouped_events)],
        "mark_event_part_count": len(grouped_events),
        "has_part_labeled_mark_events": bool(grouped_events),
        "has_existing_part_scoped_queue_items": bool(sibling_parts),
    }

    if status in CONFLICT_STATUSES or _text(item.get("cross_topic_status")) == "conflict_needs_review":
        return "conflict_needs_review", [], summary, "Known-risk conflict; do not decompose automatically."
    if not _is_whole_scope(item):
        candidate = _candidate_for_existing_part(item, grouped_events)
        summary["matching_mark_event_count"] = len(candidate["matching_mark_event_refs"])
        summary["other_part_mark_event_count"] = len(candidate["other_part_mark_event_refs"])
        return "already_part_scoped", [candidate], summary, _part_image_warning(item)
    if status not in DECOMPOSABLE_STATUSES:
        return "not_decomposable", [], summary, "Queue status is not a cross-topic or split-needed decomposition target."
    if not grouped_events:
        fallback = "needs_manual_split" if status == "split_needed_candidate" else "insufficient_part_signal"
        return fallback, [], summary, "No part-labeled mark events are available; reviewer must split manually."

    candidates = [
        _candidate_for_part(
            source_item=item,
            part_path=part_path,
            matching_refs=refs,
            all_grouped_refs=grouped_events,
            question_items=question_items,
        )
        for part_path, refs in sorted(grouped_events.items())
    ]
    if not candidates:
        return "insufficient_part_signal", [], summary, "Part labels exist but no candidate could be built."
    candidate_status = "subpart_level_candidate" if any(len(candidate["part_path"]) > 1 for candidate in candidates) else "part_level_candidate"
    return candidate_status, candidates, summary, _part_image_warning(item)


def render_part_decomposition_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    lines = [
        "# P3 Exact-Skill Part Decomposition",
        "",
        "This report proposes part/subpart review candidates only. It is not reviewed evidence and does not replace canonical image inspection.",
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in summary.items() if key != "decomposition_status_counts"],
        "",
        "## Decomposition Status Counts",
        "",
        *_counter_lines(summary.get("decomposition_status_counts") or {}),
        "",
        "## Diagnostic Notes",
        "",
        f"- Queue items: `{diagnostics.get('queue_item_count', 0)}`",
        f"- Already subpart-scoped queue items: `{diagnostics.get('already_subpart_scoped_queue_items', 0)}`",
        f"- Whole-question queue items: `{diagnostics.get('whole_question_queue_items', 0)}`",
        f"- Items with part-labeled mark events: `{diagnostics.get('items_with_part_labeled_mark_events', 0)}`",
        f"- Items with existing part-scoped sibling queue records: `{diagnostics.get('items_with_part_scoped_siblings', 0)}`",
        "- Part labels can be inferred from queue scopes and mark-event `part_path` values.",
        "- Part-level image crops are not created here; whole-question images are linked when no part crop exists.",
        "- Skill/topic suggestions come from existing queue records only and remain review context.",
        "",
        "## Representative Candidates",
        "",
    ]
    if not items:
        lines.append("- None")
    for item in items[:80]:
        lines.append(
            "- "
            f"`{item.get('decomposition_id')}` from `{item.get('source_queue_id')}` | "
            f"status `{item.get('decomposition_status')}` | part `{item.get('proposed_part_id')}` | "
            f"skills `{', '.join(item.get('candidate_source_skill_ids') or []) or 'none'}` | "
            f"confidence `{item.get('confidence')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def _candidate_for_existing_part(item: dict[str, Any], grouped_events: dict[tuple[str, ...], list[dict[str, Any]]]) -> dict[str, Any]:
    part = _text(item.get("subpart_label") or item.get("part_id"))
    part_path = (part,) if part and part != "whole" else tuple()
    matching = grouped_events.get(part_path, [])
    other = [ref for path, refs in grouped_events.items() if path != part_path for ref in refs]
    return _candidate_record(
        source_item=item,
        part_path=part_path,
        matching_refs=matching,
        other_refs=other,
        candidate_item=item,
        status="already_part_scoped",
        confidence="medium" if matching else "low",
        skill_mapping_signal=True,
    )


def _candidate_for_part(
    *,
    source_item: dict[str, Any],
    part_path: tuple[str, ...],
    matching_refs: list[dict[str, Any]],
    all_grouped_refs: dict[tuple[str, ...], list[dict[str, Any]]],
    question_items: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_item = _matching_part_item(question_items, part_path) or source_item
    other = [ref for path, refs in all_grouped_refs.items() if path != part_path for ref in refs]
    has_sibling_mapping = candidate_item is not source_item
    return _candidate_record(
        source_item=source_item,
        part_path=part_path,
        matching_refs=matching_refs,
        other_refs=other,
        candidate_item=candidate_item,
        status="subpart_level_candidate" if len(part_path) > 1 else "part_level_candidate",
        confidence="high" if has_sibling_mapping and matching_refs else "medium" if matching_refs else "low",
        skill_mapping_signal=has_sibling_mapping,
    )


def _candidate_record(
    *,
    source_item: dict[str, Any],
    part_path: tuple[str, ...],
    matching_refs: list[dict[str, Any]],
    other_refs: list[dict[str, Any]],
    candidate_item: dict[str, Any],
    status: str,
    confidence: str,
    skill_mapping_signal: bool,
) -> dict[str, Any]:
    question_id = _text(source_item.get("question_id"))
    part = part_path[0] if part_path else _text(source_item.get("part_id"))
    subpart = ".".join(part_path[1:]) if len(part_path) > 1 else None
    return {
        "decomposition_id": f"p3_part_decomp:v1:{question_id}:{'.'.join(part_path) if part_path else 'whole'}",
        "source_queue_id": _text(source_item.get("queue_id")),
        "question_id": question_id,
        "proposed_part_id": part or None,
        "proposed_subpart_id": subpart,
        "part_path": list(part_path),
        "candidate_source_skill_ids": _unique_texts(candidate_item.get("candidate_p3_skill_ids") or []),
        "candidate_topic_ids": _unique_texts(candidate_item.get("topic_routing_topic_ids") or []),
        "supporting_skill_ids": _unique_texts(candidate_item.get("supporting_candidate_skill_ids") or []),
        "source_question_asset_refs": source_item.get("source_question_asset_refs") or [],
        "source_mark_scheme_asset_refs": source_item.get("source_mark_scheme_asset_refs") or [],
        "matching_mark_event_refs": matching_refs,
        "other_part_mark_event_refs": other_refs,
        "evidence_signals": {
            "part_label_signal": bool(part_path),
            "mark_event_part_match": bool(matching_refs),
            "topic_assignment_signal": bool(candidate_item.get("topic_routing_topic_ids")),
            "skill_mapping_signal": skill_mapping_signal,
            "mark_scheme_method_signal": bool(matching_refs),
        },
        "decomposition_status": status,
        "confidence": confidence,
        "blockers": _candidate_blockers(source_item, matching_refs, skill_mapping_signal),
        "recommended_review_action": "review_part_scope_and_skill",
        "warning": "This is a decomposition candidate, not reviewed evidence.",
    }


def _candidate_blockers(source_item: dict[str, Any], matching_refs: list[dict[str, Any]], skill_mapping_signal: bool) -> list[str]:
    blockers = ["uses_whole_question_images_for_part_review"]
    if not matching_refs:
        blockers.append("no_matching_part_mark_events")
    if not skill_mapping_signal:
        blockers.append("no_existing_part_skill_mapping")
    if _text(source_item.get("proposed_route_status")) == "conflict_candidate":
        blockers.append("source_queue_conflict_candidate")
    return blockers


def _diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queue_item_count": len(items),
        "already_subpart_scoped_queue_items": sum(1 for item in items if not _is_whole_scope(item)),
        "whole_question_queue_items": sum(1 for item in items if _is_whole_scope(item)),
        "items_with_part_labeled_mark_events": sum(
            1 for item in items if _mark_events_by_part(item.get("mark_event_refs") or [])
        ),
        "items_with_part_scoped_siblings": sum(
            1
            for item in items
            if item.get("part_signal_summary", {}).get("has_existing_part_scoped_queue_items")
        ),
        "available_part_labels": dict(
            Counter(
                path[0]
                for item in items
                for path in _mark_events_by_part(item.get("mark_event_refs") or [])
                if path
            )
        ),
    }


def _part_image_warning(item: dict[str, Any]) -> str:
    return "Part-level crops are not generated; reviewer must use whole-question images and confirm the part boundary."


def _matching_part_item(question_items: list[dict[str, Any]], part_path: tuple[str, ...]) -> dict[str, Any] | None:
    if not part_path:
        return None
    for item in question_items:
        if _text(item.get("subpart_label")) == part_path[0] and not _is_whole_scope(item):
            return item
    return None


def _items_by_question(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        result[_text(item.get("question_id"))].append(item)
    return result


def _mark_events_by_part(refs: list[Any]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        part_path = tuple(_text(part) for part in ref.get("part_path") or [] if _text(part))
        if part_path:
            grouped[part_path].append(ref)
    return dict(grouped)


def _is_whole_scope(item: dict[str, Any]) -> bool:
    return _text(item.get("subpart_label") or item.get("part_id")) == "whole"


def _counter_lines(counter: dict[str, Any]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- `{key}`: {value}" for key, value in counter.items()]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _unique_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _text(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
