from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    ALLOWED_USE_CASE_KEYS,
    DEFAULT_REVIEW_BATCH_DIR,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
    REVIEW_BATCH_MANIFEST_SCHEMA,
    REVIEW_BATCH_TEMPLATE_SCHEMA,
)

REVIEW_BATCH_SCHEMA_VERSION = 1
DEFAULT_BATCH_STATUSES = ("clean_candidate", "cross_topic_candidate")
DEFAULT_EXCLUDED_BATCH_STATUSES = ("conflict_candidate", "fallback_only", "ambiguous_candidate", "blocked_candidate")
PURPOSE_STATUS_DEFAULTS = {
    "exact_skill_review": DEFAULT_BATCH_STATUSES,
    "split_review": ("split_needed_candidate",),
    "conflict_review": ("conflict_candidate",),
    "part_decomposition_review": ("cross_topic_candidate", "split_needed_candidate"),
}
ADVISORY_MARK_EVENT_WARNING = (
    "Mark-event refs are advisory-only review context. They are not authority for clean evidence, "
    "marking use, or candidate generation."
)
REVIEWER_CHECKLIST = [
    "Inspect the canonical question image.",
    "Inspect the canonical mark-scheme image.",
    "Confirm the exact P3 skill.",
    "Confirm whether whole-question or part-level scope is safe.",
    "Confirm whether P1 prerequisite/support-only material is involved.",
    "Confirm allowed use cases.",
    "Write evidence_basis in project wording.",
    "Choose route_status: clean, thin, ambiguous, blocked, deferred, review_needed, fallback_only.",
]


def build_p3_exact_skill_review_batch(
    *,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    reviewed_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    batch_id: str = "batch_0001",
    limit: int = 25,
    out_dir: str | Path = DEFAULT_REVIEW_BATCH_DIR,
    status: str | None = None,
    include_statuses: list[str] | tuple[str, ...] | None = None,
    exclude_statuses: list[str] | tuple[str, ...] | None = DEFAULT_EXCLUDED_BATCH_STATUSES,
    batch_purpose: str = "exact_skill_review",
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    reviewed_path = Path(reviewed_path)
    out_dir = Path(out_dir)
    generated_at = generated_at or _utc_now_iso()

    queue = _load_json(queue_path)
    reviewed = _load_json(reviewed_path)
    reviewed_scopes = _reviewed_scopes(reviewed)
    clean_counts = _clean_reviewed_counts(reviewed)
    items = [item for item in queue.get("items", []) if isinstance(item, dict)] if isinstance(queue, dict) else []

    selected, skipped = select_review_batch_items(
        items,
        reviewed_scopes=reviewed_scopes,
        clean_reviewed_counts=clean_counts,
        limit=limit,
        status=status,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        batch_purpose=batch_purpose,
    )
    packet = render_review_packet(
        selected,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        limit=limit,
        status=status,
        include_statuses=_effective_include_statuses(status, include_statuses, batch_purpose),
        exclude_statuses=_normalise_statuses(exclude_statuses),
        batch_purpose=batch_purpose,
    )
    template = build_decision_template(
        selected,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
    )
    manifest = build_batch_manifest(
        selected,
        skipped=skipped,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        limit=limit,
        status=status,
        include_statuses=_effective_include_statuses(status, include_statuses, batch_purpose),
        exclude_statuses=_normalise_statuses(exclude_statuses),
        batch_purpose=batch_purpose,
        clean_reviewed_counts=clean_counts,
    )

    paths = {
        "packet": out_dir / f"{batch_id}_review_packet.md",
        "template": out_dir / f"{batch_id}_decision_template.v1.json",
        "manifest": out_dir / f"{batch_id}_manifest.v1.json",
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths["packet"].write_text(packet, encoding="utf-8")
        write_atomic_json(template, paths["template"], sort_keys=True)
        write_atomic_json(manifest, paths["manifest"], sort_keys=True)

    return {
        "ok": True,
        "dry_run": dry_run,
        "batch_id": batch_id,
        "selected_count": len(selected),
        "selected_queue_ids": [item["queue_id"] for item in selected],
        "skipped_count_by_reason": dict(skipped),
        "paths": {key: str(path) for key, path in paths.items()},
        "packet": packet if dry_run else None,
        "decision_template": template if dry_run else None,
        "manifest": manifest,
    }


def select_review_batch_items(
    items: list[dict[str, Any]],
    *,
    reviewed_scopes: set[tuple[str, str]],
    clean_reviewed_counts: dict[str, int],
    limit: int,
    status: str | None = None,
    include_statuses: list[str] | tuple[str, ...] | None = None,
    exclude_statuses: list[str] | tuple[str, ...] | None = DEFAULT_EXCLUDED_BATCH_STATUSES,
    batch_purpose: str = "exact_skill_review",
) -> tuple[list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    included = _effective_include_statuses(status, include_statuses, batch_purpose)
    excluded = _normalise_statuses(exclude_statuses)
    for item in items:
        reasons = _skip_reasons(
            item,
            reviewed_scopes=reviewed_scopes,
            include_statuses=included,
            exclude_statuses=excluded,
            batch_purpose=batch_purpose,
        )
        if reasons:
            skipped.update(reasons)
            continue
        eligible.append(item)

    selected: list[dict[str, Any]] = []
    remaining = list(eligible)
    selected_questions: set[str] = set()
    selected_papers: set[str] = set()
    selected_sessions: set[str] = set()
    selected_skills: set[str] = set()

    while remaining and len(selected) < max(limit, 0):
        candidate_indices = [
            index for index, item in enumerate(remaining) if _text(item.get("question_id")) not in selected_questions
        ] or list(range(len(remaining)))
        best_index = max(
            candidate_indices,
            key=lambda index: _selection_score(
                remaining[index],
                clean_reviewed_counts=clean_reviewed_counts,
                selected_questions=selected_questions,
                selected_papers=selected_papers,
                selected_sessions=selected_sessions,
                selected_skills=selected_skills,
                batch_purpose=batch_purpose,
            ),
        )
        item = remaining.pop(best_index)
        selected.append(item)
        selected_questions.add(_text(item.get("question_id")))
        selected_papers.add(_text(item.get("paper")))
        selected_sessions.add(_text(item.get("session")))
        selected_skills.update(_p3_skill_ids(item))

    skipped["not_selected_limit"] += max(0, len(eligible) - len(selected))
    return selected, skipped


def build_decision_template(
    items: list[dict[str, Any]],
    *,
    batch_id: str,
    generated_at: str,
    queue_path: str | Path,
    reviewed_path: str | Path,
) -> dict[str, Any]:
    records = [_template_record(item, batch_id=batch_id, generated_at=generated_at) for item in items]
    return {
        "schema": REVIEW_BATCH_TEMPLATE_SCHEMA,
        "schema_version": REVIEW_BATCH_SCHEMA_VERSION,
        "artifact_kind": "human_editable_review_batch_template",
        "generated_at": generated_at,
        "batch_id": batch_id,
        "source_queue_path": str(queue_path),
        "reviewed_registry_path": str(reviewed_path),
        "warning": (
            "This is not the reviewed-decision registry and must not be consumed as clean evidence. "
            "Approved records must be manually converted into data/review/p3_exact_skill_reviewed_decisions.v1.json."
        ),
        "record_count": len(records),
        "records": records,
    }


def build_batch_manifest(
    items: list[dict[str, Any]],
    *,
    skipped: Counter[str],
    batch_id: str,
    generated_at: str,
    queue_path: str | Path,
    reviewed_path: str | Path,
    limit: int,
    status: str,
    include_statuses: list[str],
    exclude_statuses: list[str],
    batch_purpose: str,
    clean_reviewed_counts: dict[str, int],
) -> dict[str, Any]:
    selected_skills = sorted({skill_id for item in items for skill_id in _p3_skill_ids(item)})
    sparse_selected_skills = [skill_id for skill_id in selected_skills if clean_reviewed_counts.get(skill_id, 0) == 0]
    cross_topic_status_counts = Counter(_text(item.get("cross_topic_status")) or "unknown" for item in items)
    topic_routing_alignment_counts = Counter(_text(item.get("topic_routing_alignment")) or "unknown" for item in items)
    return {
        "schema": REVIEW_BATCH_MANIFEST_SCHEMA,
        "schema_version": REVIEW_BATCH_SCHEMA_VERSION,
        "artifact_kind": "review_batch_manifest",
        "generated_at": generated_at,
        "batch_id": batch_id,
        "source_queue_path": str(queue_path),
        "reviewed_registry_path": str(reviewed_path),
        "selection_limit": limit,
        "selection_filters": {
            "proposed_route_status": status,
            "include_statuses": include_statuses,
            "exclude_statuses": exclude_statuses,
            "batch_purpose": batch_purpose,
            "exclude_already_reviewed": True,
            "require_p3_candidate_skill": True,
            "require_question_asset_refs": True,
            "require_mark_scheme_asset_refs": True,
        },
        "selected_count": len(items),
        "skipped_count_by_reason": dict(skipped),
        "skip_reason_counts_are_not_mutually_exclusive": True,
        "selected_queue_ids": [item["queue_id"] for item in items],
        "selected_question_ids": sorted({_text(item.get("question_id")) for item in items if _text(item.get("question_id"))}),
        "cross_topic_summary": {
            "cross_topic_status_counts": dict(cross_topic_status_counts),
            "topic_routing_alignment_counts": dict(topic_routing_alignment_counts),
            "selected_items": [
                {
                    "queue_id": item["queue_id"],
                    "cross_topic_status": _text(item.get("cross_topic_status")) or "unknown",
                    "topic_routing_alignment": _text(item.get("topic_routing_alignment")) or "unknown",
                    "recommended_scope": _text(item.get("recommended_scope")) or "reviewer_decide",
                }
                for item in items
            ],
        },
        "skill_coverage_delta_estimate": {
            "selected_unique_p3_skill_count": len(selected_skills),
            "selected_sparse_or_zero_clean_skill_count": len(sparse_selected_skills),
            "selected_sparse_or_zero_clean_skill_ids": sparse_selected_skills,
        },
        "warning": (
            "This batch is a review packet only. It is not reviewed evidence, does not promote clean records, "
            "and is not the final Asterion p3_exact_skill_evidence_v1.json sidecar."
        ),
    }


def render_review_packet(
    items: list[dict[str, Any]],
    *,
    batch_id: str,
    generated_at: str,
    queue_path: str | Path,
    reviewed_path: str | Path,
    limit: int,
    status: str,
    include_statuses: list[str],
    exclude_statuses: list[str],
    batch_purpose: str,
) -> str:
    lines = [
        f"# P3 Exact-Skill Review Packet: {batch_id}",
        "",
        "This packet is for human review only. It does not assert clean evidence, does not update the reviewed-decision registry, and does not create the Asterion sidecar.",
        "",
        "## Batch Metadata",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Source queue: `{queue_path}`",
        f"- Reviewed registry checked for exclusions: `{reviewed_path}`",
        f"- Selection status: `{status or 'multiple'}`",
        f"- Included statuses: `{', '.join(include_statuses) or 'none'}`",
        f"- Excluded statuses: `{', '.join(exclude_statuses) or 'none'}`",
        f"- Batch purpose: `{batch_purpose}`",
        f"- Selection limit: `{limit}`",
        f"- Selected items: `{len(items)}`",
        "",
        "## Reviewer Checklist",
        "",
        *[f"- {item}" for item in REVIEWER_CHECKLIST],
        "",
        f"> {ADVISORY_MARK_EVENT_WARNING}",
        "",
        "## Review Items",
        "",
    ]
    if not items:
        lines.append("No items selected.")
        return "\n".join(lines).rstrip() + "\n"
    for index, item in enumerate(items, start=1):
        lines.extend(_packet_item_lines(index, item))
    return "\n".join(lines).rstrip() + "\n"


def _skip_reasons(
    item: dict[str, Any],
    *,
    reviewed_scopes: set[tuple[str, str]],
    include_statuses: list[str],
    exclude_statuses: list[str],
    batch_purpose: str,
) -> list[str]:
    reasons: list[str] = []
    item_status = _text(item.get("proposed_route_status"))
    if item_status not in include_statuses:
        reasons.append("status_filter")
    if item_status in exclude_statuses:
        reasons.append("excluded_status")
    if batch_purpose == "part_decomposition_review" and not item.get("proposed_part_level_candidates"):
        reasons.append("no_part_decomposition_candidates")
    scope = (_text(item.get("question_id")), _text(item.get("subpart_id")) or f"{_text(item.get('question_id'))}_whole")
    if item.get("reviewed_decision_status") == "already_reviewed" or scope in reviewed_scopes:
        reasons.append("already_reviewed")
    if not _p3_skill_ids(item):
        reasons.append("no_candidate_p3_skill")
    if not _has_existing_asset(item.get("source_question_asset_refs")):
        reasons.append("missing_question_asset")
    if not _has_existing_asset(item.get("source_mark_scheme_asset_refs")):
        reasons.append("missing_mark_scheme_asset")
    blockers = set(item.get("proposed_blockers") or [])
    if "p1_or_support_only_candidate_skill" in blockers:
        reasons.append("p1_or_support_only")
    if "mark_events_not_advisory_safe" in blockers:
        reasons.append("mark_events_not_advisory_safe")
    if "question_crop_not_high_confidence" in blockers or "mark_scheme_crop_not_high_confidence" in blockers:
        reasons.append("crop_blocker")
    return reasons


def _selection_score(
    item: dict[str, Any],
    *,
    clean_reviewed_counts: dict[str, int],
    selected_questions: set[str],
    selected_papers: set[str],
    selected_sessions: set[str],
    selected_skills: set[str],
    batch_purpose: str,
) -> tuple[int, str, str]:
    skills = _p3_skill_ids(item)
    blockers = set(item.get("proposed_blockers") or [])
    score = int(item.get("priority_score") or 0)
    score += {
        "clean_candidate": 120,
        "cross_topic_candidate": 90,
        "split_needed_candidate": 55,
        "weak_candidate": 25,
        "conflict_candidate": -80,
        "fallback_only": -100,
        "ambiguous_candidate": -120,
    }.get(_text(item.get("proposed_route_status")), 0)
    if batch_purpose == "part_decomposition_review":
        score += 120 if item.get("proposed_part_level_candidates") else -200
        score += 40 if _text(item.get("decomposition_status")) in {"part_level_candidate", "subpart_level_candidate", "already_part_scoped"} else 0
        score += sum(len(candidate.get("other_part_mark_event_refs") or []) for candidate in item.get("proposed_part_level_candidates") or [])
    score += 80 if any(clean_reviewed_counts.get(skill_id, 0) == 0 for skill_id in skills) else 0
    score += 30 if len(skills) == 1 else -25
    score += 15 if _text(_nested(item, "asterion_candidate", "candidate_id")) else 0
    score += 16 if _has_only_p3_source_skills(item) else -8
    score += len(item.get("mark_event_refs") or []) * 2
    score -= 20 if "mixed_or_ambiguous_topic" in blockers else 0
    score -= 12 * sum(1 for skill_id in skills if skill_id in selected_skills)
    score -= 30 if _text(item.get("question_id")) in selected_questions else 0
    score -= 8 if _text(item.get("paper")) in selected_papers else 0
    score -= 4 if _text(item.get("session")) in selected_sessions else 0
    return (score, _text(item.get("question_id")), _text(item.get("subpart_id")))


def _template_record(item: dict[str, Any], *, batch_id: str, generated_at: str) -> dict[str, Any]:
    return {
        "evidence_id": f"p3_exact_skill_review:{batch_id}:{_text(item.get('question_id'))}:{_text(item.get('subpart_id'))}",
        "queue_id": _text(item.get("queue_id")),
        "question_id": _text(item.get("question_id")),
        "part_id": _text(item.get("part_id")),
        "subpart_id": _text(item.get("subpart_id")),
        "paper": _text(item.get("paper")),
        "session": _text(item.get("session")),
        "variant": _text(item.get("variant")),
        "suggested_source_skill_ids": _p3_skill_ids(item),
        "suggested_primary_skill_ids": _unique_texts(item.get("primary_candidate_skill_ids") or _p3_skill_ids(item)),
        "suggested_supporting_skill_ids": _unique_texts(item.get("supporting_candidate_skill_ids") or []),
        "suggested_cross_topic_status": _text(item.get("cross_topic_status")) or "unknown",
        "suggested_recommended_scope": _text(item.get("recommended_scope")) or "reviewer_decide",
        "suggested_candidate_status": _text(item.get("proposed_route_status")) or "review_needed",
        "suggested_review_priority": _text(item.get("review_priority_group")) or "unknown",
        "suggested_scope_risk": _scope_risk(item),
        "suggested_ambiguity_reason": _text(item.get("ambiguity_reason")) or "unknown_ambiguity",
        "suggested_decomposition_status": _text(item.get("decomposition_status")) or "not_decomposable",
        "suggested_part_level_candidates": item.get("proposed_part_level_candidates") or [],
        "reviewed_source_skill_ids": [],
        "reviewed_region": None,
        "route_status": "review_needed",
        "source_question_asset_refs": item.get("source_question_asset_refs") or [],
        "source_mark_scheme_asset_refs": item.get("source_mark_scheme_asset_refs") or [],
        "mark_event_refs": _advisory_mark_event_refs(item.get("mark_event_refs") or []),
        "evidence_basis": "",
        "blockers": ["pending_human_review"],
        "allowed_use_cases": {key: False for key in sorted(ALLOWED_USE_CASE_KEYS)},
        "reviewer": {
            "reviewed_by": "",
            "reviewed_at": "",
            "review_status": "review_needed",
        },
        "provenance": {
            "batch_id": batch_id,
            "generated_at": generated_at,
            "source_queue_id": _text(item.get("queue_id")),
            "template_note": "Draft template only; manually convert approved decisions into the reviewed registry.",
        },
    }


def _packet_item_lines(index: int, item: dict[str, Any]) -> list[str]:
    skills = ", ".join(f"`{skill}`" for skill in _p3_skill_ids(item)) or "none"
    source_skills = ", ".join(f"`{skill}`" for skill in item.get("candidate_source_skill_ids") or []) or "none"
    blockers = ", ".join(f"`{blocker}`" for blocker in item.get("proposed_blockers") or []) or "none"
    reconciliation = ", ".join(f"`{flag}`" for flag in item.get("reconciliation_flags") or []) or "none"
    supporting_skills = ", ".join(f"`{skill}`" for skill in item.get("supporting_candidate_skill_ids") or []) or "none"
    cross_topic_notes = "; ".join(_text(note) for note in item.get("cross_topic_notes") or [] if _text(note)) or "none"
    lines = [
        f"### {index}. `{item.get('question_id')}` / `{item.get('subpart_id')}`",
        "",
        f"- Queue ID: `{item.get('queue_id')}`",
        f"- Question ID: `{item.get('question_id')}`",
        f"- Part/subpart: `{item.get('part_id')}` / `{item.get('subpart_id')}`",
        f"- Paper/session/variant: `{item.get('paper')}` / `{item.get('session')}` / `{item.get('variant')}`",
        f"- Candidate P3 skill IDs: {skills}",
        f"- Suggested candidate status: `{item.get('proposed_route_status') or 'unknown'}`",
        f"- Suggested review priority: `{item.get('review_priority_group') or 'unknown'}`",
        f"- Suggested ambiguity reason: `{item.get('ambiguity_reason') or 'unknown_ambiguity'}`",
        f"- Decomposition status: `{item.get('decomposition_status') or 'not_decomposable'}`",
        f"- Candidate source skill IDs, including prerequisite/support context: {source_skills}",
        f"- Primary candidate skill IDs: {', '.join(f'`{skill}`' for skill in item.get('primary_candidate_skill_ids') or []) or 'none'}",
        f"- Supporting candidate skill IDs: {supporting_skills}",
        f"- Candidate region/topic: `{json.dumps(item.get('candidate_region_topic') or {}, sort_keys=True)}`",
        f"- Topic-routing context: `{json.dumps(item.get('topic_routing') or {}, sort_keys=True)}`",
        f"- Cross-topic status: `{item.get('cross_topic_status') or 'unknown'}`",
        f"- Topic-routing topic IDs: `{json.dumps(item.get('topic_routing_topic_ids') or [])}`",
        f"- Topic-routing alignment: `{item.get('topic_routing_alignment') or 'unknown'}`",
        f"- Recommended scope: `{item.get('recommended_scope') or 'reviewer_decide'}`",
        f"- Cross-topic notes: {cross_topic_notes}",
        "Part-level decomposition candidates:",
        *_json_bullets(item.get("proposed_part_level_candidates") or []),
        f"- Content Lab blocker context: `{json.dumps(item.get('asterion_candidate') or {}, sort_keys=True)}`",
        f"- Proposed blockers: {blockers}",
        f"- Reconciliation flags: {reconciliation}",
        f"- Recommended review action: `{item.get('recommended_review_action')}`",
        "",
        "Question asset refs:",
        *_json_bullets(item.get("source_question_asset_refs") or []),
        "",
        "Mark-scheme asset refs:",
        *_json_bullets(item.get("source_mark_scheme_asset_refs") or []),
        "",
        "Advisory-only mark-event refs:",
        *_json_bullets(_advisory_mark_event_refs(item.get("mark_event_refs") or [])),
        "",
        "Reviewer checklist:",
        *[f"- [ ] {check}" for check in REVIEWER_CHECKLIST],
        "",
        "Cross-topic reviewer checklist:",
        *[f"- [ ] {check}" for check in item.get("reviewer_cross_topic_checklist") or []],
        "",
    ]
    return lines


def _effective_include_statuses(
    status: str | None,
    include_statuses: list[str] | tuple[str, ...] | None,
    batch_purpose: str,
) -> list[str]:
    if include_statuses:
        return _normalise_statuses(include_statuses)
    if status:
        return _normalise_statuses([status])
    return list(PURPOSE_STATUS_DEFAULTS.get(batch_purpose, DEFAULT_BATCH_STATUSES))


def _normalise_statuses(statuses: list[str] | tuple[str, ...] | None) -> list[str]:
    result: list[str] = []
    for status in statuses or []:
        for part in _text(status).split(","):
            text = _text(part)
            if text and text not in result:
                result.append(text)
    return result


def _scope_risk(item: dict[str, Any]) -> str:
    status = _text(item.get("proposed_route_status"))
    if status == "split_needed_candidate":
        return "scope_split_likely"
    if _text(item.get("recommended_scope")) in {"part_level", "subpart_level"}:
        return "part_or_subpart_scope_review"
    if status == "conflict_candidate":
        return "known_conflict"
    return "reviewer_decide"


def _reviewed_scopes(payload: dict[str, Any]) -> set[tuple[str, str]]:
    scopes: set[tuple[str, str]] = set()
    for record in payload.get("records", []) if isinstance(payload.get("records"), list) else []:
        if not isinstance(record, dict):
            continue
        question_id = _text(record.get("question_id"))
        subpart_id = _text(record.get("subpart_id")) or f"{question_id}_whole"
        if question_id:
            scopes.add((question_id, subpart_id))
    return scopes


def _clean_reviewed_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in payload.get("records", []) if isinstance(payload.get("records"), list) else []:
        if isinstance(record, dict) and record.get("route_status") == "clean":
            counts.update(_text(skill_id) for skill_id in record.get("reviewed_source_skill_ids") or [] if _text(skill_id))
    return dict(counts)


def _advisory_mark_event_refs(refs: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        updated = dict(ref)
        updated["review_status"] = "advisory"
        updated["advisory_only"] = True
        result.append(updated)
    return result


def _has_existing_asset(refs: Any) -> bool:
    return any(isinstance(ref, dict) and _text(ref.get("path")) and ref.get("exists", True) is not False for ref in refs or [])


def _p3_skill_ids(item: dict[str, Any]) -> list[str]:
    return [skill_id for skill_id in _unique_texts(item.get("candidate_p3_skill_ids") or []) if skill_id.startswith("9709_p3_")]


def _has_only_p3_source_skills(item: dict[str, Any]) -> bool:
    source_skill_ids = _unique_texts(item.get("candidate_source_skill_ids") or [])
    return bool(source_skill_ids) and all(skill_id.startswith("9709_p3_") for skill_id in source_skill_ids)


def _json_bullets(values: list[Any]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- `{json.dumps(value, sort_keys=True)}`" for value in values]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _text(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
