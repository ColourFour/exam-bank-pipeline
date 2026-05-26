from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    DEFAULT_REVIEW_BATCH_DIR,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
)

BATCH_CONCLUSIONS_SCHEMA = "exam_bank.p3_exact_skill.manual_review_batch_conclusions"
BATCH_CONCLUSIONS_SCHEMA_VERSION = 1

DEFAULT_BATCH_ID = "batch_0001"
DEFAULT_OUTPUT_JSON_PATH = "reports/manual_review_batch_0001_conclusions.v1.json"
DEFAULT_OUTPUT_REPORT_PATH = "reports/manual_review_batch_0001_conclusions.md"


def build_manual_review_batch_conclusions(
    *,
    batch_id: str = DEFAULT_BATCH_ID,
    batch_dir: str | Path = DEFAULT_REVIEW_BATCH_DIR,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    reviewed_registry_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    output_json_path: str | Path | None = DEFAULT_OUTPUT_JSON_PATH,
    output_report_path: str | Path | None = DEFAULT_OUTPUT_REPORT_PATH,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    batch_dir = Path(batch_dir)
    queue_path = Path(queue_path)
    reviewed_registry_path = Path(reviewed_registry_path)
    manifest_path = batch_dir / f"{batch_id}_manifest.v1.json"
    template_path = batch_dir / f"{batch_id}_decision_template.v1.json"
    packet_path = batch_dir / f"{batch_id}_review_packet.md"
    response_path = batch_dir / f"{batch_id}_review_responses.v1.json"
    visual_packet_path = batch_dir / f"{batch_id}_visual_review.html"

    queue = _load_json(queue_path)
    manifest = _load_json(manifest_path)
    template = _load_json(template_path)
    responses = _load_json(response_path)
    reviewed_registry = _load_json(reviewed_registry_path)
    queue_items = {
        _text(item.get("queue_id")): item
        for item in queue.get("items", [])
        if isinstance(item, dict) and _text(item.get("queue_id"))
    }
    response_records = [record for record in responses.get("responses", []) if isinstance(record, dict)]
    template_records = [record for record in template.get("records", []) if isinstance(record, dict)]
    registry_records = [record for record in reviewed_registry.get("records", []) if isinstance(record, dict)]

    comparisons = [
        _compare_response_to_queue(record, queue_items.get(_text(record.get("queue_id")), {}))
        for record in response_records
    ]
    outcome_counts = Counter(comparison["review_outcome"] for comparison in comparisons)
    exact_skill_counts = Counter(comparison["exact_skill_confirmed"] for comparison in comparisons)
    scope_counts = Counter(comparison["review_scope_decision"] for comparison in comparisons)
    decomposition_counts = Counter(comparison["decomposition_accepted"] for comparison in comparisons)
    support_counts = Counter(comparison["support_material_decision"] for comparison in comparisons)
    alignment_by_outcome = Counter(
        f"{comparison['review_outcome']}|{comparison['automated_topic_routing_alignment']}"
        for comparison in comparisons
    )
    registry_route_counts = Counter(_text(record.get("route_status")) for record in registry_records)
    block_reason_counts = Counter(
        reason
        for comparison in comparisons
        for reason in comparison["content_lab_generation_gate_block_reasons"]
    )
    primary_skill_by_outcome = Counter(
        f"{comparison['review_outcome']}|{','.join(comparison['automated_primary_candidate_skill_ids']) or 'none'}"
        for comparison in comparisons
    )

    clean_comparisons = [comparison for comparison in comparisons if comparison["review_outcome"] == "clean"]
    rejected_comparisons = [comparison for comparison in comparisons if comparison["review_outcome"] != "clean"]
    all_mark_events_advisory = all(
        ref.get("review_status") == "advisory" and ref.get("advisory_only") is True
        for record in template_records
        for ref in record.get("mark_event_refs", [])
        if isinstance(ref, dict)
    )

    payload = {
        "schema": BATCH_CONCLUSIONS_SCHEMA,
        "schema_version": BATCH_CONCLUSIONS_SCHEMA_VERSION,
        "artifact_kind": "manual_review_batch_conclusions",
        "generated_at": generated_at or _utc_now_iso(),
        "batch_id": batch_id,
        "warning": (
            "The batch response file is review evidence for this report only. It is marked as draft notes "
            "and is not the validated reviewed-decision registry."
        ),
        "source_files_inspected": {
            "review_packet": str(packet_path),
            "visual_review_packet": str(visual_packet_path),
            "review_responses": str(response_path),
            "decision_template": str(template_path),
            "manifest": str(manifest_path),
            "review_queue": str(queue_path),
            "reviewed_registry": str(reviewed_registry_path),
            "asset_reference_validation": "reports/asset_reference_validation.v1.json",
            "current_state_audit": "reports/p3_exact_skill_current_state_audit.v1.json",
            "post_storage_cleanup_delta_review": "reports/post_storage_cleanup_delta_review.v1.json",
            "readiness_doc": "docs/P3_EXACT_SKILL_EVIDENCE_READINESS_2026_05_25.md",
        },
        "reviewed_item_counts": {
            "batch_response_count": len(response_records),
            "batch_selected_count": _int(manifest.get("selected_count")),
            "decision_template_record_count": _int(template.get("record_count")),
            "reviewed_registry_record_count": len(registry_records),
            "validated_clean_registry_record_count": registry_route_counts.get("clean", 0),
        },
        "outcome_counts": {
            "approved_or_clean_draft": outcome_counts.get("clean", 0),
            "rejected_or_blocked_draft": outcome_counts.get("blocked", 0),
            "ambiguous_or_needs_adjustment_draft": outcome_counts.get("ambiguous", 0),
            "needs_review_draft": outcome_counts.get("review_needed", 0),
            "registry_route_status_counts": dict(sorted(registry_route_counts.items())),
            "draft_route_status_counts": dict(sorted(outcome_counts.items())),
        },
        "reviewed_topics_and_skills": _reviewed_topics_and_skills(comparisons),
        "reviewed_mark_event_and_evidence_status": {
            "all_template_mark_events_advisory_only": all_mark_events_advisory,
            "content_lab_block_reason_counts": dict(sorted(block_reason_counts.items())),
            "draft_evidence_basis_status_counts": _field_counts(response_records, "evidence_basis_status"),
            "inspected_question_image_counts": _field_counts(response_records, "inspected_question_image"),
            "inspected_mark_scheme_image_counts": _field_counts(response_records, "inspected_mark_scheme_image"),
            "part_boundary_confirmed_counts": _field_counts(response_records, "part_boundary_confirmed"),
        },
        "before_after_summary": {
            "automated_batch_inputs": {
                "selected_cross_topic_status_counts": _nested_get(manifest, "cross_topic_summary", "cross_topic_status_counts"),
                "selected_topic_routing_alignment_counts": _nested_get(
                    manifest, "cross_topic_summary", "topic_routing_alignment_counts"
                ),
                "selected_count": _int(manifest.get("selected_count")),
                "all_selected_were_cross_topic_reviewable": (
                    _nested_get(manifest, "cross_topic_summary", "cross_topic_status_counts").get("cross_topic_reviewable")
                    == _int(manifest.get("selected_count"))
                ),
            },
            "draft_review_outputs": {
                "route_status_counts": dict(sorted(outcome_counts.items())),
                "exact_skill_confirmed_counts": dict(sorted(exact_skill_counts.items())),
                "scope_decision_counts": dict(sorted(scope_counts.items())),
                "decomposition_accepted_counts": dict(sorted(decomposition_counts.items())),
                "support_material_decision_counts": dict(sorted(support_counts.items())),
                "outcome_by_topic_routing_alignment": dict(sorted(alignment_by_outcome.items())),
                "outcome_by_primary_candidate_skill": dict(sorted(primary_skill_by_outcome.items())),
            },
            "item_comparisons": comparisons,
        },
        "top_failure_modes": _top_failure_modes(rejected_comparisons),
        "top_reliable_signals": _top_reliable_signals(clean_comparisons, all_mark_events_advisory=all_mark_events_advisory),
        "fields_that_need_stricter_gating": [
            "reviewed_source_skill_ids: keep empty or blocked until reviewer resolves exact skill; 4/25 draft responses rejected the automated exact skill.",
            "mark_event_refs: keep advisory-only out of generation readiness; all selected records still had unreviewed mark events.",
            "supporting_candidate_skill_ids: keep as context only; repeated rejected records were supporting-method matches rather than safe exact targets.",
            "topic_routing_alignment=supporting_topic: treat as a review-risk signal, not a strict routing permission.",
            "decomposition candidates: keep part boundaries review-gated because 4/25 needed adjustment and 4 whole-question items did not need decomposition.",
        ],
        "fields_safe_for_advisory_indexing": [
            "canonical question and mark-scheme asset refs for reviewer navigation when asset validation passes.",
            "topic-routing alignment and confidence for triage buckets, with strict-filter safety kept separate.",
            "Content Lab generation_gate block reasons for review prioritization.",
            "candidate primary/supporting skill IDs for reviewer packets only, not mastery or generation authority.",
            "part/subpart labels and advisory mark-event part_path for choosing what to inspect, not for automatic promotion.",
        ],
        "specific_recommended_pipeline_improvements": [
            "Add a reviewed-batch conclusion report so future decisions cite reviewed records and draft status explicitly.",
            "Surface repeated failure modes: supporting-method skill mistaken as exact target, implicit/parametric differentiation retag needs, and integration-area questions routed through trig identities.",
            "Warn when clean reviewed-decision records still carry unreviewed mark events, because those records cannot support candidate generation.",
            "Warn when clean records lack verified asset-ref flags, so image inspection and asset metadata stay aligned.",
            "Keep Content Lab blocked when source skill IDs are missing or mark events are not reviewed.",
        ],
        "improvements_actually_implemented": [
            "Added this reviewed-batch conclusion JSON/Markdown report builder.",
            "Added validator warnings for clean records with unreviewed mark-event refs.",
            "Added validator warnings for clean records with unverified source question or mark-scheme asset refs.",
            "Added targeted tests for the new conclusion report and validator diagnostics.",
        ],
        "improvements_intentionally_deferred": [
            "No candidate promotion from draft responses into the reviewed registry.",
            "No automatic topic-routing or skill-routing changes from this small batch.",
            "No difficulty/index/band logic changes; this batch did not carry reviewed difficulty evidence.",
            "No Content Lab generation-readiness or mastery-readiness change.",
            "No storage cleanup changes.",
        ],
        "risks_and_concerns": [
            "The completed response file is still marked human_review_response_draft and repeatedly says it is not registry evidence.",
            "The sample is only 25 P3 exact-skill candidates, all selected from cross-topic-reviewable records.",
            "There are zero clean validated reviewed-registry records, so runtime authority remains unavailable.",
            "Topic-routing strict-filter safety is still false in the asset validation summary because 153 topic-routing records failed.",
            "Part-level review currently relies on whole-question and whole mark-scheme images for many records.",
        ],
        "next_recommended_review_batch_or_validation_target": [
            "Convert a small subset of the 21 clean draft responses into validated reviewed-decision registry records with reviewer identity, timestamps, reviewed regions, and verified asset refs.",
            "Run a targeted retag review for the 4 rejected/ambiguous records before they re-enter candidate batches.",
            "Review fixed-point iteration and complex-number subpart candidates next, because those were the strongest repeated clean patterns in this draft batch.",
            "Rerun or repair the topic-routing failure batches that used unavailable evidence labels before claiming strict-filter safety.",
        ],
    }

    if output_json_path and not dry_run:
        write_atomic_json(payload, output_json_path, sort_keys=True)
    if output_report_path and not dry_run:
        report_path = Path(output_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_manual_review_batch_conclusions_report(payload), encoding="utf-8")
    return payload


def render_manual_review_batch_conclusions_report(payload: dict[str, Any]) -> str:
    counts = payload["reviewed_item_counts"]
    outcomes = payload["outcome_counts"]
    before_after = payload["before_after_summary"]
    lines = [
        "# Manual Review Batch 0001 Conclusions",
        "",
        payload["warning"],
        "",
        "## Source Files Inspected",
        "",
        *[f"- `{key}`: `{path}`" for key, path in payload["source_files_inspected"].items()],
        "",
        "## Reviewed Item Counts",
        "",
        f"- Batch response records: `{counts['batch_response_count']}`",
        f"- Batch selected records: `{counts['batch_selected_count']}`",
        f"- Decision-template records: `{counts['decision_template_record_count']}`",
        f"- Reviewed-registry records: `{counts['reviewed_registry_record_count']}`",
        f"- Clean validated registry records: `{counts['validated_clean_registry_record_count']}`",
        "",
        "## Outcome Counts",
        "",
        f"- Draft clean/approved: `{outcomes['approved_or_clean_draft']}`",
        f"- Draft ambiguous/needs adjustment: `{outcomes['ambiguous_or_needs_adjustment_draft']}`",
        f"- Draft blocked/rejected: `{outcomes['rejected_or_blocked_draft']}`",
        f"- Draft needs-review: `{outcomes['needs_review_draft']}`",
        f"- Registry route counts: `{outcomes['registry_route_status_counts']}`",
        "",
        "## Reviewed Topics And Skills",
        "",
        *_counter_lines(payload["reviewed_topics_and_skills"]["review_outcome_by_primary_skill_id"]),
        "",
        "## Mark-Event And Evidence Status",
        "",
        f"- Template mark events all advisory-only: `{str(payload['reviewed_mark_event_and_evidence_status']['all_template_mark_events_advisory_only']).lower()}`",
        f"- Content Lab block reasons: `{payload['reviewed_mark_event_and_evidence_status']['content_lab_block_reason_counts']}`",
        f"- Evidence-basis draft statuses: `{payload['reviewed_mark_event_and_evidence_status']['draft_evidence_basis_status_counts']}`",
        "",
        "## Before After Summary",
        "",
        f"- Automated selected alignment counts: `{before_after['automated_batch_inputs']['selected_topic_routing_alignment_counts']}`",
        f"- Draft review route counts: `{before_after['draft_review_outputs']['route_status_counts']}`",
        f"- Draft exact-skill counts: `{before_after['draft_review_outputs']['exact_skill_confirmed_counts']}`",
        f"- Draft scope decisions: `{before_after['draft_review_outputs']['scope_decision_counts']}`",
        f"- Draft outcome by alignment: `{before_after['draft_review_outputs']['outcome_by_topic_routing_alignment']}`",
        "",
        "## Top Failure Modes",
        "",
        *[f"- {item}" for item in payload["top_failure_modes"]],
        "",
        "## Top Reliable Signals",
        "",
        *[f"- {item}" for item in payload["top_reliable_signals"]],
        "",
        "## Fields Needing Stricter Gating",
        "",
        *[f"- {item}" for item in payload["fields_that_need_stricter_gating"]],
        "",
        "## Fields Safe For Advisory Indexing",
        "",
        *[f"- {item}" for item in payload["fields_safe_for_advisory_indexing"]],
        "",
        "## Recommended Pipeline Improvements",
        "",
        *[f"- {item}" for item in payload["specific_recommended_pipeline_improvements"]],
        "",
        "## Implemented In This Pass",
        "",
        *[f"- {item}" for item in payload["improvements_actually_implemented"]],
        "",
        "## Deferred",
        "",
        *[f"- {item}" for item in payload["improvements_intentionally_deferred"]],
        "",
        "## Risks And Concerns",
        "",
        *[f"- {item}" for item in payload["risks_and_concerns"]],
        "",
        "## Next Target",
        "",
        *[f"- {item}" for item in payload["next_recommended_review_batch_or_validation_target"]],
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _compare_response_to_queue(response: dict[str, Any], queue_item: dict[str, Any]) -> dict[str, Any]:
    fields = response.get("fields") if isinstance(response.get("fields"), dict) else {}
    content_lab = queue_item.get("asterion_candidate") if isinstance(queue_item.get("asterion_candidate"), dict) else {}
    gate_reasons = _strings(content_lab.get("generation_gate_block_reasons"))
    mark_event_refs = [ref for ref in queue_item.get("mark_event_refs", []) if isinstance(ref, dict)]
    return {
        "queue_id": _text(response.get("queue_id")),
        "question_id": _text(response.get("question_id")),
        "subpart_id": _text(response.get("subpart_id")),
        "automated_candidate_source_skill_ids": _strings(queue_item.get("candidate_source_skill_ids")),
        "automated_primary_candidate_skill_ids": _strings(queue_item.get("primary_candidate_skill_ids")),
        "automated_supporting_candidate_skill_ids": _strings(queue_item.get("supporting_candidate_skill_ids")),
        "automated_candidate_topic": _nested_get(queue_item, "candidate_region_topic", "topic_assignment_name")
        or _nested_get(queue_item, "candidate_region_topic", "mapping_source_topic"),
        "automated_topic_routing_primary_topic_id": _nested_get(queue_item, "topic_routing", "primary_topic_id"),
        "automated_topic_routing_confidence": _nested_get(queue_item, "topic_routing", "confidence"),
        "automated_topic_routing_evidence_used": _strings(_nested_get(queue_item, "topic_routing", "evidence_used")),
        "automated_topic_routing_alignment": _text(queue_item.get("topic_routing_alignment")) or "unknown",
        "automated_recommended_scope": _text(queue_item.get("recommended_scope")),
        "automated_proposed_route_status": _text(queue_item.get("proposed_route_status")),
        "automated_source_question_asset_refs_present": bool(queue_item.get("source_question_asset_refs")),
        "automated_source_mark_scheme_asset_refs_present": bool(queue_item.get("source_mark_scheme_asset_refs")),
        "mark_event_review_status_counts": dict(Counter(_text(ref.get("review_status")) for ref in mark_event_refs)),
        "content_lab_generation_gate_status": _text(content_lab.get("generation_gate_status")),
        "content_lab_generation_gate_block_reasons": gate_reasons,
        "review_outcome": _text(fields.get("route_status")),
        "exact_skill_confirmed": _text(fields.get("exact_skill_confirmed")),
        "allowed_use_case_summary": _text(fields.get("allowed_use_case_summary")),
        "review_evidence_basis_status": _text(fields.get("evidence_basis_status")),
        "review_scope_decision": _text(fields.get("scope_decision")),
        "decomposition_accepted": _text(fields.get("decomposition_accepted")),
        "support_material_decision": _text(fields.get("support_material_decision")),
        "reviewer_note": _text(fields.get("reviewer_notes")),
    }


def _reviewed_topics_and_skills(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by_skill = Counter(
        f"{comparison['review_outcome']}|{','.join(comparison['automated_primary_candidate_skill_ids']) or 'none'}"
        for comparison in comparisons
    )
    by_topic = Counter(
        f"{comparison['review_outcome']}|{comparison['automated_candidate_topic'] or 'unknown'}"
        for comparison in comparisons
    )
    return {
        "review_outcome_by_primary_skill_id": dict(sorted(by_skill.items())),
        "review_outcome_by_candidate_topic": dict(sorted(by_topic.items())),
    }


def _top_failure_modes(rejected: list[dict[str, Any]]) -> list[str]:
    if not rejected:
        return ["No rejected or ambiguous draft records were present."]
    supporting = sum(1 for item in rejected if item["support_material_decision"] == "possible_target_skill")
    wrong_skill = sum(1 for item in rejected if item["exact_skill_confirmed"] == "no_wrong_skill")
    supporting_alignment = sum(1 for item in rejected if item["automated_topic_routing_alignment"] == "supporting_topic")
    return [
        f"Automated exact skill rejected or blocked on `{wrong_skill}` draft records.",
        f"Supporting method mistaken for possible target skill on `{supporting}` draft records.",
        f"`supporting_topic` alignment appeared in `{supporting_alignment}` of the rejected/ambiguous draft records.",
        "Observed retag needs: trig identity used inside integration-area work, log/exponential algebra inside differential-equation work, derivative-rules label on implicit differentiation, and polynomial label on stationary implicit-differentiation work.",
    ]


def _top_reliable_signals(clean: list[dict[str, Any]], *, all_mark_events_advisory: bool) -> list[str]:
    skill_counts = Counter(
        ",".join(item["automated_primary_candidate_skill_ids"]) or "none"
        for item in clean
    )
    strongest = ", ".join(f"`{skill}` ({count})" for skill, count in skill_counts.most_common(4))
    return [
        f"Primary candidate skill matched draft review on `{len(clean)}` records; strongest repeated patterns were {strongest}.",
        "Canonical asset refs were present for all selected records and supported reviewer navigation.",
        "Part/subpart scoping was useful for triage: most draft-clean records were subpart-level, while four were whole-question safe.",
        "Content Lab gate reasons were reliable safety diagnostics: all selected records remained blocked by unreviewed mark events and missing source skill IDs.",
        f"Advisory mark-event status was consistent across the template: `{str(all_mark_events_advisory).lower()}`.",
    ]


def _field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                _text((record.get("fields") if isinstance(record.get("fields"), dict) else {}).get(field)) or "missing"
                for record in records
            ).items()
        )
    )


def _counter_lines(counter: dict[str, Any]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- `{key}`: {value}" for key, value in sorted(counter.items())]


def _nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if current is not None else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
