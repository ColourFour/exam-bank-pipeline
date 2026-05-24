from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    DEFAULT_AMBIGUITY_AUDIT_JSON_PATH,
    DEFAULT_CURRENT_STATE_AUDIT_JSON_PATH,
    DEFAULT_CURRENT_STATE_AUDIT_REPORT_PATH,
    DEFAULT_REVIEWED_DECISIONS_PATH,
    DEFAULT_REVIEW_QUEUE_JSON_PATH,
)
from exam_bank.p3_exact_skill.part_decomposition import DEFAULT_PART_DECOMPOSITION_JSON_PATH

CURRENT_STATE_AUDIT_SCHEMA = "exam_bank.p3_exact_skill.current_state_audit"
CURRENT_STATE_AUDIT_SCHEMA_VERSION = 1

DEFAULT_ASTERION_SIDECAR_PATH = "output/asterion/exports/latest/p3_exact_skill_evidence_v1.json"

PREVIOUS_KNOWN_STATUS_COUNTS = {
    "total_queue_items": 749,
    "cross_topic_candidate": 568,
    "split_needed_candidate": 21,
    "conflict_candidate": 126,
    "fallback_only": 34,
    "ambiguous_candidate": 0,
    "clean_candidate": 0,
}

READINESS_VERDICTS = [
    "READY_FOR_ASTERION_CONTENT_LAB_REVIEW_DIAGNOSTICS",
    "NOT_READY_FOR_ASTERION_RUNTIME_MASTERY",
    "NOT_READY_FOR_GUARDIAN",
    "NOT_READY_FOR_CANDIDATE_GENERATION",
    "NOT_READY_FOR_SOURCE_BACKED_WORKED_EXAMPLES",
]


def build_p3_exact_skill_current_state_audit(
    *,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    ambiguity_audit_path: str | Path = DEFAULT_AMBIGUITY_AUDIT_JSON_PATH,
    part_decomposition_path: str | Path = DEFAULT_PART_DECOMPOSITION_JSON_PATH,
    reviewed_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    sidecar_path: str | Path = DEFAULT_ASTERION_SIDECAR_PATH,
    output_path: str | Path | None = DEFAULT_CURRENT_STATE_AUDIT_JSON_PATH,
    report_path: str | Path | None = DEFAULT_CURRENT_STATE_AUDIT_REPORT_PATH,
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    ambiguity_audit_path = Path(ambiguity_audit_path)
    part_decomposition_path = Path(part_decomposition_path)
    reviewed_path = Path(reviewed_path)
    sidecar_path = Path(sidecar_path)

    queue = _load_json(queue_path)
    ambiguity_audit = _load_json(ambiguity_audit_path)
    part_decomposition = _load_json(part_decomposition_path)
    reviewed = _load_json(reviewed_path)

    queue_summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    part_summary = part_decomposition.get("summary") if isinstance(part_decomposition.get("summary"), dict) else {}
    ambiguity_summary = ambiguity_audit.get("summary") if isinstance(ambiguity_audit.get("summary"), dict) else {}
    reviewed_records = [record for record in reviewed.get("records", []) if isinstance(record, dict)]
    reviewed_status_counts = dict(Counter(_text(record.get("route_status")) for record in reviewed_records))
    current_status_counts = queue_summary.get("status_counts") if isinstance(queue_summary.get("status_counts"), dict) else {}

    payload = {
        "schema": CURRENT_STATE_AUDIT_SCHEMA,
        "schema_version": CURRENT_STATE_AUDIT_SCHEMA_VERSION,
        "artifact_kind": "p3_exact_skill_current_state_audit",
        "generated_at": generated_at or _utc_now_iso(),
        "source_paths": {
            "queue": str(queue_path),
            "ambiguity_audit": str(ambiguity_audit_path),
            "part_decomposition": str(part_decomposition_path),
            "reviewed_registry": str(reviewed_path),
            "forbidden_runtime_sidecar": str(sidecar_path),
        },
        "readiness_verdicts": READINESS_VERDICTS,
        "warning": (
            "This audit is review-diagnostic metadata only. It must not be consumed as clean evidence, "
            "runtime mastery authority, Guardian authority, or candidate-generation authority."
        ),
        "summary": {
            "total_queue_items": _int(queue_summary.get("total_queue_items")),
            "status_counts": current_status_counts,
            "cross_topic_status_counts": queue_summary.get("cross_topic_status_counts") or {},
            "topic_routing_alignment_counts": queue_summary.get("topic_routing_alignment_counts") or {},
            "decomposition_status_counts": part_summary.get("decomposition_status_counts") or {},
            "reviewed_registry_route_status_counts": reviewed_status_counts,
            "clean_reviewed_registry_count": reviewed_status_counts.get("clean", 0),
            "thin_reviewed_registry_count": reviewed_status_counts.get("thin", 0),
            "ambiguous_reviewed_registry_count": reviewed_status_counts.get("ambiguous", 0),
            "blocked_reviewed_registry_count": reviewed_status_counts.get("blocked", 0),
            "deferred_reviewed_registry_count": reviewed_status_counts.get("deferred", 0),
            "review_needed_reviewed_registry_count": reviewed_status_counts.get("review_needed", 0),
            "already_reviewed_queue_scopes": _int(queue_summary.get("already_reviewed_records")),
            "advisory_only_mark_event_count": _int(queue_summary.get("advisory_only_mark_event_count")),
            "missing_question_asset_count": _int(queue_summary.get("missing_question_asset_count")),
            "missing_mark_scheme_asset_count": _int(queue_summary.get("missing_mark_scheme_asset_count")),
            "no_candidate_p3_skill_count": _int(queue_summary.get("no_candidate_skill_count")),
            "part_level_decomposition_candidate_count": _int(part_summary.get("part_level_candidate_count")),
            "subpart_level_decomposition_candidate_count": _int(part_summary.get("subpart_level_candidate_count")),
            "decomposition_candidate_count": _int(part_summary.get("decomposition_candidate_count")),
            "needs_manual_split_count": _int(part_summary.get("needs_manual_split_count")),
            "insufficient_part_signal_count": _int(part_summary.get("insufficient_part_signal_count")),
            "conflict_needs_review_count": _int(part_summary.get("conflict_needs_review_count")),
            "fallback_only_count": _int(queue_summary.get("fallback_only_candidates")),
            "ambiguous_audit_group_counts": ambiguity_summary.get("audit_group_counts") or {},
            "forbidden_runtime_sidecar_exists": sidecar_path.exists(),
        },
        "previous_known_state": PREVIOUS_KNOWN_STATUS_COUNTS,
        "comparison_to_previous_known_state": _comparison(current_status_counts, queue_summary),
        "asterion_recommendation": {
            "can_connect_now": True,
            "allowed_capacity": "Content Lab/admin/reviewer diagnostics only",
            "not_allowed_capacities": [
                "runtime mastery authority",
                "Guardian authority",
                "candidate-generation authority",
                "source-backed worked-example authority",
            ],
            "safe_to_consume_now": [
                "review status summaries",
                "candidate status",
                "cross-topic status",
                "split-needed/conflict/fallback flags",
                "proposed part-level decomposition",
                "recommended review action",
                "canonical asset refs for reviewer inspection",
                "blocker diagnostics",
            ],
            "must_not_consume_as_authority": [
                "suggested_source_skill_ids",
                "candidate skills as mastery evidence",
                "advisory mark events as source-backed example evidence",
                "OCR/native/advisory text labels",
                "browser review responses before registry validation",
                "cross-topic candidates as clean evidence",
                "decomposition candidates as reviewed part boundaries",
            ],
        },
    }

    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    if report_path and not dry_run:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(render_current_state_audit_report(payload), encoding="utf-8")
    return payload


def render_current_state_audit_report(audit: dict[str, Any]) -> str:
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    recommendation = audit.get("asterion_recommendation") if isinstance(audit.get("asterion_recommendation"), dict) else {}
    lines = [
        "# P3 Exact-Skill Current-State Audit",
        "",
        "This is a review-diagnostic audit. It is not reviewed evidence and must not be used as runtime authority.",
        "",
        "## Verdict",
        "",
        *[f"- `{verdict}`" for verdict in audit.get("readiness_verdicts", [])],
        "",
        "## Queue Counts",
        "",
        f"- Total queue items: {summary.get('total_queue_items', 0)}",
        *_counter_lines(summary.get("status_counts") or {}),
        "",
        "## Cross-Topic Counts",
        "",
        *_counter_lines(summary.get("cross_topic_status_counts") or {}),
        "",
        "## Decomposition Counts",
        "",
        *_counter_lines(summary.get("decomposition_status_counts") or {}),
        "",
        "## Reviewed Registry Counts",
        "",
        *_counter_lines(summary.get("reviewed_registry_route_status_counts") or {}),
        "",
        "## Safety Counts",
        "",
        f"- Already-reviewed queue scopes: {summary.get('already_reviewed_queue_scopes', 0)}",
        f"- Advisory-only mark-event count: {summary.get('advisory_only_mark_event_count', 0)}",
        f"- Missing question asset count: {summary.get('missing_question_asset_count', 0)}",
        f"- Missing mark-scheme asset count: {summary.get('missing_mark_scheme_asset_count', 0)}",
        f"- No candidate P3 skill count: {summary.get('no_candidate_p3_skill_count', 0)}",
        f"- Forbidden runtime sidecar exists: `{str(summary.get('forbidden_runtime_sidecar_exists', False)).lower()}`",
        "",
        "## Comparison To Previous Known State",
        "",
        *_comparison_lines(audit.get("comparison_to_previous_known_state") or {}),
        "",
        "## Asterion Recommendation",
        "",
        f"- Can connect now: `{str(recommendation.get('can_connect_now', False)).lower()}`",
        f"- Allowed capacity: {recommendation.get('allowed_capacity', '')}",
        "",
        "Safe to consume now:",
        "",
        *[f"- {item}" for item in recommendation.get("safe_to_consume_now", [])],
        "",
        "Must not consume as authority:",
        "",
        *[f"- {item}" for item in recommendation.get("must_not_consume_as_authority", [])],
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _comparison(status_counts: dict[str, Any], queue_summary: dict[str, Any]) -> dict[str, dict[str, int]]:
    current = {
        "total_queue_items": _int(queue_summary.get("total_queue_items")),
        "cross_topic_candidate": _int(status_counts.get("cross_topic_candidate")),
        "split_needed_candidate": _int(status_counts.get("split_needed_candidate")),
        "conflict_candidate": _int(status_counts.get("conflict_candidate")),
        "fallback_only": _int(status_counts.get("fallback_only")),
        "ambiguous_candidate": _int(status_counts.get("ambiguous_candidate")),
        "clean_candidate": _int(status_counts.get("clean_candidate")),
    }
    return {
        key: {
            "previous": previous,
            "current": current.get(key, 0),
            "delta": current.get(key, 0) - previous,
        }
        for key, previous in PREVIOUS_KNOWN_STATUS_COUNTS.items()
    }


def _comparison_lines(comparison: dict[str, Any]) -> list[str]:
    if not comparison:
        return ["- None"]
    return [
        f"- `{key}`: {values.get('previous', 0)} -> {values.get('current', 0)} "
        f"(delta {values.get('delta', 0):+d})"
        for key, values in comparison.items()
        if isinstance(values, dict)
    ]


def _counter_lines(counter: dict[str, Any]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- `{key}`: {value}" for key, value in sorted(counter.items())]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
