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
BATCH_0002_PURPOSE = "batch_0002_mixed_manual_review"
BATCH_0003_PURPOSE = "batch_0003_adversarial_mark_event_review"
BATCH_0002_SELECTION_CATEGORIES = {
    "reliable_pattern_confirmation",
    "known_failure_mode_probe",
    "deferred_batch_0001_clean",
    "seed_mark_event_alignment_probe",
}
BATCH_0003_SELECTION_CATEGORIES = {
    "prior_ambiguous_retag_probe",
    "prior_blocked_confirmation",
    "thin_adjacent_part_probe",
    "clean_control_mark_event_probe",
    "deferred_exact_skill_boundary_probe",
}
REVIEW_OUTCOME_CATEGORIES = {
    "clean_seed",
    "thin_control",
    "supporting_evidence_only",
    "exact_but_not_seed_quality",
    "cross_content_not_exact_skill_isolatable",
    "supporting_method_not_target_skill",
    "thin_or_adjacent_context",
    "blocked_wrong_or_unsafe_label",
    "review_needed",
}
SEED_REGISTRY_SUBPART_IDS = (
    "33summer23_q11_b",
    "31summer24_q04_b",
    "32summer23_q06_c",
    "32autumn23_q06_c",
    "33summer23_q06_b",
    "33summer23_q09_b",
    "32spring23_q05_b",
)
PURPOSE_STATUS_DEFAULTS = {
    "exact_skill_review": DEFAULT_BATCH_STATUSES,
    "split_review": ("split_needed_candidate",),
    "conflict_review": ("conflict_candidate",),
    "part_decomposition_review": ("cross_topic_candidate", "split_needed_candidate"),
    BATCH_0002_PURPOSE: ("cross_topic_candidate", "conflict_candidate", "split_needed_candidate", "fallback_only"),
    BATCH_0003_PURPOSE: ("cross_topic_candidate", "conflict_candidate", "split_needed_candidate", "fallback_only"),
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


def build_p3_exact_skill_batch_0002(
    *,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    reviewed_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    batch_0001_conclusions_path: str | Path = "reports/manual_review_batch_0001_conclusions.v1.json",
    seed_report_path: str | Path = "reports/p3_exact_skill_registry_seed_0001.v1.json",
    content_lab_path: str | Path = "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json",
    batch_id: str = "batch_0002",
    out_dir: str | Path = DEFAULT_REVIEW_BATCH_DIR,
    reports_dir: str | Path = "reports",
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    reviewed_path = Path(reviewed_path)
    conclusions_path = Path(batch_0001_conclusions_path)
    seed_report_path = Path(seed_report_path)
    content_lab_path = Path(content_lab_path)
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    generated_at = generated_at or _utc_now_iso()

    queue = _load_json(queue_path)
    reviewed = _load_json(reviewed_path)
    conclusions = _load_json(conclusions_path)
    seed_report = _load_json(seed_report_path)
    content_lab = _load_json(content_lab_path) if content_lab_path.exists() else {}
    items = [item for item in queue.get("items", []) if isinstance(item, dict)] if isinstance(queue, dict) else []

    selected, selection_notes = select_batch_0002_items(
        items,
        reviewed=reviewed,
        batch_0001_conclusions=conclusions,
        seed_report=seed_report,
    )
    template = build_decision_template(
        selected,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
    )
    skipped = Counter(selection_notes.get("skipped_count_by_reason", {}))
    manifest = build_batch_manifest(
        selected,
        skipped=skipped,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        limit=40,
        status=None,
        include_statuses=list(PURPOSE_STATUS_DEFAULTS[BATCH_0002_PURPOSE]),
        exclude_statuses=[],
        batch_purpose=BATCH_0002_PURPOSE,
        clean_reviewed_counts=_clean_reviewed_counts(reviewed),
    )
    manifest.update(_batch_0002_manifest_fields(selected, selection_notes))
    report_payload = build_batch_0002_plan_report(
        selected,
        manifest=manifest,
        reviewed=reviewed,
        content_lab=content_lab,
        generated_at=generated_at,
        source_paths={
            "queue": str(queue_path),
            "reviewed_registry": str(reviewed_path),
            "batch_0001_conclusions": str(conclusions_path),
            "seed_report": str(seed_report_path),
            "content_lab_candidates": str(content_lab_path),
        },
    )
    report_markdown = render_batch_0002_plan_report(report_payload)
    packet = render_review_packet(
        selected,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        limit=40,
        status=None,
        include_statuses=list(PURPOSE_STATUS_DEFAULTS[BATCH_0002_PURPOSE]),
        exclude_statuses=[],
        batch_purpose=BATCH_0002_PURPOSE,
    )

    validation_errors = validate_batch_0002_artifacts(manifest, template)
    if validation_errors:
        raise ValueError("Batch 0002 validation failed: " + "; ".join(validation_errors))

    paths = {
        "packet": out_dir / f"{batch_id}_review_packet.md",
        "template": out_dir / f"{batch_id}_decision_template.v1.json",
        "manifest": out_dir / f"{batch_id}_manifest.v1.json",
        "plan_report": reports_dir / "p3_exact_skill_batch_0002_plan.md",
        "plan_report_json": reports_dir / "p3_exact_skill_batch_0002_plan.v1.json",
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        paths["packet"].write_text(packet, encoding="utf-8")
        write_atomic_json(template, paths["template"], sort_keys=True)
        write_atomic_json(manifest, paths["manifest"], sort_keys=True)
        paths["plan_report"].write_text(report_markdown, encoding="utf-8")
        write_atomic_json(report_payload, paths["plan_report_json"], sort_keys=True)

    return {
        "ok": True,
        "dry_run": dry_run,
        "batch_id": batch_id,
        "selected_count": len(selected),
        "selected_queue_ids": [item["queue_id"] for item in selected],
        "skipped_count_by_reason": selection_notes.get("skipped_count_by_reason", {}),
        "category_counts": dict(Counter(_selection_value(item, "selection_category") for item in selected)),
        "paths": {key: str(path) for key, path in paths.items()},
        "manifest": manifest,
        "decision_template": template if dry_run else None,
        "packet": packet if dry_run else None,
        "plan_report": report_payload,
    }


def build_p3_exact_skill_batch_0003(
    *,
    queue_path: str | Path = DEFAULT_REVIEW_QUEUE_JSON_PATH,
    reviewed_path: str | Path = DEFAULT_REVIEWED_DECISIONS_PATH,
    content_lab_path: str | Path = "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json",
    batch_id: str = "batch_0003",
    out_dir: str | Path = DEFAULT_REVIEW_BATCH_DIR,
    reports_dir: str | Path = "reports",
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    reviewed_path = Path(reviewed_path)
    content_lab_path = Path(content_lab_path)
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    generated_at = generated_at or _utc_now_iso()

    queue = _load_json(queue_path)
    reviewed = _load_json(reviewed_path)
    content_lab = _load_json(content_lab_path) if content_lab_path.exists() else {}
    items = [item for item in queue.get("items", []) if isinstance(item, dict)] if isinstance(queue, dict) else []

    selected, selection_notes = select_batch_0003_items(items, reviewed=reviewed)
    template = build_decision_template(
        selected,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
    )
    skipped = Counter(selection_notes.get("skipped_count_by_reason", {}))
    manifest = build_batch_manifest(
        selected,
        skipped=skipped,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        limit=20,
        status=None,
        include_statuses=list(PURPOSE_STATUS_DEFAULTS[BATCH_0003_PURPOSE]),
        exclude_statuses=[],
        batch_purpose=BATCH_0003_PURPOSE,
        clean_reviewed_counts=_clean_reviewed_counts(reviewed),
    )
    manifest.update(_batch_0003_manifest_fields(selected, selection_notes))
    report_payload = build_batch_0003_plan_report(
        selected,
        manifest=manifest,
        reviewed=reviewed,
        content_lab=content_lab,
        generated_at=generated_at,
        source_paths={
            "queue": str(queue_path),
            "reviewed_registry": str(reviewed_path),
            "content_lab_candidates": str(content_lab_path),
        },
    )
    report_markdown = render_batch_0003_plan_report(report_payload)
    packet = render_review_packet(
        selected,
        batch_id=batch_id,
        generated_at=generated_at,
        queue_path=queue_path,
        reviewed_path=reviewed_path,
        limit=20,
        status=None,
        include_statuses=list(PURPOSE_STATUS_DEFAULTS[BATCH_0003_PURPOSE]),
        exclude_statuses=[],
        batch_purpose=BATCH_0003_PURPOSE,
    )

    validation_errors = validate_batch_0003_artifacts(manifest, template)
    if validation_errors:
        raise ValueError("Batch 0003 validation failed: " + "; ".join(validation_errors))

    paths = {
        "packet": out_dir / f"{batch_id}_review_packet.md",
        "template": out_dir / f"{batch_id}_decision_template.v1.json",
        "manifest": out_dir / f"{batch_id}_manifest.v1.json",
        "plan_report": reports_dir / "p3_exact_skill_batch_0003_plan.md",
        "plan_report_json": reports_dir / "p3_exact_skill_batch_0003_plan.v1.json",
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        paths["packet"].write_text(packet, encoding="utf-8")
        write_atomic_json(template, paths["template"], sort_keys=True)
        write_atomic_json(manifest, paths["manifest"], sort_keys=True)
        paths["plan_report"].write_text(report_markdown, encoding="utf-8")
        write_atomic_json(report_payload, paths["plan_report_json"], sort_keys=True)

    return {
        "ok": True,
        "dry_run": dry_run,
        "batch_id": batch_id,
        "selected_count": len(selected),
        "selected_queue_ids": [item["queue_id"] for item in selected],
        "skipped_count_by_reason": selection_notes.get("skipped_count_by_reason", {}),
        "category_counts": dict(Counter(_selection_value(item, "selection_category") for item in selected)),
        "paths": {key: str(path) for key, path in paths.items()},
        "manifest": manifest,
        "decision_template": template if dry_run else None,
        "packet": packet if dry_run else None,
        "plan_report": report_payload,
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


def select_batch_0002_items(
    items: list[dict[str, Any]],
    *,
    reviewed: dict[str, Any],
    batch_0001_conclusions: dict[str, Any],
    seed_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_subpart = {_text(item.get("subpart_id")): item for item in items if _text(item.get("subpart_id"))}
    selected: list[dict[str, Any]] = []
    used_queue_ids: set[str] = set()
    selected_questions: set[str] = set()
    skipped: Counter[str] = Counter()
    batch_0001_queue_ids = _batch_0001_queue_ids(batch_0001_conclusions)
    reviewed_index = _reviewed_registry_index(reviewed)

    def add_item(item: dict[str, Any] | None, category: str, reason: str, risks: list[str]) -> bool:
        if not item:
            skipped[f"{category}:missing_item"] += 1
            return False
        queue_id = _text(item.get("queue_id"))
        if not queue_id or queue_id in used_queue_ids:
            skipped[f"{category}:duplicate"] += 1
            return False
        if not _has_existing_asset(item.get("source_question_asset_refs")):
            skipped[f"{category}:missing_question_asset"] += 1
            return False
        if not _has_existing_asset(item.get("source_mark_scheme_asset_refs")):
            skipped[f"{category}:missing_mark_scheme_asset"] += 1
            return False
        annotated = _annotate_batch_0002_item(
            item,
            category=category,
            reason=reason,
            risks=risks,
            reviewed_index=reviewed_index,
        )
        selected.append(annotated)
        used_queue_ids.add(queue_id)
        selected_questions.add(_text(item.get("question_id")))
        return True

    for subpart_id in SEED_REGISTRY_SUBPART_IDS:
        add_item(
            by_subpart.get(subpart_id),
            "seed_mark_event_alignment_probe",
            (
                "Seed registry source-skill evidence exists; review whether advisory mark-event refs can be "
                "paired safely without changing generation readiness."
            ),
            ["reviewed_source_skill_exists", "mark_events_advisory_only", "part_level_uses_whole_question_images"],
        )

    deferred = _deferred_batch_0001_clean_items(batch_0001_conclusions, seed_report, by_subpart)
    for item, skip_reason in deferred[:6]:
        risks = ["batch_0001_clean_draft_not_promoted", "mark_events_advisory_only"]
        if "whole_question" in skip_reason:
            risks.append("whole_question_scope_requires_part_boundary_check")
        if "unknown" in skip_reason:
            risks.append("unknown_topic_alignment")
        add_item(
            item,
            "deferred_batch_0001_clean",
            f"Batch 0001 clean draft was intentionally deferred during seed promotion: {skip_reason}.",
            risks,
        )

    failure_selectors = [
        (
            3,
            "trig_supporting_integration",
            lambda item: _has_primary_skill(item, "9709_p3_3_3_identities_compound_double_angle_equations")
            and _topic_contains(item, "integration"),
            "Trigonometry appears inside integration or area work; verify it is not only supporting method context.",
        ),
        (
            2,
            "log_exp_supporting_differential_equations",
            lambda item: _has_primary_skill(item, "9709_p3_3_2_log_exponential_equations")
            and _topic_contains(item, "differential_equations"),
            "Log/exponential algebra appears inside differential-equation evidence; verify the assessed target skill.",
        ),
        (
            3,
            "derivative_rules_vs_implicit",
            lambda item: (
                _has_primary_skill(item, "9709_p3_3_4_derivative_rules")
                or _has_primary_skill(item, "9709_p3_3_4_parametric_implicit_differentiation")
            )
            and _text(item.get("recommended_review_action"))
            in {"verify_de_vs_implicit_differentiation", "verify_parametric_equation_parameter"},
            "Derivative-rule context may be mistaken for parametric or implicit differentiation.",
        ),
        (
            2,
            "polynomial_remainder_vs_differentiation",
            lambda item: _has_primary_skill(item, "9709_p3_3_1_polynomial_division_factor_remainder")
            and (_topic_contains(item, "differentiation") or _topic_contains(item, "algebra")),
            "Polynomial/remainder theorem evidence may be supporting work rather than the target skill.",
        ),
        (
            2,
            "whole_question_single_part_match",
            lambda item: _text(item.get("subpart_label")) == "whole"
            and "mixed_or_ambiguous_topic" in set(item.get("proposed_blockers") or []),
            "Whole-question candidate likely contains only one part matching the proposed skill.",
        ),
    ]
    for quota, risk_flag, predicate, reason in failure_selectors:
        candidates = [
            item
            for item in items
            if predicate(item)
            and _text(item.get("queue_id")) not in used_queue_ids
            and _text(item.get("proposed_route_status")) in PURPOSE_STATUS_DEFAULTS[BATCH_0002_PURPOSE]
        ]
        for item in _rank_batch_0002_candidates(candidates, selected_questions, prefer_batch_0001=True)[:quota]:
            add_item(
                item,
                "known_failure_mode_probe",
                reason,
                [risk_flag, "known_batch_0001_failure_mode", "do_not_default_to_clean"],
            )

    reliable_quotas = [
        ("9709_p3_3_9_complex_arithmetic_polar_form", 3, "Complex polar/modulus evidence was reliable in Batch 0001; confirm the pattern on additional items."),
        ("9709_p3_3_6_fixed_point_iteration", 2, "Fixed-point iteration evidence was reliable in Batch 0001; confirm the pattern on additional items."),
        ("9709_p3_3_3_identities_compound_double_angle_equations", 2, "Trig identities/equations were reliable when they were the target skill; confirm clean target-vs-support separation."),
        ("9709_p3_3_7_vector_lines", 2, "Vector-line evidence was reliable in Batch 0001; confirm the pattern on additional items."),
        ("9709_p3_3_4_parametric_implicit_differentiation", 3, "Parametric/implicit differentiation evidence was reliable in Batch 0001; confirm the pattern on additional items."),
    ]
    for skill_id, quota, reason in reliable_quotas:
        candidates = [
            item
            for item in items
            if _has_primary_skill(item, skill_id)
            and _text(item.get("topic_routing_alignment")) == "aligned"
            and _text(item.get("proposed_route_status")) == "cross_topic_candidate"
            and _text(item.get("recommended_review_action")) == "review_assets_and_skill"
            and _text(item.get("queue_id")) not in used_queue_ids
            and _text(item.get("queue_id")) not in batch_0001_queue_ids
        ]
        for item in _rank_batch_0002_candidates(candidates, selected_questions, prefer_batch_0001=False)[:quota]:
            add_item(
                item,
                "reliable_pattern_confirmation",
                reason,
                ["pattern_confirmed_clean_in_batch_0001", "mark_events_advisory_only"],
            )

    reliable_count = sum(
        1 for item in selected if _selection_value(item, "selection_category") == "reliable_pattern_confirmation"
    )
    if reliable_count < 12:
        reliable_skill_ids = {skill_id for skill_id, _, _ in reliable_quotas}
        top_up_candidates = [
            item
            for item in items
            if any(_has_primary_skill(item, skill_id) for skill_id in reliable_skill_ids)
            and _text(item.get("topic_routing_alignment")) == "aligned"
            and _text(item.get("proposed_route_status")) == "cross_topic_candidate"
            and _text(item.get("queue_id")) not in used_queue_ids
            and _text(item.get("queue_id")) not in batch_0001_queue_ids
        ]
        for item in _rank_batch_0002_candidates(top_up_candidates, selected_questions, prefer_batch_0001=False)[
            : 12 - reliable_count
        ]:
            add_item(
                item,
                "reliable_pattern_confirmation",
                "Top-up confidence-confirmation item from a Batch 0001 reliable skill family.",
                ["pattern_confirmed_clean_in_batch_0001", "mark_events_advisory_only"],
            )

    return selected, {"skipped_count_by_reason": dict(skipped)}


def select_batch_0003_items(
    items: list[dict[str, Any]],
    *,
    reviewed: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_subpart = {_text(item.get("subpart_id")): item for item in items if _text(item.get("subpart_id"))}
    selected: list[dict[str, Any]] = []
    used_queue_ids: set[str] = set()
    skipped: Counter[str] = Counter()
    reviewed_index = _reviewed_registry_index(reviewed)

    specs = [
        (
            "32autumn23_q09_b",
            "prior_ambiguous_retag_probe",
            "Retest Batch 0001/0002 trig-identity ambiguity where area/integration is the assessed target and trig identities are method support.",
            ["supporting_method_confusion", "integration_trig_area_target", "do_not_default_to_clean"],
        ),
        (
            "31autumn21_q07_c",
            "prior_ambiguous_retag_probe",
            "Retest Batch 0001 DE/log ambiguity where the selected part is limiting behaviour from a differential-equation solution.",
            ["supporting_method_confusion", "de_log_context", "do_not_default_to_clean"],
        ),
        (
            "32summer23_q07_b",
            "prior_ambiguous_retag_probe",
            "Retest Batch 0001 derivative-rules ambiguity where implicit differentiation and tangent conditions are the safer target.",
            ["wrong_skill_routing", "implicit_vs_derivative_rules", "do_not_default_to_clean"],
        ),
        (
            "33autumn22_q10_a",
            "prior_ambiguous_retag_probe",
            "Retest Batch 0002 log/exponential ambiguity where the subpart identifies constants in a differential-equation model.",
            ["supporting_method_confusion", "de_log_context", "do_not_default_to_clean"],
        ),
        (
            "33autumn22_q10_b",
            "prior_ambiguous_retag_probe",
            "Retest Batch 0002 log/exponential ambiguity where logarithms occur during separable differential-equation solving.",
            ["supporting_method_confusion", "de_log_context", "do_not_default_to_clean"],
        ),
        (
            "31autumn21_q04_whole",
            "prior_ambiguous_retag_probe",
            "Retest broad standard-integration routing where substitution, changed limits, and improper-limit structure need narrower treatment.",
            ["broad_integration_label", "retag_to_narrower_integration", "do_not_default_to_clean"],
        ),
        (
            "33autumn23_q07_b",
            "prior_blocked_confirmation",
            "Confirm Batch 0001 blocked polynomial/remainder route on an implicit-differentiation stationary-tangent subpart.",
            ["wrong_skill_routing", "polynomial_support_only", "do_not_default_to_clean"],
        ),
        (
            "31summer23_q02_a",
            "prior_blocked_confirmation",
            "Confirm Batch 0002 blocked parametric/implicit route on a modulus graph or linear-inequality subpart.",
            ["wrong_skill_routing", "parametric_implicit_absent", "do_not_default_to_clean"],
        ),
        (
            "31summer24_q06_c",
            "thin_adjacent_part_probe",
            "Retest thin fixed-point adjacent part against promoted part (d); exact but not enough evidence as a standalone source example.",
            ["thin_adjacent_part", "adjacent_part_contamination"],
        ),
        (
            "31summer24_q09_a",
            "thin_adjacent_part_probe",
            "Retest thin vector adjacent part against promoted part (b); one-mark scalar-product evidence remains too thin.",
            ["thin_adjacent_part", "adjacent_part_contamination"],
        ),
        (
            "32spring24_q05_b",
            "deferred_exact_skill_boundary_probe",
            "Retest clean-looking complex evidence skipped in Batch 0002 because the narrower Argand/loci boundary was not resolved.",
            ["narrower_skill_boundary", "do_not_default_to_clean"],
        ),
        (
            "33summer23_q11_b",
            "clean_control_mark_event_probe",
            "Already-promoted clean control: verify source-skill machinery still marks it safe while mark-event refs remain advisory-only.",
            ["already_promoted_clean_control", "mark_events_advisory_only", "mark_event_approval_probe"],
        ),
        (
            "31summer24_q06_d",
            "clean_control_mark_event_probe",
            "Already-promoted clean control: compare against thin adjacent part (c) and probe whether event-level approval exists.",
            ["already_promoted_clean_control", "mark_events_advisory_only", "mark_event_approval_probe"],
        ),
        (
            "31summer24_q09_b",
            "clean_control_mark_event_probe",
            "Already-promoted clean control: compare against thin adjacent part (a) and probe whether event-level approval exists.",
            ["already_promoted_clean_control", "mark_events_advisory_only", "mark_event_approval_probe"],
        ),
    ]

    for subpart_id, category, reason, risks in specs:
        item = by_subpart.get(subpart_id)
        if not item:
            skipped[f"{category}:missing_item"] += 1
            continue
        queue_id = _text(item.get("queue_id"))
        if not queue_id or queue_id in used_queue_ids:
            skipped[f"{category}:duplicate"] += 1
            continue
        if not _has_existing_asset(item.get("source_question_asset_refs")):
            skipped[f"{category}:missing_question_asset"] += 1
            continue
        if not _has_existing_asset(item.get("source_mark_scheme_asset_refs")):
            skipped[f"{category}:missing_mark_scheme_asset"] += 1
            continue
        selected.append(
            _annotate_batch_0003_item(
                item,
                category=category,
                reason=reason,
                risks=risks,
                reviewed_index=reviewed_index,
            )
        )
        used_queue_ids.add(queue_id)

    return selected, {"skipped_count_by_reason": dict(skipped)}


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
            "exclude_already_reviewed": batch_purpose
            not in {BATCH_0002_PURPOSE, BATCH_0003_PURPOSE},
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


def build_batch_0002_plan_report(
    items: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    reviewed: dict[str, Any],
    content_lab: dict[str, Any],
    generated_at: str,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    category_counts = Counter(_selection_value(item, "selection_category") for item in items)
    skill_counts = Counter(skill_id for item in items for skill_id in _p3_skill_ids(item))
    paper_counts = Counter(_text(item.get("paper")) for item in items)
    session_counts = Counter(_text(item.get("session")) for item in items)
    scope_counts = Counter(_selection_value(item, "review_scope_level") for item in items)
    risk_counts = Counter(risk for item in items for risk in _selection_list(item, "known_risk_flags"))
    registry_overlap = sum(1 for item in items if _selection_bool(item, "related_reviewed_registry_evidence_exists"))
    content_lab_candidates = content_lab.get("candidates") if isinstance(content_lab.get("candidates"), list) else []
    generation_ready_count = sum(1 for candidate in content_lab_candidates if _content_lab_generation_ready(candidate))
    empty_source_skill_count = sum(
        1 for candidate in content_lab_candidates if isinstance(candidate, dict) and not candidate.get("source_skill_ids")
    )
    return {
        "schema": "exam_bank.p3_exact_skill.batch_0002_plan",
        "schema_version": 1,
        "artifact_kind": "p3_exact_skill_batch_0002_plan",
        "generated_at": generated_at,
        "batch_id": manifest.get("batch_id"),
        "source_paths": source_paths,
        "summary": {
            "total_selected_items": len(items),
            "count_by_selection_category": dict(category_counts),
            "count_by_proposed_source_skill": dict(skill_counts),
            "count_by_paper": dict(paper_counts),
            "count_by_session": dict(session_counts),
            "part_level_vs_whole_question_count": dict(scope_counts),
            "seed_registry_overlap_count": registry_overlap,
            "known_risk_failure_mode_count": sum(count for risk, count in risk_counts.items() if "failure" in risk or "supporting" in risk or "ambiguous" in risk or "confused" in risk or "whole_question" in risk),
            "known_risk_flag_counts": dict(risk_counts),
            "expected_reviewer_workload": "30-40 item mixed manual-review batch; expect extra time on failure probes and seed mark-event alignment.",
            "reviewed_registry_record_count": len(reviewed.get("records", [])) if isinstance(reviewed.get("records"), list) else 0,
            "clean_reviewed_registry_record_count": sum(
                1
                for record in reviewed.get("records", [])
                if isinstance(record, dict) and record.get("route_status") == "clean"
            )
            if isinstance(reviewed.get("records"), list)
            else 0,
            "content_lab_candidate_count": len(content_lab_candidates),
            "content_lab_generation_ready_count": generation_ready_count,
            "content_lab_empty_source_skill_id_count": empty_source_skill_count,
        },
        "why_this_improves_on_batch_0001": [
            "Batch 0001 mostly tested clean-looking cross-topic reviewable records; Batch 0002 deliberately mixes confirmations with failure probes.",
            "Known Batch 0001 overconfidence cases are represented as probes and remain review_needed by default.",
            "Deferred clean drafts are revisited without automatic registry promotion.",
            "Seed registry records are used to inspect mark-event pairing safety while candidate_generation remains false.",
            "Unknown topic-alignment and whole-question scope risks are explicit selection metadata instead of implicit reviewer context.",
        ],
        "selected_items": [_batch_0002_manifest_item(item) for item in items],
    }


def render_batch_0002_plan_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# P3 Exact-Skill Batch 0002 Plan",
        "",
        "This is a plan and evidence-collection report only. It does not promote candidates, mark anything generation-ready, or change canonical assets.",
        "",
        "## Summary",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- Selected items: `{summary.get('total_selected_items', 0)}`",
        f"- Count by category: `{json.dumps(summary.get('count_by_selection_category') or {}, sort_keys=True)}`",
        f"- Count by proposed source skill: `{json.dumps(summary.get('count_by_proposed_source_skill') or {}, sort_keys=True)}`",
        f"- Count by paper: `{json.dumps(summary.get('count_by_paper') or {}, sort_keys=True)}`",
        f"- Count by session: `{json.dumps(summary.get('count_by_session') or {}, sort_keys=True)}`",
        f"- Part-level vs whole-question: `{json.dumps(summary.get('part_level_vs_whole_question_count') or {}, sort_keys=True)}`",
        f"- Seed-registry overlap count: `{summary.get('seed_registry_overlap_count', 0)}`",
        f"- Known risk/failure-mode count: `{summary.get('known_risk_failure_mode_count', 0)}`",
        f"- Expected reviewer workload: {summary.get('expected_reviewer_workload')}",
        f"- Content Lab generation-ready candidates: `{summary.get('content_lab_generation_ready_count', 0)}`",
        "",
        "## Why This Improves On Batch 0001",
        "",
        *[f"- {item}" for item in payload.get("why_this_improves_on_batch_0001") or []],
        "",
        "## Reviewer Guardrails",
        "",
        "- Canonical question and mark-scheme images are the source of truth.",
        "- Advisory text, OCR, topic routing, and mark-event refs are review context only.",
        "- Do not convert reviewed source-skill evidence into generation readiness.",
        "- Do not promote any selected record from this batch without a later explicit promotion pass.",
        "",
        "## Selected Items",
        "",
    ]
    for item in payload.get("selected_items") or []:
        lines.extend(
            [
                f"### `{item.get('queue_id')}`",
                "",
                f"- Category: `{item.get('selection_category')}`",
                f"- Reason: {item.get('selection_reason')}",
                f"- Proposed source skill: `{item.get('proposed_source_skill_id')}`",
                f"- Paper/session: `{item.get('paper')}` / `{item.get('session')}`",
                f"- Scope: `{item.get('review_scope_level')}`",
                f"- Related reviewed registry evidence exists: `{item.get('related_reviewed_registry_evidence_exists')}`",
                f"- Known risk flags: `{json.dumps(item.get('known_risk_flags') or [])}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_batch_0003_plan_report(
    items: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    reviewed: dict[str, Any],
    content_lab: dict[str, Any],
    generated_at: str,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    category_counts = Counter(_selection_value(item, "selection_category") for item in items)
    skill_counts = Counter(skill_id for item in items for skill_id in _p3_skill_ids(item))
    risk_counts = Counter(risk for item in items for risk in _selection_list(item, "known_risk_flags"))
    content_lab_candidates = content_lab.get("candidates") if isinstance(content_lab.get("candidates"), list) else []
    generation_ready_count = sum(1 for candidate in content_lab_candidates if _content_lab_generation_ready(candidate))
    reviewed_records = reviewed.get("records") if isinstance(reviewed.get("records"), list) else []
    return {
        "schema": "exam_bank.p3_exact_skill.batch_0003_plan",
        "schema_version": 1,
        "artifact_kind": "p3_exact_skill_batch_0003_plan",
        "generated_at": generated_at,
        "batch_id": manifest.get("batch_id"),
        "source_paths": source_paths,
        "summary": {
            "total_selected_items": len(items),
            "count_by_selection_category": dict(category_counts),
            "count_by_proposed_source_skill": dict(skill_counts),
            "known_risk_flag_counts": dict(risk_counts),
            "prior_ambiguous_or_blocked_or_thin_count": sum(
                count
                for category, count in category_counts.items()
                if category
                in {
                    "prior_ambiguous_retag_probe",
                    "prior_blocked_confirmation",
                    "thin_adjacent_part_probe",
                    "deferred_exact_skill_boundary_probe",
                }
            ),
            "clean_control_count": category_counts.get("clean_control_mark_event_probe", 0),
            "reviewed_registry_record_count": len(reviewed_records),
            "clean_reviewed_registry_record_count": sum(
                1 for record in reviewed_records if isinstance(record, dict) and record.get("route_status") == "clean"
            ),
            "content_lab_candidate_count": len(content_lab_candidates),
            "content_lab_generation_ready_count": generation_ready_count,
            "expected_reviewer_workload": "12-20 adversarial records; expect few or zero promotions.",
        },
        "review_guardrails": [
            "Do not promote broad integration labels where the narrower target is substitution, trigonometric integration, area, or improper-limit work.",
            "Do not treat log, trig, polynomial, or derivative support work as the exact assessed skill.",
            "Do not promote thin adjacent parts that are only context for a more substantial neighbouring part.",
            "Do not approve mark events from advisory refs unless the workflow has an explicit reviewed-mark-event schema and validator path.",
            "Do not change Content Lab generation readiness from this batch.",
        ],
        "selected_items": [_batch_0003_manifest_item(item) for item in items],
    }


def render_batch_0003_plan_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# P3 Exact-Skill Batch 0003 Plan",
        "",
        "This is an adversarial retagging and mark-event approval probe. It does not promote candidates, mark anything generation-ready, or change canonical assets.",
        "",
        "## Summary",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- Selected items: `{summary.get('total_selected_items', 0)}`",
        f"- Count by category: `{json.dumps(summary.get('count_by_selection_category') or {}, sort_keys=True)}`",
        f"- Count by proposed source skill: `{json.dumps(summary.get('count_by_proposed_source_skill') or {}, sort_keys=True)}`",
        f"- Prior ambiguous/blocked/thin probes: `{summary.get('prior_ambiguous_or_blocked_or_thin_count', 0)}`",
        f"- Clean controls: `{summary.get('clean_control_count', 0)}`",
        f"- Content Lab generation-ready candidates: `{summary.get('content_lab_generation_ready_count', 0)}`",
        "",
        "## Review Guardrails",
        "",
        *[f"- {item}" for item in payload.get("review_guardrails") or []],
        "",
        "## Selected Items",
        "",
    ]
    for item in payload.get("selected_items") or []:
        lines.extend(
            [
                f"### `{item.get('queue_id')}`",
                "",
                f"- Category: `{item.get('selection_category')}`",
                f"- Reason: {item.get('selection_reason')}",
                f"- Proposed source skill: `{item.get('proposed_source_skill_id')}`",
                f"- Scope: `{item.get('review_scope_level')}`",
                f"- Related reviewed evidence exists: `{item.get('related_reviewed_registry_evidence_exists')}`",
                f"- Known risk flags: `{json.dumps(item.get('known_risk_flags') or [])}`",
                f"- Mark-event approval probe: `{item.get('mark_event_approval_probe')}`",
                f"- Generation ready: `{item.get('generation_ready')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


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
        *_batch_0002_instruction_lines(items),
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


def validate_batch_0002_artifacts(manifest: dict[str, Any], template: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    selected_items = manifest.get("selected_items") if isinstance(manifest.get("selected_items"), list) else []
    template_records = template.get("records") if isinstance(template.get("records"), list) else []
    records_by_queue_id = {
        _text(record.get("queue_id")): record for record in template_records if isinstance(record, dict)
    }
    for index, item in enumerate(selected_items):
        if not isinstance(item, dict):
            errors.append(f"selected_items[{index}]:not_object")
            continue
        prefix = f"{_text(item.get('queue_id')) or f'selected_items[{index}]'}"
        category = _text(item.get("selection_category"))
        if category not in BATCH_0002_SELECTION_CATEGORIES:
            errors.append(f"{prefix}:missing_or_invalid_selection_category")
        if not _text(item.get("selection_reason")):
            errors.append(f"{prefix}:missing_selection_reason")
        if not _has_existing_asset(item.get("canonical_question_image_refs")):
            errors.append(f"{prefix}:missing_canonical_question_image_ref")
        if item.get("mark_scheme_evidence_expected", True) and not _has_existing_asset(
            item.get("canonical_mark_scheme_image_refs")
        ):
            errors.append(f"{prefix}:missing_canonical_mark_scheme_image_ref")
        if not _text(item.get("proposed_source_skill_id")) and not _text(item.get("unknown_alignment_reason")):
            errors.append(f"{prefix}:missing_proposed_source_skill_without_unknown_alignment_reason")
        if category == "seed_mark_event_alignment_probe" and item.get("generation_ready") is True:
            errors.append(f"{prefix}:seed_probe_marked_generation_ready")
        record = records_by_queue_id.get(_text(item.get("queue_id")))
        if not record:
            errors.append(f"{prefix}:missing_template_record")
            continue
        if record.get("route_status") != "review_needed":
            errors.append(f"{prefix}:template_route_status_not_review_needed")
        if category == "known_failure_mode_probe" and record.get("route_status") == "clean":
            errors.append(f"{prefix}:failure_probe_defaulted_to_clean")
        if category == "known_failure_mode_probe" and "do_not_default_to_clean" not in set(
            item.get("known_risk_flags") or []
        ):
            errors.append(f"{prefix}:failure_probe_missing_do_not_default_to_clean_risk")
        allowed = record.get("allowed_use_cases") if isinstance(record.get("allowed_use_cases"), dict) else {}
        if allowed.get("candidate_generation") is True:
            errors.append(f"{prefix}:template_candidate_generation_allowed")
    return errors


def validate_batch_0003_artifacts(manifest: dict[str, Any], template: dict[str, Any]) -> list[str]:
    errors = _validate_batch_review_artifacts(
        manifest,
        template,
        valid_categories=BATCH_0003_SELECTION_CATEGORIES,
    )
    selected_items = manifest.get("selected_items") if isinstance(manifest.get("selected_items"), list) else []
    category_counts = Counter(
        _text(item.get("selection_category")) for item in selected_items if isinstance(item, dict)
    )
    selected_count = len(selected_items)
    if selected_count < 12 or selected_count > 20:
        errors.append(f"batch_0003:selected_count_outside_target_range:{selected_count}")
    if category_counts.get("clean_control_mark_event_probe", 0) < 2:
        errors.append("batch_0003:missing_minimum_clean_controls")
    if category_counts.get("prior_ambiguous_retag_probe", 0) < 4:
        errors.append("batch_0003:missing_prior_ambiguous_retag_probes")
    if category_counts.get("prior_blocked_confirmation", 0) < 1:
        errors.append("batch_0003:missing_prior_blocked_confirmation")
    if category_counts.get("thin_adjacent_part_probe", 0) < 2:
        errors.append("batch_0003:missing_thin_adjacent_part_probes")
    if any(
        isinstance(item, dict)
        and _text(item.get("selection_category"))
        in {
            "prior_ambiguous_retag_probe",
            "prior_blocked_confirmation",
            "deferred_exact_skill_boundary_probe",
        }
        and "do_not_default_to_clean" not in set(item.get("known_risk_flags") or [])
        for item in selected_items
    ):
        errors.append("batch_0003:failure_or_boundary_probe_missing_do_not_default_to_clean")
    if any(isinstance(item, dict) and item.get("generation_ready") is True for item in selected_items):
        errors.append("batch_0003:selected_item_marked_generation_ready")
    if any(
        isinstance(item, dict)
        and item.get("mark_event_decision_default") not in {None, "advisory_only"}
        for item in selected_items
    ):
        errors.append("batch_0003:mark_event_default_not_advisory_only")
    if any(
        isinstance(item, dict)
        and item.get("control_record") is True
        and (
            _text(item.get("selection_category")) != "clean_control_mark_event_probe"
            or item.get("related_reviewed_registry_evidence_exists") is not True
            or not item.get("related_reviewed_evidence_ids")
        )
        for item in selected_items
    ):
        errors.append("batch_0003:control_record_without_existing_reviewed_decision")
    return errors


def _validate_batch_review_artifacts(
    manifest: dict[str, Any],
    template: dict[str, Any],
    *,
    valid_categories: set[str],
) -> list[str]:
    errors: list[str] = []
    selected_items = manifest.get("selected_items") if isinstance(manifest.get("selected_items"), list) else []
    template_records = template.get("records") if isinstance(template.get("records"), list) else []
    records_by_queue_id = {
        _text(record.get("queue_id")): record for record in template_records if isinstance(record, dict)
    }
    for index, item in enumerate(selected_items):
        if not isinstance(item, dict):
            errors.append(f"selected_items[{index}]:not_object")
            continue
        prefix = f"{_text(item.get('queue_id')) or f'selected_items[{index}]'}"
        category = _text(item.get("selection_category"))
        if category not in valid_categories:
            errors.append(f"{prefix}:missing_or_invalid_selection_category")
        if not _text(item.get("selection_reason")):
            errors.append(f"{prefix}:missing_selection_reason")
        if not _has_existing_asset(item.get("canonical_question_image_refs")):
            errors.append(f"{prefix}:missing_canonical_question_image_ref")
        if item.get("mark_scheme_evidence_expected", True) and not _has_existing_asset(
            item.get("canonical_mark_scheme_image_refs")
        ):
            errors.append(f"{prefix}:missing_canonical_mark_scheme_image_ref")
        if not _text(item.get("proposed_source_skill_id")) and not _text(item.get("unknown_alignment_reason")):
            errors.append(f"{prefix}:missing_proposed_source_skill_without_unknown_alignment_reason")
        record = records_by_queue_id.get(_text(item.get("queue_id")))
        if not record:
            errors.append(f"{prefix}:missing_template_record")
            continue
        if record.get("route_status") != "review_needed":
            errors.append(f"{prefix}:template_route_status_not_review_needed")
        allowed = record.get("allowed_use_cases") if isinstance(record.get("allowed_use_cases"), dict) else {}
        if allowed.get("candidate_generation") is True:
            errors.append(f"{prefix}:template_candidate_generation_allowed")
    return errors


def _batch_0002_instruction_lines(items: list[dict[str, Any]]) -> list[str]:
    if not any(_selection_value(item, "selection_category") for item in items):
        return []
    if any(_selection_payload(item, "batch_0003_selection") for item in items):
        return [
            "## Batch 0003 Review Instructions",
            "",
            "- Review exact-skill decisions and mark-event decisions separately for every record.",
            "- Use approved, retagged, deferred, blocked, or thin for the exact-skill outcome; do not infer mark-event approval from that outcome.",
            "- Keep mark-event refs advisory-only unless an explicit reviewed-mark-event schema and validator path already exist.",
            "- Treat known controls as controls only: they test that clean source-skill evidence stays distinguishable from unsafe probes.",
            "- Do not change Content Lab generation readiness from advisory mark-event refs.",
            "",
        ]
    return [
        "## Batch 0002 Review Instructions",
        "",
        "- Distinguish target skill from supporting method. Supporting algebra, trigonometry, or differentiation is not clean source-skill evidence unless the image evidence shows it is the assessed target.",
        "- Distinguish whole-question evidence from part-level evidence. Whole-question images are canonical, but a clean decision may still need a part-level scope.",
        "- Distinguish source-skill review from mark-event review. Reviewed source-skill evidence does not approve mark events.",
        "- Treat advisory text, OCR, topic routing, and mark-event refs as context. Canonical question and mark-scheme images are the source of truth.",
        "- Use clean only when the exact skill, scope, and evidence basis are clear. Use ambiguous when target-vs-support or part scope remains uncertain.",
        "- Reviewed source-skill evidence is not generation readiness. Leave candidate_generation false unless a later explicit promotion pass changes it.",
        "",
    ]


def _batch_0002_manifest_fields(items: list[dict[str, Any]], selection_notes: dict[str, Any]) -> dict[str, Any]:
    category_counts = Counter(_selection_value(item, "selection_category") for item in items)
    risk_counts = Counter(risk for item in items for risk in _selection_list(item, "known_risk_flags"))
    return {
        "selection_strategy": BATCH_0002_PURPOSE,
        "selection_categories": sorted(BATCH_0002_SELECTION_CATEGORIES),
        "review_outcome_categories": sorted(REVIEW_OUTCOME_CATEGORIES),
        "category_counts": dict(category_counts),
        "known_risk_flag_counts": dict(risk_counts),
        "selected_items": [_batch_0002_manifest_item(item) for item in items],
        "selection_notes": selection_notes,
        "batch_0002_constraints": {
            "auto_promotion_allowed": False,
            "generation_readiness_change_allowed": False,
            "canonical_image_changes_allowed": False,
            "validation_loosened": False,
            "content_lab_export_consumption_changed": False,
        },
    }


def _batch_0003_manifest_fields(items: list[dict[str, Any]], selection_notes: dict[str, Any]) -> dict[str, Any]:
    category_counts = Counter(_selection_value(item, "selection_category") for item in items)
    risk_counts = Counter(risk for item in items for risk in _selection_list(item, "known_risk_flags"))
    return {
        "selection_strategy": BATCH_0003_PURPOSE,
        "selection_categories": sorted(BATCH_0003_SELECTION_CATEGORIES),
        "review_outcome_categories": sorted(REVIEW_OUTCOME_CATEGORIES),
        "category_counts": dict(category_counts),
        "known_risk_flag_counts": dict(risk_counts),
        "selected_items": [_batch_0003_manifest_item(item) for item in items],
        "selection_notes": selection_notes,
        "batch_0003_constraints": {
            "auto_promotion_allowed": False,
            "generation_readiness_change_allowed": False,
            "canonical_image_changes_allowed": False,
            "validation_loosened": False,
            "content_lab_export_consumption_changed": False,
            "mark_event_runtime_behavior_changed": False,
        },
    }


def _batch_0002_manifest_item(item: dict[str, Any]) -> dict[str, Any]:
    proposed_skill_ids = _p3_skill_ids(item)
    alignment = _text(item.get("topic_routing_alignment")) or "unknown"
    mark_event_filtering = _part_level_mark_event_filtering(item)
    return {
        "queue_id": _text(item.get("queue_id")),
        "question_id": _text(item.get("question_id")),
        "paper": _text(item.get("paper")),
        "session": _text(item.get("session")),
        "variant": _text(item.get("variant")),
        "part_id": _text(item.get("part_id")),
        "subpart_id": _text(item.get("subpart_id")),
        "proposed_source_skill_id": proposed_skill_ids[0] if proposed_skill_ids else "",
        "proposed_source_skill_ids": proposed_skill_ids,
        "selection_category": _selection_value(item, "selection_category"),
        "selection_reason": _selection_value(item, "selection_reason"),
        "canonical_question_image_refs": item.get("source_question_asset_refs") or [],
        "canonical_mark_scheme_image_refs": item.get("source_mark_scheme_asset_refs") or [],
        "advisory_text_evidence": {
            "candidate_region_topic": item.get("candidate_region_topic") or {},
            "topic_routing": item.get("topic_routing") or {},
            "topic_routing_alignment": alignment,
            "cross_topic_status": _text(item.get("cross_topic_status")) or "unknown",
            "cross_topic_notes": item.get("cross_topic_notes") or [],
            "mark_event_refs": _advisory_mark_event_refs(item.get("mark_event_refs") or []),
            "matching_mark_event_ids": mark_event_filtering["matching_mark_event_ids"],
            "other_part_mark_event_ids": mark_event_filtering["other_part_mark_event_ids"],
            "mark_event_filter_confidence": mark_event_filtering["confidence"],
            "content_lab_blocker_context": item.get("asterion_candidate") or {},
        },
        "known_risk_flags": _selection_list(item, "known_risk_flags"),
        "review_scope_level": _selection_value(item, "review_scope_level"),
        "part_level_review": _selection_value(item, "review_scope_level") == "part_level",
        "whole_question_review": _selection_value(item, "review_scope_level") == "whole_question",
        "related_reviewed_registry_evidence_exists": _selection_bool(
            item, "related_reviewed_registry_evidence_exists"
        ),
        "related_reviewed_evidence_ids": _selection_list(item, "related_reviewed_evidence_ids"),
        "control_record": _selection_bool(item, "control_record"),
        "review_outcome_category_default": _review_outcome_category_default(item),
        "mark_event_filtering": mark_event_filtering,
        "mark_scheme_evidence_expected": True,
        "generation_ready": False,
        "unknown_alignment_reason": "topic_routing_alignment_unknown" if alignment == "unknown" else "",
    }


def _batch_0003_manifest_item(item: dict[str, Any]) -> dict[str, Any]:
    base = _batch_0002_manifest_item(item)
    base["mark_event_approval_probe"] = "mark_event_approval_probe" in set(base.get("known_risk_flags") or [])
    base["mark_event_decision_default"] = "advisory_only"
    base["exact_skill_decision_default"] = "review_needed"
    base["control_record"] = base.get("control_record") or base["selection_category"] == "clean_control_mark_event_probe"
    return base


def _annotate_batch_0002_item(
    item: dict[str, Any],
    *,
    category: str,
    reason: str,
    risks: list[str],
    reviewed_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    annotated = dict(item)
    scope = (_text(item.get("question_id")), _text(item.get("subpart_id")) or f"{_text(item.get('question_id'))}_whole")
    related = reviewed_index.get(scope, [])
    merged_risks = list(dict.fromkeys([risk for risk in risks if risk] + _derived_risk_flags(item)))
    annotated["batch_0002_selection"] = {
        "selection_category": category,
        "selection_reason": reason,
        "known_risk_flags": merged_risks,
        "review_scope_level": _review_scope_level(item),
        "related_reviewed_registry_evidence_exists": bool(related),
        "related_reviewed_evidence_ids": [_text(record.get("evidence_id")) for record in related if _text(record.get("evidence_id"))],
    }
    return annotated


def _annotate_batch_0003_item(
    item: dict[str, Any],
    *,
    category: str,
    reason: str,
    risks: list[str],
    reviewed_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    annotated = dict(item)
    scope = (_text(item.get("question_id")), _text(item.get("subpart_id")) or f"{_text(item.get('question_id'))}_whole")
    related = reviewed_index.get(scope, [])
    merged_risks = list(dict.fromkeys([risk for risk in risks if risk] + _derived_risk_flags(item)))
    annotated["batch_0003_selection"] = {
        "selection_category": category,
        "selection_reason": reason,
        "known_risk_flags": merged_risks,
        "review_scope_level": _review_scope_level(item),
        "related_reviewed_registry_evidence_exists": bool(related),
        "related_reviewed_evidence_ids": [_text(record.get("evidence_id")) for record in related if _text(record.get("evidence_id"))],
        "control_record": category == "clean_control_mark_event_probe",
    }
    return annotated


def _deferred_batch_0001_clean_items(
    conclusions: dict[str, Any], seed_report: dict[str, Any], by_subpart: dict[str, dict[str, Any]]
) -> list[tuple[dict[str, Any], str]]:
    skipped = conclusions.get("records_intentionally_skipped")
    if not isinstance(skipped, list):
        skipped = (
            seed_report.get("records_intentionally_skipped")
            if isinstance(seed_report.get("records_intentionally_skipped"), list)
            else []
        )
    preferred_reasons = {
        "clean_whole_question_candidate_deferred_for_later_pass",
        "clean_but_topic_alignment_unknown_deferred",
        "not_in_small_seed_subset",
    }
    candidates: list[tuple[dict[str, Any], str]] = []
    for record in skipped:
        if not isinstance(record, dict):
            continue
        if record.get("draft_route_status") != "clean" or record.get("skip_reason") not in preferred_reasons:
            continue
        item = by_subpart.get(_text(record.get("subpart_id")))
        if item:
            candidates.append((item, _text(record.get("skip_reason"))))
    return sorted(
        candidates,
        key=lambda pair: (
            {"clean_whole_question_candidate_deferred_for_later_pass": 0, "clean_but_topic_alignment_unknown_deferred": 1}.get(pair[1], 2),
            _text(pair[0].get("question_id")),
            _text(pair[0].get("subpart_id")),
        ),
    )


def _batch_0001_queue_ids(conclusions: dict[str, Any]) -> set[str]:
    summary = conclusions.get("before_after_summary") if isinstance(conclusions.get("before_after_summary"), dict) else {}
    items = summary.get("item_comparisons") if isinstance(summary.get("item_comparisons"), list) else []
    return {_text(item.get("queue_id")) for item in items if isinstance(item, dict) and _text(item.get("queue_id"))}


def _reviewed_registry_index(reviewed: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    records = reviewed.get("records") if isinstance(reviewed.get("records"), list) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = _text(record.get("question_id"))
        subpart_id = _text(record.get("subpart_id")) or f"{question_id}_whole"
        if question_id:
            index.setdefault((question_id, subpart_id), []).append(record)
    return index


def _rank_batch_0002_candidates(
    candidates: list[dict[str, Any]], selected_questions: set[str], *, prefer_batch_0001: bool
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            0 if (_text(item.get("question_id")) not in selected_questions or prefer_batch_0001) else 1,
            -int(item.get("priority_score") or 0),
            _text(item.get("question_id")),
            _text(item.get("subpart_id")),
        ),
    )


def _derived_risk_flags(item: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    blockers = set(item.get("proposed_blockers") or [])
    if "mixed_or_ambiguous_topic" in blockers:
        risks.append("mixed_or_ambiguous_topic")
    if _text(item.get("topic_routing_alignment")) == "unknown":
        risks.append("unknown_topic_alignment")
    if _text(item.get("subpart_label")) == "whole":
        risks.append("whole_question_review_scope")
    if _text(item.get("recommended_scope")) in {"subpart_level", "part_level"}:
        risks.append("part_level_scope_uses_whole_images")
    if item.get("mark_event_refs"):
        risks.append("mark_events_advisory_only")
    return risks


def _review_scope_level(item: dict[str, Any]) -> str:
    if _text(item.get("subpart_label")) == "whole" or _text(item.get("part_id")) == "whole":
        return "whole_question"
    return "part_level"


def _review_outcome_category_default(item: dict[str, Any]) -> str:
    selection_category = _selection_value(item, "selection_category")
    risks = set(_selection_list(item, "known_risk_flags"))
    if selection_category == "clean_control_mark_event_probe":
        return "clean_seed"
    if selection_category == "thin_adjacent_part_probe":
        return "thin_or_adjacent_context"
    if "thin_adjacent_part" in risks:
        return "exact_but_not_seed_quality"
    if "supporting_method_confusion" in risks:
        return "supporting_method_not_target_skill"
    if "wrong_skill_routing" in risks:
        return "blocked_wrong_or_unsafe_label"
    if selection_category in {"prior_ambiguous_retag_probe", "prior_blocked_confirmation"}:
        return "cross_content_not_exact_skill_isolatable"
    return "review_needed"


def _part_level_mark_event_filtering(item: dict[str, Any]) -> dict[str, Any]:
    all_refs = _advisory_mark_event_refs(item.get("mark_event_refs") or [])
    all_ids = _event_ids(all_refs)
    matching_ids: list[str] = []
    other_ids: list[str] = []
    source_subpart_label = _text(item.get("subpart_label") or item.get("part_id"))
    candidates = item.get("proposed_part_level_candidates") if isinstance(item.get("proposed_part_level_candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_path = [_text(part) for part in candidate.get("part_path") or [] if _text(part)]
        candidate_matches_source = bool(candidate_path) and source_subpart_label and candidate_path[0] == source_subpart_label
        if candidate_matches_source or candidate.get("decomposition_status") == "already_part_scoped":
            matching_ids.extend(_event_ids(candidate.get("matching_mark_event_refs") or []))
            other_ids.extend(_event_ids(candidate.get("other_part_mark_event_refs") or []))
        matching_ids = _unique_texts(matching_ids)
    if matching_ids:
        other_ids = _unique_texts(other_ids + [event_id for event_id in all_ids if event_id not in set(matching_ids)])
        confidence = "part_path_matched"
    else:
        other_ids = all_ids
        confidence = "uncertain_no_confident_part_match" if all_ids else "no_mark_events"
    return {
        "matching_mark_event_ids": matching_ids,
        "other_part_mark_event_ids": other_ids,
        "confidence": confidence,
        "warning": (
            "Only matching_mark_event_ids should be used for future part-level mark-event review probes. "
            "When confidence is uncertain, all event IDs remain outside the matching set."
        ),
    }


def _event_ids(refs: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(refs, list):
        return result
    for ref in refs:
        if isinstance(ref, dict) and _text(ref.get("event_id")):
            result.append(_text(ref.get("event_id")))
    return _unique_texts(result)


def _has_primary_skill(item: dict[str, Any], skill_id: str) -> bool:
    return skill_id in _unique_texts(item.get("primary_candidate_skill_ids") or _p3_skill_ids(item))


def _topic_contains(item: dict[str, Any], needle: str) -> bool:
    haystack = " ".join(
        [
            _text(_nested(item, "topic_routing", "primary_topic_id")),
            _text(_nested(item, "candidate_region_topic", "topic_assignment_id")),
            _text(_nested(item, "candidate_region_topic", "mapping_source_topic")),
            _text(_nested(item, "candidate_region_topic", "subtopic_id")),
        ]
    ).lower()
    return needle.lower() in haystack


def _selection_value(item: dict[str, Any], key: str) -> str:
    selection = _selection_payload(item)
    return _text(selection.get(key))


def _selection_list(item: dict[str, Any], key: str) -> list[str]:
    selection = _selection_payload(item)
    return _unique_texts(selection.get(key) or [])


def _selection_bool(item: dict[str, Any], key: str) -> bool:
    selection = _selection_payload(item)
    return bool(selection.get(key))


def _selection_payload(item: dict[str, Any], preferred_key: str | None = None) -> dict[str, Any]:
    keys = [preferred_key] if preferred_key else ["batch_0003_selection", "batch_0002_selection"]
    for key in keys:
        if key and isinstance(item.get(key), dict):
            return item[key]
    return {}


def _content_lab_generation_ready(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    gate = candidate.get("generation_gate") if isinstance(candidate.get("generation_gate"), dict) else {}
    return gate.get("blocked") is False or _text(gate.get("status")) in {"ready", "generation_ready", "generation-ready"}


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
    mark_event_filtering = _part_level_mark_event_filtering(item)
    return {
        "evidence_id": f"p3_exact_skill_review:{batch_id}:{_text(item.get('question_id'))}:{_text(item.get('subpart_id'))}",
        "queue_id": _text(item.get("queue_id")),
        "selection_category": _selection_value(item, "selection_category"),
        "selection_reason": _selection_value(item, "selection_reason"),
        "known_risk_flags": _selection_list(item, "known_risk_flags"),
        "review_scope_level": _selection_value(item, "review_scope_level"),
        "related_reviewed_registry_evidence_exists": _selection_bool(
            item, "related_reviewed_registry_evidence_exists"
        ),
        "related_reviewed_evidence_ids": _selection_list(item, "related_reviewed_evidence_ids"),
        "control_record": _selection_bool(item, "control_record"),
        "review_outcome_category_default": _review_outcome_category_default(item),
        "generation_readiness": {
            "must_remain_blocked": True,
            "candidate_generation_allowed_by_template": False,
            "review_note": "Review source-skill and mark-event evidence separately; this template does not mark generation-ready.",
        },
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
        "mark_event_filtering": mark_event_filtering,
        "matching_mark_event_ids": mark_event_filtering["matching_mark_event_ids"],
        "other_part_mark_event_ids": mark_event_filtering["other_part_mark_event_ids"],
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
        f"- Selection category: `{_selection_value(item, 'selection_category') or 'unspecified'}`",
        f"- Selection reason: {_selection_value(item, 'selection_reason') or 'unspecified'}",
        f"- Known risk flags: {', '.join(f'`{risk}`' for risk in _selection_list(item, 'known_risk_flags')) or 'none'}",
        f"- Review scope level: `{_selection_value(item, 'review_scope_level') or 'unknown'}`",
        f"- Related reviewed registry evidence exists: `{_selection_bool(item, 'related_reviewed_registry_evidence_exists')}`",
        f"- Related reviewed evidence IDs: {', '.join(f'`{evidence_id}`' for evidence_id in _selection_list(item, 'related_reviewed_evidence_ids')) or 'none'}",
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
