from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .classification import classify_question
from .config import AppConfig, load_config


RESCORE_SCHEMA_NAME = "exam_bank.topic_confidence_rescoring"
RESCORE_SCHEMA_VERSION = 1

DEFAULT_QUESTION_BANK_PATH = Path("output/json/question_bank.json")
DEFAULT_TOPIC_ROUTING_PATH = Path("data/topic_routing/question_bank.topic_routing.v1.json")
DEFAULT_TAXONOMY_ROOT = Path("exam_bank_taxonomy/canonical")
DEFAULT_OUT_DIR = Path("output/audits/topic_confidence_rescoring")

CONFIDENCE_VALUES = {"high", "medium", "low"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

BENIGN_VISUAL_OCR_FLAGS = {
    "ocr_hint_figure_regions",
    "figure_region_separated",
    "question_context_figure_inference_used",
}
QUESTION_TEXT_WEAK_FLAGS = {
    "missing_question_text",
    "question_text_missing",
    "truncated_question_text",
    "question_text_truncated",
    "short_question_text",
    "ocr_question_crop_failed",
}
CROP_UNCERTAIN_FLAGS = {
    "crop_uncertain",
    "low_confidence_question_crop",
    "crop_reaches_page_margin",
    "crop_fallback_failed",
    "missing_image_detection_failure",
}
VISUAL_DEPENDENCY_FLAGS = {
    "contains_graph_or_diagram_prompt",
    "contains_table_or_data_prompt",
    "contains_inequality_or_region_prompt",
    "diagram_required",
    "image_required",
}
MARKSCHEME_WEAK_FLAGS = {
    "legacy_markscheme_segmentation",
    "markscheme_relaxed_anchor_detection",
}
FORCED_NO_RULE_FLAGS = {
    "topic_forced_no_rule_match",
    "topic_forced_low_confidence",
    "topic_uncertain_no_rule_match",
}
CLOSE_SCORE_FLAGS = {"topic_close_score"}
STRUCTURAL_FAILURE_FLAGS = {
    "ocr_large_margin_blocked_by_structural_rejection",
    "ocr_large_margin_structural_rejection",
}

IMPORTANT_FLAG_FIELDS = {
    "ocr_hint_figure_regions": ("flag", "ocr_hint_figure_regions"),
    "crop_uncertain": ("flag", "crop_uncertain"),
    "low_confidence_question_crop": ("flag", "low_confidence_question_crop"),
    "text_only_status=review": ("field", "text_only_status", "review"),
    "question_text_trust=medium": ("field", "question_text_trust", "medium"),
    "legacy_markscheme_segmentation": ("flag", "legacy_markscheme_segmentation"),
    "markscheme_relaxed_anchor_detection": ("flag", "markscheme_relaxed_anchor_detection"),
    "topic_forced_no_rule_match": ("flag", "topic_forced_no_rule_match"),
    "topic_close_score": ("flag", "topic_close_score"),
    "ocr_large_margin_blocked_by_structural_rejection": (
        "flag",
        "ocr_large_margin_blocked_by_structural_rejection",
    ),
}

FLAG_FIELDS = (
    "review_flags",
    "extraction_quality_flags",
    "validation_flags",
    "visual_reason_flags",
    "text_fidelity_flags",
    "ocr_rejected_reasons",
    "text_candidate_decision_reasons",
)


def add_topic_confidence_rescoring_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_QUESTION_BANK_PATH, help="Path to question_bank.json.")
    parser.add_argument(
        "--topic-routing",
        type=Path,
        default=DEFAULT_TOPIC_ROUTING_PATH,
        help="Optional strict topic-routing sidecar used as additional route/review evidence.",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_ROOT,
        help="Canonical taxonomy root used to map strict topic route IDs to local topic slugs.",
    )
    parser.add_argument("--config", default="config.yaml", help="Optional config.yaml path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for report outputs.")


def run_topic_confidence_rescoring(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rescore topic confidence with typed extraction/review flag handling.")
    add_topic_confidence_rescoring_cli_arguments(parser)
    args = parser.parse_args(argv)
    return run_topic_confidence_rescoring_from_args(args)


def run_topic_confidence_rescoring_from_args(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    question_bank = read_json(args.input)
    topic_routing = read_optional_json(args.topic_routing)
    report = build_topic_confidence_rescoring_report(
        question_bank,
        config=config,
        topic_routing_payload=topic_routing,
        taxonomy_root=args.taxonomy,
        input_path=args.input,
        topic_routing_path=args.topic_routing if topic_routing is not None else None,
    )
    outputs = write_topic_confidence_rescoring_outputs(report, args.out_dir)
    summary = report["summary"]
    print(
        "Topic confidence rescoring: "
        f"{summary['total_records']} records, "
        f"low {summary['rescored_distribution']['low']['count']} "
        f"({summary['rescored_distribution']['low']['percentage']}%), "
        f"promoted low->medium {summary['promoted_low_to_medium']}, "
        f"low->high {summary['promoted_low_to_high']}."
    )
    print(f"Wrote review pack to {outputs['remaining_low_review_pack_json']}")
    return 0


def build_topic_confidence_rescoring_report(
    question_bank_payload: dict[str, Any],
    *,
    config: AppConfig | None = None,
    topic_routing_payload: dict[str, Any] | None = None,
    taxonomy_root: str | Path = DEFAULT_TAXONOMY_ROOT,
    input_path: str | Path | None = None,
    topic_routing_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    config = config or AppConfig()
    records = _question_bank_records(question_bank_payload)
    topic_routes = _topic_routes_by_question_id(topic_routing_payload or {})
    topic_slug_map = load_topic_slug_map(taxonomy_root) if topic_routes else {}

    rescore_records = [
        rescore_topic_confidence_record(
            record,
            config=config,
            topic_route=topic_routes.get(_text(record.get("question_id"))),
            topic_slug_map=topic_slug_map,
        )
        for record in records
    ]
    review_pack = [
        remaining_low_review_record(row)
        for row in rescore_records
        if row["topic_confidence_rescored"] == "low"
    ]
    summary = build_rescoring_summary(records, rescore_records)
    return {
        "schema_name": RESCORE_SCHEMA_NAME,
        "schema_version": RESCORE_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "question_bank": str(input_path) if input_path else None,
            "topic_routing": str(topic_routing_path) if topic_routing_path else None,
            "taxonomy_root": str(taxonomy_root),
        },
        "summary": summary,
        "records": rescore_records,
        "remaining_low_review_pack": review_pack,
    }


def rescore_topic_confidence_record(
    record: dict[str, Any],
    *,
    config: AppConfig | None = None,
    topic_route: dict[str, Any] | None = None,
    topic_slug_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    config = config or AppConfig()
    topic_slug_map = topic_slug_map or {}
    flags = _record_flags(record)
    typed = typed_confidence_flags(record, flags=flags)
    original = _confidence(_field(record, "topic_confidence"))
    current_topic = _text(record.get("topic"))
    classification = classify_question(
        _text(record.get("question_text")),
        _int_value(record.get("question_solution_marks")),
        config,
        context_flags=[],
        source_name=_source_name(record),
        mark_scheme_text=_text(record.get("mark_scheme_text")),
        question_ocr_text=_text(record.get("ocr_text")),
    )
    basis = topic_score_basis(
        record,
        classification,
        config=config,
        topic_route=topic_route,
        topic_slug_map=topic_slug_map,
    )
    hard_low_reasons = hard_low_confidence_reasons(record, typed, topic_route=topic_route)
    confidence_blockers = soft_confidence_blockers(record, typed, basis)
    promotion_basis = promotion_basis_reasons(basis, topic_route=topic_route)

    computed = computed_rescored_confidence(
        hard_low_reasons=hard_low_reasons,
        confidence_blockers=confidence_blockers,
        promotion_basis=promotion_basis,
        basis=basis,
    )
    if original != "low" and not hard_low_reasons:
        rescored = original
    elif original != "low" and hard_low_reasons:
        rescored = "low"
    else:
        rescored = computed

    effective_hard_low_reasons = list(hard_low_reasons)
    if rescored == "low" and not effective_hard_low_reasons:
        if "question_text_trust_unusable" in confidence_blockers:
            effective_hard_low_reasons = ["unusable_question_text_trust"]
        else:
            effective_hard_low_reasons = ["no_strong_topic_evidence"]
    confidence_changed = rescored != original
    change_reason = confidence_change_reason(
        original=original,
        rescored=rescored,
        hard_low_reasons=effective_hard_low_reasons,
        confidence_blockers=confidence_blockers,
        promotion_basis=promotion_basis,
    )
    return {
        "question_id": _text(record.get("question_id")),
        "paper": _text(record.get("paper")),
        "paper_family": _text(record.get("paper_family")),
        "session": _text(_field(record, "canonical_session")),
        "source": _source_name(record),
        "question_number": _text(record.get("question_number")),
        "current_topic_candidate": current_topic,
        "topic_confidence_original": original,
        "topic_confidence_rescored": rescored,
        "confidence_changed": confidence_changed,
        "confidence_change_reason": change_reason,
        "confidence_blockers": confidence_blockers,
        "benign_flags_ignored": typed["benign_visual_ocr_flag"],
        "hard_low_reasons": effective_hard_low_reasons if rescored == "low" else [],
        "promotion_basis": promotion_basis if CONFIDENCE_RANK[rescored] > CONFIDENCE_RANK[original] else [],
        "typed_flag_categories": typed,
        "topic_score": basis["current_topic_score"],
        "topic_score_margin_from_top": basis["current_topic_margin_from_top"],
        "classifier_top_topic": basis["classifier_top_topic"],
        "classifier_top_score": basis["classifier_top_score"],
        "classifier_top_margin": basis["classifier_top_margin"],
        "classifier_topic": basis["classifier_topic"],
        "classifier_topic_confidence": basis["classifier_topic_confidence"],
        "topic_route_confidence": basis.get("topic_route_confidence"),
        "topic_route_primary_topic_id": basis.get("topic_route_primary_topic_id"),
        "topic_route_topic_slug": basis.get("topic_route_topic_slug"),
        "topic_route_agrees_with_current": basis.get("topic_route_agrees_with_current"),
        "question_text_excerpt": _excerpt(_text(record.get("question_text"))),
        "flags": sorted(flags),
        "question_image_path": _text(record.get("question_image_path") or record.get("canonical_question_artifact")),
        "mark_scheme_image_path": _text(record.get("mark_scheme_image_path") or record.get("canonical_mark_scheme_artifact")),
    }


def computed_rescored_confidence(
    *,
    hard_low_reasons: list[str],
    confidence_blockers: list[str],
    promotion_basis: list[str],
    basis: dict[str, Any],
) -> str:
    if hard_low_reasons:
        return "low"
    if "question_text_trust_unusable" in confidence_blockers:
        return "low"
    if not promotion_basis:
        return "low"
    strong_clear = bool(basis["strong_clear_topic_evidence"])
    if strong_clear and not confidence_blockers:
        return "high"
    return "medium"


def hard_low_confidence_reasons(
    record: dict[str, Any],
    typed: dict[str, list[str]],
    *,
    topic_route: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    question_text = _text(record.get("question_text"))
    if not question_text or len(question_text) < 20:
        reasons.append("missing_or_truncated_question_text")
    if typed["structural_failure_flag"]:
        reasons.append("structural_failure_flag")
    if any(flag in typed["true_ambiguity_flag"] for flag in FORCED_NO_RULE_FLAGS):
        reasons.append("topic_forced_no_rule_match")
    if not _text(record.get("question_image_path") or record.get("canonical_question_artifact")):
        reasons.append("missing_question_crop")
    if topic_route and topic_route.get("review_required") is True:
        reasons.append("topic_route_review_required")
    return sorted(set(reasons))


def soft_confidence_blockers(record: dict[str, Any], typed: dict[str, list[str]], basis: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    text_only_status = _status(_field(record, "text_only_status"))
    text_trust = _status(_field(record, "question_text_trust"))
    if typed["crop_uncertain_flag"]:
        blockers.append("crop_uncertain_flag")
    if typed["visual_dependency_flag"] or bool(_field(record, "visual_required")) or text_only_status in {"review", "fail"}:
        blockers.append("visual_dependency_flag")
    if text_trust in {"medium", "low", "unusable"}:
        blockers.append(f"question_text_trust_{text_trust}")
    if "topic_close_score" in typed["true_ambiguity_flag"]:
        blockers.append("topic_close_score")
    if typed["markscheme_weak_flag"] and basis["topic_uses_markscheme_evidence"]:
        blockers.append("markscheme_weak_flag_used_for_topic")
    if basis["classifier_disagrees_with_current"] and basis["current_topic_score"] > 0:
        blockers.append("classifier_top_topic_differs")
    return sorted(set(blockers))


def promotion_basis_reasons(
    basis: dict[str, Any],
    *,
    topic_route: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if basis["classifier_same_topic"] and basis["classifier_topic_confidence"] in {"high", "medium"}:
        reasons.append(f"classifier_same_topic_{basis['classifier_topic_confidence']}")
    if basis["strong_clear_topic_evidence"]:
        reasons.append("strong_clear_current_topic_score")
    elif basis["strong_current_topic_evidence"]:
        reasons.append("strong_current_topic_score")
    elif basis["decent_current_topic_evidence"]:
        reasons.append("decent_current_topic_score")
    if topic_route and basis.get("topic_route_agrees_with_current") and basis.get("topic_route_confidence") in {"high", "medium"}:
        reasons.append(f"strict_topic_route_agrees_{basis['topic_route_confidence']}")
    return sorted(set(reasons))


def topic_score_basis(
    record: dict[str, Any],
    classification: Any,
    *,
    config: AppConfig,
    topic_route: dict[str, Any] | None,
    topic_slug_map: dict[str, str],
) -> dict[str, Any]:
    current_topic = _text(record.get("topic"))
    breakdown = classification.topic_evidence_details.get("topic_score_breakdown", {})
    scores = sorted(
        (
            (topic, _float_value((data or {}).get("final_score")))
            for topic, data in breakdown.items()
            if isinstance(data, dict)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    classifier_top_topic = scores[0][0] if scores else ""
    classifier_top_score = scores[0][1] if scores else 0.0
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    current_topic_score = _float_value((breakdown.get(current_topic) or {}).get("final_score")) if current_topic else 0.0
    margin_from_top = round(classifier_top_score - current_topic_score, 3)
    classifier_top_margin = round(classifier_top_score - second_score, 3)
    classifier_same_topic = _text(classification.topic) == current_topic
    classifier_disagrees = bool(current_topic and classification.topic and _text(classification.topic) != current_topic)

    route_topic_id = _text((topic_route or {}).get("primary_topic_id"))
    route_slug = topic_slug_map.get(route_topic_id, _topic_slug_from_id(route_topic_id)) if route_topic_id else ""
    route_confidence = _confidence((topic_route or {}).get("confidence")) if topic_route else None
    route_agrees = bool(route_slug and route_slug == current_topic)

    strong_clear = (
        (classifier_same_topic and classification.topic_confidence == "high")
        or (classifier_top_topic == current_topic and current_topic_score >= 12.0 and classifier_top_margin >= 3.0)
        or (route_agrees and route_confidence == "high")
    )
    strong_current = current_topic_score >= 12.0 and margin_from_top <= 5.0
    decent_current = (
        (classifier_same_topic and classification.topic_confidence in {"high", "medium"})
        or (current_topic_score >= 5.0 and margin_from_top <= 20.0)
        or (route_agrees and route_confidence in {"high", "medium"})
    )
    return {
        "classifier_topic": _text(classification.topic),
        "classifier_topic_confidence": _confidence(classification.topic_confidence),
        "classifier_same_topic": classifier_same_topic,
        "classifier_disagrees_with_current": classifier_disagrees,
        "classifier_top_topic": classifier_top_topic,
        "classifier_top_score": round(classifier_top_score, 3),
        "classifier_top_margin": classifier_top_margin,
        "current_topic_score": round(current_topic_score, 3),
        "current_topic_margin_from_top": margin_from_top,
        "strong_clear_topic_evidence": strong_clear,
        "strong_current_topic_evidence": strong_current,
        "decent_current_topic_evidence": decent_current,
        "topic_uses_markscheme_evidence": topic_depends_on_markscheme(record, current_topic, full_current_score=current_topic_score, config=config),
        "topic_route_confidence": route_confidence,
        "topic_route_primary_topic_id": route_topic_id,
        "topic_route_topic_slug": route_slug,
        "topic_route_agrees_with_current": route_agrees,
    }


def topic_depends_on_markscheme(
    record: dict[str, Any],
    current_topic: str,
    *,
    full_current_score: float,
    config: AppConfig,
) -> bool:
    if not current_topic or full_current_score < 5.0:
        return False
    text_only = classify_question(
        _text(record.get("question_text")),
        _int_value(record.get("question_solution_marks")),
        config,
        context_flags=[],
        source_name=_source_name(record),
        mark_scheme_text="",
        question_ocr_text=_text(record.get("ocr_text")),
    )
    breakdown = text_only.topic_evidence_details.get("topic_score_breakdown", {})
    text_score = _float_value((breakdown.get(current_topic) or {}).get("final_score"))
    return text_score < 5.0


def typed_confidence_flags(record: dict[str, Any], *, flags: set[str] | None = None) -> dict[str, list[str]]:
    flags = flags or _record_flags(record)
    visual_dependency = set(flags) & VISUAL_DEPENDENCY_FLAGS
    if bool(_field(record, "visual_required")):
        visual_dependency.add("visual_required")
    text_only_status = _status(_field(record, "text_only_status"))
    if text_only_status == "review":
        visual_dependency.add("text_only_status=review")
    elif text_only_status == "fail":
        visual_dependency.add("text_only_status=fail")
    question_text_weak = set(flags) & QUESTION_TEXT_WEAK_FLAGS
    text_trust = _status(_field(record, "question_text_trust"))
    if text_trust in {"low", "unusable"}:
        question_text_weak.add(f"question_text_trust={text_trust}")
    return {
        "benign_visual_ocr_flag": sorted(set(flags) & BENIGN_VISUAL_OCR_FLAGS),
        "question_text_weak_flag": sorted(question_text_weak),
        "crop_uncertain_flag": sorted(set(flags) & CROP_UNCERTAIN_FLAGS),
        "visual_dependency_flag": sorted(visual_dependency),
        "markscheme_weak_flag": sorted(set(flags) & MARKSCHEME_WEAK_FLAGS),
        "true_ambiguity_flag": sorted((set(flags) & FORCED_NO_RULE_FLAGS) | (set(flags) & CLOSE_SCORE_FLAGS)),
        "structural_failure_flag": sorted(set(flags) & STRUCTURAL_FAILURE_FLAGS),
    }


def confidence_change_reason(
    *,
    original: str,
    rescored: str,
    hard_low_reasons: list[str],
    confidence_blockers: list[str],
    promotion_basis: list[str],
) -> str:
    if original == rescored:
        return "unchanged"
    if rescored == "low":
        return "demoted_to_low:" + ",".join(hard_low_reasons or ["no_strong_topic_evidence"])
    if original == "low":
        basis = promotion_basis[0] if promotion_basis else "recoverable_low"
        cap = f"; capped_by={','.join(confidence_blockers)}" if confidence_blockers else ""
        return f"promoted_low_to_{rescored}:{basis}{cap}"
    return f"rescored_{original}_to_{rescored}"


def build_rescoring_summary(source_records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    original_counts = Counter(row["topic_confidence_original"] for row in rows)
    rescored_counts = Counter(row["topic_confidence_rescored"] for row in rows)
    remaining_low = [row for row in rows if row["topic_confidence_rescored"] == "low"]
    promoted_rows = [
        row
        for row in rows
        if row["topic_confidence_original"] == "low"
        and CONFIDENCE_RANK[row["topic_confidence_rescored"]] > CONFIDENCE_RANK["low"]
    ]
    low_reasons = Counter()
    fix_categories = Counter()
    promotion_reasons = Counter()
    for row in remaining_low:
        for reason in row["hard_low_reasons"] or ["unknown"]:
            low_reasons[reason] += 1
        fix_categories[suggested_next_fix_category(row)] += 1
    for row in promoted_rows:
        for reason in row["promotion_basis"] or [row["confidence_change_reason"]]:
            promotion_reasons[reason] += 1
        for blocker in row["confidence_blockers"]:
            if blocker in {"topic_close_score", "classifier_top_topic_differs", "markscheme_weak_flag_used_for_topic"}:
                promotion_reasons[f"medium_cap:{blocker}"] += 1
    return {
        "total_records": total,
        "original_distribution": confidence_distribution(original_counts, total),
        "rescored_distribution": confidence_distribution(rescored_counts, total),
        "promoted_low_to_medium": sum(1 for row in promoted_rows if row["topic_confidence_rescored"] == "medium"),
        "promoted_low_to_high": sum(1 for row in promoted_rows if row["topic_confidence_rescored"] == "high"),
        "remaining_low_count": len(remaining_low),
        "remaining_low_percentage": _percentage(len(remaining_low), total),
        "top_20_remaining_low_reasons": dict(low_reasons.most_common(20)),
        "remaining_low_fix_category_counts": dict(fix_categories.most_common()),
        "top_20_promotion_reasons": dict(promotion_reasons.most_common(20)),
        "important_flag_counts": important_flag_counts(source_records),
        "risky_promotion_counts": risky_promotion_counts(promoted_rows),
    }


def confidence_distribution(counts: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        value: {"count": int(counts.get(value, 0)), "percentage": _percentage(int(counts.get(value, 0)), total)}
        for value in ("high", "medium", "low")
    }


def important_flag_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, rule in IMPORTANT_FLAG_FIELDS.items():
        if rule[0] == "flag":
            flag = rule[1]
            counts[label] = sum(1 for record in records if flag in _record_flags(record))
        else:
            _kind, field, expected = rule
            counts[label] = sum(1 for record in records if _status(_field(record, field)) == expected)
    return counts


def risky_promotion_counts(promoted_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in promoted_rows:
        for blocker in row["confidence_blockers"]:
            if blocker in {"classifier_top_topic_differs", "topic_close_score", "question_text_trust_low", "question_text_trust_unusable"}:
                counter[blocker] += 1
        if float(row.get("topic_score_margin_from_top") or 0.0) > 8.0:
            counter["wide_classifier_margin_against_current_topic"] += 1
    return dict(counter.most_common())


def remaining_low_review_record(row: dict[str, Any]) -> dict[str, Any]:
    reason = ";".join(row["hard_low_reasons"] or ["no_strong_topic_evidence"])
    return {
        "question_id": row["question_id"],
        "paper": row["paper"],
        "session": row["session"],
        "source": row["source"],
        "current_topic_candidate": row["current_topic_candidate"],
        "topic_score": row["topic_score"],
        "topic_score_margin_from_top": row["topic_score_margin_from_top"],
        "classifier_top_topic": row["classifier_top_topic"],
        "classifier_top_score": row["classifier_top_score"],
        "question_text_excerpt": row["question_text_excerpt"],
        "flags": row["flags"],
        "crop_path": row["question_image_path"],
        "mark_scheme_path": row["mark_scheme_image_path"],
        "remaining_low_reason": reason,
        "suggested_next_fix_category": suggested_next_fix_category(row),
    }


def suggested_next_fix_category(row: dict[str, Any]) -> str:
    reasons = set(row["hard_low_reasons"])
    blockers = set(row["confidence_blockers"])
    flags = set(row["flags"])
    if "structural_failure_flag" in reasons:
        return "structural_extraction_failure"
    if "missing_or_truncated_question_text" in reasons or "unusable_question_text_trust" in reasons:
        return "fix_ocr_text"
    if "topic_forced_no_rule_match" in reasons or "no_strong_topic_evidence" in reasons:
        return "add_topic_rule"
    if "topic_route_review_required" in reasons:
        return "visual_required_manual_review" if "visual_dependency_flag" in blockers else "resolve_close_topic"
    if "topic_close_score" in blockers or "topic_close_score" in flags:
        return "resolve_close_topic"
    if "missing_question_crop" in reasons or "crop_uncertain_flag" in blockers:
        return "fix_question_crop"
    if "visual_dependency_flag" in blockers:
        return "visual_required_manual_review"
    if "markscheme_weak_flag_used_for_topic" in blockers:
        return "fix_markscheme_dependency"
    return "add_topic_rule"


def write_topic_confidence_rescoring_outputs(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_json": out_dir / "topic_confidence_rescoring_report.json",
        "summary_json": out_dir / "topic_confidence_rescoring_summary.json",
        "summary_markdown": out_dir / "topic_confidence_rescoring_summary.md",
        "records_csv": out_dir / "topic_confidence_rescoring_records.csv",
        "remaining_low_review_pack_json": out_dir / "remaining_low_review_pack.json",
        "remaining_low_review_pack_csv": out_dir / "remaining_low_review_pack.csv",
    }
    write_json(paths["report_json"], report)
    write_json(paths["summary_json"], report["summary"])
    paths["summary_markdown"].write_text(render_summary_markdown(report), encoding="utf-8")
    write_csv(paths["records_csv"], report["records"])
    write_json(paths["remaining_low_review_pack_json"], {"records": report["remaining_low_review_pack"]})
    write_csv(paths["remaining_low_review_pack_csv"], report["remaining_low_review_pack"])
    return {key: str(path) for key, path in paths.items()}


def render_summary_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Topic Confidence Rescoring Summary",
        "",
        f"- Total records: `{summary['total_records']}`",
        f"- Original confidence: `{json.dumps(summary['original_distribution'], sort_keys=True)}`",
        f"- Rescored confidence: `{json.dumps(summary['rescored_distribution'], sort_keys=True)}`",
        f"- Promoted low->medium: `{summary['promoted_low_to_medium']}`",
        f"- Promoted low->high: `{summary['promoted_low_to_high']}`",
        f"- Remaining low: `{summary['remaining_low_count']}` (`{summary['remaining_low_percentage']}%`)",
        "",
        "## Remaining Low Reasons",
        *[f"- `{reason}`: `{count}`" for reason, count in summary["top_20_remaining_low_reasons"].items()],
        "",
        "## Remaining Low Fix Categories",
        *[f"- `{category}`: `{count}`" for category, count in summary["remaining_low_fix_category_counts"].items()],
        "",
        "## Promotion Reasons",
        *[f"- `{reason}`: `{count}`" for reason, count in summary["top_20_promotion_reasons"].items()],
        "",
        "## Important Flag Counts",
        *[f"- `{flag}`: `{count}`" for flag, count in summary["important_flag_counts"].items()],
    ]
    return "\n".join(lines) + "\n"


def load_topic_slug_map(taxonomy_root: str | Path) -> dict[str, str]:
    root = Path(taxonomy_root)
    mapping: dict[str, str] = {}
    for path in sorted((root / "topic_filter_maps").glob("*.json")):
        payload = read_json(path)
        for topic in payload.get("topics") or []:
            if not isinstance(topic, dict):
                continue
            topic_id = _text(topic.get("topic_id"))
            if topic_id:
                mapping[topic_id] = _topic_slug_from_id(topic_id)
    return mapping


def _topic_slug_from_id(topic_id: str) -> str:
    return re.sub(r"^9709_[a-z0-9]+_topic_", "", topic_id)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return read_json(candidate)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _question_bank_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("questions")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    if isinstance(records, dict):
        return [record for record in records.values() if isinstance(record, dict)]
    return []


def _topic_routes_by_question_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if isinstance(records, dict):
        return {str(question_id): row for question_id, row in records.items() if isinstance(row, dict)}
    if isinstance(records, list):
        return {_text(row.get("question_id")): row for row in records if isinstance(row, dict) and _text(row.get("question_id"))}
    return {}


def _field(record: dict[str, Any], name: str) -> Any:
    if name in record:
        return record.get(name)
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(name)
    return None


def _record_flags(record: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    for field in FLAG_FIELDS:
        for value in _list_values(_field(record, field)):
            for part in str(value).split(";"):
                text = part.strip()
                if text:
                    flags.add(text)
    return flags


def _list_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _source_name(record: dict[str, Any]) -> str:
    return _text(_field(record, "source_pdf") or record.get("source_pdf") or record.get("paper"))


def _confidence(value: Any) -> str:
    text = _status(value)
    return text if text in CONFIDENCE_VALUES else "low"


def _status(value: Any) -> str:
    return _text(value).strip().lower()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _int_value(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100.0, 3)


def _excerpt(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


if __name__ == "__main__":
    raise SystemExit(run_topic_confidence_rescoring())
