from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TEXT_ONLY_ORDER = {"fail": 0, "review": 1, "ready": 2}
FIDELITY_ORDER = {"unusable": 0, "degraded": 1, "clean": 2}

HARD_CORRUPTION_FLAGS = {
    "math_text_corruption_detected",
    "ocr_math_notation_degraded",
    "ocr_noise_fragment_present",
    "pdf_control_garbage_detected",
    "hybrid_math_text_requires_review",
    "missing_visible_structure_in_text",
    "contains_math_text_corruption",
    "contains_pdf_control_garbage",
    "text_order_unreliable",
    "likely_needs_visual_review",
    "math_corruption_suspected",
    "broken_fraction_structure",
    "broken_superscript_or_power",
    "suspicious_symbol_run",
    "diagram_text_mixed_with_body",
    "native_compacted_math_corruption",
    "symbol_loss_detected",
    "malformed_unit_notation",
    "contains_native_compacted_math_corruption",
    "contains_symbol_loss",
    "contains_unit_corruption",
}
DIAGRAM_TABLE_FLAGS = {
    "contains_graph_or_diagram_prompt",
    "contains_table_or_data_prompt",
}
MATH_LAYOUT_FLAGS = {
    "contains_equation_layout",
    "contains_fraction_or_integral_layout",
    "contains_vector_notation",
    "contains_complex_number_notation",
    "contains_inequality_or_region_prompt",
    "contains_trig_expression",
    "contains_log_exponential_expression",
    "contains_flattened_math_structure",
}
OCR_STRUCTURAL_REJECTION_REASONS = {
    "ocr_missing_question_number",
    "ocr_lost_mark_brackets",
    "ocr_missing_subpart_labels",
    "ocr_lost_math_structure",
    "ocr_lost_visual_dependency_prompt",
    "ocr_lost_function_structure",
    "ocr_lost_greek_symbol",
    "ocr_lost_radical_structure",
    "ocr_lost_unit_structure",
    "ocr_introduced_unit_corruption",
    "ocr_introduced_symbol_loss",
    "ocr_introduced_compacted_math_corruption",
    "ocr_introduced_flattened_math_structure",
    "page_furniture_or_header_text",
    "next_question_contamination",
}
NEW_DETECTOR_FLAGS = {
    "native_compacted_math_corruption": {
        "native_compacted_math_corruption",
        "contains_native_compacted_math_corruption",
        "ocr_introduced_compacted_math_corruption",
    },
    "merged_diagram_table_prompt": {
        "contains_graph_or_diagram_prompt",
        "contains_table_or_data_prompt",
        "ocr_lost_visual_dependency_prompt",
    },
    "flattened_math_structure": {
        "flattened_math_structure",
        "contains_flattened_math_structure",
        "ocr_introduced_flattened_math_structure",
        "ocr_lost_function_structure",
        "ocr_lost_math_structure",
    },
    "symbol_loss": {
        "symbol_loss_detected",
        "contains_symbol_loss",
        "ocr_introduced_symbol_loss",
        "ocr_lost_greek_symbol",
        "ocr_lost_radical_structure",
    },
    "unit_corruption": {
        "malformed_unit_notation",
        "contains_unit_corruption",
        "ocr_introduced_unit_corruption",
        "ocr_lost_unit_structure",
    },
    "ocr_large_margin_structural_rejection": {
        "ocr_large_margin_blocked_by_structural_rejection",
        "ocr_large_margin_structural_rejection",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit text-confidence recovery between two question-bank exports.")
    parser.add_argument("--before", required=True, help="Baseline question_bank.json path.")
    parser.add_argument("--after", required=True, help="Candidate question_bank.json path.")
    parser.add_argument("--out-dir", required=True, help="Directory for audit outputs.")
    args = parser.parse_args()

    before_payload = load_payload(Path(args.before))
    after_payload = load_payload(Path(args.after))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(before_payload, after_payload)
    write_json(out_dir / "text_confidence_recovery_audit.json", report)
    write_csv(out_dir / "text_confidence_grouped_audit.csv", report["grouped_audit_rows"])
    write_csv(out_dir / "text_confidence_failure_type_audit.csv", report["failure_type_rows"])
    write_csv(out_dir / "text_confidence_promoted_records.csv", report["promoted_records"])
    write_csv(out_dir / "text_confidence_status_movements.csv", report["status_movement_rows"])
    write_csv(out_dir / "text_confidence_detector_blocks.csv", report["blocked_from_ready_by_detector_rows"])
    write_csv(out_dir / "text_confidence_ocr_decision_changes.csv", report["ocr_decision_changes"])
    write_markdown(out_dir / "text_confidence_recovery_audit.md", render_markdown(report))
    return 0


def build_report(before_payload: dict[str, Any], after_payload: dict[str, Any]) -> dict[str, Any]:
    before_records = list(before_payload.get("questions") or [])
    after_records = list(after_payload.get("questions") or [])
    before_by_id = records_by_id(before_records)
    after_by_id = records_by_id(after_records)
    shared_ids = sorted(set(before_by_id) & set(after_by_id))
    added_ids = sorted(set(after_by_id) - set(before_by_id))
    removed_ids = sorted(set(before_by_id) - set(after_by_id))

    promoted_records = [
        promoted_record(before_by_id[qid], after_by_id[qid])
        for qid in shared_ids
        if text_only_improved(before_by_id[qid], after_by_id[qid])
    ]
    failure_type_rows = build_failure_type_rows(before_records, after_records)
    grouped_rows = build_grouped_rows(before_records, after_records, promoted_records)
    guardrails = build_guardrails(before_records, after_records, promoted_records)
    movement = build_movement_rows(before_by_id, after_by_id, shared_ids)
    blocked_by_detector = build_blocked_by_detector(before_by_id, after_by_id, shared_ids)
    ocr_decision_changes = build_ocr_decision_changes(before_by_id, after_by_id, shared_ids)

    return {
        "audit_version": "text_confidence_recovery_v1",
        "before_record_count": len(before_records),
        "after_record_count": len(after_records),
        "shared_record_count": len(shared_ids),
        "added_record_ids": added_ids,
        "removed_record_ids": removed_ids,
        "before_counts": corpus_counts(before_records),
        "after_counts": corpus_counts(after_records),
        "target_pool": {
            "before_count": len([record for record in before_records if in_primary_target_pool(record)]),
            "after_count": len([record for record in after_records if in_primary_target_pool(record)]),
            "promoted_count": sum(1 for row in promoted_records if row["primary_target_pool_before"]),
        },
        "promoted_record_count": len(promoted_records),
        "promoted_records": promoted_records,
        "guardrails": guardrails,
        "status_movement_counts": movement["counts"],
        "status_movement_rows": movement["rows"],
        "blocked_from_ready_by_detector_counts": blocked_by_detector["counts"],
        "blocked_from_ready_by_detector_rows": blocked_by_detector["rows"],
        "ocr_decision_change_count": len(ocr_decision_changes),
        "ocr_decision_changes": ocr_decision_changes,
        "failure_type_counts": {
            "before": failure_type_counts(before_records),
            "after": failure_type_counts(after_records),
        },
        "grouped_audit_rows": grouped_rows,
        "failure_type_rows": failure_type_rows,
    }


def corpus_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "text_only_status": count_values(records, "text_only_status"),
        "text_fidelity_status": count_values(records, "text_fidelity_status"),
        "validation_status": count_values(records, "validation_status"),
        "mapping_status": count_values(records, "mapping_status"),
        "scope_quality_status": count_values(records, "scope_quality_status"),
        "ocr_ran_count": sum(1 for record in records if as_bool(value(record, "ocr_ran"))),
        "ocr_selected_count": sum(1 for record in records if as_bool(value(record, "ocr_selected"))),
        "missing_question_image_path_count": sum(1 for record in records if not value(record, "question_image_path")),
        "missing_mark_scheme_image_path_count": sum(1 for record in records if not value(record, "mark_scheme_image_path")),
    }


def build_guardrails(
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
    promoted_records: list[dict[str, Any]],
) -> dict[str, Any]:
    before_counts = corpus_counts(before_records)
    after_counts = corpus_counts(after_records)
    before_validation_fail = before_counts["validation_status"].get("fail", 0)
    after_validation_fail = after_counts["validation_status"].get("fail", 0)
    before_mapping_fail = before_counts["mapping_status"].get("fail", 0)
    after_mapping_fail = after_counts["mapping_status"].get("fail", 0)
    before_missing_images = (
        before_counts["missing_question_image_path_count"] + before_counts["missing_mark_scheme_image_path_count"]
    )
    after_missing_images = after_counts["missing_question_image_path_count"] + after_counts["missing_mark_scheme_image_path_count"]
    math_corrupt_ready = [
        row["question_id"]
        for row in promoted_records
        if row["after_text_only_status"] == "ready"
        and any(item in row["after_failure_type_flags"] for item in {"hard_text_corruption", "flattened_display_math"})
    ]
    diagram_table_ready = [
        row["question_id"]
        for row in promoted_records
        if row["after_text_only_status"] == "ready"
        and any(item in row["after_failure_type_flags"] for item in {"diagram_table_dependent_but_readable_text"})
    ]
    return {
        "validation_status_fail_delta": after_validation_fail - before_validation_fail,
        "mapping_status_fail_delta": after_mapping_fail - before_mapping_fail,
        "missing_image_path_delta": after_missing_images - before_missing_images,
        "known_math_corruption_promoted_to_ready_count": len(math_corrupt_ready),
        "known_math_corruption_promoted_to_ready_ids": math_corrupt_ready,
        "diagram_table_promoted_to_ready_count": len(diagram_table_ready),
        "diagram_table_promoted_to_ready_ids": diagram_table_ready,
        "passes_acceptance_guardrails": (
            after_validation_fail <= before_validation_fail
            and after_mapping_fail <= before_mapping_fail
            and after_missing_images <= before_missing_images
            and not math_corrupt_ready
            and not diagram_table_ready
        ),
    }


def build_movement_rows(
    before_by_id: dict[str, dict[str, Any]],
    after_by_id: dict[str, dict[str, Any]],
    shared_ids: list[str],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for qid in shared_ids:
        before = before_by_id[qid]
        after = after_by_id[qid]
        before_status = str(value(before, "text_only_status") or "missing")
        after_status = str(value(after, "text_only_status") or "missing")
        movement = f"{before_status}->{after_status}"
        counts[movement] += 1
        if before_status != after_status or value(before, "text_fidelity_status") != value(after, "text_fidelity_status"):
            rows.append(
                {
                    "question_id": qid,
                    "paper": value(after, "paper"),
                    "before_text_only_status": before_status,
                    "after_text_only_status": after_status,
                    "before_text_fidelity_status": value(before, "text_fidelity_status"),
                    "after_text_fidelity_status": value(after, "text_fidelity_status"),
                    "detectors": detector_hits(after),
                    "ocr_changed": value(before, "text_candidate_source") != value(after, "text_candidate_source"),
                    "reason_flags": sorted(set(list_value(after, "text_fidelity_flags") + list_value(after, "visual_reason_flags"))),
                }
            )
    return {"counts": dict(sorted(counts.items())), "rows": rows}


def build_blocked_by_detector(
    before_by_id: dict[str, dict[str, Any]],
    after_by_id: dict[str, dict[str, Any]],
    shared_ids: list[str],
) -> dict[str, Any]:
    counts: Counter[str] = Counter({detector: 0 for detector in NEW_DETECTOR_FLAGS})
    rows: list[dict[str, Any]] = []
    for qid in shared_ids:
        before = before_by_id[qid]
        after = after_by_id[qid]
        if value(before, "text_only_status") != "ready" or value(after, "text_only_status") == "ready":
            continue
        hits = detector_hits(after)
        if not hits:
            continue
        for hit in hits:
            counts[hit] += 1
        rows.append(
            {
                "question_id": qid,
                "paper": value(after, "paper"),
                "before_text_only_status": value(before, "text_only_status"),
                "after_text_only_status": value(after, "text_only_status"),
                "after_text_fidelity_status": value(after, "text_fidelity_status"),
                "detectors": hits,
                "text_fidelity_flags": list_value(after, "text_fidelity_flags"),
                "visual_reason_flags": list_value(after, "visual_reason_flags"),
                "question_text_snippet": str(value(after, "question_text") or "")[:240],
            }
        )
    return {"counts": dict(sorted(counts.items())), "rows": rows}


def build_ocr_decision_changes(
    before_by_id: dict[str, dict[str, Any]],
    after_by_id: dict[str, dict[str, Any]],
    shared_ids: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in shared_ids:
        before = before_by_id[qid]
        after = after_by_id[qid]
        if value(before, "text_candidate_source") == value(after, "text_candidate_source") and as_bool(value(before, "ocr_selected")) == as_bool(
            value(after, "ocr_selected")
        ):
            continue
        rows.append(
            {
                "question_id": qid,
                "paper": value(after, "paper"),
                "before_text_candidate_source": value(before, "text_candidate_source"),
                "after_text_candidate_source": value(after, "text_candidate_source"),
                "before_ocr_selected": value(before, "ocr_selected"),
                "after_ocr_selected": value(after, "ocr_selected"),
                "native_text_score": value(after, "native_text_score"),
                "ocr_text_score": value(after, "ocr_text_score"),
                "after_ocr_rejected_reasons": list_value(after, "ocr_rejected_reasons"),
                "after_text_candidate_decision_reasons": list_value(after, "text_candidate_decision_reasons"),
            }
        )
    return rows


def build_grouped_rows(
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
    promoted_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    promoted_by_before_group: dict[tuple[str, str], int] = Counter()
    for row in promoted_records:
        before_snapshot = row["before_snapshot"]
        for dimension, group_value in group_values(before_snapshot):
            promoted_by_before_group[(dimension, group_value)] += 1

    snapshots: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(lambda: {"before": Counter(), "after": Counter()})
    for stage, records in [("before", before_records), ("after", after_records)]:
        for record in records:
            for dimension, group_value in group_values(record):
                bucket = snapshots[(dimension, group_value)][stage]
                bucket["record_count"] += 1
                bucket[f"text_only:{value(record, 'text_only_status') or 'missing'}"] += 1
                bucket[f"text_fidelity:{value(record, 'text_fidelity_status') or 'missing'}"] += 1
                if as_bool(value(record, "ocr_selected")):
                    bucket["ocr_selected"] += 1

    rows: list[dict[str, Any]] = []
    for (dimension, group_value), stage_counts in sorted(snapshots.items()):
        before = stage_counts["before"]
        after = stage_counts["after"]
        rows.append(
            {
                "dimension": dimension,
                "group_value": group_value,
                "before_record_count": before["record_count"],
                "after_record_count": after["record_count"],
                "before_text_only_fail": before["text_only:fail"],
                "after_text_only_fail": after["text_only:fail"],
                "before_text_only_review": before["text_only:review"],
                "after_text_only_review": after["text_only:review"],
                "before_text_only_ready": before["text_only:ready"],
                "after_text_only_ready": after["text_only:ready"],
                "before_text_fidelity_degraded": before["text_fidelity:degraded"],
                "after_text_fidelity_degraded": after["text_fidelity:degraded"],
                "before_text_fidelity_clean": before["text_fidelity:clean"],
                "after_text_fidelity_clean": after["text_fidelity:clean"],
                "before_ocr_selected": before["ocr_selected"],
                "after_ocr_selected": after["ocr_selected"],
                "promoted_from_before_group": promoted_by_before_group[(dimension, group_value)],
            }
        )
    return rows


def build_failure_type_rows(before_records: list[dict[str, Any]], after_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, records in [("before", before_records), ("after", after_records)]:
        for record in records:
            flags = failure_type_flags(record)
            for failure_type in flags:
                rows.append(
                    {
                        "stage": stage,
                        "failure_type": failure_type,
                        "question_id": value(record, "question_id"),
                        "paper": value(record, "paper"),
                        "paper_year": paper_year(record),
                        "paper_family": value(record, "paper_family"),
                        "question_format_profile": value(record, "question_format_profile"),
                        "text_only_status": value(record, "text_only_status"),
                        "text_fidelity_status": value(record, "text_fidelity_status"),
                        "selected_text_score": value(record, "selected_text_score"),
                        "selected_text_score_band": selected_text_score_band(record),
                        "text_candidate_decision": value(record, "text_candidate_decision"),
                        "ocr_selected": value(record, "ocr_selected"),
                        "ocr_rejected_reasons": list_value(record, "ocr_rejected_reasons"),
                        "text_fidelity_flags": list_value(record, "text_fidelity_flags"),
                        "review_flags": list_value(record, "review_flags"),
                    }
                )
    return rows


def promoted_record(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_flags = failure_type_flags(before)
    after_flags = failure_type_flags(after)
    return {
        "question_id": value(after, "question_id"),
        "paper": value(after, "paper"),
        "paper_year": paper_year(after),
        "paper_family": value(after, "paper_family"),
        "question_number": value(after, "question_number"),
        "question_format_profile": value(after, "question_format_profile"),
        "primary_target_pool_before": in_primary_target_pool(before),
        "before_text_only_status": value(before, "text_only_status"),
        "after_text_only_status": value(after, "text_only_status"),
        "before_text_fidelity_status": value(before, "text_fidelity_status"),
        "after_text_fidelity_status": value(after, "text_fidelity_status"),
        "before_question_text_trust": value(before, "question_text_trust"),
        "after_question_text_trust": value(after, "question_text_trust"),
        "after_visual_required": value(after, "visual_required"),
        "after_visual_reason_flags": list_value(after, "visual_reason_flags"),
        "before_failure_type_flags": before_flags,
        "after_failure_type_flags": after_flags,
        "selected_text_score": value(after, "selected_text_score"),
        "selected_text_score_band": selected_text_score_band(after),
        "text_candidate_decision": value(after, "text_candidate_decision"),
        "ocr_selected": value(after, "ocr_selected"),
        "ocr_rejected_reasons": list_value(after, "ocr_rejected_reasons"),
        "safe_promotion_reasons": safe_promotion_reasons(before, after, after_flags),
        "question_text_snippet": str(value(after, "question_text") or "")[:240],
        "before_snapshot": compact_snapshot(before),
        "after_snapshot": compact_snapshot(after),
    }


def safe_promotion_reasons(before: dict[str, Any], after: dict[str, Any], after_failure_flags: list[str]) -> list[str]:
    reasons: list[str] = []
    if value(after, "mapping_status") == "pass":
        reasons.append("mapping_status_pass")
    if value(after, "validation_status") == "pass":
        reasons.append("validation_status_pass")
    if value(after, "scope_quality_status") == "clean":
        reasons.append("scope_quality_clean")
    if "hard_text_corruption" not in after_failure_flags:
        reasons.append("no_hard_text_corruption_flags")
    if value(after, "text_only_status") == "ready":
        reasons.append("text_only_ready")
    elif value(after, "visual_required") is True or as_bool(value(after, "visual_required")):
        reasons.append("kept_as_review_because_visual_required")
    else:
        reasons.append("text_usable_under_text_only_gate")
    if question_number_present(after):
        reasons.append("question_number_preserved")
    if mark_bracket_count(str(value(after, "question_text") or "")) > 0:
        reasons.append("mark_brackets_present")
    if expected_subparts_present(after):
        reasons.append("expected_subparts_preserved")
    if value(before, "text_only_status") == "fail" and value(after, "text_only_status") == "review":
        reasons.append("fail_to_review_search_hint_recovery")
    if value(before, "text_only_status") == "fail" and value(after, "text_only_status") == "ready":
        reasons.append("fail_to_ready_clean_text_recovery")
    return sorted(set(reasons))


def compact_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": value(record, "question_id"),
        "paper": value(record, "paper"),
        "paper_family": value(record, "paper_family"),
        "question_format_profile": value(record, "question_format_profile"),
        "text_fidelity_flags": list_value(record, "text_fidelity_flags"),
        "review_flags": list_value(record, "review_flags"),
        "text_candidate_decision": value(record, "text_candidate_decision"),
        "ocr_selected": value(record, "ocr_selected"),
        "ocr_rejected_reasons": list_value(record, "ocr_rejected_reasons"),
        "selected_text_score": value(record, "selected_text_score"),
    }


def group_values(record: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "paper_year", paper_year(record)
    yield "paper_family", str(value(record, "paper_family") or "missing")
    yield "question_format_profile", str(value(record, "question_format_profile") or "missing")
    for item in list_or_none(record, "text_fidelity_flags"):
        yield "text_fidelity_flags", item
    for item in list_or_none(record, "review_flags"):
        yield "review_flags", item
    yield "text_candidate_decision", str(value(record, "text_candidate_decision") or "missing")
    for item in ocr_selected_rejected_reason_values(record):
        yield "ocr_selected_rejected_reason", item
    yield "selected_text_score_band", selected_text_score_band(record)


def list_or_none(record: dict[str, Any], field: str) -> list[str]:
    items = list_value(record, field)
    return items or ["none"]


def ocr_selected_rejected_reason_values(record: dict[str, Any]) -> list[str]:
    if as_bool(value(record, "ocr_selected")):
        return ["ocr_selected"]
    reasons = list_value(record, "ocr_rejected_reasons")
    return reasons or ["no_ocr_rejected_reason"]


def failure_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(flag for record in records for flag in failure_type_flags(record))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def detector_hits(record: dict[str, Any]) -> list[str]:
    record_flags = (
        set(list_value(record, "text_fidelity_flags"))
        | set(list_value(record, "visual_reason_flags"))
        | set(list_value(record, "ocr_rejected_reasons"))
    )
    hits = [detector for detector, flags in NEW_DETECTOR_FLAGS.items() if record_flags & flags]
    return sorted(hits)


def failure_type_flags(record: dict[str, Any]) -> list[str]:
    flags: set[str] = set()
    text_only = str(value(record, "text_only_status") or "").lower()
    if text_only not in {"fail", "review"}:
        return []

    validation = str(value(record, "validation_status") or "").lower()
    mapping = str(value(record, "mapping_status") or "").lower()
    scope = str(value(record, "scope_quality_status") or "").lower()
    text_fidelity_flags = set(list_value(record, "text_fidelity_flags"))
    review_flags = set(list_value(record, "review_flags"))
    extraction_flags = set(list_value(record, "extraction_quality_flags"))
    visual_flags = set(list_value(record, "visual_reason_flags"))
    rejected_reasons = set(list_value(record, "ocr_rejected_reasons"))

    if validation != "pass" or mapping != "pass" or scope == "fail":
        flags.add("true_mapping_validation_scope_failures")
    if (text_fidelity_flags | review_flags | extraction_flags | visual_flags) & HARD_CORRUPTION_FLAGS:
        flags.add("hard_text_corruption")
    if "flattened_display_math" in extraction_flags:
        flags.add("flattened_display_math")
    if text_fidelity_flags & {"sparse_or_merged_question_text", "weak_extracted_text"} or "weak_text_structure" in extraction_flags:
        flags.add("sparse_merged_prose")
    if visual_flags & DIAGRAM_TABLE_FLAGS:
        flags.add("diagram_table_dependent_but_readable_text")
    if review_flags & {"low_confidence_question_crop", "crop_uncertain"} and not (flags & {"hard_text_corruption", "sparse_merged_prose"}):
        flags.add("low_crop_confidence_only")
    if ocr_better_than_native_but_rejected(record, rejected_reasons):
        flags.add("ocr_better_than_native_but_rejected")
    if native_better_than_ocr_but_still_low_confidence(record):
        flags.add("native_better_than_ocr_but_still_low_confidence")
    if not flags and visual_flags & MATH_LAYOUT_FLAGS:
        flags.add("flattened_display_math")
    if not flags:
        flags.add("other_text_confidence_review")
    return sorted(flags)


def ocr_better_than_native_but_rejected(record: dict[str, Any], rejected_reasons: set[str]) -> bool:
    if as_bool(value(record, "ocr_selected")):
        return False
    margin = score_margin(record)
    if margin is None or margin < 20:
        return False
    return not bool(rejected_reasons & OCR_STRUCTURAL_REJECTION_REASONS)


def native_better_than_ocr_but_still_low_confidence(record: dict[str, Any]) -> bool:
    if as_bool(value(record, "ocr_selected")):
        return False
    native_score = numeric(value(record, "native_text_score"))
    ocr_score = numeric(value(record, "ocr_text_score"))
    if native_score is None or ocr_score is None or native_score < ocr_score:
        return False
    trust = str(value(record, "question_text_trust") or "").lower()
    return str(value(record, "text_only_status") or "").lower() == "fail" and trust in {"low", "unusable"}


def in_primary_target_pool(record: dict[str, Any]) -> bool:
    return (
        value(record, "text_only_status") == "fail"
        and value(record, "mapping_status") == "pass"
        and value(record, "validation_status") == "pass"
        and value(record, "scope_quality_status") == "clean"
    )


def text_only_improved(before: dict[str, Any], after: dict[str, Any]) -> bool:
    old = TEXT_ONLY_ORDER.get(str(value(before, "text_only_status") or ""), -1)
    new = TEXT_ONLY_ORDER.get(str(value(after, "text_only_status") or ""), -1)
    return new > old


def selected_text_score_band(record: dict[str, Any]) -> str:
    score = numeric(value(record, "selected_text_score"))
    if score is None:
        return "missing"
    if score < 0:
        return "<0"
    if score < 50:
        return "0-49"
    if score < 75:
        return "50-74"
    if score < 100:
        return "75-99"
    if score < 125:
        return "100-124"
    if score < 150:
        return "125-149"
    return "150+"


def paper_year(record: dict[str, Any]) -> str:
    year = value(record, "year")
    if year:
        return str(year)
    paper = str(value(record, "paper") or "")
    match = re.match(r"^\d{2}(?:spring|summer|autumn)(?P<yy>\d{2})$", paper)
    if match:
        return str(2000 + int(match.group("yy")))
    return "missing"


def question_number_present(record: dict[str, Any]) -> bool:
    qn = str(value(record, "question_number") or "")
    text = " ".join(str(value(record, "question_text") or "").split())
    return bool(qn and re.search(rf"(?:^|\s){re.escape(qn)}(?:\s|\.|\(|$)", text))


def expected_subparts_present(record: dict[str, Any]) -> bool:
    expected = [str(item).strip("()").lower() for item in list_value(record, "subparts")]
    if not expected:
        return True
    text = str(value(record, "question_text") or "").lower()
    present = {match.lower() for match in re.findall(r"\(([a-h]|i{1,3}|iv|v|vi{0,3}|ix|x)\)", text)}
    return set(expected).issubset(present)


def mark_bracket_count(text: str) -> int:
    return len(re.findall(r"\[\d{1,2}\]", text))


def score_margin(record: dict[str, Any]) -> float | None:
    ocr_score = numeric(value(record, "ocr_text_score"))
    native_score = numeric(value(record, "native_text_score"))
    if ocr_score is None or native_score is None:
        return None
    return ocr_score - native_score


def count_values(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(value(record, field) if value(record, field) not in (None, "") else "missing") for record in records)
    return dict(sorted(counts.items()))


def records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(value(record, "question_id")): record for record in records if value(record, "question_id")}


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError(f"{path} is not a question bank JSON document")
    return payload


def value(record: dict[str, Any], field: str) -> Any:
    top = record.get(field)
    if top not in (None, ""):
        return top
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(field)
    return top


def list_value(record: dict[str, Any], field: str) -> list[str]:
    raw = value(record, field)
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    return []


def numeric(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return float(raw)
    return None


def as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return bool(raw)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(raw: Any) -> str:
    if isinstance(raw, (list, dict)):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    if raw is None:
        return ""
    return str(raw)


def render_markdown(report: dict[str, Any]) -> str:
    before = report["before_counts"]
    after = report["after_counts"]
    guardrails = report["guardrails"]
    return "\n".join(
        [
            "# Text Confidence Recovery Audit",
            "",
            f"- Records: before `{report['before_record_count']}`, after `{report['after_record_count']}`",
            f"- Text-only status: before `{json.dumps(before['text_only_status'], sort_keys=True)}`, after `{json.dumps(after['text_only_status'], sort_keys=True)}`",
            f"- Text fidelity: before `{json.dumps(before['text_fidelity_status'], sort_keys=True)}`, after `{json.dumps(after['text_fidelity_status'], sort_keys=True)}`",
            f"- OCR selected: before `{before['ocr_selected_count']}`, after `{after['ocr_selected_count']}`",
            f"- Primary target promoted: `{report['target_pool']['promoted_count']}` of `{report['target_pool']['before_count']}`",
            f"- Total promoted records: `{report['promoted_record_count']}`",
            f"- Text-only movement: `{json.dumps(report['status_movement_counts'], sort_keys=True)}`",
            f"- Blocked from ready by new detectors: `{json.dumps(report['blocked_from_ready_by_detector_counts'], sort_keys=True)}`",
            f"- OCR/native decisions changed: `{report['ocr_decision_change_count']}`",
            "",
            "## Guardrails",
            "",
            f"- Validation fail delta: `{guardrails['validation_status_fail_delta']}`",
            f"- Mapping fail delta: `{guardrails['mapping_status_fail_delta']}`",
            f"- Missing image path delta: `{guardrails['missing_image_path_delta']}`",
            f"- Known math corruption promoted to ready: `{guardrails['known_math_corruption_promoted_to_ready_count']}`",
            f"- Diagram/table dependent promoted to ready: `{guardrails['diagram_table_promoted_to_ready_count']}`",
            f"- Guardrails passed: `{guardrails['passes_acceptance_guardrails']}`",
            "",
            "## Failure Types",
            "",
            f"- Before: `{json.dumps(report['failure_type_counts']['before'], sort_keys=True)}`",
            f"- After: `{json.dumps(report['failure_type_counts']['after'], sort_keys=True)}`",
            "",
            "Detailed grouped rows, failure-type rows, detector blocks, status movements, OCR decision changes, and promoted-record reasons are in the CSV/JSON outputs next to this file.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
