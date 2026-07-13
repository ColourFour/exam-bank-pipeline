from __future__ import annotations

from .identifiers import parent_question_id
from .mark_scheme_models import MarkSchemeAnchor, MarkSchemeCropRegion, MarkSchemeTable


def validate_mark_scheme_mapping(
    canonical_number: str,
    question_subparts: list[str],
    markscheme_subparts: list[str],
    question_marks_total: int | None,
    markscheme_marks_total: int | None,
    anchor: MarkSchemeAnchor | None,
    next_anchor: MarkSchemeAnchor | None,
    regions: list[MarkSchemeCropRegion],
    flags: list[str],
    question_validation_flags: list[str] | None = None,
) -> tuple[list[str], str]:
    validation_flags: list[str] = []
    question_validation_flags = question_validation_flags or []
    if not anchor or not table_header_ok(anchor.table):
        return ["invalid_table_header"], "invalid_table_header"
    if "invalid_table_header" in flags:
        return ["invalid_table_header"], "invalid_table_header"
    if not regions:
        return ["partial_question_block"], "partial_question_block"
    if next_anchor and parent_question_id(next_anchor.question_number) == canonical_number:
        return ["partial_question_block"], "partial_question_block"
    candidates = mapping_failure_candidates(
        canonical_number=canonical_number,
        question_subparts=question_subparts,
        markscheme_subparts=markscheme_subparts,
        question_marks_total=question_marks_total,
        markscheme_marks_total=markscheme_marks_total,
        anchor=anchor,
        next_anchor=next_anchor,
        regions=regions,
        question_validation_flags=question_validation_flags,
    )
    failure_reason = select_mapping_failure_reason(candidates)
    if failure_reason:
        validation_flags.append(failure_reason)
    return validation_flags, failure_reason


def mapping_failure_candidates(
    *,
    canonical_number: str,
    question_subparts: list[str],
    markscheme_subparts: list[str],
    question_marks_total: int | None,
    markscheme_marks_total: int | None,
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
    regions: list[MarkSchemeCropRegion],
    question_validation_flags: list[str],
) -> list[str]:
    candidates: list[str] = []
    if "question_scope_contaminated" in question_validation_flags:
        candidates.append("question_scope_contaminated")
    if any(part not in markscheme_subparts for part in question_subparts):
        candidates.append("mark_scheme_part_structure_mismatch")
    if any(part not in question_subparts for part in markscheme_subparts):
        candidates.append("question_subparts_incomplete")
    if "missing_terminal_mark_total" in question_validation_flags and (markscheme_subparts or question_subparts):
        candidates.append("missing_terminal_mark_total")
    if question_marks_total is None and markscheme_marks_total is not None:
        candidates.append("question_mark_total_missing")
    if question_marks_total is not None and markscheme_marks_total is not None and question_marks_total != markscheme_marks_total:
        candidates.append("question_mark_total_mismatch")
    if "likely_truncated_question_crop" in question_validation_flags:
        candidates.append("likely_truncated_question_crop")
    if "weak_question_anchor" in question_validation_flags:
        candidates.append("weak_question_anchor")
    if block_contains_adjacent_question(canonical_number, regions, anchor, next_anchor):
        candidates.append("adjacent_question_block_selected")
    return candidates


def select_mapping_failure_reason(candidates: list[str]) -> str:
    priority = [
        "question_scope_contaminated",
        "mark_scheme_part_structure_mismatch",
        "question_subparts_incomplete",
        "missing_terminal_mark_total",
        "question_mark_total_missing",
        "question_mark_total_mismatch",
        "likely_truncated_question_crop",
        "weak_question_anchor",
        "adjacent_question_block_selected",
    ]
    return next((reason for reason in priority if reason in candidates), "")


def block_contains_adjacent_question(
    canonical_number: str,
    regions: list[MarkSchemeCropRegion],
    anchor: MarkSchemeAnchor,
    next_anchor: MarkSchemeAnchor | None,
) -> bool:
    del anchor
    if next_anchor is None:
        return False
    return any(
        region.page_number == next_anchor.page_number
        and region.bbox.y1 > next_anchor.y0 + 2
        and parent_question_id(next_anchor.question_number) != canonical_number
        for region in regions
    )


def table_header_ok(table: MarkSchemeTable | None) -> bool:
    return bool(table and table.header_detected == ["Question", "Answer", "Marks", "Guidance"])
