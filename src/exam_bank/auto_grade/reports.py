from __future__ import annotations

from pathlib import Path
from typing import Any

from exam_bank.auto_grade.eligibility import summarize_eligible_items


def write_eligible_items_summary(
    eligible_items: dict[str, Any],
    *,
    output_path: str | Path = "reports/auto_grade/eligible_items_summary.md",
    dry_run: bool = False,
) -> str:
    text = render_eligible_items_summary(eligible_items)
    if not dry_run:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def render_eligible_items_summary(eligible_items: dict[str, Any]) -> str:
    items = [item for item in eligible_items.get("items", []) if isinstance(item, dict)]
    summary = eligible_items.get("summary") if isinstance(eligible_items.get("summary"), dict) else summarize_eligible_items(items)
    status_counts = summary.get("status_counts") or {}
    top_reasons = summary.get("top_block_reasons") or {}
    lines = [
        "# Auto-Grade Eligible Items Summary",
        "",
        "Phase 1 is contracts and readiness classification only. It did not score submissions or make student-facing grading claims.",
        "",
        "## Summary",
        "",
        f"- Total records classified: {summary.get('record_count', len(items))}",
        "- Count by eligibility status:",
        *_counter_lines(status_counts),
        f"- Canonical question images present: {summary.get('canonical_question_image_present_count', 0)}",
        f"- Canonical question images missing: {summary.get('canonical_question_image_missing_count', 0)}",
        f"- Canonical mark-scheme images present: {summary.get('canonical_mark_scheme_image_present_count', 0)}",
        f"- Canonical mark-scheme images missing: {summary.get('canonical_mark_scheme_image_missing_count', 0)}",
        f"- Blocked by missing reviewed rubric: {summary.get('missing_reviewed_rubric_count', 0)}",
        f"- Candidate items for future rubric review: {summary.get('future_rubric_review_candidate_count', 0)}",
        f"- Blocked/review-only items with actionable reasons: {summary.get('blocked_or_review_only_actionable_reason_percent', 0)}%",
        "",
        "## Top Block Reasons",
        "",
        *_counter_lines(top_reasons),
        "",
        "## Student Readiness",
        "",
        f"- Student-ready items: {summary.get('student_ready_count', 0)}",
        f"- Student self-check beta items: {summary.get('student_self_check_beta_count', 0)}",
        "",
        "Phase 1 produced 0 student-ready items unless reviewed rubrics already existed and validated.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _counter_lines(values: dict[str, int]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: {values[key]}" for key in sorted(values)]
