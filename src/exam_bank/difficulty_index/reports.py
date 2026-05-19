from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from exam_bank.difficulty_index.sidecar import record_needs_review, sidecar_summary


def write_difficulty_index_reports(
    sidecar: dict[str, Any],
    *,
    reports_dir: str | Path = "reports",
    dry_run: bool = False,
) -> dict[str, Any]:
    reports_dir = Path(reports_dir)
    outputs = {
        "summary": reports_dir / "difficulty_index_summary.md",
        "by_paper": reports_dir / "difficulty_index_by_paper.md",
        "review_queue": reports_dir / "difficulty_index_review_queue.md",
    }
    rendered = {
        "summary": render_summary_report(sidecar),
        "by_paper": render_by_paper_report(sidecar),
        "review_queue": render_review_queue_report(sidecar),
    }
    if not dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)
        for key, path in outputs.items():
            path.write_text(rendered[key], encoding="utf-8")
    return {"dry_run": dry_run, "outputs": {key: str(path) for key, path in outputs.items()}}


def render_summary_report(sidecar: dict[str, Any]) -> str:
    summary = sidecar_summary(sidecar)
    generated_at = str(sidecar.get("generated_at") or "unknown")
    source_path = str(sidecar.get("source_question_bank_path") or "unknown")
    lines = [
        "# Difficulty Index v1 Summary",
        "",
        f"Generated from `{source_path}` at `{generated_at}`. See `docs/DIFFICULTY_INDEX_CONTRACT.md` for interpretation and forbidden uses.",
        "",
        "The `difficulty_index_0_100` field is an internal advisory sorting score only. It is not a psychometric measurement, and 0/100 must not be read as literal candidate success rates.",
        "",
        "`paper_relative_difficulty_band` is the downstream-friendly category: a 1-5 band assigned within each paper after sorting by the advisory index.",
        "",
        "## Counts",
        "",
        f"- Records scored: {summary['record_count']}",
        f"- Unsafe records: {summary['unsafe_count']}",
        f"- Low-confidence records: {summary['low_confidence_count']}",
        f"- Review queue records: {summary['review_queue_count']}",
        f"- Question-total / mark-scheme-total disagreements: {summary['question_total_mark_scheme_total_disagreement_count']}",
        f"- Safe for teacher filtering: {summary['teacher_filtering_safe_count']}",
        f"- Safe for student sequencing: {summary['student_sequencing_safe_count']}",
        "",
        "## Confidence",
        "",
        *_counter_lines(summary["confidence_counts"]),
        "",
        "## Paper-Relative Bands",
        "",
        *_counter_lines(summary["band_counts"]),
        "",
        "## Missing Important Features",
        "",
        *_counter_lines(summary["missing_important_features"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_by_paper_report(sidecar: dict[str, Any]) -> str:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sidecar.get("records", []):
        if isinstance(record, dict):
            by_paper[str(record.get("paper") or "unknown")].append(record)
    generated_at = str(sidecar.get("generated_at") or "unknown")
    lines = [
        "# Difficulty Index v1 By Paper",
        "",
        f"Generated at `{generated_at}`. See `docs/DIFFICULTY_INDEX_CONTRACT.md` for interpretation and forbidden uses.",
        "",
        "Questions are shown easiest to hardest within each paper. Bands are paper-relative, not global difficulty claims.",
        "",
    ]
    for paper in sorted(by_paper):
        records = sorted(
            by_paper[paper],
            key=lambda record: (
                int(record.get("paper_relative_difficulty_band") or 0),
                float(record.get("paper_relative_percentile") or 0),
                float(record.get("difficulty_index_0_100") or 0),
                str(record.get("question_id") or ""),
            ),
        )
        lines.extend([f"## {paper}", "", "| Question | ID | Index | Percentile | Band | Confidence | Warnings / Review |", "|---|---|---:|---:|---:|---|---|"])
        for record in records:
            reasons = ", ".join(list(record.get("unsafe_reasons") or []) + list(record.get("warnings") or []) + list(record.get("review_reasons") or []))
            lines.append(
                f"| {record.get('question_number')} | `{record.get('question_id')}` | {record.get('difficulty_index_0_100')} | "
                f"{record.get('paper_relative_percentile')} | {record.get('paper_relative_difficulty_band')} | "
                f"{record.get('confidence')} | {reasons or '-'} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_review_queue_report(sidecar: dict[str, Any]) -> str:
    records = [record for record in sidecar.get("records", []) if isinstance(record, dict) and record_needs_review(record)]
    records.sort(
        key=lambda record: (
            0 if record.get("confidence") == "unsafe" else 1,
            str(record.get("paper") or ""),
            int(record.get("paper_relative_difficulty_band") or 0),
            str(record.get("question_id") or ""),
        )
    )
    generated_at = str(sidecar.get("generated_at") or "unknown")
    lines = [
        "# Difficulty Index v1 Review Queue",
        "",
        f"Generated at `{generated_at}`. See `docs/DIFFICULTY_INDEX_CONTRACT.md` for interpretation and forbidden uses.",
        "",
        "Records listed here need human calibration or safety review before relying on their advisory placement.",
        "",
        f"Records needing review: {len(records)}",
        "",
        "| Paper | Question | ID | Index | Band | Confidence | Reasons |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for record in records:
        reasons = ", ".join(list(record.get("unsafe_reasons") or []) + list(record.get("warnings") or []) + list(record.get("review_reasons") or []))
        lines.append(
            f"| {record.get('paper')} | {record.get('question_number')} | `{record.get('question_id')}` | "
            f"{record.get('difficulty_index_0_100')} | {record.get('paper_relative_difficulty_band')} | "
            f"{record.get('confidence')} | {reasons or '-'} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _counter_lines(values: dict[str, int]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: {values[key]}" for key in sorted(values, key=str)]
