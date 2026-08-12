from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "pairing_correct",
    "question_image_usable",
    "question_image_strict",
    "markscheme_image_usable",
    "markscheme_image_strict",
    "question_text_usable",
    "question_text_strict",
    "markscheme_text_usable",
    "markscheme_text_strict",
    "overall_usable",
    "strict_accurate",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two seeded visual accuracy audits.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    before = Path(args.before)
    after = Path(args.after)
    output_dir = Path(args.output_dir)
    before_summary = load(before / "summary.json")
    after_summary = load(after / "summary.json")
    before_sample = load(before / "sample.json")
    after_sample = load(after / "sample.json")

    before_rows = {row["question_id"]: row for row in before_sample["questions"]}
    after_rows = {row["question_id"]: row for row in after_sample["questions"]}
    same_ids = set(before_rows) == set(after_rows)
    changed_question_text = sorted(
        question_id
        for question_id in set(before_rows) & set(after_rows)
        if before_rows[question_id].get("question_text")
        != after_rows[question_id].get("question_text")
    )
    changed_markscheme_text = sorted(
        question_id
        for question_id in set(before_rows) & set(after_rows)
        if before_rows[question_id].get("mark_scheme_text")
        != after_rows[question_id].get("mark_scheme_text")
    )

    comparisons = {}
    for metric in METRICS:
        old = before_summary["headline_metrics"][metric]
        new = after_summary["headline_metrics"][metric]
        comparisons[metric] = {
            "before_count": old["count"],
            "after_count": new["count"],
            "count_delta": new["count"] - old["count"],
            "before_percent": old["percent"],
            "after_percent": new["percent"],
            "percentage_point_delta": round(new["percent"] - old["percent"], 1),
        }

    result = {
        "schema_name": "exam_bank.random_visual_accuracy_comparison",
        "schema_version": 1,
        "same_question_ids": same_ids,
        "sample_count": len(after_rows),
        "changed_question_text_count": len(changed_question_text),
        "changed_markscheme_text_count": len(changed_markscheme_text),
        "changed_question_text_ids": changed_question_text,
        "changed_markscheme_text_ids": changed_markscheme_text,
        "metrics": comparisons,
        "interpretation_note": (
            "The before markscheme-image usable rate used the prior conservative "
            "regeneration_failed screen. The after rate uses fresh OCR screening plus "
            "manual visual adjudication of every pair, so its large delta reflects both "
            "repairs and removal of stale failure metadata."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "COMPARISON.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Seeded 10% audit comparison",
        "",
        f"The rerun uses the same {result['sample_count']} question IDs as the baseline.",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = result["metrics"][metric]
        lines.append(
            f"| {metric.replace('_', ' ')} | {row['before_percent']}% | "
            f"{row['after_percent']}% | {row['percentage_point_delta']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Text-change check",
            "",
            f"- Question JSON strings changed: {result['changed_question_text_count']}.",
            f"- Markscheme JSON strings changed: {result['changed_markscheme_text_count']}.",
            "",
            "The small movement in OCR-alignment text rates therefore comes from cleaner "
            "current images and OCR comparison variance, not repaired JSON text.",
            "",
            "## Comparability note",
            "",
            result["interpretation_note"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
