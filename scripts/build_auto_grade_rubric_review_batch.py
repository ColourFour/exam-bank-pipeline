from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.review_batch import (
    DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH,
    DEFAULT_EXCLUDED_RISK_FLAGS,
    DEFAULT_REVIEW_BATCH_PATH,
    DEFAULT_REVIEW_BATCH_REPORT_PATH,
    build_rubric_review_batch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Phase 2B human reviewed-rubric gold-set worklist.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--review-queue", type=Path, default=Path("output/auto_grade/rubric_review_queue.v1.json"))
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--eligible-items", type=Path, default=Path("output/auto_grade/eligible_items.v1.json"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_REVIEW_BATCH_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_REVIEW_BATCH_REPORT_PATH))
    parser.add_argument("--draft-output", type=Path, default=Path(DEFAULT_DRAFT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--max-rubrics", type=int, default=25)
    parser.add_argument("--max-events", type=int, default=75)
    parser.add_argument(
        "--paper-family",
        action="append",
        default=None,
        help="Paper family to include. Repeat or pass comma-separated values. Defaults to p1,p3.",
    )
    parser.add_argument(
        "--exclude-risk-flags",
        default=",".join(DEFAULT_EXCLUDED_RISK_FLAGS),
        help="Comma-separated risk/blocker flags to exclude. Defaults to unknown_mark_codes,total_mismatch.",
    )
    parser.add_argument("--include-medium-priority", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paper_families = _split_repeated(args.paper_family) or ["p1", "p3"]
    excluded = _split_csv(args.exclude_risk_flags)
    result = build_rubric_review_batch(
        question_bank_path=args.question_bank,
        review_queue_path=args.review_queue,
        mark_events_path=args.mark_events,
        eligible_items_path=args.eligible_items,
        output_path=args.output,
        report_path=args.report,
        draft_output_path=args.draft_output,
        max_rubrics=args.max_rubrics,
        max_events=args.max_events,
        paper_families=paper_families,
        exclude_risk_flags=excluded,
        include_medium_priority=args.include_medium_priority,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "review_batch": str(args.output),
                "draft_reviewed_rubrics": str(args.draft_output),
                "report": str(args.report),
                "summary": result["batch"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _split_repeated(values: list[str] | None) -> list[str]:
    if not values:
        return []
    output: list[str] = []
    for value in values:
        output.extend(_split_csv(value))
    return output


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
