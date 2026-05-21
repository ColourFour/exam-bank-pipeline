from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.constants import DEFAULT_RUBRIC_REVIEW_QUEUE_PATH, DEFAULT_RUBRIC_REVIEW_QUEUE_REPORT_PATH
from exam_bank.auto_grade.reviewed_rubrics import build_rubric_review_queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 2A human rubric review queue.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--eligible-items", type=Path, default=Path("output/auto_grade/eligible_items.v1.json"))
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--topic-routing", type=Path, default=Path("output/json/question_bank.topic_routing.v1.json"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_RUBRIC_REVIEW_QUEUE_PATH))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_RUBRIC_REVIEW_QUEUE_REPORT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_rubric_review_queue(
        question_bank_path=args.question_bank,
        eligible_items_path=args.eligible_items,
        mark_events_path=args.mark_events,
        topic_routing_path=args.topic_routing,
        output_path=args.output,
        report_path=args.report,
        dry_run=args.dry_run,
    )
    print(json.dumps({"review_queue": str(args.output), "report": str(args.report), "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
