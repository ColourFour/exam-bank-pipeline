from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.auto_grade.constants import (
    DEFAULT_ELIGIBLE_ITEMS_PATH,
    DEFAULT_REVIEWED_RUBRICS_PATH,
    DEFAULT_SUMMARY_REPORT_PATH,
)
from exam_bank.auto_grade.eligibility import build_eligible_items
from exam_bank.auto_grade.reports import write_eligible_items_summary
from exam_bank.auto_grade.validation import validate_eligible_items


def run_build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed auto-grade eligible_items.v1 artifact.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_ELIGIBLE_ITEMS_PATH))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--reviewed-rubrics", type=Path, default=Path(DEFAULT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--topic-routing", type=Path, default=Path("output/json/question_bank.topic_routing.v1.json"))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_SUMMARY_REPORT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_eligible_items(
        question_bank_path=args.question_bank,
        output_path=args.output,
        artifact_root=args.artifact_root,
        reviewed_rubrics_path=args.reviewed_rubrics,
        mark_events_path=args.mark_events,
        topic_routing_path=args.topic_routing,
        dry_run=args.dry_run,
    )
    write_eligible_items_summary(payload, output_path=args.report, dry_run=args.dry_run)
    print(json.dumps({"eligible_items": str(args.output), "report": str(args.report), "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


def run_validate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate auto-grade eligible_items.v1 artifact.")
    parser.add_argument("--eligible-items", type=Path, default=Path(DEFAULT_ELIGIBLE_ITEMS_PATH))
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--reviewed-rubrics", type=Path, default=Path(DEFAULT_REVIEWED_RUBRICS_PATH))
    parser.add_argument("--output", type=Path, default=Path("output/auto_grade/eligible_items.validation.v1.json"))
    parser.add_argument("--no-existence-check", action="store_true")
    args = parser.parse_args(argv)

    report = validate_eligible_items(
        eligible_items_path=args.eligible_items,
        question_bank_path=args.question_bank,
        artifact_root=args.artifact_root,
        reviewed_rubrics_path=args.reviewed_rubrics,
        check_artifact_existence=not args.no_existence_check,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1
