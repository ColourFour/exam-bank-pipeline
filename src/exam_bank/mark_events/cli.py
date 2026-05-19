from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.mark_events.reports import write_mark_event_reports
from exam_bank.mark_events.sidecar import build_mark_events_sidecar, sidecar_summary
from exam_bank.mark_events.validation import validate_mark_events


def run_build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build advisory deterministic mark-event sidecar from question_bank.json.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--out", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--report", type=Path, default=Path("output/reports/mark_events_audit.md"))
    parser.add_argument("--review-queue", type=Path, default=Path("output/reports/mark_events_review_queue.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    sidecar = build_mark_events_sidecar(
        question_bank_path=args.question_bank,
        artifact_root=args.artifact_root,
        output_path=args.out,
        dry_run=args.dry_run,
    )
    reports = write_mark_event_reports(
        sidecar,
        audit_report_path=args.report,
        review_queue_path=args.review_queue,
        dry_run=args.dry_run,
    )
    print(json.dumps({"sidecar": str(args.out), "summary": sidecar_summary(sidecar), "reports": reports}, indent=2, sort_keys=True))
    return 0


def run_validate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate advisory deterministic mark-event sidecar.")
    parser.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    parser.add_argument("--mark-events", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("output/json/question_bank.mark_events.validation.v1.json"))
    args = parser.parse_args(argv)

    report = validate_mark_events(
        question_bank_path=args.question_bank,
        mark_events_path=args.mark_events,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1

