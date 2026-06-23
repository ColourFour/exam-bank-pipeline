from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.submissions.outgoing_email import (
    build_outgoing_email_queue,
    send_outgoing_email_fake,
    write_outgoing_email_dry_run_report,
)


def build_queue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build controlled outgoing email queue artifacts.")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    parser.add_argument("--approval-csv", type=Path, default=None)
    return parser


def dry_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write outgoing email dry-run delivery report.")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    return parser


def fake_send_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record approved outgoing queue items with the local fake adapter.")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    parser.add_argument("--use-fake-adapter", action="store_true")
    return parser


def build_queue_main(argv: list[str] | None = None) -> int:
    args = build_queue_parser().parse_args(argv)
    result = build_outgoing_email_queue(
        assignment_id=args.assignment_id,
        submission_output_root=args.submission_output_root,
        reports_root=args.reports_root,
        approval_csv=args.approval_csv,
    )
    summary = result["summary"]
    print(
        json.dumps(
            {
                "assignment_id": result["assignment_id"],
                "approval_template": str(result["approval_template"]),
                "approved_drafts": str(result["approved_drafts"]),
                "blocked_drafts": str(result["blocked_drafts"]),
                "outgoing_queue": str(result["outgoing_queue"]),
                "outgoing_email_summary": str(result["outgoing_email_summary"]),
                "outgoing_email_audit": str(result["outgoing_email_audit"]),
                "drafts_seen": summary.drafts_seen,
                "approved_count": summary.approved_count,
                "blocked_count": summary.blocked_count,
                "queued_count": summary.queued_count,
                "dry_run_ready_count": summary.dry_run_ready_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def dry_run_main(argv: list[str] | None = None) -> int:
    args = dry_run_parser().parse_args(argv)
    result = write_outgoing_email_dry_run_report(
        assignment_id=args.assignment_id,
        submission_output_root=args.submission_output_root,
        reports_root=args.reports_root,
    )
    print(json.dumps({"assignment_id": result["assignment_id"], "dry_run_report": str(result["dry_run_report"]), "rows": result["rows"]}, indent=2, sort_keys=True))
    return 0


def fake_send_main(argv: list[str] | None = None) -> int:
    args = fake_send_parser().parse_args(argv)
    result = send_outgoing_email_fake(
        assignment_id=args.assignment_id,
        submission_output_root=args.submission_output_root,
        use_fake_adapter=args.use_fake_adapter,
    )
    print(
        json.dumps(
            {
                "assignment_id": result["assignment_id"],
                "fake_sent_messages": str(result["fake_sent_messages"]),
                "sent_count": result["sent_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
