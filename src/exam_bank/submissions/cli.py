from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from exam_bank.submissions.ingest import ingest_assignment_submissions, parse_datetime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local assignment PDF submissions without email, OCR, or grading.")
    parser.add_argument("--assignment", type=Path, required=True, help="Assignment metadata JSON.")
    parser.add_argument("--roster", type=Path, required=True, help="Roster CSV.")
    parser.add_argument("--submissions-dir", type=Path, required=True, help="Local folder of submission files.")
    parser.add_argument("--output-root", type=Path, default=Path("output/submissions"), help="Ignored output root.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"), help="Ignored reports root.")
    parser.add_argument("--received-at", default="", help="Optional ISO timestamp override for every local file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    received_at: datetime | None = parse_datetime(args.received_at) if args.received_at else None
    result = ingest_assignment_submissions(
        assignment_path=args.assignment,
        roster_path=args.roster,
        submissions_dir=args.submissions_dir,
        output_root=args.output_root,
        reports_root=args.reports_root,
        received_at_override=received_at,
    )
    print(
        json.dumps(
            {
                "assignment_id": result["assignment_id"],
                "manifest": str(result["manifest"]),
                "audit_log": str(result["audit_log"]),
                "completion_report": str(result["completion_report"]),
                "accepted_count": len(result["accepted"]),
                "rejected_count": len(result["rejected"]),
                "completion_rows": len(result["completion_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
