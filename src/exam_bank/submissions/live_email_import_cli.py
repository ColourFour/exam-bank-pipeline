from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.submissions.live_email_import import import_live_email_submissions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import assignment submission email through a read-only connector.")
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--connector-config", type=Path, required=True)
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    parser.add_argument("--dry-run", action="store_true", help="Retained for explicitness; dry-run is the default.")
    parser.add_argument("--apply", action="store_true", help="Materialize scoped messages and run Phase 4/Phase 1 intake.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = import_live_email_submissions(
        assignment_path=args.assignment,
        roster_path=args.roster,
        connector_config_path=args.connector_config,
        submission_output_root=args.submission_output_root,
        reports_root=args.reports_root,
        apply=args.apply,
    )
    summary = result["summary"]
    print(
        json.dumps(
            {
                "assignment_id": result["assignment_id"],
                "dry_run": result["dry_run"],
                "apply": result["apply"],
                "messages_found": summary["messages_found"],
                "likely_accepted_count": summary["likely_accepted_count"],
                "likely_quarantined_count": summary["likely_quarantined_count"],
                "likely_rejected_count": summary["likely_rejected_count"],
                "apply_recommended": summary["apply_recommended"],
                "live_email_dir": str(result["live_email_dir"]),
                "readiness_report": str(result["readiness_report"]),
                "audit_log": str(result["audit_log"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
