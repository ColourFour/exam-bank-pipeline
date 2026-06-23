from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.submissions.email_intake import ingest_email_fixtures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest synthetic email submission fixtures without connecting to email.")
    parser.add_argument("--assignment", type=Path, required=True, help="Assignment metadata JSON.")
    parser.add_argument("--roster", type=Path, required=True, help="Roster CSV.")
    parser.add_argument("--email-fixtures-dir", type=Path, required=True, help="Synthetic email fixture directory.")
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"), help="Ignored output root.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"), help="Ignored reports root.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and evaluate fixtures without copying PDFs or invoking Phase 1.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = ingest_email_fixtures(
        assignment_path=args.assignment,
        roster_path=args.roster,
        email_fixtures_dir=args.email_fixtures_dir,
        output_root=args.submission_output_root,
        reports_root=args.reports_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
