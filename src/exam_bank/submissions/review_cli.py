from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.submissions.review_queue import build_submission_review_queue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build teacher review and manual grading-prep artifacts for local submissions.")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_submission_review_queue(
        assignment_id=args.assignment_id,
        submission_output_root=args.submission_output_root,
        reports_root=args.reports_root,
    )
    summary = result["summary"]
    print(
        json.dumps(
            {
                "assignment_id": result["assignment_id"],
                "review_queue": str(result["review_queue"]),
                "grading_prep": str(result["grading_prep"]),
                "review_summary": str(result["review_summary"]),
                "review_queue_csv": str(result["review_queue_csv"]),
                "teacher_notes_template": str(result["teacher_notes_template"]),
                "accepted_count": summary.accepted_count,
                "needs_review_count": summary.needs_review_count,
                "manual_grading_placeholders_count": summary.manual_grading_placeholders_count,
                "rejected_count": result["rejected_count"],
                "missing_count": result["missing_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
