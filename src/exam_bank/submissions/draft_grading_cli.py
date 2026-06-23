from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.submissions.draft_grading import build_submission_draft_grades


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 3 teacher-facing draft grading artifacts for accepted submissions.")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--submission-output-root", type=Path, default=Path("output/submissions"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/submissions"))
    parser.add_argument("--reviewed-rubrics-path", type=Path, default=Path("output/auto_grade/reviewed_rubrics.v1.json"))
    parser.add_argument("--mark-events-path", type=Path, default=Path("output/json/question_bank.mark_events.v1.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_submission_draft_grades(
        assignment_id=args.assignment_id,
        submission_output_root=args.submission_output_root,
        reports_root=args.reports_root,
        reviewed_rubrics_path=args.reviewed_rubrics_path,
        mark_events_path=args.mark_events_path,
    )
    summary = result["summary"]
    print(
        json.dumps(
            {
                "assignment_id": result["assignment_id"],
                "extraction_results": str(result["extraction_results"]),
                "draft_grading_results": str(result["draft_grading_results"]),
                "draft_grading_summary": str(result["draft_grading_summary"]),
                "teacher_grading_review_packet": str(result["teacher_grading_review_packet"]),
                "draft_grading_summary_csv": str(result["draft_grading_summary_csv"]),
                "submissions_attempted": summary.submissions_attempted,
                "drafts_created": summary.drafts_created,
                "failed_count": summary.failed_count,
                "low_confidence_count": summary.low_confidence_count,
                "medium_confidence_count": summary.medium_confidence_count,
                "high_confidence_count": summary.high_confidence_count,
                "teacher_review_required_count": summary.teacher_review_required_count,
                "student_facing_count": summary.student_facing_count,
                "draft_scores_assigned": result["draft_scores_assigned"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
