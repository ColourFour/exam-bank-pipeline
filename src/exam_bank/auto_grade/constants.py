from __future__ import annotations

ELIGIBILITY_STATUSES = {
    "blocked",
    "review_only",
    "teacher_beta",
    "student_self_check_beta",
    "student_ready",
}

STUDENT_SAFE_STATUSES = {"student_self_check_beta", "student_ready"}
GRADING_STATUSES = {"teacher_beta", "student_self_check_beta", "student_ready"}

DEFAULT_ELIGIBLE_ITEMS_PATH = "output/auto_grade/eligible_items.v1.json"
DEFAULT_REVIEWED_RUBRICS_PATH = "output/auto_grade/reviewed_rubrics.v1.json"
DEFAULT_SUMMARY_REPORT_PATH = "reports/auto_grade/eligible_items_summary.md"
