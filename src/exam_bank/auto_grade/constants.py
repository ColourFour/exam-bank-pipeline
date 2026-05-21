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
REVIEWED_RUBRICS_SCHEMA = "exam_bank.auto_grade.reviewed_rubrics"
REVIEWED_RUBRICS_VALIDATION_SCHEMA = "exam_bank.auto_grade.reviewed_rubrics.validation"
RUBRIC_REVIEW_QUEUE_SCHEMA = "exam_bank.auto_grade.rubric_review_queue"
REVIEWED_RUBRIC_SCHEMA_VERSION = 1
ALLOWED_RUBRIC_MARK_CODES = {"M", "A", "B", "E", "DM", "FT", "unknown"}
APPROVED_REVIEW_STATUSES = {"approved"}

DEFAULT_ELIGIBLE_ITEMS_PATH = "output/auto_grade/eligible_items.v1.json"
DEFAULT_REVIEWED_RUBRICS_PATH = "output/auto_grade/reviewed_rubrics.v1.json"
DEFAULT_RUBRIC_REVIEW_QUEUE_PATH = "output/auto_grade/rubric_review_queue.v1.json"
DEFAULT_SUMMARY_REPORT_PATH = "reports/auto_grade/eligible_items_summary.md"
DEFAULT_RUBRIC_REVIEW_QUEUE_REPORT_PATH = "reports/auto_grade/rubric_review_queue_summary.md"
DEFAULT_REVIEWED_RUBRICS_VALIDATION_REPORT_PATH = "reports/auto_grade/reviewed_rubrics_validation.md"
