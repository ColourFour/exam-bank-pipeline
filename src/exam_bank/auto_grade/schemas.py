from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exam_bank.auto_grade.constants import APPROVED_REVIEW_STATUSES


@dataclass(frozen=True)
class ReviewedRubric:
    rubric_id: str
    question_id: str
    total_marks: int | None
    reviewed_by: str
    reviewed_at: str
    review_status: str = ""
    safe_for_auto_grade_lab: bool = False
    safe_for_teacher_beta: bool = False
    safe_for_student_self_check: bool = False

    @property
    def is_approved(self) -> bool:
        return bool(
            self.rubric_id
            and self.reviewed_by
            and self.reviewed_at
            and self.review_status in APPROVED_REVIEW_STATUSES
            and self.safe_for_auto_grade_lab
        )

    @property
    def approved_for(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.safe_for_teacher_beta:
            values.append("teacher_beta")
        if self.safe_for_student_self_check:
            values.append("student_self_check_beta")
        return tuple(values)


def load_reviewed_rubrics(payload: Any) -> tuple[dict[str, ReviewedRubric], list[str]]:
    if not payload:
        return {}, []
    if isinstance(payload, dict):
        records = payload.get("rubrics")
        if records is None:
            records = payload.get("records")
    else:
        records = []
    errors: list[str] = []
    rubrics: dict[str, ReviewedRubric] = {}
    if not isinstance(records, list):
        return {}, ["reviewed_rubrics_records_not_list"]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"reviewed_rubric_not_object:{index}")
            continue
        question_id = str(record.get("source_question_id") or record.get("question_id") or "").strip()
        rubric_id = str(record.get("rubric_id") or "").strip()
        approved_for = tuple(str(value) for value in record.get("approved_for") or [] if str(value))
        reviewed_by = str(record.get("reviewed_by") or "").strip()
        reviewed_at = str(record.get("reviewed_at") or "").strip()
        total_marks = _int_or_none(record.get("total_marks"))
        review_status = str(record.get("review_status") or ("approved" if approved_for else "")).strip()
        safe_for_teacher_beta = _bool(record.get("safe_for_teacher_beta")) or "teacher_beta" in approved_for
        safe_for_student_self_check = _bool(record.get("safe_for_student_self_check")) or "student_self_check_beta" in approved_for
        safe_for_auto_grade_lab = _bool(record.get("safe_for_auto_grade_lab")) or safe_for_teacher_beta or safe_for_student_self_check
        if not question_id:
            errors.append(f"reviewed_rubric_missing_question_id:{index}")
            continue
        rubric = ReviewedRubric(
            rubric_id=rubric_id,
            question_id=question_id,
            total_marks=total_marks,
            reviewed_by=reviewed_by,
            reviewed_at=reviewed_at,
            review_status=review_status,
            safe_for_auto_grade_lab=safe_for_auto_grade_lab,
            safe_for_teacher_beta=safe_for_teacher_beta,
            safe_for_student_self_check=safe_for_student_self_check,
        )
        if not rubric.is_approved:
            errors.append(f"reviewed_rubric_missing_approval_metadata:{index}:{question_id}")
        rubrics[question_id] = rubric
    return rubrics, errors


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
