from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewedRubric:
    rubric_id: str
    question_id: str
    total_marks: int | None
    reviewed_by: str
    reviewed_at: str
    approved_for: tuple[str, ...]

    @property
    def is_approved(self) -> bool:
        return bool(self.rubric_id and self.reviewed_by and self.reviewed_at)


def load_reviewed_rubrics(payload: Any) -> tuple[dict[str, ReviewedRubric], list[str]]:
    if not payload:
        return {}, []
    records = payload.get("rubrics") or payload.get("records") if isinstance(payload, dict) else []
    errors: list[str] = []
    rubrics: dict[str, ReviewedRubric] = {}
    if not isinstance(records, list):
        return {}, ["reviewed_rubrics_records_not_list"]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"reviewed_rubric_not_object:{index}")
            continue
        question_id = str(record.get("question_id") or "").strip()
        rubric_id = str(record.get("rubric_id") or "").strip()
        approved_for = tuple(str(value) for value in record.get("approved_for") or [] if str(value))
        reviewed_by = str(record.get("reviewed_by") or "").strip()
        reviewed_at = str(record.get("reviewed_at") or "").strip()
        total_marks = _int_or_none(record.get("total_marks"))
        if not question_id:
            errors.append(f"reviewed_rubric_missing_question_id:{index}")
            continue
        rubric = ReviewedRubric(
            rubric_id=rubric_id,
            question_id=question_id,
            total_marks=total_marks,
            reviewed_by=reviewed_by,
            reviewed_at=reviewed_at,
            approved_for=approved_for,
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
