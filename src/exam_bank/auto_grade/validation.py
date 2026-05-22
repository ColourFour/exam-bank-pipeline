from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade import (
    AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA,
    AUTO_GRADE_SCHEMA_VERSION,
    AUTO_GRADE_VALIDATION_SCHEMA,
)
from exam_bank.auto_grade.constants import (
    DEFAULT_REVIEWED_RUBRICS_PATH,
    ELIGIBILITY_STATUSES,
    GRADING_STATUSES,
    STUDENT_SAFE_STATUSES,
)
from exam_bank.auto_grade.reviewed_rubrics import approved_question_ids_from_reviewed_rubrics
from exam_bank.auto_grade.schemas import load_reviewed_rubrics


def validate_eligible_items(
    *,
    eligible_items_path: str | Path = "output/auto_grade/eligible_items.v1.json",
    question_bank_path: str | Path = "output/json/question_bank.json",
    artifact_root: str | Path = "output",
    reviewed_rubrics_path: str | Path | None = None,
    check_artifact_existence: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    eligible = _load_json(eligible_items_path)
    question_bank = _load_json(question_bank_path)
    questions = _question_records(question_bank)
    question_ids = [str(record.get("question_id") or "") for record in questions]
    question_id_set = {question_id for question_id in question_ids if question_id}
    reviewed_rubrics_path = _reviewed_rubrics_path_for_validation(eligible, reviewed_rubrics_path)
    rubrics_payload = _load_optional_json(reviewed_rubrics_path)
    rubrics, rubric_errors = load_reviewed_rubrics(rubrics_payload)
    approved_question_ids = approved_question_ids_from_reviewed_rubrics(
        rubrics_payload,
        question_bank_payload=question_bank,
    ) if rubrics_payload else set()
    rubrics = {question_id: rubric for question_id, rubric in rubrics.items() if question_id in approved_question_ids}
    errors: list[str] = []
    warnings: list[str] = list(rubric_errors)

    if eligible.get("schema") != AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA:
        errors.append(f"schema_mismatch:expected={AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA}:actual={eligible.get('schema')}")
    if eligible.get("schema_name") not in (None, AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA):
        errors.append(f"schema_name_mismatch:expected={AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA}:actual={eligible.get('schema_name')}")
    if int(eligible.get("schema_version") or 0) != AUTO_GRADE_SCHEMA_VERSION:
        errors.append(f"schema_version_mismatch:expected={AUTO_GRADE_SCHEMA_VERSION}:actual={eligible.get('schema_version')}")
    items = eligible.get("items")
    if not isinstance(items, list):
        items = []
        errors.append("items_not_list")
    if eligible.get("record_count") != len(items):
        errors.append(f"record_count_mismatch:declared={eligible.get('record_count')}:actual={len(items)}")
    if len(items) != len(questions):
        errors.append(f"question_bank_record_count_mismatch:eligible={len(items)}:question_bank={len(questions)}")

    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"item_not_object:{index}")
            continue
        _validate_item(
            index,
            item,
            question_id_set=question_id_set,
            seen=seen,
            artifact_root=Path(artifact_root),
            rubrics=rubrics,
            check_artifact_existence=check_artifact_existence,
            errors=errors,
            warnings=warnings,
        )

    missing_ids = sorted(question_id_set - seen)
    if missing_ids:
        errors.append(f"missing_question_ids:{','.join(missing_ids[:25])}")

    report = {
        "schema": AUTO_GRADE_VALIDATION_SCHEMA,
        "schema_name": AUTO_GRADE_VALIDATION_SCHEMA,
        "schema_version": AUTO_GRADE_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    if output_path:
        write_atomic_json(report, output_path, sort_keys=True)
    return report


def _validate_item(
    index: int,
    item: dict[str, Any],
    *,
    question_id_set: set[str],
    seen: set[str],
    artifact_root: Path,
    rubrics: dict[str, Any],
    check_artifact_existence: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    question_id = str(item.get("question_id") or "")
    if not question_id:
        errors.append(f"missing_question_id:{index}")
    elif question_id in seen:
        errors.append(f"duplicate_question_id:{index}:{question_id}")
    elif question_id not in question_id_set:
        errors.append(f"mismatched_question_id:{index}:{question_id}")
    seen.add(question_id)

    status = item.get("eligibility_status")
    if not status:
        errors.append(f"missing_eligibility_status:{index}:{question_id}")
    elif status not in ELIGIBILITY_STATUSES:
        errors.append(f"invalid_eligibility_status:{index}:{question_id}:{status}")

    block_reasons = item.get("block_reasons")
    if not isinstance(block_reasons, list):
        errors.append(f"block_reasons_not_list:{index}:{question_id}")
        block_reasons = []
    if status == "blocked" and not block_reasons:
        errors.append(f"blocked_item_without_block_reasons:{index}:{question_id}")
    if status == "review_only" and not block_reasons:
        warnings.append(f"review_only_item_without_actionable_reasons:{index}:{question_id}")

    for field in ("canonical_question_artifact", "canonical_mark_scheme_artifact"):
        value = str(item.get(field) or "").strip()
        if not value:
            errors.append(f"missing_artifact_path:{index}:{question_id}:{field}")
            continue
        if check_artifact_existence and not _path_exists(value, artifact_root):
            errors.append(f"artifact_file_missing:{index}:{question_id}:{field}:{value}")

    if status in GRADING_STATUSES:
        rubric = rubrics.get(question_id)
        if not rubric or not rubric.is_approved:
            errors.append(f"unsafe_promotion_without_reviewed_rubric:{index}:{question_id}:{status}")
        elif status == "teacher_beta" and not rubric.safe_for_teacher_beta:
            errors.append(f"teacher_beta_without_rubric_approval:{index}:{question_id}:{status}")
        if status in STUDENT_SAFE_STATUSES:
            errors.append(f"student_safe_status_forbidden_in_phase_2a:{index}:{question_id}:{status}")
            if not rubric:
                errors.append(f"student_safe_without_reviewed_rubric:{index}:{question_id}:{status}")
            elif status not in set(rubric.approved_for):
                errors.append(f"student_safe_without_rubric_approval:{index}:{question_id}:{status}")

    if status in STUDENT_SAFE_STATUSES and not item.get("learning_target_ids"):
        errors.append(f"student_safe_without_learning_targets:{index}:{question_id}:{status}")

    rubric = rubrics.get(question_id)
    total_marks = _int_or_none(item.get("total_marks"))
    if rubric and rubric.total_marks is not None and total_marks is not None and rubric.total_marks != total_marks:
        errors.append(
            f"reviewed_rubric_total_mismatch:{index}:{question_id}:rubric={rubric.total_marks}:item={total_marks}"
        )


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_optional_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return _load_json(path)


def _reviewed_rubrics_path_for_validation(
    eligible: dict[str, Any],
    reviewed_rubrics_path: str | Path | None,
) -> str | Path | None:
    if reviewed_rubrics_path is not None:
        return reviewed_rubrics_path
    source_sidecars = eligible.get("source_sidecars")
    if isinstance(source_sidecars, dict):
        source_path = str(source_sidecars.get("reviewed_rubrics_path") or "").strip()
        if source_path:
            return source_path
    return DEFAULT_REVIEWED_RUBRICS_PATH


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _path_exists(path_value: str, artifact_root: Path) -> bool:
    path = Path(path_value)
    if path.is_absolute():
        return path.is_file()
    return (artifact_root / path).is_file() or path.is_file()


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
