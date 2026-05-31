from __future__ import annotations

from typing import Any, Iterable, Literal, TypedDict


CourseId = Literal["p1", "p3", "m1", "s1"]
ReviewStatus = Literal["reviewed", "needs_review", "candidate", "blocked"]

COURSE_IDS: tuple[CourseId, ...] = ("p1", "p3", "m1", "s1")

PAPER_FAMILY_TO_COURSE_ID: dict[str, CourseId] = {
    "p1": "p1",
    "p3": "p3",
    "p4": "m1",
    "p5": "s1",
    "m1": "m1",
    "s1": "s1",
}

COURSE_ID_TO_PAPER_FAMILIES: dict[CourseId, tuple[str, ...]] = {
    "p1": ("p1",),
    "p3": ("p3",),
    "m1": ("p4",),
    "s1": ("p5",),
}

COURSE_COMPONENT_NAMES: dict[CourseId, str] = {
    "p1": "Pure Mathematics 1",
    "p3": "Pure Mathematics 3",
    "m1": "Mechanics 1",
    "s1": "Probability & Statistics 1",
}

COURSE_EMPTY_STATE_MESSAGE = "No reviewed exam-bank records available yet."
REVIEW_STATUSES: set[str] = {"reviewed", "needs_review", "candidate", "blocked"}
CONTENT_LAB_SCHEMA_NAME = "asterion.content_lab_candidates"
RUNTIME_SAFE_CANDIDATE_SCHEMA_NAME = "asterion.student_runtime_safe_candidates"


class ExamBankCourseRecord(TypedDict, total=False):
    id: str
    course_id: CourseId
    paper: str
    component_name: str
    topic_id: str
    source_exam: str
    question_image_path: str
    mark_scheme_image_path: str
    student_runtime_safe: bool
    review_status: ReviewStatus


def course_registry() -> list[dict[str, Any]]:
    return [
        {
            "course_id": course_id,
            "component_name": COURSE_COMPONENT_NAMES[course_id],
            "paper_families": list(COURSE_ID_TO_PAPER_FAMILIES[course_id]),
            "empty_state_message": COURSE_EMPTY_STATE_MESSAGE,
        }
        for course_id in COURSE_IDS
    ]


def normalize_course_id(value: Any) -> CourseId | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in COURSE_IDS:
        return normalized  # type: ignore[return-value]
    return None


def course_id_for_paper_family(value: Any) -> CourseId | None:
    normalized = str(value or "").strip().lower()
    return PAPER_FAMILY_TO_COURSE_ID.get(normalized)


def course_id_for_record(record: dict[str, Any]) -> CourseId | None:
    explicit = _first_text(record, ("course_id", "courseId"))
    if explicit:
        return normalize_course_id(explicit)
    return course_id_for_paper_family(_first_text(record, ("paper_family", "paperFamily", "component")))


def component_name_for_course(course_id: Any) -> str | None:
    normalized = normalize_course_id(course_id)
    if not normalized:
        return None
    return COURSE_COMPONENT_NAMES[normalized]


def component_name_for_record(record: dict[str, Any]) -> str | None:
    explicit = _first_text(record, ("component_name", "componentName"))
    if explicit:
        return explicit
    return component_name_for_course(course_id_for_record(record))


def source_exam_for_record(record: dict[str, Any]) -> str | None:
    return _first_text(record, ("source_exam", "sourceExam", "paper"))


def topic_id_for_record(record: dict[str, Any]) -> str | None:
    return _first_text(record, ("topic_id", "topicId", "primary_topic_id", "primaryTopicId"))


def question_image_path_for_record(record: dict[str, Any]) -> str | None:
    return _first_text(
        record,
        (
            "question_image_path",
            "questionImagePath",
            "canonical_question_artifact",
            "canonicalQuestionArtifact",
            "canonical_question_image_path",
        ),
    )


def mark_scheme_image_path_for_record(record: dict[str, Any]) -> str | None:
    return _first_text(
        record,
        (
            "mark_scheme_image_path",
            "markSchemeImagePath",
            "canonical_mark_scheme_artifact",
            "canonicalMarkSchemeArtifact",
            "canonical_mark_scheme_image_path",
        ),
    )


def is_content_lab_candidate_record(record: dict[str, Any]) -> bool:
    if record.get("schema_name") == CONTENT_LAB_SCHEMA_NAME or record.get("schema") == CONTENT_LAB_SCHEMA_NAME:
        return True
    if "generation_gate" in record or "role_statuses" in record or "candidate_selection" in record:
        return True
    candidate_id = _first_text(record, ("candidate_id", "candidateId"))
    return bool(candidate_id and candidate_id.startswith("content_lab_"))


def student_runtime_safe_for_record(record: dict[str, Any]) -> bool:
    if is_content_lab_candidate_record(record):
        return False
    course_id = course_id_for_record(record)
    if not course_id:
        return False
    explicit = _first_present(record, ("student_runtime_safe", "studentRuntimeSafe"))
    if explicit is not None:
        return explicit is True
    roles = record.get("usage_roles") if isinstance(record.get("usage_roles"), dict) else {}
    return course_id == "p3" and roles.get("canonical_practice") == "allow"


def student_runtime_ready_for_record(record: dict[str, Any]) -> bool:
    return student_runtime_safe_for_record(record) and review_status_for_record(record) == "reviewed"


def review_status_for_record(record: dict[str, Any]) -> ReviewStatus:
    explicit = _first_text(record, ("review_status", "reviewStatus"))
    if explicit in REVIEW_STATUSES:
        return explicit  # type: ignore[return-value]
    if is_content_lab_candidate_record(record):
        return "candidate"
    if student_runtime_safe_for_record(record):
        return "reviewed"
    quality_gate = record.get("quality_gate") if isinstance(record.get("quality_gate"), dict) else {}
    reason_codes = _strings(quality_gate.get("reason_codes"))
    hard_block_reasons = {
        "canonical_assets_missing_or_unhashed",
        "missing_question_image_path",
        "missing_question_image_file",
        "missing_mark_scheme_image_path",
        "missing_mark_scheme_image_file",
        "marks_inconsistent",
        "scope_quality_status_fail",
    }
    roles = record.get("usage_roles") if isinstance(record.get("usage_roles"), dict) else {}
    if (
        quality_gate.get("canonical_assets_ok") is False
        or roles.get("canonical_practice") == "block"
        and bool(set(reason_codes).intersection(hard_block_reasons))
    ):
        return "blocked"
    return "needs_review"


def to_exam_bank_course_record(record: dict[str, Any]) -> ExamBankCourseRecord | None:
    course_id = course_id_for_record(record)
    record_id = _first_text(record, ("id", "question_id", "questionId"))
    if not course_id or not record_id or is_content_lab_candidate_record(record):
        return None
    result: ExamBankCourseRecord = {
        "id": record_id,
        "course_id": course_id,
        "component_name": COURSE_COMPONENT_NAMES[course_id],
        "student_runtime_safe": student_runtime_safe_for_record(record),
        "review_status": review_status_for_record(record),
    }
    for key, value in (
        ("paper", _first_text(record, ("paper",))),
        ("topic_id", topic_id_for_record(record)),
        ("source_exam", source_exam_for_record(record)),
        ("question_image_path", question_image_path_for_record(record)),
        ("mark_scheme_image_path", mark_scheme_image_path_for_record(record)),
    ):
        if value:
            result[key] = value  # type: ignore[literal-required]
    return result


def validate_exam_bank_course_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _first_text(record, ("id", "question_id", "questionId")):
        errors.append("missing_id")
    if not course_id_for_record(record):
        errors.append("invalid_course_id")
    explicit_review = _first_text(record, ("review_status", "reviewStatus"))
    if explicit_review and explicit_review not in REVIEW_STATUSES:
        errors.append(f"invalid_review_status:{explicit_review}")
    if is_content_lab_candidate_record(record) and _first_present(record, ("student_runtime_safe", "studentRuntimeSafe")) is True:
        errors.append("content_lab_candidate_cannot_be_student_runtime")
    if student_runtime_safe_for_record(record):
        if review_status_for_record(record) != "reviewed":
            errors.append("student_runtime_safe_review_status_not_reviewed")
        if not question_image_path_for_record(record):
            errors.append("student_runtime_safe_missing_question_image_path")
        if not mark_scheme_image_path_for_record(record):
            errors.append("student_runtime_safe_missing_mark_scheme_image_path")
    return errors


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        return []
    schema = payload.get("schema_name") or payload.get("schema")
    if schema in {CONTENT_LAB_SCHEMA_NAME, RUNTIME_SAFE_CANDIDATE_SCHEMA_NAME}:
        return []
    records = payload.get("questions") or payload.get("records")
    if isinstance(records, dict):
        return [record for record in records.values() if isinstance(record, dict)]
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    return []


def filter_records_by_course(
    records: Iterable[dict[str, Any]],
    course_id: Any,
    *,
    student_runtime_only: bool = False,
) -> list[dict[str, Any]]:
    normalized = normalize_course_id(course_id)
    if not normalized:
        return []
    return [
        record
        for record in records
        if course_id_for_record(record) == normalized
        and (not student_runtime_only or student_runtime_ready_for_record(record))
    ]


def filter_records_by_paper_or_component(
    records: Iterable[dict[str, Any]],
    *,
    paper: str | None = None,
    component_name: str | None = None,
    course_id: Any = None,
    student_runtime_only: bool = False,
) -> list[dict[str, Any]]:
    result = list(records)
    if course_id is not None:
        result = filter_records_by_course(result, course_id, student_runtime_only=False)
    if paper:
        wanted_paper = paper.strip().lower()
        result = [record for record in result if _first_text(record, ("paper", "source_exam", "sourceExam")).lower() == wanted_paper]
    if component_name:
        wanted_component = component_name.strip().lower()
        result = [record for record in result if (component_name_for_record(record) or "").lower() == wanted_component]
    if student_runtime_only:
        result = [record for record in result if student_runtime_ready_for_record(record)]
    return result


def student_runtime_course_records(payload: Any, course_id: Any, *, paper: str | None = None) -> list[ExamBankCourseRecord]:
    records = filter_records_by_paper_or_component(
        records_from_payload(payload),
        course_id=course_id,
        paper=paper,
        student_runtime_only=True,
    )
    return [course_record for record in records if (course_record := to_exam_bank_course_record(record))]


def course_counts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    result = []
    for course_id in COURSE_IDS:
        course_rows = filter_records_by_course(rows, course_id)
        safe_rows = [record for record in course_rows if student_runtime_ready_for_record(record)]
        result.append(
            {
                "course_id": course_id,
                "component_name": COURSE_COMPONENT_NAMES[course_id],
                "paper_families": list(COURSE_ID_TO_PAPER_FAMILIES[course_id]),
                "record_count": len(course_rows),
                "student_runtime_safe_record_count": len(safe_rows),
                "empty_state_message": COURSE_EMPTY_STATE_MESSAGE,
            }
        )
    return result


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        return [str(item) for item in value if str(item)]
    return [str(value)]
