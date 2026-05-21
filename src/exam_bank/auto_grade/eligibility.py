from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade import AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA, AUTO_GRADE_SCHEMA_VERSION
from exam_bank.auto_grade.constants import DEFAULT_REVIEWED_RUBRICS_PATH
from exam_bank.auto_grade.schemas import ReviewedRubric, load_reviewed_rubrics


def build_eligible_items(
    *,
    question_bank_path: str | Path = "output/json/question_bank.json",
    output_path: str | Path | None = "output/auto_grade/eligible_items.v1.json",
    artifact_root: str | Path = "output",
    reviewed_rubrics_path: str | Path | None = DEFAULT_REVIEWED_RUBRICS_PATH,
    mark_events_path: str | Path | None = "output/json/question_bank.mark_events.v1.json",
    topic_routing_path: str | Path | None = "output/json/question_bank.topic_routing.v1.json",
    generated_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    question_bank = _load_json(question_bank_path)
    questions = _question_records(question_bank)
    rubrics_payload = _load_optional_json(reviewed_rubrics_path)
    rubrics, rubric_errors = load_reviewed_rubrics(rubrics_payload)
    mark_events = _records_by_question_id(_load_optional_json(mark_events_path).get("records", []))
    topic_routing = _topic_routing_by_question_id(_load_optional_json(topic_routing_path).get("records", {}))

    items = [
        build_eligible_item(
            question,
            artifact_root=Path(artifact_root),
            reviewed_rubric=rubrics.get(str(question.get("question_id") or "")),
            mark_event=mark_events.get(str(question.get("question_id") or "")),
            topic_route=topic_routing.get(str(question.get("question_id") or "")),
            question_index=index,
        )
        for index, question in enumerate(questions)
    ]
    payload = {
        "schema": AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA,
        "schema_name": AUTO_GRADE_ELIGIBLE_ITEMS_SCHEMA,
        "schema_version": AUTO_GRADE_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now_iso(),
        "source_question_bank_path": _rel_path(question_bank_path),
        "source_question_bank_sha256": _sha256_file(question_bank_path),
        "source_question_bank_schema": {
            "schema_name": question_bank.get("schema_name") if isinstance(question_bank, dict) else None,
            "schema_version": question_bank.get("schema_version") if isinstance(question_bank, dict) else None,
        },
        "source_sidecars": {
            "reviewed_rubrics_path": _rel_path(reviewed_rubrics_path) if reviewed_rubrics_path else None,
            "reviewed_rubrics_loaded": bool(rubrics),
            "reviewed_rubric_error_count": len(rubric_errors),
            "mark_events_path": _rel_path(mark_events_path) if mark_events_path else None,
            "topic_routing_path": _rel_path(topic_routing_path) if topic_routing_path else None,
        },
        "record_count": len(items),
        "summary": summarize_eligible_items(items),
        "items": items,
    }
    if output_path and not dry_run:
        write_atomic_json(payload, output_path, sort_keys=True)
    return payload


def build_eligible_item(
    question: dict[str, Any],
    *,
    artifact_root: Path,
    reviewed_rubric: ReviewedRubric | None = None,
    mark_event: dict[str, Any] | None = None,
    topic_route: dict[str, Any] | None = None,
    question_index: int = 0,
) -> dict[str, Any]:
    question_id = str(question.get("question_id") or f"record_{question_index:04d}")
    notes = question.get("notes") if isinstance(question.get("notes"), dict) else {}
    question_artifact = _first_path(
        question.get("canonical_question_artifact"),
        question.get("question_image_path"),
        question.get("question_image_paths"),
    )
    mark_scheme_artifact = _first_path(question.get("mark_scheme_image_path"), question.get("mark_scheme_image_paths"))
    question_exists = _path_exists(question_artifact, artifact_root) if question_artifact else False
    mark_scheme_exists = _path_exists(mark_scheme_artifact, artifact_root) if mark_scheme_artifact else False
    total_marks = _first_int(
        question.get("question_solution_marks"),
        question.get("marks"),
        notes.get("question_solution_marks"),
    )
    mark_scheme_total = _first_int(
        _nested(notes, "mark_scheme_structure_detected", "mark_scheme_total_detected"),
        notes.get("mark_scheme_total_detected"),
        mark_event.get("total_marks_detected") if mark_event else None,
    )
    question_total = _first_int(
        _nested(notes, "question_structure_detected", "question_total_detected"),
        notes.get("question_total_detected"),
        mark_event.get("question_total_detected") if mark_event else None,
        total_marks,
    )
    block_reasons: list[str] = []
    if not question_artifact:
        block_reasons.append("missing_canonical_question_artifact_path")
    elif not question_exists:
        block_reasons.append("missing_canonical_question_image")
    if not mark_scheme_artifact:
        block_reasons.append("missing_canonical_mark_scheme_artifact_path")
    elif not mark_scheme_exists:
        block_reasons.append("missing_canonical_mark_scheme_image")
    if mark_scheme_total is not None and question_total is not None and mark_scheme_total != question_total:
        block_reasons.append("unresolved_total_mismatch")
    if mark_event and mark_event.get("total_marks_match") is False:
        block_reasons.append("advisory_mark_event_total_mismatch")
    if str(notes.get("mapping_status") or "").lower() == "fail":
        block_reasons.append("canonical_mapping_status_fail")
    if str(notes.get("validation_status") or "").lower() == "fail":
        block_reasons.append("canonical_validation_status_fail")

    learning_target_ids = _learning_target_ids(question, topic_route)
    if not learning_target_ids:
        block_reasons.append("missing_learning_target_ids_for_student_modes")
    if _topic_route_requires_review(topic_route):
        block_reasons.append("topic_routing_not_safe_for_student_learning_target_feedback")

    has_reviewed_rubric = reviewed_rubric is not None and reviewed_rubric.is_approved
    if not has_reviewed_rubric:
        block_reasons.append("missing_reviewed_rubric")
        block_reasons.append("advisory_mark_events_not_scoring_contract")
    elif reviewed_rubric.total_marks is not None and total_marks is not None and reviewed_rubric.total_marks != total_marks:
        block_reasons.append("reviewed_rubric_total_mismatch")

    hard_blocks = {
        "missing_canonical_question_artifact_path",
        "missing_canonical_question_image",
        "missing_canonical_mark_scheme_artifact_path",
        "missing_canonical_mark_scheme_image",
        "unresolved_total_mismatch",
        "advisory_mark_event_total_mismatch",
        "canonical_mapping_status_fail",
        "canonical_validation_status_fail",
        "reviewed_rubric_total_mismatch",
    }
    if set(block_reasons) & hard_blocks:
        status = "blocked"
    elif has_reviewed_rubric and "teacher_beta" in reviewed_rubric.approved_for:
        status = "teacher_beta"
    else:
        status = "review_only"

    return {
        "question_id": question_id,
        "paper": question.get("paper"),
        "paper_family": question.get("paper_family"),
        "question_number": question.get("question_number"),
        "canonical_question_artifact": question_artifact,
        "canonical_mark_scheme_artifact": mark_scheme_artifact,
        "canonical_question_artifact_exists": question_exists,
        "canonical_mark_scheme_artifact_exists": mark_scheme_exists,
        "total_marks": total_marks,
        "supported_submission_modes": [],
        "supported_grading_mode": "reviewed_rubric_beta" if status == "teacher_beta" else "none",
        "rubric_id": reviewed_rubric.rubric_id if has_reviewed_rubric else None,
        "learning_target_ids": learning_target_ids,
        "eligibility_status": status,
        "block_reasons": _dedupe(block_reasons),
        "reviewed_by": reviewed_rubric.reviewed_by if has_reviewed_rubric else None,
        "reviewed_at": reviewed_rubric.reviewed_at if has_reviewed_rubric else None,
    }


def summarize_eligible_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(item.get("eligibility_status") or "missing") for item in items)
    reason_counts = Counter(reason for item in items for reason in item.get("block_reasons") or [])
    blocked_or_review = [item for item in items if item.get("eligibility_status") in {"blocked", "review_only"}]
    actionable = [item for item in blocked_or_review if item.get("block_reasons")]
    return {
        "record_count": len(items),
        "status_counts": dict(status_counts),
        "top_block_reasons": dict(reason_counts.most_common(20)),
        "canonical_question_image_present_count": sum(1 for item in items if item.get("canonical_question_artifact_exists") is True),
        "canonical_question_image_missing_count": sum(1 for item in items if item.get("canonical_question_artifact_exists") is not True),
        "canonical_mark_scheme_image_present_count": sum(1 for item in items if item.get("canonical_mark_scheme_artifact_exists") is True),
        "canonical_mark_scheme_image_missing_count": sum(1 for item in items if item.get("canonical_mark_scheme_artifact_exists") is not True),
        "missing_reviewed_rubric_count": reason_counts.get("missing_reviewed_rubric", 0),
        "future_rubric_review_candidate_count": sum(
            1
            for item in items
            if item.get("eligibility_status") == "review_only" and "missing_reviewed_rubric" in set(item.get("block_reasons") or [])
        ),
        "blocked_or_review_only_with_actionable_reasons_count": len(actionable),
        "blocked_or_review_only_actionable_reason_percent": round((len(actionable) / len(blocked_or_review)) * 100, 2)
        if blocked_or_review
        else 100.0,
        "student_ready_count": status_counts.get("student_ready", 0),
        "student_self_check_beta_count": status_counts.get("student_self_check_beta", 0),
    }


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _records_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    if isinstance(records, dict):
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}
    if not isinstance(records, list):
        return {}
    return {str(record.get("question_id")): record for record in records if isinstance(record, dict) and record.get("question_id")}


def _topic_routing_by_question_id(records: Any) -> dict[str, dict[str, Any]]:
    return _records_by_question_id(records)


def _learning_target_ids(question: dict[str, Any], topic_route: dict[str, Any] | None) -> list[str]:
    ids: list[str] = []
    for key in ("learning_target_ids", "skill_ids", "subtopic_ids"):
        value = question.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).strip())
    if topic_route and topic_route.get("review_required") is not True and topic_route.get("confidence") == "high":
        topic_id = str(topic_route.get("primary_topic_id") or "").strip()
        if topic_id:
            ids.append(topic_id)
    return sorted(set(ids))


def _topic_route_requires_review(topic_route: dict[str, Any] | None) -> bool:
    if not topic_route:
        return True
    return topic_route.get("review_required") is True or str(topic_route.get("confidence") or "").lower() != "high"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_optional_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return _load_json(path)


def _first_path(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def _path_exists(path_value: str, artifact_root: Path) -> bool:
    path = Path(path_value)
    if path.is_absolute():
        return path.is_file()
    return (artifact_root / path).is_file() or path.is_file()


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value in ("", None):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _dedupe(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(str(value) for value in values if str(value)))


def _rel_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
