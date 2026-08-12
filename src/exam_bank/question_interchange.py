from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from typing import Any

from .atomic_json import write_atomic_json


QUESTION_EXPORT_SCHEMA_NAME = "exam_bank.interchange.questions"
QUESTION_EXPORT_SCHEMA_VERSION = 1
QUESTION_EXPORT_VALIDATION_SCHEMA_NAME = "exam_bank.interchange.questions.validation"
QUESTION_SCHEMA_NAME = "exam_bank.interchange.question"
QUESTION_SCHEMA_VERSION = 1
QUESTION_SCHEMA_ID = "question.v1.schema.json"
SOURCE_QUESTION_BANK_SCHEMA_NAME = "exam_bank.question_bank"
SOURCE_QUESTION_BANK_SCHEMA_VERSION = 2
DEFAULT_QUESTION_BANK_PATH = Path("output/json/question_bank.json")
DEFAULT_QUESTION_EXPORT_PATH = Path("output/interchange/questions.v1.json")
QUESTION_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / QUESTION_SCHEMA_ID

_QUESTION_KEYS = {
    "schema_name",
    "schema_version",
    "question_id",
    "paper",
    "question_number",
    "paper_family",
    "topic",
    "subtopic",
    "question_text",
    "question_image",
    "mark_scheme_image",
    "mark_scheme_text",
    "max_marks",
    "rubric_status",
    "rubric",
    "quality",
    "metadata",
}
_QUALITY_KEYS = {
    "mapping_status",
    "validation_status",
    "question_text_trust",
    "visual_curation_status",
    "text_only_status",
    "mark_scheme_confidence",
    "review_flags",
}
_RUBRIC_KEYS = {
    "mark_id",
    "mark_code",
    "mark_type",
    "max_marks",
    "criteria",
    "depends_on",
    "follow_through",
    "accepted_evidence",
    "metadata",
}


class QuestionInterchangeError(ValueError):
    pass


def export_question_interchange(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    output_path: str | Path = DEFAULT_QUESTION_EXPORT_PATH,
    artifact_root: str | Path | None = None,
    generated_at: str | None = None,
    check_assets: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    output_path = Path(output_path)
    question_bank = _read_json_object(question_bank_path)
    source_questions = _source_questions(question_bank)
    effective_artifact_root = Path(artifact_root) if artifact_root is not None else question_bank_path.parent.parent
    questions = [_to_interchange_question(record, index=index) for index, record in enumerate(source_questions)]
    questions.sort(key=lambda item: item["question_id"])

    payload = {
        "schema_name": QUESTION_EXPORT_SCHEMA_NAME,
        "schema_version": QUESTION_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "source": {
            "schema_name": question_bank["schema_name"],
            "schema_version": question_bank["schema_version"],
            "path": _relative_path(question_bank_path, output_path.parent),
            "sha256": _sha256_file(question_bank_path),
            "record_count": len(source_questions),
        },
        "question_schema": {
            "id": QUESTION_SCHEMA_ID,
            "sha256": _question_schema_sha256(),
        },
        "asset_root": _relative_path(effective_artifact_root, output_path.parent),
        "record_count": len(questions),
        "questions": questions,
    }
    report = validate_question_interchange(
        payload,
        document_path=output_path,
        artifact_root=effective_artifact_root,
        check_assets=check_assets,
    )
    if not report["ok"]:
        raise QuestionInterchangeError("; ".join(str(error) for error in report["errors"][:10]))
    write_atomic_json(payload, output_path, sort_keys=True)
    return payload


def validate_question_interchange_file(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    check_assets: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    payload = _read_json_object(path)
    return validate_question_interchange(
        payload,
        document_path=path,
        artifact_root=artifact_root,
        check_assets=check_assets,
    )


def validate_question_interchange(
    payload: Any,
    *,
    document_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    check_assets: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        errors.append("document_not_object")
        return _validation_report(errors=errors, warnings=warnings, record_count=0)

    expected_keys = {
        "schema_name",
        "schema_version",
        "generated_at",
        "source",
        "question_schema",
        "asset_root",
        "record_count",
        "questions",
    }
    _check_unknown_keys(payload, expected_keys, "document", errors)
    if payload.get("schema_name") != QUESTION_EXPORT_SCHEMA_NAME:
        errors.append("schema_name_mismatch")
    if payload.get("schema_version") != QUESTION_EXPORT_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if not _is_datetime_string(payload.get("generated_at")):
        errors.append("generated_at_invalid")
    if not _nonempty_string(payload.get("asset_root")):
        errors.append("asset_root_missing")

    _validate_source(payload.get("source"), errors)
    _validate_question_schema_reference(payload.get("question_schema"), errors)

    questions_value = payload.get("questions")
    questions = questions_value if isinstance(questions_value, list) else []
    if not isinstance(questions_value, list):
        errors.append("questions_not_array")
    record_count = payload.get("record_count")
    if not _is_nonnegative_int(record_count):
        errors.append("record_count_invalid")
    elif record_count != len(questions):
        errors.append(f"record_count_mismatch:declared={record_count}:actual={len(questions)}")
    source_value = payload.get("source")
    if isinstance(source_value, dict) and _is_nonnegative_int(source_value.get("record_count")):
        source_record_count = source_value["record_count"]
        if source_record_count != len(questions):
            errors.append(
                f"source:record_count_mismatch:declared={source_record_count}:actual={len(questions)}"
            )

    resolved_asset_root = _resolve_asset_root(
        payload,
        document_path=Path(document_path) if document_path is not None else None,
        artifact_root=Path(artifact_root) if artifact_root is not None else None,
    )
    if check_assets and resolved_asset_root is None:
        errors.append("asset_root_unresolvable")

    seen_question_ids: set[str] = set()
    for index, question in enumerate(questions):
        prefix = f"question[{index}]"
        _validate_question(question, prefix=prefix, errors=errors, warnings=warnings)
        if not isinstance(question, dict):
            continue
        question_id = str(question.get("question_id") or "")
        if question_id in seen_question_ids:
            errors.append(f"{prefix}:duplicate_question_id:{question_id}")
        elif question_id:
            seen_question_ids.add(question_id)
        if check_assets and resolved_asset_root is not None:
            _check_question_assets(question, prefix=prefix, asset_root=resolved_asset_root, errors=errors)

    return _validation_report(errors=errors, warnings=warnings, record_count=len(questions))


def _source_questions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_name") != SOURCE_QUESTION_BANK_SCHEMA_NAME:
        raise QuestionInterchangeError(f"Unexpected question-bank schema: {payload.get('schema_name')!r}")
    if payload.get("schema_version") != SOURCE_QUESTION_BANK_SCHEMA_VERSION:
        raise QuestionInterchangeError(f"Unsupported question-bank schema version: {payload.get('schema_version')!r}")
    records = payload.get("questions")
    if not isinstance(records, list):
        raise QuestionInterchangeError("Question bank must contain a questions array")
    declared_count = payload.get("record_count")
    if not _is_nonnegative_int(declared_count) or declared_count != len(records):
        raise QuestionInterchangeError(
            f"Question-bank record_count mismatch: declared={declared_count!r}, actual={len(records)}"
        )
    if any(not isinstance(record, dict) for record in records):
        raise QuestionInterchangeError("Question-bank questions must all be objects")
    question_ids = [str(record.get("question_id") or "").strip() for record in records]
    if any(not question_id for question_id in question_ids):
        raise QuestionInterchangeError("Question-bank question_id values must be non-empty")
    if len(set(question_ids)) != len(question_ids):
        raise QuestionInterchangeError("Question-bank question_id values must be unique")
    return records


def _to_interchange_question(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    question_id = _required_string(record.get("question_id"), f"question[{index}].question_id")
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    rubric = _normalize_rubric(record.get("rubric"), question_id=question_id)
    explicit_rubric_status = str(record.get("rubric_status") or "").strip()
    if explicit_rubric_status:
        rubric_status = explicit_rubric_status
    else:
        rubric_status = "included" if rubric else "not_included"
    mark_scheme_confidence = _optional_confidence(
        record.get("mark_scheme_confidence_score", notes.get("mark_scheme_confidence_score"))
    )
    review_flags = _string_list(notes.get("review_flags"))
    review_flags = _dedupe([*review_flags, *_string_list(notes.get("validation_flags"))])

    return {
        "schema_name": QUESTION_SCHEMA_NAME,
        "schema_version": QUESTION_SCHEMA_VERSION,
        "question_id": question_id,
        "paper": _required_string(
            record.get("paper") or record.get("canonical_paper_id"),
            f"{question_id}.paper",
        ),
        "question_number": _required_string(record.get("question_number"), f"{question_id}.question_number"),
        "paper_family": str(record.get("paper_family") or ""),
        "topic": str(record.get("topic") or ""),
        "subtopic": str(notes.get("subtopic") or record.get("subtopic") or ""),
        "question_text": str(record.get("question_text") or ""),
        "question_image": _first_asset_path(
            record.get("canonical_question_artifact"),
            record.get("question_image_path"),
            record.get("question_image_paths"),
            label=f"{question_id}.question_image",
        ),
        "mark_scheme_image": _first_asset_path(
            record.get("canonical_mark_scheme_artifact"),
            record.get("mark_scheme_image_path"),
            record.get("mark_scheme_image_paths"),
            label=f"{question_id}.mark_scheme_image",
        ),
        "mark_scheme_text": str(record.get("mark_scheme_text") or ""),
        "max_marks": _first_optional_int(
            record.get("question_solution_marks"),
            record.get("marks"),
            notes.get("question_total_detected"),
        ),
        "rubric_status": rubric_status,
        "rubric": rubric,
        "quality": {
            "mapping_status": str(notes.get("mapping_status") or ""),
            "validation_status": str(notes.get("validation_status") or ""),
            "question_text_trust": str(
                record.get("question_text_trust") or notes.get("question_text_trust") or ""
            ),
            "visual_curation_status": str(
                record.get("visual_curation_status") or notes.get("visual_curation_status") or ""
            ),
            "text_only_status": str(record.get("text_only_status") or notes.get("text_only_status") or ""),
            "mark_scheme_confidence": mark_scheme_confidence,
            "review_flags": review_flags,
        },
        "metadata": {
            "canonical_paper_id": str(record.get("canonical_paper_id") or record.get("paper") or ""),
            "canonical_session": str(record.get("canonical_session") or ""),
            "canonical_year_folder": str(record.get("canonical_year_folder") or ""),
            "difficulty": str(record.get("difficulty") or notes.get("difficulty") or ""),
            "difficulty_score": record.get("difficulty_score", notes.get("difficulty_score")),
            "difficulty_band": str(record.get("difficulty_band") or ""),
            "subparts": record.get("subparts") if isinstance(record.get("subparts"), list) else [],
            "page_refs": record.get("page_refs") if isinstance(record.get("page_refs"), dict) else {},
            "topic_confidence": str(notes.get("topic_confidence") or ""),
            "topic_uncertain": bool(notes.get("topic_uncertain", False)),
        },
    }


def _normalize_rubric(value: Any, *, question_id: str) -> list[dict[str, Any]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise QuestionInterchangeError(f"{question_id}.rubric must be an array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise QuestionInterchangeError(f"{question_id}.rubric[{index}] must be an object")
        mark_id = _required_string(raw.get("mark_id") or raw.get("event_id"), f"{question_id}.rubric[{index}].mark_id")
        mark_code = _required_string(
            raw.get("mark_code") or raw.get("mark_code_raw"),
            f"{question_id}.rubric[{index}].mark_code",
        )
        max_marks = _required_positive_int(
            raw.get("max_marks", raw.get("mark_value", raw.get("points"))),
            f"{question_id}.rubric[{index}].max_marks",
        )
        normalized.append(
            {
                "mark_id": mark_id,
                "mark_code": mark_code,
                "mark_type": _required_string(
                    raw.get("mark_type") or mark_code.rstrip("0123456789") or "unknown",
                    f"{question_id}.rubric[{index}].mark_type",
                ),
                "max_marks": max_marks,
                "criteria": str(
                    raw.get("criteria")
                    or raw.get("description")
                    or raw.get("answer_text")
                    or raw.get("condition_text")
                    or ""
                ),
                "depends_on": _dedupe(
                    _string_list(raw.get("depends_on") or raw.get("depends_on_event_ids"))
                ),
                "follow_through": bool(raw.get("follow_through") or raw.get("is_follow_through")),
                "accepted_evidence": _string_list(raw.get("accepted_evidence")),
                "metadata": {
                    "review_status": str(raw.get("review_status") or ""),
                    "learning_target_ids": _string_list(raw.get("learning_target_ids")),
                    "alternative_methods": raw.get("alternative_methods")
                    if isinstance(raw.get("alternative_methods"), list)
                    else [],
                },
            }
        )
    return normalized


def _validate_source(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("source_not_object")
        return
    expected = {"schema_name", "schema_version", "path", "sha256", "record_count"}
    _check_unknown_keys(value, expected, "source", errors)
    if value.get("schema_name") != SOURCE_QUESTION_BANK_SCHEMA_NAME:
        errors.append("source:schema_name_mismatch")
    if value.get("schema_version") != SOURCE_QUESTION_BANK_SCHEMA_VERSION:
        errors.append("source:schema_version_mismatch")
    if not _nonempty_string(value.get("path")):
        errors.append("source:path_missing")
    if not _sha256_string(value.get("sha256")):
        errors.append("source:sha256_invalid")
    if not _is_nonnegative_int(value.get("record_count")):
        errors.append("source:record_count_invalid")


def _validate_question_schema_reference(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("question_schema_not_object")
        return
    _check_unknown_keys(value, {"id", "sha256"}, "question_schema", errors)
    if value.get("id") != QUESTION_SCHEMA_ID:
        errors.append("question_schema:id_mismatch")
    if not _sha256_string(value.get("sha256")):
        errors.append("question_schema:sha256_invalid")
        return
    try:
        expected_sha256 = _question_schema_sha256()
    except FileNotFoundError:
        errors.append("question_schema:local_schema_missing")
        return
    if value.get("sha256") != expected_sha256:
        errors.append("question_schema:sha256_mismatch")


def _validate_question(
    value: Any,
    *,
    prefix: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}:not_object")
        return
    _check_unknown_keys(value, _QUESTION_KEYS, prefix, errors)
    if value.get("schema_name") != QUESTION_SCHEMA_NAME:
        errors.append(f"{prefix}:schema_name_mismatch")
    if value.get("schema_version") != QUESTION_SCHEMA_VERSION:
        errors.append(f"{prefix}:schema_version_mismatch")
    for field in ("question_id", "paper", "question_number"):
        if not _nonempty_string(value.get(field)):
            errors.append(f"{prefix}:{field}_missing")
    for field in ("paper_family", "topic", "subtopic", "question_text", "mark_scheme_text"):
        if not isinstance(value.get(field), str):
            errors.append(f"{prefix}:{field}_not_string")
    for field in ("question_image", "mark_scheme_image"):
        path_value = value.get(field)
        if path_value is not None:
            path_error = _asset_path_error(path_value)
            if path_error:
                errors.append(f"{prefix}:{field}_{path_error}")
        else:
            warnings.append(f"{prefix}:{field}_missing")
    max_marks = value.get("max_marks")
    if max_marks is not None and not _is_nonnegative_int(max_marks):
        errors.append(f"{prefix}:max_marks_invalid")

    rubric = value.get("rubric")
    if not isinstance(rubric, list):
        errors.append(f"{prefix}:rubric_not_array")
        rubric = []
    rubric_status = value.get("rubric_status")
    if rubric_status not in {"not_included", "included", "review_required"}:
        errors.append(f"{prefix}:rubric_status_invalid")
    if rubric and rubric_status == "not_included":
        errors.append(f"{prefix}:rubric_status_not_included_with_items")
    if not rubric and rubric_status == "included":
        errors.append(f"{prefix}:rubric_status_included_without_items")
    if not rubric:
        warnings.append(f"{prefix}:rubric_not_included")
    seen_mark_ids: set[str] = set()
    for rubric_index, item in enumerate(rubric):
        item_prefix = f"{prefix}.rubric[{rubric_index}]"
        _validate_rubric_item(item, prefix=item_prefix, errors=errors)
        if isinstance(item, dict):
            mark_id = str(item.get("mark_id") or "")
            if mark_id in seen_mark_ids:
                errors.append(f"{item_prefix}:duplicate_mark_id:{mark_id}")
            elif mark_id:
                seen_mark_ids.add(mark_id)

    quality = value.get("quality")
    if not isinstance(quality, dict):
        errors.append(f"{prefix}:quality_not_object")
    else:
        _check_unknown_keys(quality, _QUALITY_KEYS, f"{prefix}.quality", errors)
        for field in _QUALITY_KEYS - {"mark_scheme_confidence", "review_flags"}:
            if not isinstance(quality.get(field), str):
                errors.append(f"{prefix}.quality:{field}_not_string")
        confidence = quality.get("mark_scheme_confidence")
        if confidence is not None and not _is_confidence(confidence):
            errors.append(f"{prefix}.quality:mark_scheme_confidence_invalid")
        if not _is_string_array(quality.get("review_flags")):
            errors.append(f"{prefix}.quality:review_flags_not_string_array")
    if not isinstance(value.get("metadata"), dict):
        errors.append(f"{prefix}:metadata_not_object")


def _validate_rubric_item(value: Any, *, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}:not_object")
        return
    _check_unknown_keys(value, _RUBRIC_KEYS, prefix, errors)
    for field in ("mark_id", "mark_code", "mark_type"):
        if not _nonempty_string(value.get(field)):
            errors.append(f"{prefix}:{field}_missing")
    if not _is_positive_int(value.get("max_marks")):
        errors.append(f"{prefix}:max_marks_invalid")
    if not isinstance(value.get("criteria"), str):
        errors.append(f"{prefix}:criteria_not_string")
    for field in ("depends_on", "accepted_evidence"):
        if not _is_string_array(value.get(field)):
            errors.append(f"{prefix}:{field}_not_string_array")
    if not isinstance(value.get("follow_through"), bool):
        errors.append(f"{prefix}:follow_through_not_boolean")
    if not isinstance(value.get("metadata"), dict):
        errors.append(f"{prefix}:metadata_not_object")


def _check_question_assets(
    question: dict[str, Any],
    *,
    prefix: str,
    asset_root: Path,
    errors: list[str],
) -> None:
    resolved_root = asset_root.resolve()
    for field in ("question_image", "mark_scheme_image"):
        value = question.get(field)
        if value is None or _asset_path_error(value):
            continue
        candidate = (resolved_root / PurePosixPath(value)).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            errors.append(f"{prefix}:{field}_escapes_asset_root")
            continue
        if not candidate.is_file():
            errors.append(f"{prefix}:{field}_file_missing:{value}")


def _resolve_asset_root(
    payload: dict[str, Any],
    *,
    document_path: Path | None,
    artifact_root: Path | None,
) -> Path | None:
    if artifact_root is not None:
        return artifact_root
    asset_root = payload.get("asset_root")
    if document_path is None or not _nonempty_string(asset_root):
        return None
    return document_path.parent / str(asset_root)


def _asset_path_error(value: Any) -> str:
    if not _nonempty_string(value):
        return "invalid"
    text = str(value)
    if "\\" in text:
        return "not_posix"
    if "://" in text:
        return "not_relative"
    posix = PurePosixPath(text)
    if posix.is_absolute() or PureWindowsPath(text).is_absolute():
        return "not_relative"
    if ".." in posix.parts:
        return "contains_parent_traversal"
    return ""


def _first_asset_path(*values: Any, label: str) -> str | None:
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if not _nonempty_string(candidate):
                continue
            path_error = _asset_path_error(candidate)
            if path_error:
                raise QuestionInterchangeError(f"{label} {path_error}: {candidate!r}")
            return str(candidate)
    return None


def _validation_report(*, errors: list[str], warnings: list[str], record_count: int) -> dict[str, Any]:
    return {
        "schema_name": QUESTION_EXPORT_VALIDATION_SCHEMA_NAME,
        "schema_version": 1,
        "ok": not errors,
        "record_count": record_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QuestionInterchangeError(f"JSON document must be an object: {path}")
    return payload


def _question_schema_bytes() -> bytes:
    try:
        return resources.files("exam_bank").joinpath("schemas", QUESTION_SCHEMA_ID).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError):
        if QUESTION_SCHEMA_PATH.is_file():
            return QUESTION_SCHEMA_PATH.read_bytes()
        raise FileNotFoundError(f"Missing packaged Question schema: {QUESTION_SCHEMA_ID}") from None


def _question_schema_sha256() -> str:
    return hashlib.sha256(_question_schema_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve()).replace(os.sep, "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _required_string(value: Any, label: str) -> str:
    if not _nonempty_string(value):
        raise QuestionInterchangeError(f"Missing {label}")
    return str(value).strip()


def _required_positive_int(value: Any, label: str) -> int:
    converted = _optional_int(value)
    if converted is None or converted < 1:
        raise QuestionInterchangeError(f"Invalid {label}: {value!r}")
    return converted


def _first_optional_int(*values: Any) -> int | None:
    for value in values:
        converted = _optional_int(value)
        if converted is not None:
            return converted
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _optional_confidence(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return round(confidence, 3) if 0 <= confidence <= 1 else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _check_unknown_keys(value: dict[str, Any], expected: set[str], prefix: str, errors: list[str]) -> None:
    for key in sorted(set(value) - expected):
        errors.append(f"{prefix}:unknown_field:{key}")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_datetime_string(value: Any) -> bool:
    if not _nonempty_string(value):
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _sha256_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _is_string_array(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def run_export(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export canonical exam-bank questions through the v1 interchange contract.")
    parser.add_argument("--input", type=Path, default=DEFAULT_QUESTION_BANK_PATH, help="Canonical question_bank.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_QUESTION_EXPORT_PATH, help="Question export JSON path.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Root for question and mark-scheme assets. Defaults to the question bank's output root.",
    )
    parser.add_argument("--check-assets", action="store_true", help="Fail when a referenced image file is missing.")
    args = parser.parse_args(argv)
    try:
        payload = export_question_interchange(
            question_bank_path=args.input,
            output_path=args.output,
            artifact_root=args.artifact_root,
            check_assets=bool(args.check_assets),
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError, QuestionInterchangeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "record_count": payload["record_count"],
                "question_schema_sha256": payload["question_schema"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_validate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a v1 exam-bank Question interchange export.")
    parser.add_argument("--input", type=Path, default=DEFAULT_QUESTION_EXPORT_PATH, help="Question export JSON path.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Optional local asset-root override.")
    parser.add_argument("--check-assets", action="store_true", help="Fail when a referenced image file is missing.")
    args = parser.parse_args(argv)
    try:
        report = validate_question_interchange_file(
            args.input,
            artifact_root=args.artifact_root,
            check_assets=bool(args.check_assets),
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError, QuestionInterchangeError) as exc:
        report = _validation_report(errors=[str(exc)], warnings=[], record_count=0)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(run_export())
