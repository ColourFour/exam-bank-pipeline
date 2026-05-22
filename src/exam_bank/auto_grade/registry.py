from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.auto_grade.constants import (
    APPROVED_REVIEW_STATUSES,
    DEFAULT_REVIEWED_RUBRICS_PATH,
    REVIEWED_RUBRIC_SCHEMA_VERSION,
    REVIEWED_RUBRICS_SCHEMA,
)
from exam_bank.auto_grade.reviewed_rubrics import validate_reviewed_rubrics_payload


DEFAULT_REGISTRY_REPORT_PATH = "reports/auto_grade/reviewed_rubrics_registry_summary.md"


def promote_reviewed_rubrics_registry(
    *,
    source_reviewed_rubrics_path: str | Path,
    question_bank_path: str | Path,
    output_path: str | Path = DEFAULT_REVIEWED_RUBRICS_PATH,
    mode: str = "replace-approved",
    report_path: str | Path | None = DEFAULT_REGISTRY_REPORT_PATH,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if mode not in {"replace-approved", "merge-approved"}:
        raise ValueError(f"unsupported promotion mode: {mode}")

    source_reviewed_rubrics_path = Path(source_reviewed_rubrics_path)
    question_bank_path = Path(question_bank_path)
    output_path = Path(output_path)
    source = _load_json(source_reviewed_rubrics_path)
    question_bank = _load_json(question_bank_path)
    question_by_id = {str(record.get("question_id") or ""): record for record in _question_records(question_bank)}
    source_rubrics = _rubric_records(source)
    selected = [deepcopy(rubric) for rubric in source_rubrics if _is_approved_for_registry(rubric)]

    if mode == "merge-approved" and output_path.exists():
        existing = _load_json(output_path)
        merged: dict[str, dict[str, Any]] = {
            str(rubric.get("source_question_id") or ""): deepcopy(rubric)
            for rubric in _rubric_records(existing)
            if _is_approved_for_registry(rubric)
        }
        for rubric in selected:
            merged[str(rubric.get("source_question_id") or "")] = rubric
        final_rubrics = [merged[key] for key in sorted(merged)]
    else:
        final_rubrics = sorted(selected, key=lambda rubric: str(rubric.get("source_question_id") or ""))

    event_count = sum(len(rubric.get("events") or []) for rubric in final_rubrics if isinstance(rubric, dict))
    teacher_beta_count = sum(1 for rubric in final_rubrics if rubric.get("safe_for_teacher_beta") is True)
    student_self_check_count = sum(1 for rubric in final_rubrics if rubric.get("safe_for_student_self_check") is True)
    reviewers = sorted(
        {
            str(rubric.get("reviewed_by") or "").strip()
            for rubric in final_rubrics
            if str(rubric.get("reviewed_by") or "").strip()
        }
    )
    approved_question_ids = sorted(str(rubric.get("source_question_id") or "") for rubric in final_rubrics)
    excluded_incomplete = len(source_rubrics) - len(selected)

    payload = {
        "schema": REVIEWED_RUBRICS_SCHEMA,
        "schema_version": REVIEWED_RUBRIC_SCHEMA_VERSION,
        "artifact_kind": "live_approved_reviewed_rubric_registry",
        "generated_at": generated_at or _utc_now_iso(),
        "source_reviewed_rubrics_path": _rel_path(source_reviewed_rubrics_path),
        "source_reviewed_rubrics_sha256": _sha256_file(source_reviewed_rubrics_path),
        "source_review_batch_path": source.get("source_review_batch_path") if isinstance(source, dict) else None,
        "source_question_bank_path": _rel_path(question_bank_path),
        "source_question_bank_sha256": _sha256_file(question_bank_path),
        "promotion_mode": mode,
        "reviewer_identities_represented": reviewers,
        "approved_rubric_count": len(final_rubrics),
        "excluded_incomplete_count": excluded_incomplete,
        "teacher_beta_safe_count": teacher_beta_count,
        "student_self_check_safe_count": student_self_check_count,
        "student_ready_count": 0,
        "rubric_count": len(final_rubrics),
        "event_count": event_count,
        "summary": {
            "approved_count": len(final_rubrics),
            "teacher_beta_safe_count": teacher_beta_count,
            "student_self_check_safe_count": student_self_check_count,
            "student_ready_count": 0,
            "excluded_incomplete_count": excluded_incomplete,
            "source_review_status_counts": dict(
                Counter(str(rubric.get("review_status") or "missing") for rubric in source_rubrics)
            ),
        },
        "rubrics": final_rubrics,
    }

    errors, warnings, accepted_ids = validate_reviewed_rubrics_payload(
        payload,
        question_by_id=question_by_id,
        phase="2D",
    )
    if errors:
        return _promotion_result(
            ok=False,
            output_path=output_path,
            source_path=source_reviewed_rubrics_path,
            payload=payload,
            errors=errors,
            warnings=warnings,
            accepted_ids=accepted_ids,
            report_path=report_path,
            wrote_registry=False,
        )

    write_atomic_json(payload, output_path, sort_keys=True)
    return _promotion_result(
        ok=True,
        output_path=output_path,
        source_path=source_reviewed_rubrics_path,
        payload=payload,
        errors=[],
        warnings=warnings,
        accepted_ids=accepted_ids,
        report_path=report_path,
        wrote_registry=True,
    )


def render_reviewed_rubrics_registry_summary(result: dict[str, Any]) -> str:
    payload = result.get("registry") if isinstance(result.get("registry"), dict) else {}
    source = payload.get("source_reviewed_rubrics_path") or result.get("source_reviewed_rubrics_path")
    output = result.get("output_path")
    promoted_ids = result.get("approved_question_ids") or []
    lines = [
        "# Reviewed Rubrics Registry Summary",
        "",
        "This is a teacher-beta reviewed-rubric registry only. It is not a student-facing self-check release.",
        "",
        "## Sources",
        "",
        f"- Source draft/workspace path: `{source}`",
        f"- Source review batch path: `{payload.get('source_review_batch_path') or 'not recorded'}`",
        f"- Live registry path: `{output}`",
        f"- Promotion mode: `{payload.get('promotion_mode') or 'unknown'}`",
        "",
        "## Promotion Result",
        "",
        f"- OK: {result.get('ok')}",
        f"- Validation status: {'passed' if result.get('ok') else 'failed'}",
        f"- Approved rubrics promoted: {payload.get('approved_rubric_count', 0)}",
        f"- Excluded incomplete rubrics: {payload.get('excluded_incomplete_count', 0)}",
        f"- Teacher-beta-safe rubric count: {payload.get('teacher_beta_safe_count', 0)}",
        f"- Student-self-check-safe rubric count: {payload.get('student_self_check_safe_count', 0)}",
        f"- Student-ready count: {payload.get('student_ready_count', 0)}",
        "",
        "## Promoted Question IDs",
        "",
        *_list_lines(promoted_ids),
        "",
        "## Validation Messages",
        "",
        "- Errors:",
        *_list_lines(result.get("errors") or []),
        "- Warnings:",
        *_list_lines(result.get("warnings") or []),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _promotion_result(
    *,
    ok: bool,
    output_path: Path,
    source_path: Path,
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    accepted_ids: list[str],
    report_path: str | Path | None,
    wrote_registry: bool,
) -> dict[str, Any]:
    result = {
        "ok": ok,
        "wrote_registry": wrote_registry,
        "output_path": _rel_path(output_path),
        "source_reviewed_rubrics_path": _rel_path(source_path),
        "approved_question_ids": accepted_ids,
        "approved_rubric_count": len(accepted_ids),
        "excluded_incomplete_count": payload.get("excluded_incomplete_count", 0),
        "teacher_beta_safe_count": payload.get("teacher_beta_safe_count", 0),
        "student_self_check_safe_count": payload.get("student_self_check_safe_count", 0),
        "student_ready_count": payload.get("student_ready_count", 0),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "registry": payload,
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_reviewed_rubrics_registry_summary(result), encoding="utf-8")
    return result


def _is_approved_for_registry(rubric: Any) -> bool:
    if not isinstance(rubric, dict):
        return False
    return (
        str(rubric.get("review_status") or "") in APPROVED_REVIEW_STATUSES
        and rubric.get("safe_for_auto_grade_lab") is True
    )


def _rubric_records(payload: Any) -> list[dict[str, Any]]:
    records = payload.get("rubrics") if isinstance(payload, dict) else []
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _question_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("questions") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _list_lines(values: list[str]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{value}`" for value in values]
