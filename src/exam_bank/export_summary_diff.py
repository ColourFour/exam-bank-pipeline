from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


SUMMARY_CATEGORIES = (
    "readiness_counts",
    "role_gate_counts",
    "missing_image_counts",
    "topic_safety_counts",
    "reason_code_counts",
)
MAX_RENDERED_SECTION_CHANGES = 30


class ExportSummaryDiffError(ValueError):
    """Raised when two export summaries cannot be compared."""


def compare_export_summaries(before_path: str | Path, after_path: str | Path) -> dict[str, Any]:
    before = summarize_export(before_path)
    after = summarize_export(after_path)
    if before["schema_name"] != after["schema_name"]:
        raise ExportSummaryDiffError(
            f"Cannot compare different schema_name values: {before['schema_name']} vs {after['schema_name']}"
        )
    return {
        "before": before,
        "after": after,
        "deltas": {
            category: _counter_delta(before[category], after[category])
            for category in SUMMARY_CATEGORIES
        },
    }


def summarize_export(path: str | Path) -> dict[str, Any]:
    export_path = Path(path)
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportSummaryDiffError(f"Invalid JSON: {export_path}") from exc
    if not isinstance(payload, dict):
        raise ExportSummaryDiffError(f"Export payload must be a JSON object: {export_path}")

    schema_name = str(payload.get("schema_name") or "unknown")
    if schema_name == "exam_bank.question_bank":
        summary = _summarize_question_bank(payload)
    elif schema_name == "asterion.question_bank":
        summary = _summarize_asterion_question_bank(payload)
    elif schema_name == "asterion.content_lab_candidates":
        summary = _summarize_content_lab_candidates(payload)
    elif schema_name == "exam_bank.topic_routing_sidecar":
        summary = _summarize_topic_routing_sidecar(payload)
    else:
        raise ExportSummaryDiffError(f"Unsupported export schema_name: {schema_name}")

    summary.update(
        {
            "path": str(export_path),
            "schema_name": schema_name,
            "schema_version": payload.get("schema_version"),
            "record_count": _record_count(payload, summary["record_container"]),
            "generated_at": _generated_at(payload),
            "run_id": _run_id(payload),
        }
    )
    return summary


def render_export_summary_diff(report: dict[str, Any]) -> str:
    before = report["before"]
    after = report["after"]
    lines = [
        "Export summary diff",
        f"Before: {before['path']}",
        f"After:  {after['path']}",
        f"Schema: {before['schema_name']} v{_display(before['schema_version'])} -> v{_display(after['schema_version'])}",
        f"Generated: {_display(before['generated_at'])} -> {_display(after['generated_at'])}",
        f"Run ID: {_display(before['run_id'])} -> {_display(after['run_id'])}",
        f"Record count: {_format_delta(before['record_count'], after['record_count'])}",
    ]
    sections = [
        ("Readiness counts", "readiness_counts"),
        ("Role-gate counts", "role_gate_counts"),
        ("Missing image counts", "missing_image_counts"),
        ("Topic safety metadata", "topic_safety_counts"),
        ("Major reason-code counts", "reason_code_counts"),
    ]
    for title, category in sections:
        lines.append("")
        lines.append(f"{title}:")
        deltas = report["deltas"][category]
        if not deltas:
            lines.append("  no comparable counts")
            continue
        changed = _changed_delta_items(deltas)
        if not changed:
            lines.append("  no changes")
            continue
        rendered = changed[:MAX_RENDERED_SECTION_CHANGES]
        for key, values in rendered:
            lines.append(f"  {key}: {_format_delta(values['before'], values['after'])}")
        omitted = len(changed) - len(rendered)
        if omitted > 0:
            lines.append(f"  ... {omitted} more changed count(s) omitted")
    return "\n".join(lines) + "\n"


def _summarize_question_bank(payload: dict[str, Any]) -> dict[str, Any]:
    questions = _list_records(payload.get("questions"))
    readiness: Counter[str] = Counter()
    missing_images: Counter[str] = Counter()
    topic_safety: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    for record in questions:
        notes = _dict_value(record.get("notes"))
        for field in [
            "validation_status",
            "mapping_status",
            "scope_quality_status",
            "text_fidelity_status",
            "visual_curation_status",
            "text_only_status",
            "question_text_role",
            "question_text_trust",
            "topic_trust_status",
        ]:
            value = _field(record, notes, field)
            if value not in (None, "", []):
                readiness[f"{field}.{_safe_value(value)}"] += 1

        for field in ["topic_confidence", "topic_uncertain", "visual_required"]:
            value = _field(record, notes, field)
            if value not in (None, "", []):
                topic_safety[f"{field}.{_safe_value(value)}"] += 1

        if not _path_values(record, "question_image_paths", "question_image_path", "canonical_question_artifact"):
            missing_images["missing_question_image_path"] += 1
        if not _path_values(record, "mark_scheme_image_paths", "mark_scheme_image_path", "canonical_mark_scheme_artifact"):
            missing_images["missing_mark_scheme_image_path"] += 1

        for field in [
            "mapping_failure_reason",
            "ocr_failure_reason",
            "paper_total_focus_reason",
        ]:
            value = _field(record, notes, field)
            if value:
                reasons[f"{field}.{_safe_value(value)}"] += 1
        for field in [
            "visual_reason_flags",
            "review_flags",
            "validation_flags",
            "text_fidelity_flags",
            "extraction_quality_flags",
            "text_candidate_decision_reasons",
            "ocr_rejected_reasons",
            "difficulty_review_flags",
        ]:
            _count_values(reasons, field, _field(record, notes, field))

    qa_summary = _dict_value(_dict_value(payload.get("run_manifest")).get("qa_summary"))
    _merge_prefixed_counts(readiness, "qa.validation_status_counts", qa_summary.get("validation_status_counts"))
    _merge_prefixed_counts(readiness, "qa.mapping_status_counts", qa_summary.get("mapping_status_counts"))
    _merge_prefixed_counts(readiness, "qa.text_only_status_counts", qa_summary.get("text_only_status_counts"))
    _merge_prefixed_counts(readiness, "qa.visual_curation_status_counts", qa_summary.get("visual_curation_status_counts"))
    _merge_prefixed_counts(missing_images, "qa.artifact_path_counts", qa_summary.get("artifact_path_counts"))

    return _summary_payload("questions", readiness, Counter(), missing_images, topic_safety, reasons)


def _summarize_asterion_question_bank(payload: dict[str, Any]) -> dict[str, Any]:
    questions = _list_records(payload.get("questions"))
    readiness: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    missing_images: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    for record in questions:
        gate = _dict_value(record.get("quality_gate"))
        for field, value in gate.items():
            if field != "reason_codes" and isinstance(value, (bool, str, int)):
                readiness[f"quality_gate.{field}.{_safe_value(value)}"] += 1
        for role, status in _dict_value(record.get("usage_roles")).items():
            roles[f"usage_roles.{role}.{_safe_value(status)}"] += 1
        for reason in _string_values(gate.get("reason_codes")):
            reasons[f"quality_gate.{reason}"] += 1
            if reason.startswith("missing_question_image"):
                missing_images[reason] += 1
            if reason.startswith("missing_mark_scheme_image"):
                missing_images[reason] += 1
        for subpart in _list_records(record.get("subparts")):
            review_status = subpart.get("review_status")
            if review_status:
                readiness[f"subparts.review_status.{_safe_value(review_status)}"] += 1
            for event in _list_records(subpart.get("mark_events")):
                status = event.get("review_status")
                if status:
                    readiness[f"mark_events.review_status.{_safe_value(status)}"] += 1
                quarantine_reason = event.get("quarantine_reason")
                if quarantine_reason:
                    reasons[f"mark_events.quarantine_reason.{_safe_value(quarantine_reason)}"] += 1
    return _summary_payload("questions", readiness, roles, missing_images, Counter(), reasons)


def _summarize_content_lab_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = _list_records(payload.get("candidates"))
    readiness: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for candidate in candidates:
        for field in ["review_status"]:
            value = candidate.get(field)
            if value:
                readiness[f"{field}.{_safe_value(value)}"] += 1
        gate = _dict_value(candidate.get("generation_gate"))
        status = gate.get("status")
        if status:
            readiness[f"generation_gate.status.{_safe_value(status)}"] += 1
        _count_values(reasons, "generation_gate.blocker", gate.get("blockers"))
        _count_values(reasons, "generation_gate.reason", gate.get("reason_codes"))
        for role, status in _dict_value(candidate.get("role_statuses")).items():
            roles[f"role_statuses.{role}.{_safe_value(status)}"] += 1
    return _summary_payload("candidates", readiness, roles, Counter(), Counter(), reasons)


def _summarize_topic_routing_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    records = _dict_value(payload.get("records"))
    readiness: Counter[str] = Counter()
    topic_safety: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    run_summary = _dict_value(_dict_value(payload.get("metadata")).get("run_summary"))

    for key in [
        "successful_records",
        "failed_records",
        "review_required_records",
        "strict_filter_records",
        "safe_for_strict_filters",
    ]:
        value = run_summary.get(key)
        if isinstance(value, bool):
            topic_safety[f"{key}.{_safe_value(value)}"] += 1
        elif isinstance(value, int):
            topic_safety[key] += value
    _merge_prefixed_counts(reasons, "failure", run_summary.get("failures_by_reason"))

    for record in records.values():
        if not isinstance(record, dict):
            continue
        confidence = record.get("confidence")
        if confidence:
            readiness[f"confidence.{_safe_value(confidence)}"] += 1
        if "review_required" in record:
            readiness[f"review_required.{_safe_value(record.get('review_required'))}"] += 1
        if record.get("primary_topic_id"):
            topic_safety["records_with_primary_topic_id"] += 1
        if isinstance(record.get("topic_distribution"), list) and record["topic_distribution"]:
            topic_safety["records_with_topic_distribution"] += 1
        _count_values(reasons, "review_reason", record.get("review_reasons"))
        error = _dict_value(record.get("error"))
        if error.get("type"):
            reasons[f"error.{_safe_value(error['type'])}"] += 1
    return _summary_payload("records", readiness, Counter(), Counter(), topic_safety, reasons)


def _summary_payload(
    record_container: str,
    readiness: Counter[str],
    roles: Counter[str],
    missing_images: Counter[str],
    topic_safety: Counter[str],
    reasons: Counter[str],
) -> dict[str, Any]:
    return {
        "record_container": record_container,
        "readiness_counts": _sorted_counter(readiness),
        "role_gate_counts": _sorted_counter(roles),
        "missing_image_counts": _sorted_counter(missing_images),
        "topic_safety_counts": _sorted_counter(topic_safety),
        "reason_code_counts": _sorted_counter(reasons),
    }


def _record_count(payload: dict[str, Any], container: str) -> int:
    value = payload.get("record_count")
    if isinstance(value, int):
        return value
    records = payload.get(container)
    if isinstance(records, (list, dict)):
        return len(records)
    return 0


def _generated_at(payload: dict[str, Any]) -> str:
    run_manifest = _dict_value(payload.get("run_manifest"))
    metadata = _dict_value(payload.get("metadata"))
    metadata_manifest = _dict_value(metadata.get("run_manifest"))
    return str(
        payload.get("generated_at")
        or run_manifest.get("generated_at")
        or metadata_manifest.get("run_timestamp")
        or metadata.get("generated_at")
        or ""
    )


def _run_id(payload: dict[str, Any]) -> str:
    run_manifest = _dict_value(payload.get("run_manifest"))
    metadata = _dict_value(payload.get("metadata"))
    metadata_manifest = _dict_value(metadata.get("run_manifest"))
    return str(run_manifest.get("run_id") or metadata_manifest.get("run_id") or metadata.get("run_id") or "")


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, dict[str, int]]:
    keys = sorted(set(before) | set(after))
    return {
        key: {"before": int(before.get(key, 0)), "after": int(after.get(key, 0))}
        for key in keys
    }


def _changed_delta_items(deltas: dict[str, dict[str, int]]) -> list[tuple[str, dict[str, int]]]:
    changed = [
        (key, values)
        for key, values in deltas.items()
        if values["before"] != values["after"]
    ]
    return sorted(
        changed,
        key=lambda item: (-abs(item[1]["after"] - item[1]["before"]), item[0]),
    )


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _list_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _field(record: dict[str, Any], notes: dict[str, Any], field: str) -> Any:
    if field in record:
        return record.get(field)
    return notes.get(field)


def _path_values(record: dict[str, Any], list_field: str, *single_fields: str) -> list[str]:
    values = _string_values(record.get(list_field))
    for field in single_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return values


def _string_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _count_values(counter: Counter[str], prefix: str, value: Any) -> None:
    for item in _string_values(value):
        counter[f"{prefix}.{_safe_value(item)}"] += 1


def _merge_prefixed_counts(counter: Counter[str], prefix: str, counts: Any) -> None:
    if not isinstance(counts, dict):
        return
    for key, value in counts.items():
        if isinstance(value, int):
            counter[f"{prefix}.{_safe_value(key)}"] += value


def _safe_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text if text else "unknown"


def _display(value: Any) -> str:
    text = str(value).strip()
    return text if text else "n/a"


def _format_delta(before: int, after: int) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{before} -> {after} ({sign}{delta})"
