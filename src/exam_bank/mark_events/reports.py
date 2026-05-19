from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.mark_events import MARK_EVENTS_REVIEW_QUEUE_SCHEMA_NAME, MARK_EVENTS_SCHEMA_VERSION
from exam_bank.mark_events.sidecar import sidecar_summary


def write_mark_event_reports(
    sidecar: dict[str, Any],
    *,
    audit_report_path: str | Path | None = None,
    review_queue_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    outputs: list[str] = []
    if audit_report_path:
        if not dry_run:
            path = Path(audit_report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_audit_report(sidecar), encoding="utf-8")
        outputs.append(str(audit_report_path))
    if review_queue_path:
        queue = build_review_queue(sidecar)
        if not dry_run:
            write_atomic_json(queue, review_queue_path)
        outputs.append(str(review_queue_path))
    return {"dry_run": dry_run, "outputs": outputs, "report_count": len(outputs)}


def render_audit_report(sidecar: dict[str, Any]) -> str:
    summary = sidecar_summary(sidecar)
    lines = [
        "# Mark-Event Extraction Audit",
        "",
        "This report is advisory-only. Canonical mark-scheme images remain the source of truth, and generated mark events must not be used for student marking.",
        "",
        "## Summary",
        "",
        f"- Records processed: {summary['record_count']}",
        f"- Mark events detected: {summary['event_count']}",
        f"- Safe for advisory use: {summary['safe_for_advisory_count']}",
        f"- Safe for marking use: {summary['safe_for_marking_count']}",
        f"- Total mismatches: {summary['total_mismatch_count']}",
        f"- Question-total / mark-scheme-total disagreements: {summary['question_total_disagreement_count']}",
        f"- Missing mark-scheme images: {summary['missing_mark_scheme_image_count']}",
        f"- Records with no detected mark events: {summary['no_detected_mark_events_count']}",
        f"- Ambiguous part mapping records: {summary['ambiguous_part_mapping_count']}",
        "",
        "## Extraction Status",
        "",
        *_counter_lines(summary["extraction_status_counts"]),
        "",
        "## Mark Codes",
        "",
        *_counter_lines(summary["mark_code_counts"]),
        "",
        "## Mark Behavior",
        "",
        f"- Follow-through mark events: {summary['follow_through_count']}",
        f"- Dependent mark events: {summary['dependent_count']}",
        "",
        "## Unsafe Advisory Reasons",
        "",
        *_counter_lines(summary["unsafe_for_advisory_reasons"]),
        "",
        "## Top Raw Patterns Not Yet Parsed",
        "",
        *_counter_lines(summary["top_unparsed_patterns"]),
        "",
        "## Candidate Records For Topic/Difficulty Review",
        "",
    ]
    candidates = _candidate_records(sidecar)
    lines.append(f"- Advisory-safe records with method/explanation/dependency evidence: {len(candidates)}")
    for record in candidates[:25]:
        lines.append(
            f"- `{record.get('question_id')}`: {len(record.get('mark_events') or [])} events, "
            f"codes {_compact_code_counts(record)}"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_review_queue(sidecar: dict[str, Any]) -> dict[str, Any]:
    records = []
    for record in sidecar.get("records", []):
        if not isinstance(record, dict):
            continue
        needs_review = (
            record.get("safe_for_advisory_use") is not True
            or record.get("extraction_status") in {"partial", "review", "failed"}
            or bool(record.get("review_flags"))
        )
        if not needs_review:
            continue
        records.append(
            {
                "question_id": record.get("question_id"),
                "paper": record.get("paper"),
                "question_number": record.get("question_number"),
                "extraction_status": record.get("extraction_status"),
                "safe_for_advisory_use": record.get("safe_for_advisory_use"),
                "safe_for_marking_use": record.get("safe_for_marking_use"),
                "review_flags": record.get("review_flags", []),
                "source_mark_scheme_image_path": record.get("source_mark_scheme_image_path"),
                "total_marks_detected": record.get("total_marks_detected"),
                "total_marks_expected": record.get("total_marks_expected"),
                "question_total_detected": record.get("question_total_detected"),
                "total_marks_match": record.get("total_marks_match"),
                "mark_event_count": len(record.get("mark_events") or []),
                "unparsed_evidence": (record.get("unparsed_evidence") or [])[:5],
            }
        )
    return {
        "schema_name": MARK_EVENTS_REVIEW_QUEUE_SCHEMA_NAME,
        "schema_version": MARK_EVENTS_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "source_schema": {
            "schema_name": sidecar.get("schema_name"),
            "schema_version": sidecar.get("schema_version"),
            "record_count": sidecar.get("record_count"),
        },
        "record_count": len(records),
        "records": records,
    }


def _candidate_records(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    useful = {"method", "dependent_method", "explanation", "follow_through"}
    for record in sidecar.get("records", []):
        if not isinstance(record, dict) or record.get("safe_for_advisory_use") is not True:
            continue
        if any(event.get("mark_type") in useful for event in record.get("mark_events", [])):
            output.append(record)
    return output


def _compact_code_counts(record: dict[str, Any]) -> str:
    merged: dict[str, int] = {}
    for summary in record.get("part_summaries", []):
        for key, value in (summary.get("mark_code_counts") or {}).items():
            merged[key] = merged.get(key, 0) + int(value)
    return ", ".join(f"{key}:{merged[key]}" for key in sorted(merged)) or "none"


def _counter_lines(values: dict[str, int]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- `{key}`: {values[key]}" for key in sorted(values)]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
