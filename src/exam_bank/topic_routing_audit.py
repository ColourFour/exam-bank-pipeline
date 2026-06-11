from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_SCHEMA_NAME = "exam_bank.topic_routing_baseline_audit"
AUDIT_SCHEMA_VERSION = 1

DEFAULT_QUESTION_BANK = Path("output/json/question_bank.json")
DEFAULT_TOPIC_ROUTING = Path("output/json/question_bank.topic_routing.v1.json")
DEFAULT_MARK_EVENTS = Path("output/json/question_bank.mark_events.v1.json")
DEFAULT_CATALOG = Path("output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json")
DEFAULT_RUNTIME = Path("output/asterion/exports/latest/asterion_question_bank_v1.json")
DEFAULT_AUTO_GRADE = Path("output/auto_grade/eligible_items.v1.json")

COURSE_METADATA_FIELDS = ("course_id", "component_name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit current topic-routing baseline without mutating generated outputs.")
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--topic-routing", type=Path, default=DEFAULT_TOPIC_ROUTING)
    parser.add_argument("--mark-events", type=Path, default=DEFAULT_MARK_EVENTS)
    parser.add_argument("--asterion-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--asterion-runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--auto-grade-eligibility", type=Path, default=DEFAULT_AUTO_GRADE)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for stable JSON output.")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional path for Markdown output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_topic_routing_baseline_audit(
        question_bank_path=args.question_bank,
        topic_routing_path=args.topic_routing,
        mark_events_path=args.mark_events,
        asterion_catalog_path=args.asterion_catalog,
        asterion_runtime_path=args.asterion_runtime,
        auto_grade_eligibility_path=args.auto_grade_eligibility,
    )
    if args.json_out:
        write_json(args.json_out, audit)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    return 0


def build_topic_routing_baseline_audit(
    *,
    question_bank_path: Path = DEFAULT_QUESTION_BANK,
    topic_routing_path: Path = DEFAULT_TOPIC_ROUTING,
    mark_events_path: Path = DEFAULT_MARK_EVENTS,
    asterion_catalog_path: Path = DEFAULT_CATALOG,
    asterion_runtime_path: Path = DEFAULT_RUNTIME,
    auto_grade_eligibility_path: Path = DEFAULT_AUTO_GRADE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    qbank_payload = read_required_json(question_bank_path)
    topic_payload = read_required_json(topic_routing_path)
    qbank_records = records_from_payload(qbank_payload, "questions")
    route_rows = route_records_from_payload(topic_payload)
    qbank_by_id = {
        str(record.get("question_id")): record
        for record in qbank_records
        if isinstance(record, dict) and str(record.get("question_id") or "").strip()
    }

    optional_inputs = {
        "mark_events": file_state(mark_events_path),
        "asterion_catalog": file_state(asterion_catalog_path),
        "asterion_runtime": file_state(asterion_runtime_path),
        "auto_grade_eligibility": file_state(auto_grade_eligibility_path),
    }
    mark_events_payload = read_optional_json(mark_events_path)
    catalog_payload = read_optional_json(asterion_catalog_path)
    runtime_payload = read_optional_json(asterion_runtime_path)
    auto_grade_payload = read_optional_json(auto_grade_eligibility_path)

    question_bank = summarize_question_bank(qbank_records)
    topic_routing = summarize_topic_routing(topic_payload, route_rows, qbank_by_id)
    failure_analysis = summarize_failures(route_rows)
    review_required = summarize_review_required(route_rows, qbank_by_id)
    mark_events = summarize_mark_events(mark_events_payload, mark_events_path)
    asterion = summarize_asterion(catalog_payload, runtime_payload)
    auto_grade = summarize_auto_grade(auto_grade_payload, auto_grade_eligibility_path)

    summary = {
        "question_bank_total_records": question_bank["total_records"],
        "topic_routing_total_records": topic_routing["total_route_records"],
        "topic_routing_failed_records": topic_routing["failed_count"],
        "topic_routing_review_required_records": topic_routing["review_required_count"],
        "topic_routing_strict_filter_candidates": topic_routing["strict_filter_candidate_count"],
        "topic_routing_safe_for_strict_filters": topic_routing["safe_for_strict_filters"],
        "asterion_catalog_count": asterion["all_course_catalog_count"],
        "asterion_student_runtime_count": asterion["student_runtime_count"],
        "asterion_p3_runtime_count": asterion["p3_runtime_count"],
        "auto_grade_eligibility_file_found": auto_grade["file_found"],
    }
    return {
        "schema_name": AUDIT_SCHEMA_NAME,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "question_bank": file_state(question_bank_path),
            "topic_routing": file_state(topic_routing_path),
            **optional_inputs,
        },
        "summary": summary,
        "question_bank_baseline": question_bank,
        "topic_routing_sidecar_baseline": topic_routing,
        "failure_analysis": failure_analysis,
        "review_required_analysis": review_required,
        "mark_events_snapshot": mark_events,
        "asterion_readiness_snapshot": asterion,
        "auto_grade_readiness_snapshot": auto_grade,
    }


def summarize_question_bank(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_records": len(records),
        "text_only_status_counts": sorted_counter(record.get("text_only_status") for record in records),
        "question_crop_confidence_counts": sorted_counter(get_field(record, "question_crop_confidence") for record in records),
        "visual_required_true_count": sum(1 for record in records if record.get("visual_required") is True),
        "usable_question_text_count": sum(1 for record in records if has_text(record.get("question_text"))),
        "usable_ocr_text_count": sum(1 for record in records if has_text(record.get("ocr_text"))),
        "usable_mark_scheme_text_count": sum(1 for record in records if has_text(record.get("mark_scheme_text"))),
        "crop_reference_count": sum(1 for record in records if has_crop_reference(record)),
    }


def summarize_topic_routing(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    qbank_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    route_ids = [str(row.get("question_id") or "") for row in rows]
    id_counts = Counter(route_ids)
    duplicate_ids = sorted(question_id for question_id, count in id_counts.items() if question_id and count > 1)
    missing_ids = sorted(set(qbank_by_id) - {question_id for question_id in route_ids if question_id})
    failed_count = sum(1 for row in rows if is_failed_route(row))
    strict_filter_count = sum(1 for row in rows if is_strict_filter_candidate(row))
    missing_metadata_count = sum(1 for row in rows if missing_course_metadata(row))
    missing_hash_count = sum(1 for row in rows if not has_evidence_packet_hash(row))
    malformed_hash_count = sum(
        1
        for row in rows
        if row.get("evidence_packet_hash") not in (None, "")
        and not has_evidence_packet_hash(row)
    )
    repaired_rows = [row for row in rows if row.get("evidence_used_repaired") is True]
    dropped_counter: Counter[str] = Counter()
    for row in repaired_rows:
        for field in row.get("evidence_used_dropped") or []:
            dropped_counter[str(field)] += 1
    return {
        "total_route_records": len(rows),
        "declared_record_count": payload.get("record_count"),
        "failed_count": failed_count,
        "review_required_count": sum(1 for row in rows if row.get("review_required") is True),
        "strict_filter_candidate_count": strict_filter_count,
        "safe_for_strict_filters": failed_count == 0 and strict_filter_count > 0,
        "confidence_counts": sorted_counter(row.get("confidence") for row in rows),
        "missing_course_metadata_count": missing_metadata_count,
        "stale_missing_newer_metadata_count": missing_metadata_count,
        "missing_evidence_packet_hash_count": missing_hash_count,
        "malformed_evidence_packet_hash_count": malformed_hash_count,
        "stale_missing_metadata_or_packet_hash_count": sum(
            1 for row in rows if missing_course_metadata(row) or not has_evidence_packet_hash(row)
        ),
        "evidence_used_repaired_count": len(repaired_rows),
        "top_dropped_evidence_fields": dict(sorted(dropped_counter.items())),
        "repaired_strict_filter_candidate_count": sum(1 for row in repaired_rows if is_strict_filter_candidate(row)),
        "repaired_review_required_or_error_count": sum(
            1 for row in repaired_rows if row.get("review_required") is True or is_failed_route(row)
        ),
        "duplicate_question_id_count": len(duplicate_ids),
        "duplicate_question_ids": duplicate_ids,
        "missing_question_id_count": len(missing_ids),
        "missing_question_ids": missing_ids,
    }


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if is_failed_route(row)]
    messages = Counter(failure_message(row) for row in failures)
    unsupported_counter: Counter[str] = Counter()
    unsupported_count = 0
    for row in failures:
        message = failure_message(row)
        if "evidence_used" in message and ("not supplied" in message or "unsupported" in message):
            unsupported_count += 1
            for field in re.findall(r"'([^']+)'", message):
                unsupported_counter[field] += 1
    return {
        "failed_count": len(failures),
        "unique_failure_messages": dict(sorted(messages.items())),
        "unique_failure_message_count": len(messages),
        "unsupported_evidence_used_failure_count": unsupported_count,
        "batch_amplification_estimate": {
            "unique_failure_messages": len(messages),
            "amplified_records_over_unique_messages": max(0, len(failures) - len(messages)),
            "max_records_for_one_message": max(messages.values(), default=0),
        },
        "top_unsupported_evidence_fields": dict(sorted(unsupported_counter.items())),
    }


def summarize_review_required(rows: list[dict[str, Any]], qbank_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    review_rows = [row for row in rows if row.get("review_required") is True]
    reason_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    visual_overlap = 0
    weak_overlap = 0
    for row in review_rows:
        reasons = row.get("review_reasons") if isinstance(row.get("review_reasons"), list) else []
        if not reasons:
            reason_counter["<missing>"] += 1
            bucket_counter["unknown"] += 1
        for reason in reasons:
            text = str(reason)
            reason_counter[text] += 1
            bucket_counter[infer_review_bucket(text)] += 1
        qbank_record = qbank_by_id.get(str(row.get("question_id") or ""))
        if qbank_record:
            if qbank_record.get("visual_required") is True:
                visual_overlap += 1
            if weak_text_or_crop_readiness(qbank_record):
                weak_overlap += 1
    return {
        "review_required_count": len(review_rows),
        "review_required_reason_counts": dict(sorted(reason_counter.items())),
        "normalized_bucket_counts": dict(sorted(bucket_counter.items())),
        "visual_required_overlap_count": visual_overlap,
        "weak_text_or_crop_readiness_overlap_count": weak_overlap,
    }


def summarize_mark_events(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {"file_found": False, "path": str(path), "record_count": None}
    records = records_from_payload(payload, "records")
    return {
        "file_found": True,
        "path": str(path),
        "record_count": len(records),
        "extraction_status_counts": sorted_counter(record.get("extraction_status") for record in records),
        "safe_for_marking_use_true_count": sum(1 for record in records if record.get("safe_for_marking_use") is True),
        "safe_for_advisory_use_true_count": sum(1 for record in records if record.get("safe_for_advisory_use") is True),
    }


def summarize_asterion(catalog_payload: dict[str, Any] | None, runtime_payload: dict[str, Any] | None) -> dict[str, Any]:
    catalog_questions = records_from_payload(catalog_payload or {}, "questions") if catalog_payload else []
    runtime_questions = records_from_payload(runtime_payload or {}, "questions") if runtime_payload else []
    all_questions = catalog_questions or runtime_questions
    return {
        "all_course_catalog_file_found": catalog_payload is not None,
        "all_course_catalog_count": len(catalog_questions) if catalog_payload is not None else None,
        "student_runtime_file_found": runtime_payload is not None,
        "student_runtime_count": len(runtime_questions) if runtime_payload is not None else None,
        "p3_runtime_count": sum(1 for row in runtime_questions if row.get("paper_family") == "p3") if runtime_payload is not None else None,
        "topic_route_filter_ok_true_count": sum(1 for row in all_questions if isinstance(row.get("topic_route"), dict) and row["topic_route"].get("filter_ok") is True),
        "route_review_error_or_stale_blocked_count": sum(1 for row in all_questions if route_blocked(row)),
    }


def summarize_auto_grade(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "file_found": False,
            "path": str(path),
            "status_counts": {},
            "top_blocker_counts": {},
            "note": "No current auto-grade eligibility file was found.",
        }
    items = records_from_payload(payload, "items")
    blocker_counter: Counter[str] = Counter()
    for item in items:
        for reason in item.get("block_reasons") or []:
            blocker_counter[str(reason)] += 1
    return {
        "file_found": True,
        "path": str(path),
        "record_count": len(items),
        "status_counts": sorted_counter(item.get("eligibility_status") for item in items),
        "top_blocker_counts": dict(sorted(blocker_counter.most_common(), key=lambda pair: (-pair[1], pair[0]))[:20]),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    failure = audit["failure_analysis"]
    auto = audit["auto_grade_readiness_snapshot"]
    lines = [
        "# Topic Routing Baseline Audit",
        "",
        f"- Generated at: `{audit['generated_at']}`",
        f"- Question-bank records: `{summary['question_bank_total_records']}`",
        f"- Topic-route records: `{summary['topic_routing_total_records']}`",
        f"- Failed routes: `{summary['topic_routing_failed_records']}`",
        f"- Review-required routes: `{summary['topic_routing_review_required_records']}`",
        f"- Strict-filter candidates: `{summary['topic_routing_strict_filter_candidates']}`",
        f"- Safe for strict filters: `{summary['topic_routing_safe_for_strict_filters']}`",
        f"- Asterion catalog records: `{summary['asterion_catalog_count']}`",
        f"- Asterion student runtime records: `{summary['asterion_student_runtime_count']}`",
        f"- Asterion P3 runtime records: `{summary['asterion_p3_runtime_count']}`",
        f"- Unique failure messages: `{failure['unique_failure_message_count']}`",
        f"- Unsupported evidence_used failures: `{failure['unsupported_evidence_used_failure_count']}`",
        f"- Auto-grade eligibility file found: `{auto['file_found']}`",
        "",
    ]
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_required_json(path)


def file_state(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def records_from_payload(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    records = payload.get(key)
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    if isinstance(records, dict):
        return [record for record in records.values() if isinstance(record, dict)]
    return []


def route_records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records")
    rows: list[dict[str, Any]] = []
    if isinstance(records, dict):
        for question_id in sorted(records):
            value = records[question_id]
            if isinstance(value, dict):
                rows.append({"question_id": str(question_id), **value})
    elif isinstance(records, list):
        rows = [record for record in records if isinstance(record, dict)]
    return rows


def sorted_counter(values: Any) -> dict[str, int]:
    counter = Counter(str(value) if value not in (None, "") else "<missing>" for value in values)
    return dict(sorted(counter.items()))


def get_field(record: dict[str, Any], name: str) -> Any:
    if name in record:
        return record.get(name)
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(name)
    return None


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_crop_reference(record: dict[str, Any]) -> bool:
    for field in ("question_image_path", "canonical_question_artifact"):
        if has_text(record.get(field)):
            return True
    paths = record.get("question_image_paths")
    return isinstance(paths, list) and any(has_text(path) for path in paths)


def is_failed_route(row: dict[str, Any]) -> bool:
    return isinstance(row.get("error"), dict) or str(row.get("routing_source") or "").endswith("_error")


def failure_message(row: dict[str, Any]) -> str:
    error = row.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or "unknown")
    return "unknown"


def is_strict_filter_candidate(row: dict[str, Any]) -> bool:
    return (
        not is_failed_route(row)
        and row.get("review_required") is False
        and row.get("confidence") in {"high", "medium"}
        and isinstance(row.get("primary_topic_id"), str)
        and bool(row.get("topic_distribution"))
    )


def missing_course_metadata(row: dict[str, Any]) -> bool:
    return any(not str(row.get(field) or "").strip() for field in COURSE_METADATA_FIELDS)


def has_evidence_packet_hash(row: dict[str, Any]) -> bool:
    value = row.get("evidence_packet_hash")
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def infer_review_bucket(reason: str) -> str:
    text = reason.lower()
    if "schema" in text or "validation" in text:
        return "schema_validation_error"
    if "visual" in text and ("text" in text or "image" in text or "provided" in text):
        return "visual_required_without_sufficient_text_evidence"
    if "weak" in text or "missing text" in text or "insufficient text" in text:
        return "weak_or_missing_text_evidence"
    if "ambiguous" in text or "multi-topic" in text or "multiple topic" in text:
        return "ambiguous_multi_topic_fit"
    return "unknown"


def weak_text_or_crop_readiness(record: dict[str, Any]) -> bool:
    return record.get("text_only_status") != "ready" or get_field(record, "question_crop_confidence") != "high"


def route_blocked(row: dict[str, Any]) -> bool:
    route = row.get("topic_route")
    if not isinstance(route, dict):
        return False
    reasons = row.get("quality_gate", {}).get("reason_codes") if isinstance(row.get("quality_gate"), dict) else []
    return (
        route.get("review_required") is True
        or route.get("filter_ok") is False
        or str(route.get("routing_source") or "").endswith("_error")
        or any("topic" in str(reason) for reason in (reasons or []))
    )


if __name__ == "__main__":
    raise SystemExit(main())
