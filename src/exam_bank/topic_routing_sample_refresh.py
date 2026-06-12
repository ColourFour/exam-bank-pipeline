from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .topic_routing_audit import (
    failure_message,
    has_evidence_packet_hash,
    is_failed_route,
    is_strict_filter_candidate,
    missing_course_metadata,
    route_records_from_payload,
)


DEFAULT_PRODUCTION_SIDECAR = Path("output/json/question_bank.topic_routing.v1.json")
DEFAULT_SAMPLE_SIZE = 80


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select and audit a controlled topic-routing sample refresh. Typical sequence: select sample IDs, "
            "run topic-route-ai to a /tmp sidecar, run delta, then confirm the production sidecar was not overwritten."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="Select deterministic topic-routing refresh sample IDs.")
    select.add_argument("--sidecar", type=Path, default=DEFAULT_PRODUCTION_SIDECAR)
    select.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    select.add_argument("--ids-out", type=Path, default=Path("/tmp/topic_routing_sample_refresh_pr6_ids.txt"))
    select.add_argument("--json-out", type=Path, default=Path("/tmp/topic_routing_sample_refresh_pr6_sample.json"))

    delta = subparsers.add_parser("delta", help="Compare production sidecar rows with refreshed sample rows.")
    delta.add_argument("--before-sidecar", type=Path, default=DEFAULT_PRODUCTION_SIDECAR)
    delta.add_argument("--after-sidecar", type=Path, required=True)
    delta.add_argument("--sample-ids", type=Path, default=Path("/tmp/topic_routing_sample_refresh_pr6_ids.txt"))
    delta.add_argument("--json-out", type=Path, default=Path("/tmp/topic_routing_sample_refresh_delta_pr6.json"))
    delta.add_argument("--markdown-out", type=Path, default=Path("/tmp/topic_routing_sample_refresh_delta_pr6.md"))

    triage = subparsers.add_parser("triage", help="Triage review-required rows in a refreshed sample sidecar.")
    triage.add_argument("--sidecar", type=Path, default=Path("/tmp/question_bank.topic_routing.sample_refresh.pr6.json"))
    triage.add_argument("--sample-ids", type=Path, default=Path("/tmp/topic_routing_sample_refresh_pr6_ids.txt"))
    triage.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    triage.add_argument("--json-out", type=Path, default=Path("/tmp/topic_routing_sample_refresh_triage_pr7.json"))
    triage.add_argument("--markdown-out", type=Path, default=Path("/tmp/topic_routing_sample_refresh_triage_pr7.md"))

    visual = subparsers.add_parser("visual-evidence", help="Audit visual-required topic-routing evidence gaps.")
    visual.add_argument("--question-bank", type=Path, default=Path("output/json/question_bank.json"))
    visual.add_argument("--production-sidecar", type=Path, default=DEFAULT_PRODUCTION_SIDECAR)
    visual.add_argument("--sample-sidecar", type=Path, default=Path("/tmp/question_bank.topic_routing.sample_refresh.pr6.json"))
    visual.add_argument("--triage", type=Path, default=Path("/tmp/topic_routing_sample_refresh_triage_pr7.json"))
    visual.add_argument("--taxonomy", type=Path, default=Path("exam_bank_taxonomy/canonical"))
    visual.add_argument("--json-out", type=Path, default=Path("/tmp/topic_routing_visual_evidence_audit_pr8.json"))
    visual.add_argument("--markdown-out", type=Path, default=Path("/tmp/topic_routing_visual_evidence_audit_pr8.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        report = select_topic_routing_sample(
            sidecar_path=args.sidecar,
            sample_size=args.sample_size,
        )
        write_json(args.json_out, report)
        write_lines(args.ids_out, report["sample_ids"])
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "delta":
        sample_ids = read_id_file(args.sample_ids)
        report = build_topic_routing_sample_delta(
            before_sidecar_path=args.before_sidecar,
            after_sidecar_path=args.after_sidecar,
            sample_ids=sample_ids,
        )
        write_json(args.json_out, report)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_delta_markdown(report), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "triage":
        sample_ids = read_id_file(args.sample_ids)
        report = build_topic_routing_sample_triage(
            sidecar_path=args.sidecar,
            sample_ids=sample_ids,
            question_bank_path=args.question_bank,
        )
        write_json(args.json_out, report)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_triage_markdown(report), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    if args.command == "visual-evidence":
        report = build_visual_required_evidence_audit(
            question_bank_path=args.question_bank,
            production_sidecar_path=args.production_sidecar,
            sample_sidecar_path=args.sample_sidecar,
            triage_path=args.triage,
            taxonomy_root=args.taxonomy,
        )
        write_json(args.json_out, report)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_visual_evidence_markdown(report), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def select_topic_routing_sample(
    *,
    sidecar_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    payload = read_json(sidecar_path)
    records = _records_dict(payload)
    failed_ids = sorted(question_id for question_id, row in records.items() if is_failed_route(row))
    review_ids = sorted(
        question_id
        for question_id, row in records.items()
        if row.get("review_required") is True and not is_failed_route(row)
    )
    strict_ids = sorted(question_id for question_id, row in records.items() if is_strict_filter_candidate(row))

    sample_ids: list[str] = []
    sample_ids.extend(failed_ids)
    remaining = max(0, sample_size - len(sample_ids))
    review_take = min(len(review_ids), remaining // 2 if strict_ids else remaining)
    sample_ids.extend(review_ids[:review_take])
    remaining = max(0, sample_size - len(sample_ids))
    sample_ids.extend(strict_ids[:remaining])
    if len(sample_ids) < sample_size:
        seen = set(sample_ids)
        for question_id in sorted(records):
            if question_id not in seen:
                sample_ids.append(question_id)
                seen.add(question_id)
            if len(sample_ids) >= sample_size:
                break
    sample_ids = sample_ids[:sample_size]
    return {
        "schema_name": "exam_bank.topic_routing_sample_refresh_selection",
        "schema_version": 1,
        "generated_at": now_iso(),
        "input_sidecar": str(sidecar_path),
        "summary": {
            "sample_size": len(sample_ids),
            "failed_ids_included": len([question_id for question_id in sample_ids if question_id in set(failed_ids)]),
            "review_required_ids_included": len([question_id for question_id in sample_ids if question_id in set(review_ids)]),
            "strict_filter_ids_included": len([question_id for question_id in sample_ids if question_id in set(strict_ids)]),
        },
        "source_counts": {
            "failed": len(failed_ids),
            "review_required_non_failed": len(review_ids),
            "strict_filter_candidate": len(strict_ids),
        },
        "sample_ids": sample_ids,
    }


def build_topic_routing_sample_delta(
    *,
    before_sidecar_path: Path,
    after_sidecar_path: Path,
    sample_ids: list[str],
) -> dict[str, Any]:
    before_payload = read_json(before_sidecar_path)
    before = _records_dict(before_payload)
    after_payload = read_optional_json(after_sidecar_path)
    after = _records_dict(after_payload) if after_payload else {}
    before_rows = [before[question_id] for question_id in sample_ids if question_id in before]
    after_rows = [after[question_id] for question_id in sample_ids if question_id in after]
    before_summary = summarize_rows(before_rows)
    after_file_found = after_payload is not None
    after_summary = summarize_rows(after_rows) if after_file_found else unavailable_summary()
    batch_summary = summarize_batches(after_payload or {}) if after_file_found else unavailable_batch_summary()
    computed_deltas = build_computed_deltas(before_summary, after_summary) if after_file_found else unavailable_deltas()
    return {
        "schema_name": "exam_bank.topic_routing_sample_refresh_delta",
        "schema_version": 1,
        "generated_at": now_iso(),
        "inputs": {
            "before_sidecar": {"path": str(before_sidecar_path), "exists": before_sidecar_path.exists()},
            "after_sidecar": {"path": str(after_sidecar_path), "exists": after_sidecar_path.exists()},
        },
        "summary": {
            "sample_size": len(sample_ids),
            "after_file_found": after_file_found,
            "after_status": "computed" if after_file_found else "unavailable",
            "behavioral_conclusion": (
                "computed_from_refreshed_sidecar"
                if after_file_found
                else "not_computed_provider_backed_refresh_did_not_run"
            ),
            "failed_before": before_summary["failed_count"],
            "failed_after": after_summary["failed_count"],
            "review_required_before": before_summary["review_required_count"],
            "review_required_after": after_summary["review_required_count"],
            "strict_filter_candidates_before": before_summary["strict_filter_candidate_count"],
            "strict_filter_candidates_after": after_summary["strict_filter_candidate_count"],
            "missing_evidence_packet_hash_before": before_summary["missing_evidence_packet_hash_count"],
            "missing_evidence_packet_hash_after": after_summary["missing_evidence_packet_hash_count"],
            "evidence_used_repaired_after": after_summary["evidence_used_repaired_count"],
            "valid_sibling_records_preserved_after_failure": batch_summary["valid_sibling_records_preserved_after_failure"],
            "unknown_returned_records_after": batch_summary["unknown_returned_records"],
            "missing_returned_records_after": batch_summary["missing_records"],
            "duplicate_returned_records_after": batch_summary["duplicate_records"],
        },
        "before": before_summary,
        "after": after_summary,
        "deltas": computed_deltas,
        "after_batch_metadata": batch_summary,
        "sample_ids": sample_ids,
    }


def build_topic_routing_sample_triage(
    *,
    sidecar_path: Path,
    sample_ids: list[str],
    question_bank_path: Path,
) -> dict[str, Any]:
    sidecar_payload = read_json(sidecar_path)
    qbank_payload = read_json(question_bank_path)
    routes = _records_dict(sidecar_payload)
    qbank = {
        str(record.get("question_id")): record
        for record in _question_bank_records(qbank_payload)
        if str(record.get("question_id") or "").strip()
    }
    sample_rows = [routes[question_id] for question_id in sample_ids if question_id in routes]
    row_summary = summarize_rows(sample_rows)
    review_rows = [
        build_review_triage_row(question_id, routes[question_id], qbank.get(question_id, {}))
        for question_id in sample_ids
        if question_id in routes and routes[question_id].get("review_required") is True
    ]
    bucket_counts = Counter(str(row["inferred_bucket"]) for row in review_rows)
    reason_counts = Counter(reason for row in review_rows for reason in row["review_reasons"])
    recommendation = recommend_next_step(bucket_counts)
    return {
        "schema_name": "exam_bank.topic_routing_sample_refresh_triage",
        "schema_version": 1,
        "generated_at": now_iso(),
        "inputs": {
            "sidecar": {"path": str(sidecar_path), "exists": sidecar_path.exists()},
            "sample_ids": {"count": len(sample_ids)},
            "question_bank": {"path": str(question_bank_path), "exists": question_bank_path.exists()},
        },
        "summary": {
            "sample_size": len(sample_ids),
            "matched_route_count": len(sample_rows),
            "failed_count": row_summary["failed_count"],
            "review_required_count": row_summary["review_required_count"],
            "strict_filter_candidate_count": row_summary["strict_filter_candidate_count"],
            "missing_evidence_packet_hash_count": row_summary["missing_evidence_packet_hash_count"],
            "evidence_used_repaired_count": row_summary["evidence_used_repaired_count"],
            "top_dropped_evidence_fields": row_summary["top_dropped_evidence_fields"],
            "review_required_bucket_counts": dict(sorted(bucket_counts.items())),
            "recommended_next_step": recommendation,
        },
        "review_required_reason_counts": dict(sorted(reason_counts.items())),
        "review_required_records": review_rows,
    }


def build_review_triage_row(question_id: str, route: dict[str, Any], qbank_record: dict[str, Any]) -> dict[str, Any]:
    review_reasons = [str(reason) for reason in route.get("review_reasons") or []]
    bucket = infer_review_triage_bucket(review_reasons, route=route, qbank_record=qbank_record)
    return {
        "question_id": question_id,
        "course_id": route.get("course_id") or qbank_record.get("course_id"),
        "component_name": route.get("component_name") or qbank_record.get("component_name"),
        "paper_family": route.get("paper_family") or qbank_record.get("paper_family"),
        "text_only_status": qbank_record.get("text_only_status"),
        "visual_required": bool(qbank_record.get("visual_required")),
        "question_crop_confidence": _field_with_notes(qbank_record, "question_crop_confidence"),
        "available_evidence_fields": available_evidence_fields(qbank_record),
        "primary_topic_id": route.get("primary_topic_id"),
        "confidence": route.get("confidence"),
        "review_reasons": review_reasons,
        "inferred_bucket": bucket,
        "recommended_next_step": recommended_step_for_bucket(bucket),
    }


def build_visual_required_evidence_audit(
    *,
    question_bank_path: Path,
    production_sidecar_path: Path,
    sample_sidecar_path: Path,
    triage_path: Path,
    taxonomy_root: Path,
) -> dict[str, Any]:
    qbank_payload = read_json(question_bank_path)
    records = _question_bank_records(qbank_payload)
    qbank_by_id = {
        str(record.get("question_id")): record
        for record in records
        if str(record.get("question_id") or "").strip()
    }
    production_routes = _records_dict(read_json(production_sidecar_path))
    sample_payload = read_optional_json(sample_sidecar_path)
    sample_routes = _records_dict(sample_payload) if sample_payload else {}
    triage_payload = read_optional_json(triage_path)

    visual_records = [record for record in records if record.get("visual_required") is True]
    visual_rows = [
        build_visual_evidence_row(
            record,
            production_routes.get(str(record.get("question_id") or ""), {}),
            sample_routes.get(str(record.get("question_id") or ""), {}),
            taxonomy_root=taxonomy_root,
        )
        for record in visual_records
    ]
    candidate_counts = Counter(row["fix_category"] for row in visual_rows)
    examples = representative_visual_examples(visual_rows)
    production_overlap = review_overlap_summary(production_routes, qbank_by_id)
    sample_overlap = triage_overlap_summary(triage_payload, qbank_by_id) if triage_payload else unavailable_overlap_summary()
    return {
        "schema_name": "exam_bank.topic_routing_visual_evidence_audit",
        "schema_version": 1,
        "generated_at": now_iso(),
        "inputs": {
            "question_bank": {"path": str(question_bank_path), "exists": question_bank_path.exists()},
            "production_sidecar": {"path": str(production_sidecar_path), "exists": production_sidecar_path.exists()},
            "sample_sidecar": {"path": str(sample_sidecar_path), "exists": sample_sidecar_path.exists()},
            "triage": {"path": str(triage_path), "exists": triage_path.exists()},
        },
        "summary": {
            "total_question_bank_records": len(records),
            "visual_required_count": len(visual_records),
            "visual_text_only_status_counts": dict(sorted(Counter(str(record.get("text_only_status") or "<missing>") for record in visual_records).items())),
            "visual_question_crop_confidence_counts": dict(sorted(Counter(str(_field_with_notes(record, "question_crop_confidence") or "<missing>") for record in visual_records).items())),
            "visual_question_crop_ref_count": sum(1 for record in visual_records if has_question_crop_ref(record)),
            "visual_mark_scheme_crop_ref_count": sum(1 for record in visual_records if has_mark_scheme_crop_ref(record)),
            "visual_ocr_text_available_count": sum(1 for record in visual_records if text_present(record.get("ocr_text"))),
            "visual_trusted_question_text_available_count": sum(1 for record in visual_records if trusted_question_text_available(record)),
            "visual_mark_scheme_text_available_count": sum(1 for record in visual_records if text_present(record.get("mark_scheme_text"))),
            "visual_search_hint_only_count": sum(1 for row in visual_rows if row["raw_evidence_fields"] == ["search_hint"]),
        },
        "packet_evidence_gap": {
            "evidence_exists_but_withheld_count": sum(1 for row in visual_rows if row["withheld_evidence_fields"]),
            "packet_no_meaningful_text_evidence_count": sum(1 for row in visual_rows if not row["packet_supplied_evidence_fields"]),
            "packet_only_mark_scheme_text_count": sum(1 for row in visual_rows if row["packet_supplied_evidence_fields"] == ["mark_scheme_text"]),
            "packet_only_search_hint_count": sum(1 for row in visual_rows if row["packet_supplied_evidence_fields"] == ["question_text"] and row["raw_evidence_fields"] == ["search_hint"]),
            "ocr_fallback_supplied_count": sum(1 for row in visual_rows if row["ocr_fallback_supplied"]),
            "search_hint_fallback_supplied_count": sum(1 for row in visual_rows if row["search_hint_fallback_supplied"]),
            "crop_refs_exist_but_no_usable_text_count": sum(
                1
                for row in visual_rows
                if row["has_question_crop_ref"] and row["has_mark_scheme_crop_ref"] and not row["packet_supplied_evidence_fields"]
            ),
        },
        "review_required_overlap": {
            "production_sidecar": production_overlap,
            "sample_triage": sample_overlap,
        },
        "candidate_fix_categories": [
            {"category": category, "impact_count": count}
            for category, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "representative_examples": examples,
    }


def build_visual_evidence_row(
    record: dict[str, Any],
    production_route: dict[str, Any],
    sample_route: dict[str, Any],
    *,
    taxonomy_root: Path,
) -> dict[str, Any]:
    question_id = str(record.get("question_id") or "")
    raw_fields = raw_visual_evidence_fields(record)
    packet_snapshot = packet_evidence_snapshot(record, taxonomy_root=taxonomy_root)
    packet_fields = packet_snapshot["fields"]
    packet_sources = packet_snapshot["sources"]
    withheld = [field for field in raw_fields if packet_field_name(field) not in packet_fields]
    route = sample_route or production_route
    category = infer_visual_fix_category(record, route, raw_fields, packet_fields, withheld)
    return {
        "question_id": question_id,
        "course_id": route.get("course_id") or record.get("course_id"),
        "component_name": route.get("component_name") or record.get("component_name"),
        "text_only_status": record.get("text_only_status"),
        "visual_required": bool(record.get("visual_required")),
        "question_crop_confidence": _field_with_notes(record, "question_crop_confidence"),
        "raw_evidence_fields": raw_fields,
        "packet_supplied_evidence_fields": packet_fields,
        "packet_evidence_sources": packet_sources,
        "ocr_fallback_supplied": packet_sources.get("ocr_text") == "ocr_fallback",
        "search_hint_fallback_supplied": packet_sources.get("question_text") == "search_hint_fallback",
        "withheld_evidence_fields": withheld,
        "has_question_crop_ref": has_question_crop_ref(record),
        "has_mark_scheme_crop_ref": has_mark_scheme_crop_ref(record),
        "current_route_review_required": route.get("review_required"),
        "current_route_status": "review_required" if route.get("review_required") is True else ("failed" if is_failed_route(route) else "strict_or_accepted"),
        "primary_topic_id": route.get("primary_topic_id"),
        "confidence": route.get("confidence"),
        "review_reasons": route.get("review_reasons") or [],
        "fix_category": category,
    }


def raw_visual_evidence_fields(record: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    role = str(record.get("question_text_role") or _field_with_notes(record, "question_text_role") or "").lower()
    if text_present(record.get("question_text")):
        fields.append("search_hint" if role == "search_hint" else "question_text")
    if text_present(record.get("ocr_text")):
        fields.append("ocr_text")
    if text_present(record.get("mark_scheme_text")):
        fields.append("mark_scheme_text")
    return fields


def packet_evidence_fields(record: dict[str, Any], *, taxonomy_root: Path) -> list[str]:
    return packet_evidence_snapshot(record, taxonomy_root=taxonomy_root)["fields"]


def packet_evidence_snapshot(record: dict[str, Any], *, taxonomy_root: Path) -> dict[str, Any]:
    from .topic_routing import build_topic_routing_question_packet

    try:
        packet = build_topic_routing_question_packet(record, taxonomy_root=taxonomy_root).packet
    except Exception:
        return {"fields": [], "sources": {}}
    evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
    available = packet.get("available_evidence_fields")
    fields = [str(field) for field in available] if isinstance(available, list) else sorted(str(field) for field in evidence)
    sources = packet.get("evidence_sources") if isinstance(packet.get("evidence_sources"), dict) else {}
    return {
        "fields": sorted(fields),
        "sources": {str(field): str(source) for field, source in sources.items()},
    }


def packet_field_name(raw_field: str) -> str:
    return "question_text" if raw_field == "search_hint" else raw_field


def infer_visual_fix_category(
    record: dict[str, Any],
    route: dict[str, Any],
    raw_fields: list[str],
    packet_fields: list[str],
    withheld: list[str],
) -> str:
    reasons = " ".join(str(reason) for reason in route.get("review_reasons") or []).lower()
    if "taxonomy" in reasons or "multiple topic" in reasons or "multi-topic" in reasons or "ambiguous" in reasons:
        return "taxonomy/topic ambiguity, not evidence quality"
    if withheld and any(field in withheld for field in ["question_text", "mark_scheme_text"]):
        return "packet can include existing safe text currently withheld"
    if withheld and any(field in withheld for field in ["ocr_text", "search_hint"]):
        return "packet can include existing OCR/search hint safely"
    if has_question_crop_ref(record) and not raw_fields:
        return "crop exists but OCR/text is missing"
    if _field_with_notes(record, "question_crop_confidence") == "low":
        return "crop confidence is low and likely needs recrop/re-extraction"
    if route.get("review_required") is True and bool(record.get("visual_required")):
        return "genuinely visual/math-diagram dependent and should remain review-required"
    return "no immediate visual evidence fix indicated"


def review_overlap_summary(routes: dict[str, dict[str, Any]], qbank_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    review_ids = [question_id for question_id, route in routes.items() if route.get("review_required") is True]
    return overlap_counts(review_ids, qbank_by_id)


def triage_overlap_summary(payload: dict[str, Any], qbank_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = payload.get("review_required_records") if isinstance(payload, dict) else []
    ids = [str(row.get("question_id")) for row in rows if isinstance(row, dict) and row.get("question_id")]
    return overlap_counts(ids, qbank_by_id)


def overlap_counts(question_ids: list[str], qbank_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = [qbank_by_id.get(question_id, {}) for question_id in question_ids]
    return {
        "review_required_count": len(question_ids),
        "visual_required_count": sum(1 for record in records if record.get("visual_required") is True),
        "text_only_review_or_fail_count": sum(1 for record in records if record.get("text_only_status") in {"review", "fail"}),
        "low_crop_confidence_count": sum(1 for record in records if _field_with_notes(record, "question_crop_confidence") == "low"),
        "missing_trusted_question_text_count": sum(1 for record in records if not trusted_question_text_available(record)),
        "missing_ocr_text_count": sum(1 for record in records if not text_present(record.get("ocr_text"))),
        "missing_mark_scheme_text_count": sum(1 for record in records if not text_present(record.get("mark_scheme_text"))),
    }


def unavailable_overlap_summary() -> dict[str, Any]:
    return {
        "review_required_count": None,
        "visual_required_count": None,
        "text_only_review_or_fail_count": None,
        "low_crop_confidence_count": None,
        "missing_trusted_question_text_count": None,
        "missing_ocr_text_count": None,
        "missing_mark_scheme_text_count": None,
    }


def representative_visual_examples(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for row in sorted(rows, key=lambda item: (item["fix_category"], item["question_id"])):
        if row["fix_category"] not in seen_categories:
            examples.append(row)
            seen_categories.add(row["fix_category"])
        if len(examples) >= limit:
            return examples
    for row in sorted(rows, key=lambda item: item["question_id"]):
        if row not in examples:
            examples.append(row)
        if len(examples) >= limit:
            break
    return examples


def infer_review_triage_bucket(
    review_reasons: list[str],
    *,
    route: dict[str, Any],
    qbank_record: dict[str, Any],
) -> str:
    text = " ".join(review_reasons).lower()
    if "weak" in text or "missing text" in text or "no text" in text:
        return "weak_or_missing_text_evidence"
    if "visual" in text or bool(qbank_record.get("visual_required")):
        if "insufficient" in text or "without image" in text or "no image" in text or "not provided" in text:
            return "visual_required_without_sufficient_text_evidence"
    if "ambiguous" in text or "multi-topic" in text or "multiple topic" in text or "spans" in text:
        return "ambiguous_multi_topic_fit"
    if "context" in text or "insufficient" in text or "unable to determine" in text or "cannot confirm" in text:
        return "insufficient_question_context"
    if (
        "taxonomy" in text
        or "topic" in text
        or route.get("primary_topic_id") is None
        or route.get("confidence") == "low"
    ):
        return "taxonomy_or_topic_fit_unclear"
    return "unknown"


def available_evidence_fields(record: dict[str, Any]) -> list[str]:
    fields = []
    for field in ["question_text", "ocr_text", "mark_scheme_text"]:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            fields.append(field)
    return fields


def text_present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def trusted_question_text_available(record: dict[str, Any]) -> bool:
    status = str(record.get("text_only_status") or _field_with_notes(record, "text_only_status") or "").lower()
    trust = str(record.get("question_text_trust") or _field_with_notes(record, "question_text_trust") or "").lower()
    role = str(record.get("question_text_role") or _field_with_notes(record, "question_text_role") or "").lower()
    return (
        text_present(record.get("question_text"))
        and status == "ready"
        and trust in {"high", "medium"}
        and role not in {"untrusted_math_text", "search_hint"}
    )


def has_question_crop_ref(record: dict[str, Any]) -> bool:
    return any(
        [
            text_present(record.get("question_image_path")),
            text_present(record.get("canonical_question_artifact")),
            any(text_present(path) for path in record.get("question_image_paths") or []),
        ]
    )


def has_mark_scheme_crop_ref(record: dict[str, Any]) -> bool:
    return any(
        [
            text_present(record.get("mark_scheme_image_path")),
            text_present(record.get("canonical_mark_scheme_artifact")),
            any(text_present(path) for path in record.get("mark_scheme_image_paths") or []),
        ]
    )


def recommend_next_step(bucket_counts: Counter[str]) -> str:
    if not bucket_counts:
        return "no change needed / honest review-required"
    if bucket_counts.get("weak_or_missing_text_evidence") or bucket_counts.get("insufficient_question_context"):
        return "packet/text improvement"
    if bucket_counts.get("visual_required_without_sufficient_text_evidence"):
        return "crop/OCR improvement"
    if bucket_counts.get("taxonomy_or_topic_fit_unclear"):
        return "taxonomy review"
    if bucket_counts.get("ambiguous_multi_topic_fit"):
        return "prompt v2"
    return "no change needed / honest review-required"


def recommended_step_for_bucket(bucket: str) -> str:
    return {
        "weak_or_missing_text_evidence": "packet/text improvement",
        "visual_required_without_sufficient_text_evidence": "crop/OCR improvement",
        "ambiguous_multi_topic_fit": "prompt v2",
        "insufficient_question_context": "packet/text improvement",
        "taxonomy_or_topic_fit_unclear": "taxonomy review",
        "unknown": "no change needed / honest review-required",
    }.get(bucket, "no change needed / honest review-required")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_messages = Counter(failure_message(row) for row in rows if is_failed_route(row))
    dropped_counter: Counter[str] = Counter()
    for row in rows:
        for field in row.get("evidence_used_dropped") or []:
            dropped_counter[str(field)] += 1
    return {
        "record_count": len(rows),
        "failed_count": sum(1 for row in rows if is_failed_route(row)),
        "review_required_count": sum(1 for row in rows if row.get("review_required") is True),
        "strict_filter_candidate_count": sum(1 for row in rows if is_strict_filter_candidate(row)),
        "missing_evidence_packet_hash_count": sum(1 for row in rows if not has_evidence_packet_hash(row)),
        "stale_missing_metadata_or_packet_hash_count": sum(
            1 for row in rows if missing_course_metadata(row) or not has_evidence_packet_hash(row)
        ),
        "evidence_used_repaired_count": sum(1 for row in rows if row.get("evidence_used_repaired") is True),
        "top_dropped_evidence_fields": dict(sorted(dropped_counter.items())),
        "unique_failure_messages": dict(sorted(failure_messages.items())),
        "unique_failure_message_count": len(failure_messages),
    }


def unavailable_summary() -> dict[str, Any]:
    return {
        "record_count": None,
        "failed_count": None,
        "review_required_count": None,
        "strict_filter_candidate_count": None,
        "missing_evidence_packet_hash_count": None,
        "stale_missing_metadata_or_packet_hash_count": None,
        "evidence_used_repaired_count": None,
        "top_dropped_evidence_fields": {},
        "unique_failure_messages": None,
        "unique_failure_message_count": None,
    }


def build_computed_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    fields = [
        "failed_count",
        "review_required_count",
        "strict_filter_candidate_count",
        "missing_evidence_packet_hash_count",
        "stale_missing_metadata_or_packet_hash_count",
        "evidence_used_repaired_count",
        "unique_failure_message_count",
    ]
    return {field: int(after[field]) - int(before[field]) for field in fields}


def unavailable_deltas() -> dict[str, str]:
    return {
        "failed_count": "not_computed",
        "review_required_count": "not_computed",
        "strict_filter_candidate_count": "not_computed",
        "missing_evidence_packet_hash_count": "not_computed",
        "stale_missing_metadata_or_packet_hash_count": "not_computed",
        "evidence_used_repaired_count": "not_computed",
        "unique_failure_message_count": "not_computed",
    }


def summarize_batches(payload: dict[str, Any]) -> dict[str, Any]:
    batches = (((payload.get("metadata") or {}).get("run_manifest") or {}).get("batches") or [])
    valid_with_invalid = False
    unknown = missing = duplicate = 0
    salvaged = 0
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        valid = int(batch.get("valid_records") or 0)
        invalid = int(batch.get("invalid_records") or 0)
        unknown += int(batch.get("unknown_returned_records") or 0)
        missing += int(batch.get("missing_records") or 0)
        duplicate += int(batch.get("duplicate_records") or 0)
        if batch.get("batch_salvaged") is True:
            salvaged += 1
        if valid > 0 and invalid > 0:
            valid_with_invalid = True
    return {
        "batch_count": len([batch for batch in batches if isinstance(batch, dict)]),
        "salvaged_batch_count": salvaged,
        "valid_sibling_records_preserved_after_failure": valid_with_invalid,
        "unknown_returned_records": unknown,
        "missing_records": missing,
        "duplicate_records": duplicate,
    }


def unavailable_batch_summary() -> dict[str, Any]:
    return {
        "batch_count": None,
        "salvaged_batch_count": None,
        "valid_sibling_records_preserved_after_failure": None,
        "unknown_returned_records": None,
        "missing_records": None,
        "duplicate_records": None,
    }


def render_delta_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    if summary["after_file_found"] is False:
        return "\n".join(
            [
                "# Topic Routing Sample Refresh Delta",
                "",
                f"- Sample size: `{summary['sample_size']}`",
                "- Provider-backed refresh did not run or did not produce the requested refreshed sidecar.",
                "- After metrics are unavailable; no behavioral conclusion or improvement claim can be made.",
                f"- Failed before/after: `{summary['failed_before']}` -> `unavailable`",
                f"- Review-required before/after: `{summary['review_required_before']}` -> `unavailable`",
                f"- Strict-filter candidates before/after: `{summary['strict_filter_candidates_before']}` -> `unavailable`",
                f"- Missing evidence packet hash before/after: `{summary['missing_evidence_packet_hash_before']}` -> `unavailable`",
                "- Evidence-used repaired after: `unavailable`",
                "",
            ]
        )
    return "\n".join(
        [
            "# Topic Routing Sample Refresh Delta",
            "",
            f"- Sample size: `{summary['sample_size']}`",
            f"- Failed before/after: `{summary['failed_before']}` -> `{summary['failed_after']}`",
            f"- Review-required before/after: `{summary['review_required_before']}` -> `{summary['review_required_after']}`",
            f"- Strict-filter candidates before/after: `{summary['strict_filter_candidates_before']}` -> `{summary['strict_filter_candidates_after']}`",
            f"- Missing evidence packet hash before/after: `{summary['missing_evidence_packet_hash_before']}` -> `{summary['missing_evidence_packet_hash_after']}`",
            f"- Evidence-used repaired after: `{summary['evidence_used_repaired_after']}`",
            f"- Valid siblings preserved after a failed row: `{summary['valid_sibling_records_preserved_after_failure']}`",
            f"- Unknown/missing/duplicate returned records after: `{summary['unknown_returned_records_after']}` / `{summary['missing_returned_records_after']}` / `{summary['duplicate_returned_records_after']}`",
            "",
        ]
    )


def render_triage_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Topic Routing Sample Refresh Triage",
        "",
        f"- Sample size: `{summary['sample_size']}`",
        f"- Failed records: `{summary['failed_count']}`",
        f"- Review-required records: `{summary['review_required_count']}`",
        f"- Strict-filter candidates: `{summary['strict_filter_candidate_count']}`",
        f"- Missing evidence packet hash: `{summary['missing_evidence_packet_hash_count']}`",
        f"- Evidence-used repaired: `{summary['evidence_used_repaired_count']}`",
        f"- Top dropped evidence fields: `{summary['top_dropped_evidence_fields']}`",
        f"- Recommended next step: `{summary['recommended_next_step']}`",
        "",
        "## Review Buckets",
        "",
    ]
    for bucket, count in summary["review_required_bucket_counts"].items():
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Review-Required Records", ""])
    for row in report["review_required_records"]:
        reasons = "; ".join(row["review_reasons"]) or "<missing>"
        lines.append(
            f"- `{row['question_id']}`: `{row['inferred_bucket']}`; "
            f"confidence `{row['confidence']}`; topic `{row['primary_topic_id']}`; reasons: {reasons}"
        )
    lines.append("")
    return "\n".join(lines)


def render_visual_evidence_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    gap = report["packet_evidence_gap"]
    lines = [
        "# Topic Routing Visual Evidence Audit",
        "",
        f"- Total question-bank records: `{summary['total_question_bank_records']}`",
        f"- Visual-required records: `{summary['visual_required_count']}`",
        f"- Visual text-only statuses: `{summary['visual_text_only_status_counts']}`",
        f"- Visual crop confidence: `{summary['visual_question_crop_confidence_counts']}`",
        f"- OCR text available: `{summary['visual_ocr_text_available_count']}`",
        f"- Trusted question text available: `{summary['visual_trusted_question_text_available_count']}`",
        f"- Mark-scheme text available: `{summary['visual_mark_scheme_text_available_count']}`",
        "",
        "## Packet Evidence Gap",
        "",
        f"- Evidence exists but withheld: `{gap['evidence_exists_but_withheld_count']}`",
        f"- Packet has no meaningful text evidence: `{gap['packet_no_meaningful_text_evidence_count']}`",
        f"- Packet only mark-scheme text: `{gap['packet_only_mark_scheme_text_count']}`",
        f"- Packet only search hint: `{gap['packet_only_search_hint_count']}`",
        f"- OCR fallback supplied: `{gap['ocr_fallback_supplied_count']}`",
        f"- Search-hint fallback supplied: `{gap['search_hint_fallback_supplied_count']}`",
        f"- Crop refs exist but no usable packet text: `{gap['crop_refs_exist_but_no_usable_text_count']}`",
        "",
        "## Candidate Fix Categories",
        "",
    ]
    for item in report["candidate_fix_categories"]:
        lines.append(f"- `{item['category']}`: `{item['impact_count']}`")
    lines.extend(["", "## Representative Examples", ""])
    for row in report["representative_examples"]:
        lines.append(
            f"- `{row['question_id']}`: `{row['fix_category']}`; raw `{row['raw_evidence_fields']}`; "
            f"packet `{row['packet_supplied_evidence_fields']}`; route `{row['current_route_status']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _records_dict(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records") if isinstance(payload, dict) else {}
    if isinstance(records, dict):
        return {str(question_id): row for question_id, row in records.items() if isinstance(row, dict)}
    return {str(row.get("question_id")): row for row in route_records_from_payload(payload) if row.get("question_id")}


def _question_bank_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("questions") if isinstance(payload, dict) else []
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    if isinstance(records, dict):
        return [record for record in records.values() if isinstance(record, dict)]
    return []


def _field_with_notes(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record.get(field)
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(field)
    return None


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def read_id_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
