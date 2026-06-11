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


def _records_dict(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records") if isinstance(payload, dict) else {}
    if isinstance(records, dict):
        return {str(question_id): row for question_id, row in records.items() if isinstance(row, dict)}
    return {str(row.get("question_id")): row for row in route_records_from_payload(payload) if row.get("question_id")}


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
