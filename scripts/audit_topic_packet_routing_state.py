#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.topic_packets import (
    load_packet_taxonomy,
    load_topic_bank_reviewed_decisions,
    load_topic_overlap_review_decisions,
    normalize_packet_topic,
)
from exam_bank.topic_packets import _packet_family_for_component


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_BANK = ROOT / "output/json/question_bank.json"
DEFAULT_TAXONOMY = ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json"
DEFAULT_REVIEWED_DECISIONS = ROOT / "data/review/canonical/topic/topic_bank_reviewed_decisions.v1.json"
DEFAULT_OVERLAP_REVIEW = ROOT / "data/review/canonical/topic/topic_overlap_review_current.v1.json"
DEFAULT_PACKETS_ROOT = ROOT / "output/topic_packets"
SCHEMA_NAME = "exam_bank.topic_packet_routing_state_audit"
SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit topic-packet routing state and packet summary provenance.")
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--reviewed-decisions", type=Path, default=DEFAULT_REVIEWED_DECISIONS)
    parser.add_argument("--topic-overlap-review", type=Path, default=DEFAULT_OVERLAP_REVIEW)
    parser.add_argument("--packets-root", type=Path, default=DEFAULT_PACKETS_ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when warnings are present.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        question_bank_path=args.question_bank,
        taxonomy_path=args.taxonomy,
        reviewed_decisions_path=args.reviewed_decisions,
        topic_overlap_review_path=args.topic_overlap_review,
        packets_root=args.packets_root,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 1 if args.strict and report["warnings"] else 0


def build_report(
    *,
    question_bank_path: Path,
    taxonomy_path: Path,
    reviewed_decisions_path: Path,
    topic_overlap_review_path: Path,
    packets_root: Path,
) -> dict[str, Any]:
    question_bank_payload = read_json(question_bank_path)
    records = list(question_bank_payload.get("questions") or [])
    taxonomy = load_packet_taxonomy(taxonomy_path)
    reviewed_decisions = (
        load_topic_bank_reviewed_decisions(reviewed_decisions_path, records=records, taxonomy=taxonomy)
        if reviewed_decisions_path.is_file()
        else {}
    )
    overlap_reviews = (
        load_topic_overlap_review_decisions(topic_overlap_review_path, records=records, taxonomy=taxonomy)
        if topic_overlap_review_path.is_file()
        else {}
    )

    family_mismatches: list[dict[str, Any]] = []
    covered_mismatches: list[dict[str, Any]] = []
    reviewed_mismatches: list[dict[str, Any]] = []
    unreviewed_mismatches: list[dict[str, Any]] = []
    for record in records:
        row = routing_mismatch_row(record, taxonomy)
        if row is None:
            continue
        family_mismatches.append(row)
        review = overlap_reviews.get(row["question_id"])
        if review is not None and review.paper_family == row["source_component_family"]:
            covered_mismatches.append(
                row
                | {
                    "review_status": review.status,
                    "review_primary_topic": review.primary_topic,
                    "review_source": review.source,
                }
            )
            continue
        decision = reviewed_decisions.get(row["question_id"])
        if decision is not None and decision.action in {"keep", "relabel", "exclude"}:
            reviewed_mismatches.append(
                row
                | {
                    "reviewed_decision_action": decision.action,
                    "reviewed_decision_source": decision.source,
                    "reviewed_topic": decision.reviewed_topic,
                }
            )
            continue
        unreviewed_mismatches.append(row)

    packet_state = packet_manifest_state(packets_root)
    warnings = audit_warnings(
        topic_overlap_review_path=topic_overlap_review_path,
        unreviewed_mismatches=unreviewed_mismatches,
        packet_state=packet_state,
    )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "question_bank": str(question_bank_path),
            "taxonomy": str(taxonomy_path),
            "reviewed_decisions": str(reviewed_decisions_path),
            "topic_overlap_review": str(topic_overlap_review_path),
            "packets_root": str(packets_root),
        },
        "summary": {
            "question_bank_records": len(records),
            "reviewed_decisions_loaded": len(reviewed_decisions),
            "topic_overlap_reviews_loaded": len(overlap_reviews),
            "raw_family_mismatch_count": len(family_mismatches),
            "raw_family_mismatch_covered_by_overlap_review_count": len(covered_mismatches),
            "raw_family_mismatch_covered_by_reviewed_decision_count": len(reviewed_mismatches),
            "raw_family_mismatch_unreviewed_count": len(unreviewed_mismatches),
            "packet_manifest_count": packet_state["manifest_count"],
            "packet_manifest_without_overlap_path_count": packet_state["manifest_without_overlap_path_count"],
            "summary_packet_count": packet_state["summary_packet_count"],
            "summary_scope_type": packet_state["summary_scope_type"],
            "warning_count": len(warnings),
        },
        "warnings": warnings,
        "family_mismatch_counts": dict(sorted(Counter(row["family_pair"] for row in family_mismatches).items())),
        "raw_family_mismatches": family_mismatches,
        "covered_raw_family_mismatches": covered_mismatches,
        "reviewed_raw_family_mismatches": reviewed_mismatches,
        "unreviewed_raw_family_mismatches": unreviewed_mismatches,
        "packet_state": packet_state,
    }


def routing_mismatch_row(record: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any] | None:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    normalization = normalize_packet_topic(
        component_code=notes.get("source_paper_code") or record.get("paper"),
        current_family=record.get("paper_family"),
        raw_topic=record.get("topic"),
        taxonomy=taxonomy,
    )
    source_family = _packet_family_for_component(normalization.source_component)
    if not source_family or not normalization.expected_family or source_family == normalization.expected_family:
        return None
    return {
        "question_id": str(record.get("question_id") or ""),
        "paper": str(record.get("paper") or ""),
        "source_component": normalization.source_component,
        "source_component_family": source_family,
        "normalized_packet_family": normalization.expected_family,
        "raw_topic": normalization.raw_topic,
        "normalized_topic": normalization.expected_topic,
        "normalization_reason": normalization.reason,
        "family_pair": f"{source_family}->{normalization.expected_family}",
    }


def packet_manifest_state(packets_root: Path) -> dict[str, Any]:
    summary_path = packets_root / "topic_packet_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(packets_root.glob("*/*/manifest.json")):
        manifest = read_json(manifest_path)
        records = manifest.get("included_records") or []
        manifests.append(
            {
                "path": str(manifest_path),
                "paper_family": str(manifest.get("paper_family") or ""),
                "topic_id": str(manifest.get("topic_id") or ""),
                "generated_at": str(manifest.get("generated_at") or ""),
                "question_count": int(manifest.get("question_count") or 0),
                "topic_overlap_review_path": str(manifest.get("topic_overlap_review_path") or ""),
                "topic_overlap_reviewed_record_count": sum(
                    1 for record in records if record.get("topic_overlap_review_status")
                ),
            }
        )

    summary_scope = summary.get("run_scope") if isinstance(summary.get("run_scope"), dict) else {}
    summary_packets = summary.get("packets_generated") if isinstance(summary.get("packets_generated"), list) else []
    generated_dates = Counter(row["generated_at"][:10] for row in manifests if row["generated_at"])
    return {
        "summary_path": str(summary_path),
        "summary_exists": summary_path.is_file(),
        "summary_generated_at": str(summary.get("generated_at") or ""),
        "summary_scope_type": str(summary_scope.get("scope_type") or "missing"),
        "summary_is_global": summary.get("is_global_run") if "is_global_run" in summary else None,
        "summary_packet_count": len(summary_packets),
        "summary_topic_overlap_review_path": str(summary.get("topic_overlap_review_path") or ""),
        "manifest_count": len(manifests),
        "manifest_generated_date_counts": dict(sorted(generated_dates.items())),
        "manifest_without_overlap_path_count": sum(1 for row in manifests if not row["topic_overlap_review_path"]),
        "manifests": manifests,
    }


def audit_warnings(
    *,
    topic_overlap_review_path: Path,
    unreviewed_mismatches: list[dict[str, Any]],
    packet_state: dict[str, Any],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if not topic_overlap_review_path.is_file():
        warnings.append({"code": "missing_topic_overlap_review", "path": str(topic_overlap_review_path)})
    if unreviewed_mismatches:
        warnings.append(
            {
                "code": "unreviewed_raw_family_mismatches",
                "count": len(unreviewed_mismatches),
                "question_ids": [row["question_id"] for row in unreviewed_mismatches[:50]],
            }
        )
    if packet_state["summary_scope_type"] == "missing":
        warnings.append({"code": "summary_missing_run_scope", "path": packet_state["summary_path"]})
    elif packet_state["summary_scope_type"] != "global":
        warnings.append({"code": "summary_is_scoped", "path": packet_state["summary_path"]})
    if packet_state["summary_packet_count"] != packet_state["manifest_count"]:
        warnings.append(
            {
                "code": "summary_manifest_count_mismatch",
                "summary_packet_count": packet_state["summary_packet_count"],
                "manifest_count": packet_state["manifest_count"],
            }
        )
    if packet_state["manifest_without_overlap_path_count"]:
        warnings.append(
            {
                "code": "manifests_missing_overlap_review_path",
                "count": packet_state["manifest_without_overlap_path_count"],
            }
        )
    if len(packet_state["manifest_generated_date_counts"]) > 1:
        warnings.append(
            {
                "code": "manifests_generated_on_multiple_dates",
                "date_counts": packet_state["manifest_generated_date_counts"],
            }
        )
    return warnings


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Topic Packet Routing State Audit",
        "",
        "## Summary",
        "",
        f"- Question-bank records: {summary['question_bank_records']}",
        f"- Reviewed topic-bank decisions loaded: {summary['reviewed_decisions_loaded']}",
        f"- Topic-overlap reviews loaded: {summary['topic_overlap_reviews_loaded']}",
        f"- Raw family mismatches: {summary['raw_family_mismatch_count']}",
        f"- Covered by overlap review: {summary['raw_family_mismatch_covered_by_overlap_review_count']}",
        f"- Covered by reviewed decision: {summary['raw_family_mismatch_covered_by_reviewed_decision_count']}",
        f"- Unreviewed mismatches: {summary['raw_family_mismatch_unreviewed_count']}",
        f"- Packet manifests: {summary['packet_manifest_count']}",
        f"- Manifests without overlap path: {summary['packet_manifest_without_overlap_path_count']}",
        f"- Summary scope: {summary['summary_scope_type']}",
        "",
        "## Warnings",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- `{warning['code']}`" for warning in report["warnings"])
    else:
        lines.append("- None")
    lines.extend(["", "## Unreviewed Family Mismatches", ""])
    if report["unreviewed_raw_family_mismatches"]:
        for row in report["unreviewed_raw_family_mismatches"]:
            lines.append(
                f"- `{row['question_id']}` {row['family_pair']} raw `{row['raw_topic']}` -> `{row['normalized_topic']}`"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
