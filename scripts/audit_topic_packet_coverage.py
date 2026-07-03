#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.topic_packets import load_packet_taxonomy, normalize_packet_topic


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit generated topic packet coverage against question_bank.json.")
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json"))
    parser.add_argument("--report", type=Path, required=True, help="Report path prefix, without .json/.md suffix.")
    args = parser.parse_args()

    report = build_report(
        question_bank_path=args.question_bank,
        packets_root=args.packets,
        taxonomy_path=args.taxonomy,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    (args.report.with_suffix(".json")).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.report.with_suffix(".md")).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return 1 if report["hard_requirement_failures"] else 0


def build_report(*, question_bank_path: Path, packets_root: Path, taxonomy_path: Path) -> dict[str, Any]:
    question_bank = json.loads(question_bank_path.read_text(encoding="utf-8"))
    records = question_bank.get("questions") or []
    records_by_id = {str(record.get("question_id")): record for record in records}
    taxonomy = load_packet_taxonomy(taxonomy_path)
    summary_path = packets_root / "topic_packet_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    included: dict[str, dict[str, Any]] = {}
    duplicate_included: list[str] = []
    manifest_count = 0
    pdf_count = 0
    for manifest_path in sorted(packets_root.glob("**/manifest.json")):
        manifest_count += 1
        if (manifest_path.parent / "topic_packet.pdf").is_file():
            pdf_count += 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        family = str(manifest.get("paper_family") or "")
        topic = str(manifest.get("topic_id") or "")
        for item in manifest.get("included_records") or []:
            question_id = str(item.get("question_id") or "")
            if question_id in included:
                duplicate_included.append(question_id)
            included[question_id] = {
                "question_id": question_id,
                "family": family,
                "topic": topic,
                "section": str(item.get("section") or ""),
                "review_reasons": list(item.get("review_reasons") or []),
                "manifest_path": str(manifest_path),
                "pdf_path": str(manifest_path.parent / "topic_packet.pdf"),
            }

    skipped_records = summary.get("skipped_records") or []
    excluded = {str(item.get("question_id") or ""): item for item in skipped_records if item.get("question_id")}
    bank_ids = set(records_by_id)
    included_ids = set(included)
    excluded_ids = set(excluded)
    unexplained = sorted(bank_ids - included_ids - excluded_ids)
    included_excluded_overlap = sorted(included_ids & excluded_ids)

    included_by_year: dict[str, Counter[str]] = defaultdict(Counter)
    excluded_by_year: Counter[str] = Counter()
    for question_id in included_ids:
        included_by_year[_record_year(records_by_id.get(question_id, {}))][included[question_id]["section"]] += 1
    for question_id in excluded_ids:
        excluded_by_year[_record_year(records_by_id.get(question_id, {}))] += 1

    usable_p4_misrouted: list[dict[str, str]] = []
    expected_p4_topics = {
        "kinematics_of_motion_in_a_straight_line",
        "forces_and_equilibrium",
        "newtons_laws_of_motion",
        "energy_work_and_power",
        "momentum",
    }
    p4_topics_generated = {
        item["topic"]
        for item in included.values()
        if item["family"] == "p4"
    }
    for question_id, record in records_by_id.items():
        normalization = normalize_packet_topic(
            component_code=_source_component(record),
            current_family=record.get("paper_family"),
            raw_topic=record.get("topic"),
            taxonomy=taxonomy,
        )
        if normalization.expected_family != "p4":
            continue
        if question_id in excluded_ids:
            continue
        included_item = included.get(question_id)
        if not included_item or included_item["family"] != "p4":
            usable_p4_misrouted.append(
                {
                    "question_id": question_id,
                    "expected_topic": normalization.expected_topic,
                    "actual_family": included_item["family"] if included_item else "",
                    "actual_topic": included_item["topic"] if included_item else "",
                }
            )

    hard_failures: list[str] = []
    if unexplained:
        hard_failures.append("unexplained_skipped_records")
    if included_excluded_overlap:
        hard_failures.append("records_both_included_and_excluded")
    if duplicate_included:
        hard_failures.append("duplicate_included_records")
    if usable_p4_misrouted:
        hard_failures.append("usable_p4_records_not_in_p4_packets")
    if not expected_p4_topics <= p4_topics_generated:
        hard_failures.append("missing_expected_p4_topics")

    counts_by_section = Counter(item["section"] for item in included.values())
    counts_by_family = Counter(item["family"] for item in included.values())
    counts_by_family_topic = Counter(f"{item['family']}/{item['topic']}" for item in included.values())

    return {
        "schema_name": "exam_bank.topic_packet_generation_coverage",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_bank_path": str(question_bank_path),
        "packets_root": str(packets_root),
        "summary_path": str(summary_path),
        "total_question_bank_records": len(records_by_id),
        "total_included": len(included_ids),
        "approved_count": int(counts_by_section.get("approved", 0)),
        "review_required_count": int(counts_by_section.get("review_required", 0)),
        "excluded_count": len(excluded_ids),
        "unexplained_skipped_count": len(unexplained),
        "manifest_count": manifest_count,
        "topic_packet_pdf_count": pdf_count,
        "p4_packet_count": len({item["topic"] for item in included.values() if item["family"] == "p4"}),
        "p4_topics_generated": sorted(p4_topics_generated),
        "expected_p4_topics_present": sorted(expected_p4_topics & p4_topics_generated),
        "counts_by_year": _counts_by_year(records_by_id, included, excluded),
        "counts_by_family": dict(sorted(counts_by_family.items())),
        "counts_by_family_topic": dict(sorted(counts_by_family_topic.items())),
        "counts_by_section": dict(sorted(counts_by_section.items())),
        "included_counts_by_year_section": {
            year: dict(counter) for year, counter in sorted(included_by_year.items())
        },
        "excluded_counts_by_year": dict(sorted(excluded_by_year.items())),
        "excluded_records": [
            {
                "question_id": question_id,
                "year": _record_year(records_by_id.get(question_id, {})),
                "paper_family": str(records_by_id.get(question_id, {}).get("paper_family") or ""),
                "reason": str(excluded[question_id].get("reason") or ""),
                "assigned_topic_id": str(excluded[question_id].get("assigned_topic_id") or ""),
            }
            for question_id in sorted(excluded_ids)
        ],
        "excluded_reason_counts": dict(sorted(Counter(str(item.get("reason") or "") for item in excluded.values()).items())),
        "unexplained_skipped_records": unexplained,
        "included_excluded_overlap": included_excluded_overlap,
        "duplicate_included_records": sorted(set(duplicate_included)),
        "usable_p4_misrouted_records": usable_p4_misrouted,
        "old_baseline": {
            "question_bank_records": 3549,
            "included_in_any_topic_packet": 1084,
            "release_packets": 129,
            "review_required_packets": 955,
            "skipped": 2465,
            "p4_packet_output": 0,
        },
        "hard_requirement_failures": hard_failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Topic Packet Generation Coverage - 2026-06-27",
        "",
        "## Summary",
        "",
        f"- Question-bank records: {report['total_question_bank_records']}",
        f"- Included: {report['total_included']}",
        f"- Approved: {report['approved_count']}",
        f"- Review required: {report['review_required_count']}",
        f"- Excluded: {report['excluded_count']}",
        f"- Unexplained skipped: {report['unexplained_skipped_count']}",
        f"- Packet manifests: {report['manifest_count']}",
        f"- Topic packet PDFs: {report['topic_packet_pdf_count']}",
        f"- P4 packet count: {report['p4_packet_count']}",
        "",
        "## Baseline Comparison",
        "",
        "| Metric | Old baseline | New output |",
        "| --- | ---: | ---: |",
        f"| Included | {report['old_baseline']['included_in_any_topic_packet']} | {report['total_included']} |",
        f"| Approved/release | {report['old_baseline']['release_packets']} | {report['approved_count']} |",
        f"| Review required | {report['old_baseline']['review_required_packets']} | {report['review_required_count']} |",
        f"| Skipped/excluded | {report['old_baseline']['skipped']} | {report['excluded_count']} |",
        f"| P4 packets | {report['old_baseline']['p4_packet_output']} | {report['p4_packet_count']} |",
        "",
        "## P4 Topics",
        "",
    ]
    lines.extend(f"- {topic}" for topic in report["p4_topics_generated"])
    lines.extend(["", "## Excluded Reasons", ""])
    lines.extend(f"- {reason}: {count}" for reason, count in report["excluded_reason_counts"].items())
    lines.extend(["", "## Hard Requirement Failures", ""])
    if report["hard_requirement_failures"]:
        lines.extend(f"- {item}" for item in report["hard_requirement_failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Counts By Family", ""])
    lines.extend(f"- {family}: {count}" for family, count in report["counts_by_family"].items())
    lines.extend(["", "## Counts By Section", ""])
    lines.extend(f"- {section}: {count}" for section, count in report["counts_by_section"].items())
    lines.append("")
    return "\n".join(lines)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_question_bank_records": report["total_question_bank_records"],
        "total_included": report["total_included"],
        "approved_count": report["approved_count"],
        "review_required_count": report["review_required_count"],
        "excluded_count": report["excluded_count"],
        "unexplained_skipped_count": report["unexplained_skipped_count"],
        "p4_packet_count": report["p4_packet_count"],
        "hard_requirement_failures": report["hard_requirement_failures"],
    }


def _counts_by_year(
    records_by_id: dict[str, dict[str, Any]],
    included: dict[str, dict[str, Any]],
    excluded: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    years = sorted({_record_year(record) for record in records_by_id.values()})
    result: dict[str, dict[str, int]] = {}
    for year in years:
        bank_count = sum(1 for record in records_by_id.values() if _record_year(record) == year)
        included_count = sum(1 for question_id in included if _record_year(records_by_id.get(question_id, {})) == year)
        excluded_count = sum(1 for question_id in excluded if _record_year(records_by_id.get(question_id, {})) == year)
        result[year] = {
            "question_bank": bank_count,
            "included": included_count,
            "excluded": excluded_count,
            "accounted": included_count + excluded_count,
        }
    return result


def _record_year(record: dict[str, Any]) -> str:
    text = str(record.get("paper") or record.get("question_id") or "")
    match = re.search(r"(\d{2})(?!.*\d)", text)
    if not match:
        return "unknown"
    year = int(match.group(1))
    return str(2000 + year if year < 70 else 1900 + year)


def _source_component(record: dict[str, Any]) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    for value in [notes.get("source_paper_code"), record.get("source_paper_code"), record.get("component")]:
        if value:
            return str(value)
    paper = str(record.get("paper") or record.get("question_id") or "")
    match = re.match(r"(\d{2})", paper)
    return match.group(1) if match else ""


if __name__ == "__main__":
    raise SystemExit(main())
