from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.topic_packets import load_packet_taxonomy, normalize_packet_topic, normalize_paper_family


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_BANK = ROOT / "output/json/question_bank.json"
DEFAULT_TOPIC_ROUTING = ROOT / "data/topic_routing/question_bank.topic_routing.v1.json"
DEFAULT_TAXONOMY = ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json"
DEFAULT_JSON_OUT = ROOT / "reports/topic_packet_preflight_audit_2026_06_27.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports/topic_packet_preflight_audit_2026_06_27.md"

SCHEMA_NAME = "exam_bank.topic_packet_preflight_audit"
SCHEMA_VERSION = 1

RELEASE_EXPECTATIONS = [
    ("mapping_status", "pass", "mapping_status_not_pass"),
    ("validation_status", "pass", "validation_status_not_pass"),
    ("scope_quality_status", "clean", "scope_quality_status_not_clean"),
    ("question_crop_confidence", "high", "question_crop_confidence_not_high"),
    ("visual_curation_status", "ready", "visual_curation_status_not_ready"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit topic-packet readiness before regenerating packet PDFs.")
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--topic-routing", type=Path, default=DEFAULT_TOPIC_ROUTING)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when warning conditions are found. Reports are still written first.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_audit(
        question_bank_path=args.question_bank,
        topic_routing_path=args.topic_routing,
        taxonomy_path=args.taxonomy,
    )
    write_json(args.json_out, audit)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    return 2 if args.strict and audit["warnings"] else 0


def build_audit(*, question_bank_path: Path, topic_routing_path: Path, taxonomy_path: Path) -> dict[str, Any]:
    question_bank_payload = read_json(question_bank_path)
    topic_routing_payload = read_json(topic_routing_path)

    question_records = question_bank_records(question_bank_payload)
    route_records = topic_routing_records(topic_routing_payload)
    taxonomy = load_packet_taxonomy(taxonomy_path)

    enriched_records = [enrich_question_record(record, taxonomy) for record in question_records]
    route_rows = [enrich_route_record(question_id, route) for question_id, route in route_records.items()]

    qbank_years = sorted({row["year"] for row in enriched_records if row["year"] is not None})
    route_years = sorted({row["year"] for row in route_rows if row["year"] is not None})
    years_absent_from_routing = [year for year in qbank_years if year not in route_years]

    warnings = []
    if years_absent_from_routing:
        warnings.append(
            {
                "code": "question_bank_years_absent_from_topic_routing",
                "message": "One or more question-bank years have no corresponding records in durable topic routing.",
                "years": years_absent_from_routing,
            }
        )
    if any(row["expected_family"] != row["current_family"] for row in enriched_records if row["expected_family"]):
        warnings.append(
            {
                "code": "family_normalization_changes_expected_packet_family",
                "message": "Some records resolve to a different packet family after topic-taxonomy normalization.",
            }
        )

    invalid_major_topic = [row for row in enriched_records if row["current_topic_valid"] is False]
    unresolved_after_normalization = [row for row in enriched_records if row["normalization_status"] != "resolved"]
    mapping_fail = [row for row in enriched_records if row["mapping_status"] == "fail"]
    validation_fail = [row for row in enriched_records if row["validation_status"] == "fail"]

    release_candidates = [row for row in enriched_records if row["packet_candidate_status"] == "release_candidate"]
    review_candidates = [row for row in enriched_records if row["packet_candidate_status"] == "review_required_candidate"]
    skipped_candidates = [row for row in enriched_records if row["packet_candidate_status"].startswith("blocked_")]

    by_year_family_component = group_counts(enriched_records, ("year", "current_family", "source_component"))
    routing_by_year_family_component = group_counts(route_rows, ("year", "route_family", "source_component"))

    summary = {
        "question_bank_records": len(enriched_records),
        "topic_routing_records": len(route_rows),
        "question_bank_year_range": year_range(qbank_years),
        "topic_routing_year_range": year_range(route_years),
        "question_bank_years_absent_from_topic_routing": years_absent_from_routing,
        "warning_count": len(warnings),
        "invalid_major_topic_count": len(invalid_major_topic),
        "unresolved_after_normalization_count": len(unresolved_after_normalization),
        "mapping_status_fail_count": len(mapping_fail),
        "validation_status_fail_count": len(validation_fail),
        "release_candidate_count_after_normalization": len(release_candidates),
        "review_required_candidate_count_after_normalization": len(review_candidates),
        "blocked_candidate_count_after_normalization": len(skipped_candidates),
    }

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "question_bank": display_path(question_bank_path),
            "topic_routing": display_path(topic_routing_path),
            "taxonomy": display_path(taxonomy_path),
        },
        "summary": summary,
        "warnings": warnings,
        "question_bank_by_year_family_component": by_year_family_component,
        "topic_routing_by_year_family_component": routing_by_year_family_component,
        "topic_routing_coverage": {
            "question_bank_years": qbank_years,
            "topic_routing_years": route_years,
            "years_absent_from_routing": years_absent_from_routing,
            "question_bank_counts_by_year": counter_dict(row["year"] for row in enriched_records),
            "topic_routing_counts_by_year": counter_dict(row["year"] for row in route_rows),
        },
        "invalid_major_topics": summarize_issue_rows(invalid_major_topic),
        "blocked_by_mapping_status_fail": summarize_issue_rows(mapping_fail),
        "blocked_by_validation_status_fail": summarize_issue_rows(validation_fail),
        "normalization": {
            "status_counts": counter_dict(row["normalization_status"] for row in enriched_records),
            "expected_family_counts": counter_dict(row["expected_family"] for row in enriched_records),
            "family_changes": summarize_issue_rows(
                [row for row in enriched_records if row["expected_family"] and row["expected_family"] != row["current_family"]]
            ),
            "expected_family_topic_counts": group_counts(enriched_records, ("expected_family", "expected_topic")),
            "unresolved_records": summarize_issue_rows(unresolved_after_normalization),
        },
        "candidate_counts": {
            "status_counts": counter_dict(row["packet_candidate_status"] for row in enriched_records),
            "status_by_year": group_counts(enriched_records, ("year", "packet_candidate_status")),
            "status_by_family": group_counts(enriched_records, ("expected_family", "packet_candidate_status")),
            "release_quality_reason_counts": counter_dict(
                reason for row in enriched_records for reason in row["release_quality_reasons"]
            ),
        },
    }


def enrich_question_record(record: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    question_id = text(record.get("question_id"))
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    year = int_or_none(record.get("canonical_year_folder")) or parse_year_from_question_id(question_id)
    source_component = text(notes.get("source_paper_code")) or parse_component_from_paper(text(record.get("paper")))
    current_family = normalize_family(record.get("paper_family"))
    raw_topic = text(record.get("topic"))
    normalization = normalize_packet_topic(
        component_code=source_component,
        current_family=current_family,
        raw_topic=raw_topic,
        taxonomy=taxonomy,
    )

    mapping_status = status_value(record, "mapping_status")
    validation_status = status_value(record, "validation_status")
    release_quality_reasons = release_quality_reasons_for(record)
    if normalization.status != "resolved":
        candidate_status = "blocked_unresolved_topic"
    elif mapping_status == "fail":
        candidate_status = "blocked_mapping_status_fail"
    elif validation_status == "fail":
        candidate_status = "blocked_validation_status_fail"
    elif release_quality_reasons:
        candidate_status = "review_required_candidate"
    else:
        candidate_status = "release_candidate"

    return {
        "question_id": question_id,
        "year": year,
        "paper": text(record.get("paper")),
        "source_component": source_component,
        "current_family": current_family,
        "raw_paper_family": text(record.get("paper_family")),
        "raw_topic": raw_topic,
        "current_topic_valid": normalization.current_topic_valid,
        "expected_family": normalization.expected_family,
        "expected_topic": normalization.expected_topic,
        "normalization_status": normalization.status,
        "normalization_reason": normalization.reason,
        "mapping_status": mapping_status,
        "validation_status": validation_status,
        "release_quality_reasons": release_quality_reasons,
        "packet_candidate_status": candidate_status,
    }


def enrich_route_record(question_id: str, route: dict[str, Any]) -> dict[str, Any]:
    paper = text(route.get("paper"))
    return {
        "question_id": question_id,
        "year": parse_year_from_question_id(question_id) or parse_year_from_paper(paper),
        "paper": paper,
        "source_component": parse_component_from_paper(paper),
        "route_family": normalize_family(route.get("paper_family")),
        "primary_topic_id": text(route.get("primary_topic_id")),
        "review_required": route.get("review_required") is True,
        "confidence": text(route.get("confidence")),
    }


def release_quality_reasons_for(record: dict[str, Any]) -> list[str]:
    reasons = []
    for key, expected, reason in RELEASE_EXPECTATIONS:
        if status_value(record, key) != expected:
            reasons.append(reason)
    if status_value(record, "text_only_status") == "fail":
        reasons.append("text_only_status_fail")
    return reasons


def summarize_issue_rows(rows: list[dict[str, Any]], *, limit: int = 100) -> dict[str, Any]:
    return {
        "count": len(rows),
        "by_year": counter_dict(row["year"] for row in rows),
        "by_current_family": counter_dict(row["current_family"] for row in rows),
        "by_expected_family": counter_dict(row["expected_family"] for row in rows),
        "by_component": counter_dict(row["source_component"] for row in rows),
        "by_raw_topic": top_counter_dict(row["raw_topic"] for row in rows),
        "sample_records": [
            {
                "question_id": row["question_id"],
                "year": row["year"],
                "paper": row["paper"],
                "source_component": row["source_component"],
                "current_family": row["current_family"],
                "raw_topic": row["raw_topic"],
                "expected_family": row["expected_family"],
                "expected_topic": row["expected_topic"],
                "normalization_status": row["normalization_status"],
                "mapping_status": row["mapping_status"],
                "validation_status": row["validation_status"],
                "packet_candidate_status": row["packet_candidate_status"],
            }
            for row in rows[:limit]
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Topic Packet Preflight Audit",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "## Inputs",
        "",
        f"- Question bank: `{audit['inputs']['question_bank']}`",
        f"- Topic routing: `{audit['inputs']['topic_routing']}`",
        f"- Taxonomy: `{audit['inputs']['taxonomy']}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Question-bank records | {summary['question_bank_records']} |",
        f"| Topic-routing records | {summary['topic_routing_records']} |",
        f"| Invalid current major topics | {summary['invalid_major_topic_count']} |",
        f"| Unresolved after normalization | {summary['unresolved_after_normalization_count']} |",
        f"| Blocked by `mapping_status=fail` | {summary['mapping_status_fail_count']} |",
        f"| Blocked by `validation_status=fail` | {summary['validation_status_fail_count']} |",
        f"| Release candidates after normalization | {summary['release_candidate_count_after_normalization']} |",
        f"| Review-required candidates after normalization | {summary['review_required_candidate_count_after_normalization']} |",
        f"| Blocked candidates after normalization | {summary['blocked_candidate_count_after_normalization']} |",
        "",
        "## Warnings",
        "",
    ]
    if audit["warnings"]:
        for warning in audit["warnings"]:
            years = warning.get("years")
            suffix = f" Years: {format_year_ranges(years)}." if years else ""
            lines.append(f"- `{warning['code']}`: {warning['message']}{suffix}")
    else:
        lines.append("- None.")

    coverage = audit["topic_routing_coverage"]
    lines.extend(
        [
            "",
            "## Routing Coverage",
            "",
            f"- Question-bank year range: {summary['question_bank_year_range']}",
            f"- Topic-routing year range: {summary['topic_routing_year_range']}",
            f"- Question-bank years absent from durable routing: {format_year_ranges(coverage['years_absent_from_routing'])}",
            "",
            "### Counts By Year",
            "",
            "| Year | Question bank | Topic routing |",
            "|---:|---:|---:|",
        ]
    )
    all_years = sorted(set(coverage["question_bank_counts_by_year"]) | set(coverage["topic_routing_counts_by_year"]), key=int)
    for year in all_years:
        lines.append(
            f"| {year} | {coverage['question_bank_counts_by_year'].get(year, 0)} | "
            f"{coverage['topic_routing_counts_by_year'].get(year, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Question Bank By Year/Family/Component",
            "",
            "| Year | Family | Component | Count |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in audit["question_bank_by_year_family_component"]:
        lines.append(f"| {row['year']} | {row['current_family']} | {row['source_component']} | {row['count']} |")

    lines.extend(
        [
            "",
            "## Topic Routing By Year/Family/Component",
            "",
            "| Year | Family | Component | Count |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in audit["topic_routing_by_year_family_component"]:
        lines.append(f"| {row['year']} | {row['route_family']} | {row['source_component']} | {row['count']} |")

    lines.extend(render_issue_section("Invalid Current Major Topics", audit["invalid_major_topics"]))
    lines.extend(render_issue_section("Blocked By Mapping Status Fail", audit["blocked_by_mapping_status_fail"]))
    lines.extend(render_issue_section("Blocked By Validation Status Fail", audit["blocked_by_validation_status_fail"]))

    lines.extend(
        [
            "",
            "## Expected Packet Family/Topic After Normalization",
            "",
            "Normalization first tries the source component family when the topic is valid there, then the record's current packet family, then unique non-ambiguous taxonomy matches.",
            "",
            "### Normalization Status",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    for status, count in audit["normalization"]["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "### Expected Family/Topic Counts",
            "",
            "| Expected family | Expected topic | Count |",
            "|---|---|---:|",
        ]
    )
    for row in audit["normalization"]["expected_family_topic_counts"]:
        lines.append(f"| {row['expected_family']} | {row['expected_topic']} | {row['count']} |")

    lines.extend(render_issue_section("Records Whose Expected Family Changes", audit["normalization"]["family_changes"]))
    lines.extend(render_issue_section("Unresolved After Normalization", audit["normalization"]["unresolved_records"]))

    lines.extend(
        [
            "",
            "## Release Vs Review Candidate Counts",
            "",
            "| Candidate status | Count |",
            "|---|---:|",
        ]
    )
    for status, count in audit["candidate_counts"]["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "### Release Quality Reason Counts",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in audit["candidate_counts"]["release_quality_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "1. Review any topic-routing refresh conflicts before promoting additional release packets.",
            "2. Keep mapping and validation failures out of release packets unless they are separately remediated.",
            "3. Generate review-required packets from normalized candidates before promoting additional release packets.",
            "",
        ]
    )
    return "\n".join(lines)


def render_issue_section(title: str, issue: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        f"Count: {issue['count']}",
        "",
        "### Top Raw Topics",
        "",
        "| Raw topic | Count |",
        "|---|---:|",
    ]
    for topic, count in issue["by_raw_topic"].items():
        lines.append(f"| {topic} | {count} |")
    lines.extend(
        [
            "",
            "### By Current Family",
            "",
            "| Family | Count |",
            "|---|---:|",
        ]
    )
    for family, count in issue["by_current_family"].items():
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "### Sample Records",
            "",
            "| Question ID | Year | Component | Current family | Raw topic | Expected family | Expected topic | Status |",
            "|---|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in issue["sample_records"][:25]:
        lines.append(
            f"| {row['question_id']} | {row['year']} | {row['source_component']} | "
            f"{row['current_family']} | {row['raw_topic']} | {row['expected_family']} | "
            f"{row['expected_topic']} | {row['packet_candidate_status']} |"
        )
    return lines


def question_bank_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return [item for item in payload["questions"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Question bank must be a dict with questions[] or a record list.")


def topic_routing_records(payload: Any) -> dict[str, dict[str, Any]]:
    records = payload.get("records") if isinstance(payload, dict) else payload
    if isinstance(records, dict):
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}
    if isinstance(records, list):
        output = {}
        for item in records:
            if isinstance(item, dict) and item.get("question_id"):
                output[str(item["question_id"])] = item
        return output
    raise ValueError("Topic routing must contain records as an object or list.")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def group_counts(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    counts = Counter(tuple(row.get(field) for field in fields) for row in rows)
    result = []
    for key, count in sorted(counts.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        item = {field: key[index] for index, field in enumerate(fields)}
        item["count"] = count
        result.append(item)
    return result


def counter_dict(values: Any) -> dict[str, int]:
    counter = Counter("" if value is None else str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def top_counter_dict(values: Any, *, limit: int = 30) -> dict[str, int]:
    counter = Counter("" if value is None else str(value) for value in values)
    return dict(counter.most_common(limit))


def status_value(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value in (None, "") and isinstance(record.get("notes"), dict):
        value = record["notes"].get(key)
    return text(value).lower()


def parse_component_from_paper(paper: str) -> str:
    match = re.match(r"(?P<component>\d+)", paper or "")
    return match.group("component") if match else ""


def parse_year_from_question_id(question_id: str) -> int | None:
    match = re.search(r"(?P<year>\d{2})_q\d+", question_id or "")
    if not match:
        return None
    year = int(match.group("year"))
    return 2000 + year


def parse_year_from_paper(paper: str) -> int | None:
    match = re.search(r"(?P<year>\d{2})$", paper or "")
    if not match:
        return None
    return 2000 + int(match.group("year"))


def year_range(years: list[int]) -> str:
    if not years:
        return ""
    return f"{min(years)}-{max(years)}"


def format_year_ranges(years: list[int] | list[str]) -> str:
    if not years:
        return "none"
    nums = sorted(int(year) for year in years)
    ranges: list[str] = []
    start = prev = nums[0]
    for year in nums[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = year
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def normalize_family(value: Any) -> str:
    return normalize_paper_family(value)


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def text(value: Any) -> str:
    return str(value or "").strip()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
