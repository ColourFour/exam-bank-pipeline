from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .canonical_sample import (
    CANONICAL_YEAR_END,
    CANONICAL_YEAR_START,
    DEFAULT_SAMPLE_OUTPUT_PATH,
    PAPER_TYPE_LABELS,
    REQUIRED_FIELDS,
    TARGET_PAPER_TYPES,
    load_jsonl,
    write_json,
)


DEFAULT_AUDIT_JSON_PATH = Path("output/samples/canonical_sample_audit.json")
DEFAULT_AUDIT_MD_PATH = Path("output/samples/canonical_sample_audit.md")
QUESTION_TEXT_FIELDS = ("question_text", "combined_question_text", "questions")
TOPIC_FIELDS = (
    "topic",
    "topics",
    "topic_ids",
    "topic_routing",
    "topic_routing_topic_ids",
    "topic_routing_alignment",
)


def audit_canonical_sample(
    rows: list[dict[str, Any]],
    *,
    year_start: int = CANONICAL_YEAR_START,
    year_end: int = CANONICAL_YEAR_END,
) -> dict[str, Any]:
    issues_by_item = _item_issues(rows, year_start=year_start, year_end=year_end)
    duplicate_issues = _duplicate_issues(rows)
    for row_index, issues in duplicate_issues.items():
        issues_by_item[row_index].extend(issues)

    coverage_matrix = _coverage_matrix(rows, year_start=year_start, year_end=year_end)
    valid_items = [
        _item_ref(index, row)
        for index, row in enumerate(rows)
        if not any(issue["severity"] == "error" for issue in issues_by_item[index])
    ]
    invalid_items = [
        {
            **_item_ref(index, row),
            "issues": sorted(issues_by_item[index], key=lambda issue: (issue["severity"], issue["code"])),
            "issue_count": len(issues_by_item[index]),
            "error_count": sum(1 for issue in issues_by_item[index] if issue["severity"] == "error"),
        }
        for index, row in enumerate(rows)
        if any(issue["severity"] == "error" for issue in issues_by_item[index])
    ]
    issue_counts = Counter(issue["code"] for issues in issues_by_item.values() for issue in issues)
    top_problematic = sorted(
        invalid_items,
        key=lambda item: (-item["error_count"], -item["issue_count"], str(item.get("id") or "")),
    )[:20]

    return {
        "schema_name": "exam_bank.canonical_sample_audit",
        "schema_version": 1,
        "source": {
            "sample_jsonl": str(DEFAULT_SAMPLE_OUTPUT_PATH),
            "year_start": year_start,
            "year_end": year_end,
            "target_paper_types": PAPER_TYPE_LABELS,
            "audit_note": (
                "This audit does not rewrite, regenerate, or correct the sample. Rows are checked exactly "
                "as present in the canonical JSONL."
            ),
        },
        "summary": {
            "total_items_checked": len(rows),
            "valid_items": len(valid_items),
            "invalid_items": len(invalid_items),
            "missing_coverage_count": coverage_matrix["missing_count"],
            "duplicate_detection_violations": sum(
                count for code, count in issue_counts.items() if code.startswith("duplicate_")
            ),
            "dataset_usable_for_downstream_training": len(invalid_items) == 0,
            "dataset_usable_for_manifest_selection": coverage_matrix["missing_count"] == 0
            and not any(code.startswith("duplicate_") for code in issue_counts),
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "coverage_matrix": coverage_matrix,
        "valid_item_refs": valid_items,
        "invalid_item_count": len(invalid_items),
        "top_20_problematic_entries": top_problematic,
    }


def build_and_write_audit(
    *,
    sample_path: Path = DEFAULT_SAMPLE_OUTPUT_PATH,
    audit_json_path: Path = DEFAULT_AUDIT_JSON_PATH,
    audit_md_path: Path = DEFAULT_AUDIT_MD_PATH,
) -> dict[str, Any]:
    rows = load_jsonl(sample_path)
    audit = audit_canonical_sample(rows)
    audit["source"]["sample_jsonl"] = str(sample_path)
    audit["outputs"] = {"audit_json": str(audit_json_path), "audit_markdown": str(audit_md_path)}
    write_json(audit_json_path, audit)
    audit_md_path.parent.mkdir(parents=True, exist_ok=True)
    audit_md_path.write_text(render_markdown_audit(audit), encoding="utf-8")
    return audit


def render_markdown_audit(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    matrix = audit["coverage_matrix"]
    lines = [
        "# Canonical Sample Audit",
        "",
        "## Summary",
        "",
        f"- Total items checked: `{summary['total_items_checked']}`",
        f"- Valid items: `{summary['valid_items']}`",
        f"- Invalid items: `{summary['invalid_items']}`",
        f"- Missing coverage cells: `{summary['missing_coverage_count']}`",
        f"- Duplicate detection violations: `{summary['duplicate_detection_violations']}`",
        f"- Usable for downstream training: `{summary['dataset_usable_for_downstream_training']}`",
        f"- Usable as a manifest selection set: `{summary['dataset_usable_for_manifest_selection']}`",
        "",
        "## Issue Counts",
        "",
    ]
    if audit["issue_counts"]:
        for code, count in audit["issue_counts"].items():
            lines.append(f"- `{code}`: `{count}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Coverage Matrix", ""])
    header = "| Year | " + " | ".join(PAPER_TYPE_LABELS[paper] for paper in TARGET_PAPER_TYPES) + " |"
    separator = "| --- | " + " | ".join("---" for _ in TARGET_PAPER_TYPES) + " |"
    lines.extend([header, separator])
    for year, cells in matrix["years"].items():
        values = []
        for paper in TARGET_PAPER_TYPES:
            label = PAPER_TYPE_LABELS[paper]
            cell = cells[label]
            values.append("present" if cell["count"] == 1 else f"missing ({cell['count']})")
        lines.append(f"| {year} | " + " | ".join(values) + " |")

    lines.extend(["", "## Top 20 Problematic Entries", ""])
    if audit["top_20_problematic_entries"]:
        for item in audit["top_20_problematic_entries"]:
            codes = ", ".join(issue["code"] for issue in item["issues"])
            lines.append(
                f"- `{item.get('id')}` year `{item.get('year')}` type `{item.get('paper_type')}`: {codes}"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit canonical one-paper-per-type sample without modifying it.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_OUTPUT_PATH)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON_PATH)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD_PATH)
    args = parser.parse_args(argv)
    audit = build_and_write_audit(sample_path=args.sample, audit_json_path=args.audit_json, audit_md_path=args.audit_md)
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    return 0


def _item_issues(
    rows: list[dict[str, Any]],
    *,
    year_start: int,
    year_end: int,
) -> dict[int, list[dict[str, str]]]:
    issues_by_item: dict[int, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if row.get(field) in (None, ""):
                issues_by_item[index].append(_issue("missing_required_field", f"Missing required field `{field}`."))
        year = row.get("year")
        paper = row.get("paper")
        if not isinstance(year, int) or not (year_start <= year <= year_end):
            issues_by_item[index].append(_issue("invalid_year_range", "Year is missing or outside 2008-2020."))
        if paper not in TARGET_PAPER_TYPES:
            issues_by_item[index].append(_issue("invalid_paper_type", "Paper type is not one of P1, P3, M1, S1."))
        if not _has_question_text(row):
            issues_by_item[index].append(_issue("missing_question_text", "No question text field is present."))
        if not _has_url(row.get("question_paper_url")):
            issues_by_item[index].append(_issue("broken_question_paper_reference", "Question-paper reference is missing or not a URL."))
        if not _has_url(row.get("mark_scheme_url")):
            issues_by_item[index].append(_issue("missing_mark_scheme", "Mark-scheme reference is missing or not a URL."))
        if not _has_topic_fields(row):
            issues_by_item[index].append(_issue("missing_topic_fields", "No topic-routing fields are present."))
        if _id_year_mismatch(row):
            issues_by_item[index].append(_issue("year_type_mismatch", "Identifier does not match row year."))
        if _id_paper_mismatch(row):
            issues_by_item[index].append(_issue("year_type_mismatch", "Identifier does not match row paper type."))
    return issues_by_item


def _duplicate_issues(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, str]]]:
    issues: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_id: dict[str, list[int]] = defaultdict(list)
    by_year_type: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    row_keys: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_id = row.get("id")
        if isinstance(row_id, str):
            by_id[row_id].append(index)
        by_year_type[(row.get("year"), row.get("paper"))].append(index)
        row_keys[json.dumps(row, sort_keys=True, separators=(",", ":"))].append(index)

    for indexes in by_id.values():
        if len(indexes) > 1:
            for index in indexes:
                issues[index].append(_issue("duplicate_id", "Duplicate paper id detected."))
    for indexes in by_year_type.values():
        if len(indexes) > 1:
            for index in indexes:
                issues[index].append(_issue("duplicate_year_paper_type", "More than one row for this year and paper type."))
    for indexes in row_keys.values():
        if len(indexes) > 1:
            for index in indexes:
                issues[index].append(_issue("duplicate_row", "Identical duplicate row detected."))
    return issues


def _coverage_matrix(rows: list[dict[str, Any]], *, year_start: int, year_end: int) -> dict[str, Any]:
    counts = Counter((row.get("year"), row.get("paper")) for row in rows)
    years: dict[str, Any] = {}
    missing_count = 0
    for year in range(year_start, year_end + 1):
        years[str(year)] = {}
        for paper in TARGET_PAPER_TYPES:
            label = PAPER_TYPE_LABELS[paper]
            count = counts[(year, paper)]
            if count != 1:
                missing_count += 1
            years[str(year)][label] = {"paper": paper, "count": count, "status": "present" if count == 1 else "missing"}
    return {"missing_count": missing_count, "years": years}


def _item_ref(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": index,
        "id": row.get("id"),
        "year": row.get("year"),
        "paper_type": PAPER_TYPE_LABELS.get(str(row.get("paper")), row.get("paper")),
        "paper": row.get("paper"),
        "component": row.get("component"),
        "session_code": row.get("session_code"),
    }


def _issue(code: str, message: str) -> dict[str, str]:
    return {"severity": "error", "code": code, "message": message}


def _has_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _has_question_text(row: dict[str, Any]) -> bool:
    for field in QUESTION_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _has_topic_fields(row: dict[str, Any]) -> bool:
    return any(field in row and row.get(field) not in (None, "", []) for field in TOPIC_FIELDS)


def _id_year_mismatch(row: dict[str, Any]) -> bool:
    row_id = str(row.get("id") or "")
    year = row.get("year")
    return isinstance(year, int) and f"{year % 100:02d}" not in row_id


def _id_paper_mismatch(row: dict[str, Any]) -> bool:
    row_id = str(row.get("id") or "")
    paper = str(row.get("paper") or "")
    return bool(paper) and paper not in row_id
