from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from exam_bank.advisory_evidence.common import load_json, rel_path
from exam_bank.advisory_evidence.constants import (
    EXAMINER_DIFFICULTY_PATH,
    EXAMINER_LINKS_PATH,
    EXAMINER_PARSED_DIR,
    EXAMINER_REPORT_INVENTORY,
    GRADE_THRESHOLD_CONTEXT_PATH,
    GRADE_THRESHOLD_INVENTORY,
    GRADE_THRESHOLD_LINKS_PATH,
    REPORTS_DIR,
    TOPIC_EVIDENCE_PATH,
)


def build_review_reports(
    *,
    advisory_root: str | Path = "output/advisory_evidence",
    output_dir: str | Path = REPORTS_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(advisory_root)
    output_dir = Path(output_dir)
    data = _load_report_data(root)
    reports = {
        "examiner_report_extraction_status.md": _examiner_status_report(data),
        "grade_threshold_extraction_status.md": _threshold_status_report(data),
        "advisory_topic_prediction_review.md": _topic_review_report(data),
        "advisory_difficulty_prediction_review.md": _difficulty_review_report(data),
        "unlinked_examiner_report_entries.md": _unlinked_report(data),
        "low_confidence_predictions.md": _low_confidence_report(data),
    }
    outputs: list[str] = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in reports.items():
            path = output_dir / filename
            path.write_text(content, encoding="utf-8")
            outputs.append(rel_path(path))
    else:
        outputs = [rel_path(output_dir / filename) for filename in reports]
    return {"dry_run": dry_run, "outputs": outputs, "report_count": len(reports)}


def _load_report_data(root: Path) -> dict[str, Any]:
    return {
        "examiner_inventory": load_json(root / EXAMINER_REPORT_INVENTORY.relative_to("output/advisory_evidence"), default={"documents": []}),
        "threshold_inventory": load_json(root / GRADE_THRESHOLD_INVENTORY.relative_to("output/advisory_evidence"), default={"documents": []}),
        "examiner_links": load_json(root / EXAMINER_LINKS_PATH.relative_to("output/advisory_evidence"), default={"links": []}),
        "threshold_links": load_json(root / GRADE_THRESHOLD_LINKS_PATH.relative_to("output/advisory_evidence"), default={"links": []}),
        "topic": load_json(root / TOPIC_EVIDENCE_PATH.relative_to("output/advisory_evidence"), default={"records": []}),
        "difficulty": load_json(root / EXAMINER_DIFFICULTY_PATH.relative_to("output/advisory_evidence"), default={"records": []}),
        "context": load_json(root / GRADE_THRESHOLD_CONTEXT_PATH.relative_to("output/advisory_evidence"), default={"contexts": []}),
        "examiner_parsed": [
            load_json(path)
            for path in sorted((root / EXAMINER_PARSED_DIR.relative_to("output/advisory_evidence")).glob("*.json"))
            if path.exists()
        ],
    }


def _examiner_status_report(data: dict[str, Any]) -> str:
    documents = data["examiner_inventory"].get("documents", [])
    links = data["examiner_links"].get("links", [])
    status = Counter(link.get("match_status") for link in links)
    low = sum(1 for link in links if link.get("evidence_level") in {"low", "none"})
    return "\n".join(
        [
            "# Examiner Report Extraction Status",
            "",
            f"- Inventoried examiner reports: {len(documents)}",
            f"- Extractable native text reports: {sum(1 for doc in documents if doc.get('can_extract_native_text'))}",
            f"- Parsed question-comment links: {len(links)}",
            f"- Linked comments: {status.get('linked', 0)}",
            f"- Ambiguous comments: {status.get('ambiguous', 0)}",
            f"- Unlinked comments: {status.get('unlinked', 0)}",
            f"- Low/no-evidence comments: {low}",
            "",
            "Advisory evidence is review support only and must not replace canonical images or mark schemes.",
        ]
    )


def _threshold_status_report(data: dict[str, Any]) -> str:
    documents = data["threshold_inventory"].get("documents", [])
    links = data["threshold_links"].get("links", [])
    status = Counter(link.get("match_status") for link in links)
    contexts = data["context"].get("contexts", [])
    lines = [
        "# Grade Threshold Extraction Status",
        "",
        f"- Inventoried grade-threshold PDFs: {len(documents)}",
        f"- Extractable native text PDFs: {sum(1 for doc in documents if doc.get('can_extract_native_text'))}",
        f"- Component threshold links: {len(links)}",
        f"- Linked components: {status.get('linked', 0)}",
        f"- Unlinked components: {status.get('unlinked', 0)}",
        f"- Component context records: {len(contexts)}",
        "",
        "Grade thresholds provide paper/component context only, not individual-question difficulty.",
        "",
        "## Grade-Threshold Unlinked Queue",
        "",
        f"- Unlinked components requiring review: {status.get('unlinked', 0)}",
        "",
    ]
    for link in [link for link in links if link.get("match_status") != "linked"]:
        lines.append(
            f"- `{link.get('normalized_key')}`: {link.get('match_status')} component `{link.get('component')}` "
            f"session `{link.get('session')}` from `{link.get('source_path')}`"
        )
    return "\n".join(lines)


def _topic_review_report(data: dict[str, Any]) -> str:
    records = data["topic"].get("records", [])
    confidence = Counter(record.get("topic_evidence", {}).get("confidence") for record in records)
    duplicate_lines = _duplicate_summary_lines("topic evidence", records)
    return "\n".join(
        [
            "# Advisory Topic Prediction Review",
            "",
            f"- Topic evidence records: {len(records)}",
            f"- High confidence: {confidence.get('high', 0)}",
            f"- Medium confidence: {confidence.get('medium', 0)}",
            f"- Low confidence: {confidence.get('low', 0)}",
            f"- Review required: {sum(1 for record in records if record.get('topic_evidence', {}).get('review_required'))}",
            "",
            "## Duplicate Advisory Evidence Summary",
            "",
            *duplicate_lines,
        ]
    )


def _difficulty_review_report(data: dict[str, Any]) -> str:
    records = data["difficulty"].get("records", [])
    signal = Counter(record.get("examiner_report_difficulty", {}).get("item_signal") for record in records)
    contexts = data["context"].get("contexts", [])
    duplicate_lines = _duplicate_summary_lines("difficulty evidence", records)
    return "\n".join(
        [
            "# Advisory Difficulty Prediction Review",
            "",
            f"- Examiner difficulty records: {len(records)}",
            f"- Hard signals: {signal.get('hard', 0)}",
            f"- Easy signals: {signal.get('easy', 0)}",
            f"- Mixed signals: {signal.get('mixed', 0)}",
            f"- Unknown signals: {signal.get('unknown', 0)}",
            f"- Grade-threshold context records: {len(contexts)}",
            "",
            "Threshold context is intentionally separate from item-level difficulty.",
            "",
            "## Duplicate Advisory Evidence Summary",
            "",
            *duplicate_lines,
        ]
    )


def _unlinked_report(data: dict[str, Any]) -> str:
    links = [link for link in data["examiner_links"].get("links", []) if link.get("match_status") != "linked"]
    reason_counts = Counter(_unlinked_reason(link) for link in links)
    component_counts = Counter(str(link.get("component") or "unknown") for link in links)
    session_counts = Counter(str(link.get("session") or "unknown") for link in links)
    low_links = [link for link in data["examiner_links"].get("links", []) if link.get("evidence_level") in {"low", "none"}]
    low_sections = _low_or_no_evidence_sections(data)
    lines = [
        "# Unlinked Examiner Report Entries",
        "",
        f"- Entries requiring review: {len(links)}",
        "",
        "## Summary By Reason",
        "",
        *_counter_lines(reason_counts),
        "",
        "## Summary By Component",
        "",
        *_counter_lines(component_counts),
        "",
        "## Summary By Session",
        "",
        *_counter_lines(session_counts),
        "",
        "## Detailed Entries",
        "",
    ]
    for link in links:
        lines.append(f"- `{link.get('normalized_key')}`: {link.get('match_status')} from `{link.get('source_path')}`")
    lines.extend(
        [
            "",
            "## Low/No-Evidence Section Queue",
            "",
            f"- Low/no-evidence linked or unlinked question comments: {len(low_links)}",
            f"- Low/no-evidence parsed component sections: {len(low_sections)}",
            "",
        ]
    )
    for link in low_links:
        lines.append(
            f"- `{link.get('normalized_key')}`: evidence `{link.get('evidence_level')}` warnings `{', '.join(link.get('warnings', []))}` "
            f"from `{link.get('source_path')}`"
        )
    for section in low_sections:
        lines.append(
            f"- component `{section['component']}` session `{section['session']}` evidence `{section['evidence_level']}` from `{section['source_path']}`"
        )
    return "\n".join(lines)


def _low_confidence_report(data: dict[str, Any]) -> str:
    topic = [
        record
        for record in data["topic"].get("records", [])
        if record.get("topic_evidence", {}).get("confidence") in {"low", "unknown"} or record.get("topic_evidence", {}).get("review_required")
    ]
    difficulty = [
        record
        for record in data["difficulty"].get("records", [])
        if record.get("examiner_report_difficulty", {}).get("confidence") in {"low", "unknown"}
        or record.get("examiner_report_difficulty", {}).get("review_required")
    ]
    lines = [
        "# Low Confidence Predictions",
        "",
        f"- Topic records needing review: {len(topic)}",
        f"- Difficulty records needing review: {len(difficulty)}",
        "",
    ]
    for record in topic[:100]:
        lines.append(f"- topic `{record.get('question_id')}`")
    for record in difficulty[:100]:
        lines.append(f"- difficulty `{record.get('question_id')}`")
    return "\n".join(lines)


def _duplicate_summary_lines(label: str, records: list[dict[str, Any]]) -> list[str]:
    question_counts = Counter(str(record.get("question_id") or "") for record in records if record.get("question_id"))
    identity_counts = Counter(
        (
            str(record.get("question_id") or ""),
            str(record.get("normalized_key") or record.get("source_path") or ""),
        )
        for record in records
        if record.get("question_id")
    )
    duplicates = [(question_id, count) for question_id, count in sorted(question_counts.items()) if count > 1]
    identity_duplicates = [(question_id, identity, count) for (question_id, identity), count in sorted(identity_counts.items()) if identity and count > 1]
    lines = [
        f"- Duplicate {label} question IDs: {len(duplicates)}",
        f"- Duplicate {label} source identities: {len(identity_duplicates)}",
    ]
    for question_id, count in duplicates[:100]:
        lines.append(f"- `{question_id}` appears {count} times")
    for question_id, identity, count in identity_duplicates[:100]:
        lines.append(f"- `{question_id}` with source identity `{identity}` appears {count} times")
    return lines


def _counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- `{key}`: {count}" for key, count in sorted(counter.items())]


def _unlinked_reason(link: dict[str, Any]) -> str:
    warnings = [str(warning) for warning in link.get("warnings", []) if warning]
    return ",".join(warnings) if warnings else str(link.get("match_status") or "unknown")


def _low_or_no_evidence_sections(data: dict[str, Any]) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for parsed in data.get("examiner_parsed", []):
        for component in parsed.get("components", []):
            evidence_level = str(component.get("evidence_level") or "normal")
            if evidence_level not in {"low", "none"}:
                continue
            sections.append(
                {
                    "source_path": str(parsed.get("source_path") or ""),
                    "session": str(parsed.get("session") or ""),
                    "component": str(component.get("component") or ""),
                    "evidence_level": evidence_level,
                }
            )
    return sections
