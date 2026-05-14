from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from exam_bank.advisory_evidence.common import load_json, rel_path
from exam_bank.advisory_evidence.constants import (
    EXAMINER_DIFFICULTY_PATH,
    EXAMINER_LINKS_PATH,
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
    return "\n".join(
        [
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
        ]
    )


def _topic_review_report(data: dict[str, Any]) -> str:
    records = data["topic"].get("records", [])
    confidence = Counter(record.get("topic_evidence", {}).get("confidence") for record in records)
    return "\n".join(
        [
            "# Advisory Topic Prediction Review",
            "",
            f"- Topic evidence records: {len(records)}",
            f"- High confidence: {confidence.get('high', 0)}",
            f"- Medium confidence: {confidence.get('medium', 0)}",
            f"- Low confidence: {confidence.get('low', 0)}",
            f"- Review required: {sum(1 for record in records if record.get('topic_evidence', {}).get('review_required'))}",
        ]
    )


def _difficulty_review_report(data: dict[str, Any]) -> str:
    records = data["difficulty"].get("records", [])
    signal = Counter(record.get("examiner_report_difficulty", {}).get("item_signal") for record in records)
    contexts = data["context"].get("contexts", [])
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
        ]
    )


def _unlinked_report(data: dict[str, Any]) -> str:
    links = [link for link in data["examiner_links"].get("links", []) if link.get("match_status") != "linked"]
    lines = ["# Unlinked Examiner Report Entries", "", f"- Entries requiring review: {len(links)}", ""]
    for link in links[:200]:
        lines.append(f"- `{link.get('normalized_key')}`: {link.get('match_status')} from `{link.get('source_path')}`")
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

