from __future__ import annotations

from pathlib import Path
from typing import Any

from exam_bank.advisory_evidence.common import load_json, records_from_question_bank, rel_path, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    EXAMINER_DIFFICULTY_PATH,
    EXAMINER_LINKS_PATH,
    EXAMINER_PARSED_DIR,
    FINAL_SIDECAR_PATH,
    FINAL_SIDECAR_SCHEMA,
    GRADE_THRESHOLD_CONTEXT_PATH,
    GRADE_THRESHOLD_LINKS_PATH,
    TOPIC_EVIDENCE_PATH,
)
from exam_bank.advisory_evidence.signals import _examiner_comments_by_key
from exam_bank.atomic_json import write_atomic_json


def build_final_sidecar(
    *,
    advisory_root: str | Path = "output/advisory_evidence",
    question_bank_path: str | Path = "output/json/question_bank.json",
    output_path: str | Path = FINAL_SIDECAR_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(advisory_root)
    question_records = records_from_question_bank(load_json(question_bank_path, default={"questions": []}))
    examiner_comments = _examiner_comments_by_key(root / EXAMINER_PARSED_DIR.relative_to("output/advisory_evidence"))
    examiner_links = load_json(root / EXAMINER_LINKS_PATH.relative_to("output/advisory_evidence"), default={"links": []}).get("links", [])
    threshold_links = load_json(root / GRADE_THRESHOLD_LINKS_PATH.relative_to("output/advisory_evidence"), default={"links": []}).get("links", [])
    topic_records = load_json(root / TOPIC_EVIDENCE_PATH.relative_to("output/advisory_evidence"), default={"records": []}).get("records", [])
    difficulty_records = load_json(root / EXAMINER_DIFFICULTY_PATH.relative_to("output/advisory_evidence"), default={"records": []}).get("records", [])
    contexts = load_json(root / GRADE_THRESHOLD_CONTEXT_PATH.relative_to("output/advisory_evidence"), default={"contexts": []}).get("contexts", [])

    examiner_by_question = _examiner_by_question(examiner_links, examiner_comments)
    threshold_by_question = _threshold_by_question(threshold_links)
    topic_by_question = {record["question_id"]: record.get("topic_evidence", {}) for record in topic_records if record.get("question_id")}
    difficulty_by_question = {
        record["question_id"]: record.get("examiner_report_difficulty", {}) for record in difficulty_records if record.get("question_id")
    }
    context_by_component_key = {
        f"{context.get('syllabus')}_{context.get('year')}_{context.get('session')}_{context.get('component')}": context for context in contexts
    }

    records: list[dict[str, Any]] = []
    for question in question_records:
        question_id = str(question.get("question_id") or "")
        threshold = threshold_by_question.get(question_id)
        context = context_by_component_key.get(threshold.get("normalized_key")) if threshold else None
        records.append(
            {
                "question_id": question_id,
                "advisory_evidence": {
                    "examiner_report": examiner_by_question.get(question_id, {"available": False, "warnings": []}),
                    "grade_threshold_context": _grade_context_payload(threshold, context),
                },
                "topic_evidence": topic_by_question.get(question_id, {}),
                "examiner_report_difficulty": difficulty_by_question.get(question_id, {}),
            }
        )

    payload = {
        "schema": FINAL_SIDECAR_SCHEMA,
        "generated_at": utc_now_iso(),
        "source_question_bank": rel_path(question_bank_path),
        "records_count": len(records),
        "records": records,
        "warnings": [],
    }
    if not dry_run:
        write_atomic_json(payload, output_path)
    return payload


def _examiner_by_question(links: list[dict[str, Any]], comments_by_key: dict[str, str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for link in links:
        if link.get("match_status") != "linked" or not link.get("candidate_question_ids"):
            continue
        question_id = link["candidate_question_ids"][0]
        key = str(link.get("normalized_key") or "")
        output[question_id] = {
            "available": True,
            "normalized_key": key,
            "component": link.get("component", ""),
            "question_number": link.get("question_number"),
            "comment_text": comments_by_key.get(key, ""),
            "evidence_level": link.get("evidence_level", "normal"),
            "source_path": link.get("source_path", ""),
            "warnings": list(link.get("warnings", [])),
        }
    return output


def _threshold_by_question(links: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for link in links:
        if link.get("match_status") != "linked":
            continue
        for question_id in link.get("candidate_question_ids", []):
            output[question_id] = link
    return output


def _grade_context_payload(threshold: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
    if not threshold:
        return {"available": False, "warnings": []}
    return {
        "available": True,
        "normalized_key": threshold.get("normalized_key", ""),
        "component": threshold.get("component", ""),
        "component_max_raw": threshold.get("max_raw_mark"),
        "thresholds": threshold.get("thresholds", {}),
        "component_context_label": context.get("component_context_label") if context else "paper_context_unknown",
        "threshold_ratios": context.get("threshold_ratios", {}) if context else {},
        "source_path": threshold.get("source_path", ""),
        "warnings": list(threshold.get("warnings", [])) + (list(context.get("warnings", [])) if context else []),
    }

