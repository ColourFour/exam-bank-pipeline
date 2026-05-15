from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exam_bank.advisory_evidence.common import load_json, records_from_question_bank, rel_path, utc_now_iso
from exam_bank.advisory_evidence.constants import (
    CONFIDENCE_LABELS,
    CONTEXT_LABELS,
    EVIDENCE_LEVELS,
    EXAMINER_DIFFICULTY_PATH,
    EXAMINER_DIFFICULTY_SCHEMA,
    EXAMINER_LINKS_PATH,
    EXAMINER_LINKS_SCHEMA,
    FINAL_SIDECAR_PATH,
    FINAL_SIDECAR_SCHEMA,
    GRADE_THRESHOLD_CONTEXT_PATH,
    GRADE_THRESHOLD_CONTEXT_SCHEMA,
    GRADE_THRESHOLD_LINKS_PATH,
    GRADE_THRESHOLD_LINKS_SCHEMA,
    ITEM_SIGNALS,
    LINK_STATUSES,
    PROTECTED_PATHS,
    TOPIC_EVIDENCE_PATH,
    TOPIC_EVIDENCE_SCHEMA,
    VALIDATION_SCHEMA,
)
from exam_bank.atomic_json import write_atomic_json
from exam_bank.config import AppConfig


def validate_advisory_evidence(
    *,
    advisory_root: str | Path = "output/advisory_evidence",
    question_bank_path: str | Path = "output/json/question_bank.json",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(advisory_root)
    question_ids = _question_ids(question_bank_path)
    allowed_topics = _allowed_topic_ids()
    errors: list[str] = []
    warnings: list[str] = []

    _validate_links(root / EXAMINER_LINKS_PATH.relative_to("output/advisory_evidence"), EXAMINER_LINKS_SCHEMA, question_ids, errors, warnings)
    _validate_links(root / GRADE_THRESHOLD_LINKS_PATH.relative_to("output/advisory_evidence"), GRADE_THRESHOLD_LINKS_SCHEMA, question_ids, errors, warnings)
    _validate_topic_evidence(root / TOPIC_EVIDENCE_PATH.relative_to("output/advisory_evidence"), question_ids, allowed_topics, errors, warnings)
    _validate_examiner_difficulty(root / EXAMINER_DIFFICULTY_PATH.relative_to("output/advisory_evidence"), question_ids, errors, warnings)
    _validate_threshold_context(root / GRADE_THRESHOLD_CONTEXT_PATH.relative_to("output/advisory_evidence"), errors, warnings)
    _validate_final_sidecar(root / FINAL_SIDECAR_PATH.relative_to("output/advisory_evidence"), question_ids, allowed_topics, errors, warnings)

    report = {
        "schema": VALIDATION_SCHEMA,
        "generated_at": utc_now_iso(),
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "protected_paths": [rel_path(path) for path in PROTECTED_PATHS],
        "notes": [
            "Protected path mutation must also be checked with git status/diff in implementation passes.",
            "Advisory validation checks schema, links, enums, topic IDs, and threshold-only item difficulty.",
        ],
    }
    if output_path:
        write_atomic_json(report, output_path)
    return report


def _question_ids(question_bank_path: str | Path) -> set[str]:
    records = records_from_question_bank(load_json(question_bank_path, default={"questions": []}))
    return {str(record.get("question_id")) for record in records if record.get("question_id")}


def _allowed_topic_ids() -> set[str]:
    allowed: set[str] = set()
    for topics in AppConfig().paper_family_taxonomy.values():
        allowed.update(topics.keys())
    return allowed


def _load_stage(path: Path, expected_schema: str, errors: list[str], warnings: list[str], *, required: bool = False) -> dict[str, Any] | None:
    if not path.exists():
        (errors if required else warnings).append(f"missing_stage_output:{rel_path(path)}")
        return None
    payload = load_json(path)
    if payload.get("schema") != expected_schema:
        errors.append(f"schema_mismatch:{rel_path(path)}:expected={expected_schema}:actual={payload.get('schema')}")
    return payload


def _validate_links(path: Path, expected_schema: str, question_ids: set[str], errors: list[str], warnings: list[str]) -> None:
    payload = _load_stage(path, expected_schema, errors, warnings)
    if not payload:
        return
    for index, link in enumerate(payload.get("links", [])):
        status = link.get("match_status")
        if status not in LINK_STATUSES:
            errors.append(f"invalid_link_status:{rel_path(path)}:{index}:{status}")
        for question_id in link.get("candidate_question_ids", []):
            if question_id not in question_ids:
                errors.append(f"orphan_link_candidate:{rel_path(path)}:{index}:{question_id}")
    _warn_duplicate_linked_candidates(path, payload.get("links", []), warnings)


def _validate_topic_evidence(path: Path, question_ids: set[str], allowed_topics: set[str], errors: list[str], warnings: list[str]) -> None:
    payload = _load_stage(path, TOPIC_EVIDENCE_SCHEMA, errors, warnings)
    if not payload:
        return
    for index, record in enumerate(payload.get("records", [])):
        question_id = record.get("question_id")
        if question_id not in question_ids:
            errors.append(f"orphan_topic_evidence:{index}:{question_id}")
        evidence = record.get("topic_evidence") if isinstance(record.get("topic_evidence"), dict) else {}
        confidence = evidence.get("confidence")
        if confidence not in CONFIDENCE_LABELS:
            errors.append(f"invalid_topic_confidence:{index}:{confidence}")
        for topic_id in evidence.get("predicted_topic_ids", []):
            if topic_id not in allowed_topics:
                errors.append(f"invalid_topic_id:{index}:{topic_id}")
    _warn_duplicate_records("topic_evidence", payload.get("records", []), warnings)


def _validate_examiner_difficulty(path: Path, question_ids: set[str], errors: list[str], warnings: list[str]) -> None:
    payload = _load_stage(path, EXAMINER_DIFFICULTY_SCHEMA, errors, warnings)
    if not payload:
        return
    for index, record in enumerate(payload.get("records", [])):
        question_id = record.get("question_id")
        if question_id not in question_ids:
            errors.append(f"orphan_examiner_difficulty:{index}:{question_id}")
        evidence = record.get("examiner_report_difficulty") if isinstance(record.get("examiner_report_difficulty"), dict) else {}
        if evidence.get("item_signal") not in ITEM_SIGNALS:
            errors.append(f"invalid_item_signal:{index}:{evidence.get('item_signal')}")
        if evidence.get("confidence") not in CONFIDENCE_LABELS:
            errors.append(f"invalid_difficulty_confidence:{index}:{evidence.get('confidence')}")
        if evidence.get("evidence_level") not in EVIDENCE_LEVELS:
            errors.append(f"invalid_evidence_level:{index}:{evidence.get('evidence_level')}")
        sources = set(evidence.get("evidence_sources", []))
        if evidence.get("item_signal") in {"easy", "moderate", "hard", "mixed"} and sources == {"grade_threshold_context"}:
            errors.append(f"threshold_only_item_difficulty:{index}:{question_id}")
    _warn_duplicate_records("examiner_difficulty", payload.get("records", []), warnings)


def _validate_threshold_context(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = _load_stage(path, GRADE_THRESHOLD_CONTEXT_SCHEMA, errors, warnings)
    if not payload:
        return
    for index, context in enumerate(payload.get("contexts", [])):
        if context.get("component_context_label") not in CONTEXT_LABELS:
            errors.append(f"invalid_context_label:{index}:{context.get('component_context_label')}")
        if context.get("confidence") not in CONFIDENCE_LABELS:
            errors.append(f"invalid_context_confidence:{index}:{context.get('confidence')}")
        if "item_signal" in context:
            errors.append(f"grade_threshold_context_contains_item_signal:{index}")


def _validate_final_sidecar(path: Path, question_ids: set[str], allowed_topics: set[str], errors: list[str], warnings: list[str]) -> None:
    payload = _load_stage(path, FINAL_SIDECAR_SCHEMA, errors, warnings)
    if not payload:
        return
    for index, record in enumerate(payload.get("records", [])):
        question_id = record.get("question_id")
        if question_id not in question_ids:
            errors.append(f"orphan_final_sidecar_record:{index}:{question_id}")
        if any(field in record for field in ["question_text", "mark_scheme_text", "canonical_question_artifact"]):
            errors.append(f"canonical_replacement_field_in_final_sidecar:{index}:{question_id}")
        topic = record.get("topic_evidence") if isinstance(record.get("topic_evidence"), dict) else {}
        for topic_id in topic.get("predicted_topic_ids", []):
            if topic_id not in allowed_topics:
                errors.append(f"invalid_final_topic_id:{index}:{topic_id}")


def _warn_duplicate_linked_candidates(path: Path, links: list[dict[str, Any]], warnings: list[str]) -> None:
    occurrences: dict[str, list[str]] = defaultdict(list)
    for link in links:
        if link.get("match_status") != "linked":
            continue
        identity = str(link.get("normalized_key") or link.get("source_path") or "unknown_source")
        for question_id in link.get("candidate_question_ids", []):
            occurrences[str(question_id)].append(identity)
    for question_id, identities in sorted(occurrences.items()):
        if len(identities) > 1:
            warnings.append(
                f"duplicate_linked_candidate_id:{rel_path(path)}:{question_id}:count={len(identities)}:sources={','.join(sorted(set(identities)))}"
            )


def _warn_duplicate_records(kind: str, records: list[dict[str, Any]], warnings: list[str]) -> None:
    by_question = Counter(str(record.get("question_id") or "") for record in records if record.get("question_id"))
    by_identity = Counter(
        (
            str(record.get("question_id") or ""),
            str(record.get("normalized_key") or record.get("source_path") or ""),
        )
        for record in records
        if record.get("question_id")
    )
    for question_id, count in sorted(by_question.items()):
        if count > 1:
            warnings.append(f"duplicate_{kind}_question_id:{question_id}:count={count}")
    for (question_id, identity), count in sorted(by_identity.items()):
        if identity and count > 1:
            warnings.append(f"duplicate_{kind}_source_identity:{question_id}:{identity}:count={count}")
