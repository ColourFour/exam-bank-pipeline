from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI

from .atomic_json import write_atomic_json
from .deepseek_enrich import load_question_bank
from .topic_packets import (
    DEFAULT_CANONICAL_TAXONOMY_ROOT,
    DEFAULT_QUESTION_BANK_PATH,
    DEFAULT_TAXONOMY_PATH,
    REVIEWED_DECISION_ACTIONS,
    TopicPacketError,
    load_packet_taxonomy,
    load_topic_bank_reviewed_decisions,
    normalize_paper_family,
)
from .topic_routing_audit import route_records_from_payload


TOPIC_REVIEW_BATCH_SCHEMA = "exam_bank.topic_review.auto_review_batch"
TOPIC_REVIEW_BATCH_SCHEMA_VERSION = 1
TOPIC_REVIEW_DECISION_VERSION = "topic_review_auto_decision_v1"
TOPIC_REVIEW_IMPORT_SCHEMA = "exam_bank.topic_review.auto_reviewed_decisions"
TOPIC_REVIEW_IMPORT_SCHEMA_VERSION = 1
TOPIC_BANK_REVIEWED_DECISIONS_SCHEMA = "exam_bank.topic_bank_reviewed_decisions"
TOPIC_BANK_REVIEWED_DECISIONS_SCHEMA_VERSION = 1
TOPIC_REVIEW_SOURCE = "automated_agentic_review"
TOPIC_REVIEW_PROMPT_VERSION = "topic_review_9709_2026_2027_v1"
DEFAULT_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_TOPIC_ROUTING_PATH = Path("data/topic_routing/question_bank.topic_routing.v1.json")
DEFAULT_REVIEWED_DECISIONS_PATH = Path("data/review/topic_bank_reviewed_decisions.v1.json")
DEFAULT_OUTPUT_DIR = Path("data/review/topic_review_batches")
CURRENT_SYLLABUS_REFERENCE = "Cambridge International AS & A Level Mathematics 9709 syllabus for 2026 and 2027"
CURRENT_SYLLABUS_URL = "https://www.cambridgeinternational.org/Images/697427-2026-2027-syllabus.pdf"
CURRENT_SYLLABUS_STATUSES = {
    "current_relevant",
    "legacy_but_relevant",
    "outdated_not_relevant",
    "not_in_2026_syllabus",
    "ambiguous",
}
AUTO_APPROVED_SYLLABUS_STATUSES = {"current_relevant", "legacy_but_relevant"}
AUTO_EXCLUDE_SYLLABUS_STATUSES = {"outdated_not_relevant", "not_in_2026_syllabus"}
MODEL_ACTIONS = REVIEWED_DECISION_ACTIONS | {"pending"}


class TopicReviewLoopError(RuntimeError):
    pass


def add_topic_review_batch_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK_PATH)
    parser.add_argument("--topic-routing", type=Path, default=DEFAULT_TOPIC_ROUTING_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--canonical-taxonomy-root", type=Path, default=DEFAULT_CANONICAL_TAXONOMY_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--reviewed-decisions", type=Path, default=DEFAULT_REVIEWED_DECISIONS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-family", choices=["p1", "p3", "p4", "p5"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-already-reviewed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def add_topic_review_run_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--provider", default="openai")


def add_topic_review_import_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")


def add_topic_review_merge_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reviewed-file", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")


def build_topic_review_batch_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_topic_review_batch(
        question_bank_path=args.question_bank,
        topic_routing_path=args.topic_routing,
        taxonomy_path=args.taxonomy,
        canonical_taxonomy_root=args.canonical_taxonomy_root,
        artifact_root=args.artifact_root,
        reviewed_decisions_path=args.reviewed_decisions,
        out_dir=args.out_dir,
        paper_family=args.paper_family,
        limit=args.limit,
        include_already_reviewed=bool(args.include_already_reviewed),
        dry_run=bool(args.dry_run),
    )


def run_topic_reviews_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return run_topic_reviews(
        batch_path=args.batch,
        out_path=args.out,
        max_records=args.max_records,
        dry_run=bool(args.dry_run),
        model=args.model,
        provider=args.provider,
    )


def import_topic_review_decisions_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return import_topic_review_decisions(
        decisions_path=args.decisions,
        batch_path=args.batch,
        out_review_file=args.out,
        artifact_root=args.artifact_root,
        taxonomy_path=args.taxonomy,
        confidence_threshold=args.confidence_threshold,
        dry_run=bool(args.dry_run),
    )


def merge_topic_review_decisions_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return merge_topic_review_decision_files(
        reviewed_files=args.reviewed_file,
        out_review_file=args.out,
        dry_run=bool(args.dry_run),
    )


def build_topic_review_batch(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    topic_routing_path: str | Path = DEFAULT_TOPIC_ROUTING_PATH,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    canonical_taxonomy_root: str | Path = DEFAULT_CANONICAL_TAXONOMY_ROOT,
    artifact_root: str | Path = Path("output"),
    reviewed_decisions_path: str | Path | None = DEFAULT_REVIEWED_DECISIONS_PATH,
    out_dir: str | Path = DEFAULT_OUTPUT_DIR,
    paper_family: str | None = None,
    limit: int | None = None,
    include_already_reviewed: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if limit is not None and limit < 0:
        raise TopicReviewLoopError("--limit must be zero or greater.")
    question_bank_path = Path(question_bank_path)
    topic_routing_path = Path(topic_routing_path)
    taxonomy_path = Path(taxonomy_path)
    canonical_taxonomy_root = Path(canonical_taxonomy_root)
    artifact_root = Path(artifact_root)
    out_dir = Path(out_dir)

    records = load_question_bank(question_bank_path)
    questions = {
        str(record.get("question_id") or "").strip(): record
        for record in records
        if str(record.get("question_id") or "").strip()
    }
    taxonomy = load_packet_taxonomy(taxonomy_path)
    reviewed = (
        load_topic_bank_reviewed_decisions(reviewed_decisions_path, records=records, taxonomy=taxonomy)
        if reviewed_decisions_path and Path(reviewed_decisions_path).exists()
        else {}
    )
    routing_payload = _read_json(topic_routing_path)
    routes = [row for row in route_records_from_payload(routing_payload) if row.get("review_required") is True]
    if paper_family:
        family = normalize_paper_family(paper_family)
        routes = [row for row in routes if normalize_paper_family(row.get("paper_family")) == family]

    skipped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for route in routes:
        question_id = str(route.get("question_id") or "").strip()
        question = questions.get(question_id)
        if not question:
            skipped.append({"question_id": question_id, "reason": "unknown_question_id"})
            continue
        if not include_already_reviewed and question_id in reviewed:
            skipped.append({"question_id": question_id, "reason": "already_reviewed"})
            continue
        row = _batch_row(
            route=route,
            question=question,
            taxonomy=taxonomy,
            canonical_taxonomy_root=canonical_taxonomy_root,
            artifact_root=artifact_root,
            question_bank_root=question_bank_path.parent,
        )
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break

    batch_id = _batch_id(rows, topic_routing_path=topic_routing_path)
    for index, row in enumerate(rows, start=1):
        row["batch_id"] = batch_id
        row["batch_index"] = index
    manifest = {
        "schema_name": TOPIC_REVIEW_BATCH_SCHEMA,
        "schema_version": TOPIC_REVIEW_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": _utc_now_iso(),
        "dry_run": dry_run,
        "source_files": {
            "question_bank": str(question_bank_path),
            "topic_routing": str(topic_routing_path),
            "taxonomy": str(taxonomy_path),
            "canonical_taxonomy_root": str(canonical_taxonomy_root),
            "artifact_root": str(artifact_root),
            "reviewed_decisions": str(reviewed_decisions_path or ""),
        },
        "syllabus_baseline": {
            "syllabus_code": "9709",
            "reference": CURRENT_SYLLABUS_REFERENCE,
            "url": CURRENT_SYLLABUS_URL,
            "valid_for": ["2026", "2027"],
        },
        "review_required_input_count": len(routes),
        "selected_count": len(rows),
        "skipped_count": len(skipped),
        "skipped_reason_counts": dict(Counter(row["reason"] for row in skipped)),
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "prompt_version": TOPIC_REVIEW_PROMPT_VERSION,
    }
    payload = {
        "schema_name": TOPIC_REVIEW_BATCH_SCHEMA,
        "schema_version": TOPIC_REVIEW_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "manifest": manifest,
        "reviewer_prompt": topic_review_prompt(),
        "decision_schema": topic_review_decision_schema(),
        "rows": rows,
        "skipped_rows": skipped,
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_atomic_json(payload, out_dir / "topic_review_batch.json", sort_keys=True)
        write_atomic_json(manifest, out_dir / "topic_review_manifest.json", sort_keys=True)
    return payload


def run_topic_reviews(
    *,
    batch_path: Path,
    out_path: Path,
    max_records: int | None = None,
    dry_run: bool = False,
    model: str = "gpt-5-mini",
    provider: str = "openai",
) -> dict[str, Any]:
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows", []) if isinstance(row, dict)]
    done = _existing_decision_question_ids(out_path)
    pending = [row for row in rows if str(row.get("question_id") or "") not in done]
    if max_records is not None:
        if max_records < 0:
            raise TopicReviewLoopError("--max-records must be zero or greater.")
        pending = pending[:max_records]
    manifest = {
        "schema_name": "exam_bank.topic_review.runner_manifest",
        "provider": provider,
        "model": model,
        "prompt_version": TOPIC_REVIEW_PROMPT_VERSION,
        "dry_run": dry_run,
        "pending_count": len(pending),
        "resumed_count": len(done),
        "created_at": _utc_now_iso(),
    }
    if dry_run:
        return manifest
    if provider != "openai":
        raise TopicReviewLoopError("topic review runner supports provider=openai only")
    if not os.environ.get("OPENAI_API_KEY"):
        raise TopicReviewLoopError("topic review runner requires OPENAI_API_KEY")

    client = OpenAI()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            try:
                decision = _request_openai_topic_review(client=client, model=model, row=row)
            except Exception as exc:
                decision = _blocked_error_decision(row, provider=provider, model=model, error=exc)
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return manifest


def import_topic_review_decisions(
    *,
    decisions_path: Path,
    batch_path: Path,
    out_review_file: Path,
    artifact_root: Path = Path("output"),
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    dry_run: bool = False,
) -> dict[str, Any]:
    batch = _read_json(batch_path)
    batch_rows = {str(row.get("question_id") or ""): row for row in batch.get("rows", []) if isinstance(row, dict)}
    taxonomy = load_packet_taxonomy(taxonomy_path)
    decisions = _read_jsonl(decisions_path)
    accepted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"decision:{index}:not_object")
            continue
        question_id = str(decision.get("question_id") or "").strip()
        row = batch_rows.get(question_id)
        if not row:
            errors.append(f"decision:{index}:{question_id or 'missing'}:unknown_question_id")
            continue
        validation_errors = validate_topic_review_decision(
            decision,
            batch_row=row,
            taxonomy=taxonomy,
            artifact_root=artifact_root,
            confidence_threshold=confidence_threshold,
        )
        if _decision_action(decision) == "pending":
            pending.append(_pending_report_row(decision, validation_errors))
            warnings.extend(f"decision:{index}:{question_id}:{error}" for error in validation_errors)
            continue
        if validation_errors:
            errors.extend(f"decision:{index}:{question_id}:{error}" for error in validation_errors)
            continue
        normalized = _topic_bank_decision_from_auto_decision(decision, row, taxonomy)
        previous = seen.get(question_id)
        if previous and previous != normalized:
            errors.append(f"decision:{index}:{question_id}:duplicate_conflicting_decision")
            continue
        if not previous:
            seen[question_id] = normalized
            accepted.append(normalized)

    syllabus_counts = Counter(record.get("current_syllabus_status", "") for record in accepted)
    payload = {
        "schema_name": TOPIC_BANK_REVIEWED_DECISIONS_SCHEMA,
        "schema_version": TOPIC_BANK_REVIEWED_DECISIONS_SCHEMA_VERSION,
        "artifact_kind": "automated_agentic_topic_bank_decision_input",
        "description": "Automated image-and-syllabus reviewed correction layer for topic packets. Does not modify question_bank.json.",
        "generated_at": _utc_now_iso(),
        "review_source": TOPIC_REVIEW_SOURCE,
        "source_batch_id": batch.get("batch_id"),
        "source_batch_path": str(batch_path),
        "source_decisions_path": str(decisions_path),
        "confidence_threshold": confidence_threshold,
        "syllabus_baseline": (batch.get("manifest") or {}).get("syllabus_baseline", {}),
        "record_count": len(accepted),
        "current_syllabus_status_counts": dict(sorted(syllabus_counts.items())),
        "records": accepted,
    }
    report = {
        "ok": not errors,
        "dry_run": dry_run,
        "decision_count": len(decisions),
        "accepted_count": len(accepted),
        "pending_count": len(pending),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "pending": pending,
        "out_review_file": str(out_review_file),
        "current_syllabus_status_counts": payload["current_syllabus_status_counts"],
    }
    if not dry_run and not errors:
        write_atomic_json(payload, out_review_file, sort_keys=True)
    return report


def merge_topic_review_decision_files(
    *,
    reviewed_files: list[Path],
    out_review_file: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for path in reviewed_files:
        if not path.exists():
            errors.append(f"reviewed_file_missing:{path}")
            continue
        payload = _read_json(path)
        raw_records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(raw_records, list):
            errors.append(f"reviewed_file_missing_records:{path}")
            continue
        for record in raw_records:
            if not isinstance(record, dict):
                errors.append(f"reviewed_file_record_not_object:{path}")
                continue
            question_id = str(record.get("question_id") or "").strip()
            if not question_id:
                errors.append(f"reviewed_file_record_missing_question_id:{path}")
                continue
            existing = seen.get(question_id)
            if existing and existing != record:
                errors.append(f"duplicate_conflicting_decision:{question_id}")
                continue
            if not existing:
                seen[question_id] = record
                records.append(record)
    payload = {
        "schema_name": TOPIC_BANK_REVIEWED_DECISIONS_SCHEMA,
        "schema_version": TOPIC_BANK_REVIEWED_DECISIONS_SCHEMA_VERSION,
        "artifact_kind": "merged_topic_bank_reviewed_decision_input",
        "description": "Merged manual and automated topic-bank reviewed decisions for packet generation.",
        "generated_at": _utc_now_iso(),
        "source_review_files": [str(path) for path in reviewed_files],
        "record_count": len(records),
        "records": records,
    }
    report = {
        "ok": not errors,
        "dry_run": dry_run,
        "reviewed_file_count": len(reviewed_files),
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "out_review_file": str(out_review_file),
    }
    if not dry_run and not errors:
        write_atomic_json(payload, out_review_file, sort_keys=True)
    return report


def validate_topic_review_decision(
    decision: dict[str, Any],
    *,
    batch_row: dict[str, Any],
    taxonomy: dict[str, Any],
    artifact_root: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[str]:
    errors: list[str] = []
    for field in topic_review_decision_schema()["required"]:
        if field not in decision:
            errors.append(f"missing_required_field:{field}")
    if decision.get("decision_version") != TOPIC_REVIEW_DECISION_VERSION:
        errors.append("invalid_decision_version")
    if decision.get("review_source") != TOPIC_REVIEW_SOURCE:
        errors.append("invalid_review_source")
    if str(decision.get("question_id") or "") != str(batch_row.get("question_id") or ""):
        errors.append("question_id_does_not_match_batch")
    action = _decision_action(decision)
    if action not in MODEL_ACTIONS:
        errors.append("invalid_action")
    syllabus_status = str(decision.get("current_syllabus_status") or "").strip()
    if syllabus_status not in CURRENT_SYLLABUS_STATUSES:
        errors.append("invalid_current_syllabus_status")
    confidence = _float(decision.get("confidence"))
    if confidence < confidence_threshold:
        errors.append("confidence_below_threshold")
    if _strings(decision.get("risk_flags")):
        errors.append("blocking_risk_flags_present")
    if not str(decision.get("explanation") or "").strip():
        errors.append("missing_explanation")
    if decision.get("reviewer_model") in (None, ""):
        errors.append("missing_reviewer_model")
    if decision.get("prompt_version") != TOPIC_REVIEW_PROMPT_VERSION:
        errors.append("invalid_prompt_version")
    if action == "pending":
        errors.append("decision_pending")

    release_override = decision.get("release_override")
    if not isinstance(release_override, bool):
        errors.append("release_override_not_boolean")
    if action in {"keep", "relabel"} and syllabus_status not in AUTO_APPROVED_SYLLABUS_STATUSES:
        errors.append("syllabus_status_not_auto_approvable")
    if action == "exclude" and syllabus_status not in AUTO_EXCLUDE_SYLLABUS_STATUSES:
        errors.append("exclude_requires_outdated_syllabus_status")

    family = str(batch_row.get("paper_family") or "")
    reviewed_topic = _reviewed_topic_for_decision(decision, batch_row)
    if action in {"keep", "relabel"}:
        if not reviewed_topic:
            errors.append("missing_reviewed_topic")
        elif _resolve_reviewed_topic(reviewed_topic, family, taxonomy) is None:
            errors.append("reviewed_topic_not_allowed_for_family")
    if action == "relabel" and not str(decision.get("reviewed_topic") or "").strip():
        errors.append("relabel_missing_reviewed_topic")

    evidence_errors = _validate_evidence_refs(decision.get("evidence_refs"), artifact_root=artifact_root)
    errors.extend(evidence_errors)
    if release_override is True:
        ref_types = _evidence_ref_types(decision.get("evidence_refs"))
        if "canonical_question_image" not in ref_types:
            errors.append("release_override_missing_question_image_evidence")
        if "canonical_mark_scheme_image" not in ref_types:
            errors.append("release_override_missing_mark_scheme_image_evidence")
    return errors


def topic_review_prompt() -> str:
    return (
        "You are reviewing CAIE 9709 Mathematics topic routing for printable topic packets. "
        "Use the 2026-2027 syllabus as the current baseline. Inspect the canonical question image and mark-scheme image. "
        "Choose exactly one decision_action: keep, relabel, exclude, or pending. "
        "Use keep only when the current topic is correct for the normalized paper family. "
        "Use relabel only when another supplied allowed topic is clearly correct. "
        "Use exclude when the question is outdated or not relevant to the 2026-2027 syllabus. "
        "Use pending for ambiguity, poor image evidence, mixed-topic uncertainty, or any curriculum doubt. "
        "Set release_override true only when both canonical images were inspected and are usable. "
        "Return JSON matching the supplied schema. Do not invent topic IDs."
    )


def topic_review_decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "decision_version",
            "review_source",
            "question_id",
            "decision_action",
            "reviewed_topic",
            "current_syllabus_status",
            "release_override",
            "confidence",
            "evidence_refs",
            "risk_flags",
            "explanation",
            "reviewer_model",
            "prompt_version",
        ],
        "properties": {
            "decision_version": {"const": TOPIC_REVIEW_DECISION_VERSION},
            "review_source": {"const": TOPIC_REVIEW_SOURCE},
            "question_id": {"type": "string"},
            "decision_action": {"enum": sorted(MODEL_ACTIONS)},
            "reviewed_topic": {"type": "string"},
            "reviewed_subtopic": {"type": "string"},
            "reviewed_skill": {"type": "string"},
            "current_syllabus_status": {"enum": sorted(CURRENT_SYLLABUS_STATUSES)},
            "release_override": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_refs": {"type": "array"},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"},
            "reviewer_model": {"type": "string"},
            "prompt_version": {"const": TOPIC_REVIEW_PROMPT_VERSION},
        },
    }


def _batch_row(
    *,
    route: dict[str, Any],
    question: dict[str, Any],
    taxonomy: dict[str, Any],
    canonical_taxonomy_root: Path,
    artifact_root: Path,
    question_bank_root: Path,
) -> dict[str, Any]:
    family = normalize_paper_family(question.get("paper_family") or route.get("paper_family"))
    route_family = normalize_paper_family(route.get("paper_family"))
    current_packet_topic = str(route.get("packet_topic_id") or question.get("topic") or "").strip()
    q_path = _first_existing_path(_question_image_paths(question), artifact_root, question_bank_root)
    ms_path = _first_existing_path(_mark_scheme_image_paths(question), artifact_root, question_bank_root)
    allowed_topics = _allowed_packet_topics(taxonomy, family)
    return {
        "question_id": str(question.get("question_id") or route.get("question_id") or ""),
        "paper": question.get("paper") or route.get("paper"),
        "paper_family": family,
        "route_paper_family": route_family,
        "question_number": question.get("question_number") or route.get("question_number"),
        "current_packet_topic": current_packet_topic,
        "current_canonical_topic_id": route.get("primary_topic_id"),
        "topic_distribution": route.get("topic_distribution") or [],
        "review_reasons": route.get("review_reasons") or [],
        "route_confidence": route.get("confidence"),
        "routing_source": route.get("routing_source"),
        "question_text": question.get("question_text") or "",
        "ocr_text": question.get("ocr_text") or "",
        "mark_scheme_text": question.get("mark_scheme_text") or "",
        "canonical_question_image_path": str(q_path) if q_path else "",
        "canonical_mark_scheme_image_path": str(ms_path) if ms_path else "",
        "image_evidence_available": bool(q_path and ms_path),
        "allowed_topics": allowed_topics,
        "syllabus_baseline": {
            "reference": CURRENT_SYLLABUS_REFERENCE,
            "url": CURRENT_SYLLABUS_URL,
            "topic_map_root": str(canonical_taxonomy_root),
        },
        "selection_eligibility": {
            "eligible_for_auto_approval": bool(q_path and ms_path and current_packet_topic and allowed_topics),
            "reasons": _eligibility_reasons(q_path=q_path, ms_path=ms_path, current_packet_topic=current_packet_topic),
        },
    }


def _topic_bank_decision_from_auto_decision(
    decision: dict[str, Any],
    row: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    action = _decision_action(decision)
    reviewed_topic = _reviewed_topic_for_decision(decision, row)
    if action in {"keep", "relabel"}:
        resolved_topic = _resolve_reviewed_topic(reviewed_topic, str(row.get("paper_family") or ""), taxonomy)
        reviewed_topic = resolved_topic or reviewed_topic
    reviewed_at = str(decision.get("reviewed_at") or _utc_now_iso())
    return {
        "question_id": row["question_id"],
        "action": action,
        "reviewed_topic": reviewed_topic if action != "exclude" else "",
        "reviewed_subtopic": str(decision.get("reviewed_subtopic") or ""),
        "reviewed_skill": str(decision.get("reviewed_skill") or ""),
        "reason": str(decision.get("explanation") or ""),
        "reviewer": f"{TOPIC_REVIEW_SOURCE}:{decision.get('reviewer_model')}",
        "reviewed_at": reviewed_at,
        "source": TOPIC_REVIEW_SOURCE,
        "current_syllabus_status": decision["current_syllabus_status"],
        "release_override": bool(decision.get("release_override")),
        "confidence": float(decision.get("confidence") or 0),
        "evidence_refs": decision.get("evidence_refs") or [],
        "risk_flags": _strings(decision.get("risk_flags")),
        "reviewer_model": str(decision.get("reviewer_model") or ""),
        "prompt_version": str(decision.get("prompt_version") or ""),
        "source_batch_id": row.get("batch_id"),
        "source_batch_index": row.get("batch_index"),
    }


def _request_openai_topic_review(*, client: OpenAI, model: str, row: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": topic_review_prompt() + "\n\nBatch row:\n" + json.dumps(row, indent=2, sort_keys=True)}
    ]
    for path_field in ("canonical_question_image_path", "canonical_mark_scheme_image_path"):
        path = str(row.get(path_field) or "")
        if path:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    parsed.setdefault("reviewer_model", model)
    parsed.setdefault("prompt_version", TOPIC_REVIEW_PROMPT_VERSION)
    return parsed


def _blocked_error_decision(row: dict[str, Any], *, provider: str, model: str, error: Exception) -> dict[str, Any]:
    return {
        "decision_version": TOPIC_REVIEW_DECISION_VERSION,
        "review_source": TOPIC_REVIEW_SOURCE,
        "question_id": row.get("question_id"),
        "decision_action": "pending",
        "reviewed_topic": "",
        "current_syllabus_status": "ambiguous",
        "release_override": False,
        "confidence": 0.0,
        "evidence_refs": [],
        "risk_flags": ["provider_error"],
        "explanation": f"Provider call failed: {type(error).__name__}: {error}",
        "reviewer_model": model,
        "reviewer_provider": provider,
        "prompt_version": TOPIC_REVIEW_PROMPT_VERSION,
    }


def _validate_evidence_refs(value: Any, *, artifact_root: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["missing_evidence_refs"]
    errors: list[str] = []
    for ref in value:
        if not isinstance(ref, dict):
            errors.append("evidence_ref_not_object")
            continue
        if str(ref.get("type") or "") not in {"canonical_question_image", "canonical_mark_scheme_image", "syllabus_reference"}:
            errors.append("unsupported_evidence_ref_type")
        if ref.get("type") in {"canonical_question_image", "canonical_mark_scheme_image"}:
            path = Path(str(ref.get("path") or ""))
            if not path.is_absolute():
                path = artifact_root / path
            if not path.is_file():
                errors.append(f"evidence_path_not_found:{ref.get('path')}")
    return errors


def _evidence_ref_types(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(ref.get("type") or "") for ref in value if isinstance(ref, dict)}


def _allowed_packet_topics(taxonomy: dict[str, Any], family: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (topic_family, topic_id), topic in sorted(taxonomy["topics"].items()):
        if topic_family != family:
            continue
        rows.append(
            {
                "topic_id": str(topic_id),
                "topic_label": str(topic.get("topic_label") or topic_id),
                "canonical_topic_id": str(topic.get("canonical_topic_id") or ""),
            }
        )
    return rows


def _resolve_reviewed_topic(value: str, family: str, taxonomy: dict[str, Any]) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    slug = _slug(text)
    alias = taxonomy["topic_aliases"].get((family, slug))
    if alias:
        return alias
    if (family, text) in taxonomy["topics"]:
        return text
    for (topic_family, topic_id), topic in taxonomy["topics"].items():
        if topic_family == family and str(topic.get("canonical_topic_id") or "") == text:
            return topic_id
    return None


def _reviewed_topic_for_decision(decision: dict[str, Any], row: dict[str, Any]) -> str:
    if _decision_action(decision) == "keep" and not str(decision.get("reviewed_topic") or "").strip():
        return str(row.get("current_packet_topic") or "")
    return str(decision.get("reviewed_topic") or "").strip()


def _decision_action(decision: dict[str, Any]) -> str:
    return str(decision.get("decision_action") or decision.get("action") or "").strip().lower()


def _question_image_paths(record: dict[str, Any]) -> list[str]:
    paths = record.get("question_image_paths")
    if isinstance(paths, list):
        return [str(path) for path in paths if str(path)]
    path = str(record.get("question_image_path") or record.get("canonical_question_artifact") or "")
    return [path] if path else []


def _mark_scheme_image_paths(record: dict[str, Any]) -> list[str]:
    paths = record.get("mark_scheme_image_paths")
    if isinstance(paths, list):
        return [str(path) for path in paths if str(path)]
    path = str(record.get("mark_scheme_image_path") or record.get("canonical_mark_scheme_artifact") or "")
    return [path] if path else []


def _first_existing_path(paths: Sequence[str], artifact_root: Path, question_bank_root: Path) -> Path | None:
    for raw in paths:
        path = Path(raw)
        candidates = [path] if path.is_absolute() else [artifact_root / path, question_bank_root / path, path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _eligibility_reasons(*, q_path: Path | None, ms_path: Path | None, current_packet_topic: str) -> list[str]:
    reasons: list[str] = []
    if not q_path:
        reasons.append("missing_canonical_question_image")
    if not ms_path:
        reasons.append("missing_canonical_mark_scheme_image")
    if not current_packet_topic:
        reasons.append("missing_current_packet_topic")
    return reasons


def _existing_decision_question_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("question_id") or "").strip()
        for row in _read_jsonl(path)
        if isinstance(row, dict) and str(row.get("question_id") or "").strip()
    }


def _pending_report_row(decision: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "question_id": str(decision.get("question_id") or ""),
        "decision_action": _decision_action(decision),
        "current_syllabus_status": decision.get("current_syllabus_status"),
        "confidence": decision.get("confidence"),
        "errors": errors,
    }


def _batch_id(rows: Sequence[dict[str, Any]], *, topic_routing_path: Path) -> str:
    digest = json.dumps(
        [row.get("question_id") for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return f"topic_review_{hashlib.sha256((str(topic_routing_path) + digest).encode('utf-8')).hexdigest()[:12]}"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TopicReviewLoopError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in candidate.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
