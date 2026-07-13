from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .atomic_json import write_atomic_json
from .deepseek_enrich import load_question_bank
from .paper_components import normalize_component_code as _normalize_component_code
from .paper_components import packet_family_for_component as _packet_family_for_component
from .topic_packets import (
    DEFAULT_QUESTION_BANK_PATH,
    DEFAULT_TAXONOMY_PATH,
    load_packet_taxonomy,
    normalize_packet_topic,
    normalize_paper_family,
)


VISUAL_AUDIT_BATCH_SCHEMA = "exam_bank.visual_topic_audit.batch"
VISUAL_AUDIT_BATCH_SCHEMA_VERSION = 1
VISUAL_AUDIT_DECISION_VERSION = "visual_topic_audit_decision_v1"
VISUAL_AUDIT_DECISION_IMPORT_SCHEMA = "exam_bank.visual_topic_audit.decision_import"
VISUAL_AUDIT_PROMPT_VERSION = "visual_topic_audit_9709_v1"
DEFAULT_PACKET_AUDIT_REPORT = Path("reports/topic_packet_paper_topic_audit_20260706.json")
DEFAULT_TOPIC_PACKET_SUMMARY = Path("output/topic_packets/topic_packet_summary.json")
DEFAULT_BASE_OVERLAP_REVIEW = Path("data/review/topic_overlap_review_merged_p1_p3_p5_2026_07_03.json")
DEFAULT_MERGED_OVERLAP_REVIEW = Path("data/review/topic_overlap_review_merged_p1_p3_p4_p5_2026_07_06.json")
DEFAULT_VISUAL_AUDIT_OUT_DIR = Path("data/review/visual_topic_audit_2026_07_06")

DECISION_STATUSES = {
    "keep",
    "relabel_primary",
    "add_secondary_topic",
    "relabel_primary_add_secondary",
    "exclude_current_syllabus",
    "genuine_exception",
    "pending",
}
NON_IMPORTED_STATUSES = {"pending", "genuine_exception"}
IMAGE_EVIDENCE_TYPES = {"canonical_question_image", "canonical_mark_scheme_image"}


class VisualTopicAuditError(RuntimeError):
    pass


def add_visual_topic_audit_cli_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="visual_topic_audit_command", required=True)

    build = subparsers.add_parser("build-batch", help="Build an image-backed visual topic audit batch.")
    build.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK_PATH)
    build.add_argument("--packet-audit", type=Path, default=DEFAULT_PACKET_AUDIT_REPORT)
    build.add_argument("--packet-summary", type=Path, default=DEFAULT_TOPIC_PACKET_SUMMARY)
    build.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    build.add_argument("--artifact-root", type=Path, default=Path("output"))
    build.add_argument("--existing-overlap-review", type=Path, default=DEFAULT_BASE_OVERLAP_REVIEW)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_VISUAL_AUDIT_OUT_DIR)
    build.add_argument("--queue", choices=["missing", "ge3", "both"], default="both")
    build.add_argument("--paper-family", choices=["p1", "p3", "p4", "p5"], default=None)
    build.add_argument("--limit-papers", type=int, default=None)
    build.add_argument("--limit-questions", type=int, default=None)
    build.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="Run AI-assisted visual audit decisions for a batch.")
    run.add_argument("--batch", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-records", type=int, default=None)
    run.add_argument("--model", default="gpt-5-mini")
    run.add_argument("--provider", default="openai")
    run.add_argument("--dry-run", action="store_true")

    import_decisions = subparsers.add_parser(
        "import-decisions",
        help="Validate visual audit decisions and merge imported corrections into a topic-overlap sidecar.",
    )
    import_decisions.add_argument("--batch", type=Path, required=True)
    import_decisions.add_argument("--decisions", type=Path, required=True)
    import_decisions.add_argument("--base-overlap-review", type=Path, default=DEFAULT_BASE_OVERLAP_REVIEW)
    import_decisions.add_argument("--out", type=Path, default=DEFAULT_MERGED_OVERLAP_REVIEW)
    import_decisions.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    import_decisions.add_argument("--artifact-root", type=Path, default=Path("output"))
    import_decisions.add_argument("--dry-run", action="store_true")


def run_visual_topic_audit_from_args(args: argparse.Namespace) -> dict[str, Any]:
    command = args.visual_topic_audit_command
    if command == "build-batch":
        batch = build_visual_topic_audit_batch(
            question_bank_path=args.question_bank,
            packet_audit_path=args.packet_audit,
            packet_summary_path=args.packet_summary,
            taxonomy_path=args.taxonomy,
            artifact_root=args.artifact_root,
            existing_overlap_review_path=args.existing_overlap_review,
            queue=args.queue,
            paper_family=args.paper_family,
            limit_papers=args.limit_papers,
            limit_questions=args.limit_questions,
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            write_atomic_json(batch, args.out_dir / "visual_topic_audit_batch.json", sort_keys=True)
            (args.out_dir / "visual_topic_audit_batch.md").write_text(
                render_visual_topic_audit_batch_markdown(batch),
                encoding="utf-8",
            )
        return batch
    if command == "run":
        return run_visual_topic_audit_reviews(
            batch_path=args.batch,
            out_path=args.out,
            max_records=args.max_records,
            model=args.model,
            provider=args.provider,
            dry_run=bool(args.dry_run),
        )
    if command == "import-decisions":
        return import_visual_topic_audit_decisions(
            batch_path=args.batch,
            decisions_path=args.decisions,
            base_overlap_review_path=args.base_overlap_review,
            out_overlap_review_path=args.out,
            taxonomy_path=args.taxonomy,
            artifact_root=args.artifact_root,
            dry_run=bool(args.dry_run),
        )
    raise VisualTopicAuditError(f"Unhandled visual topic audit command: {command}")


def build_visual_topic_audit_batch(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK_PATH,
    packet_audit_path: str | Path = DEFAULT_PACKET_AUDIT_REPORT,
    packet_summary_path: str | Path = DEFAULT_TOPIC_PACKET_SUMMARY,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    artifact_root: str | Path = Path("output"),
    existing_overlap_review_path: str | Path | None = DEFAULT_BASE_OVERLAP_REVIEW,
    queue: str = "both",
    paper_family: str | None = None,
    limit_papers: int | None = None,
    limit_questions: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if queue not in {"missing", "ge3", "both"}:
        raise VisualTopicAuditError("queue must be one of: missing, ge3, both.")
    if limit_papers is not None and limit_papers < 0:
        raise VisualTopicAuditError("limit_papers must be zero or greater.")
    if limit_questions is not None and limit_questions < 0:
        raise VisualTopicAuditError("limit_questions must be zero or greater.")

    question_bank_path = Path(question_bank_path)
    packet_audit_path = Path(packet_audit_path)
    packet_summary_path = Path(packet_summary_path)
    taxonomy_path = Path(taxonomy_path)
    artifact_root = Path(artifact_root)
    taxonomy = load_packet_taxonomy(taxonomy_path)
    records = load_question_bank(question_bank_path)
    audit_payload = _read_json(packet_audit_path)
    packet_summary = _read_optional_json(packet_summary_path)
    existing_overlap = _existing_overlap_by_question(existing_overlap_review_path)

    selected_family = normalize_paper_family(paper_family) if paper_family else None
    paper_anomalies = _paper_anomalies_from_audit(audit_payload, queue=queue)
    if selected_family:
        paper_anomalies = [row for row in paper_anomalies if row["paper_family"] == selected_family]
    if limit_papers is not None:
        paper_anomalies = paper_anomalies[:limit_papers]

    coverage_by_paper = _coverage_by_paper(audit_payload, packet_summary)
    records_by_paper = _records_by_packet_paper(records, taxonomy=taxonomy)
    rows: list[dict[str, Any]] = []
    skipped_papers: list[dict[str, Any]] = []
    for anomaly in paper_anomalies:
        key = (anomaly["paper_family"], anomaly["paper"])
        paper_records = records_by_paper.get(key, [])
        if not paper_records:
            skipped_papers.append({**anomaly, "reason": "no_matching_question_records"})
            continue
        allowed_topics = _allowed_topics(taxonomy, anomaly["paper_family"])
        for record in paper_records:
            row = _visual_audit_row(
                record,
                anomaly=anomaly,
                coverage_counts=coverage_by_paper.get(key, {}),
                taxonomy=taxonomy,
                artifact_root=artifact_root,
                question_bank_root=question_bank_path.parent,
                allowed_topics=allowed_topics,
                existing_overlap=existing_overlap.get(str(record.get("question_id") or "")),
            )
            rows.append(row)
            if limit_questions is not None and len(rows) >= limit_questions:
                break
        if limit_questions is not None and len(rows) >= limit_questions:
            break

    batch_id = _batch_id(rows)
    for index, row in enumerate(rows, start=1):
        row["batch_id"] = batch_id
        row["batch_index"] = index

    return {
        "schema_name": VISUAL_AUDIT_BATCH_SCHEMA,
        "schema_version": VISUAL_AUDIT_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": _utc_now_iso(),
        "dry_run": dry_run,
        "queue": queue,
        "source_files": {
            "question_bank": str(question_bank_path),
            "packet_audit": str(packet_audit_path),
            "packet_summary": str(packet_summary_path),
            "taxonomy": str(taxonomy_path),
            "artifact_root": str(artifact_root),
            "existing_overlap_review": str(existing_overlap_review_path or ""),
        },
        "selection": {
            "paper_family": selected_family or "",
            "limit_papers": limit_papers,
            "limit_questions": limit_questions,
            "selected_paper_count": len(paper_anomalies),
            "selected_question_count": len(rows),
            "skipped_paper_count": len(skipped_papers),
            "anomaly_type_counts": dict(Counter(kind for row in paper_anomalies for kind in row["anomaly_types"])),
            "row_family_counts": dict(Counter(row["paper_family"] for row in rows)),
        },
        "review_policy": (
            "Use canonical question and mark-scheme images. Primary topic is the dominant mark-bearing topic; "
            "secondary topics count only when substantially assessed. Do not add missing topics from incidental method use. "
            "Treat exact count 3 as a watchlist signal, not an automatic defect."
        ),
        "decision_version": VISUAL_AUDIT_DECISION_VERSION,
        "prompt_version": VISUAL_AUDIT_PROMPT_VERSION,
        "decision_schema": visual_topic_audit_decision_schema(),
        "rows": rows,
        "skipped_papers": skipped_papers,
    }


def run_visual_topic_audit_reviews(
    *,
    batch_path: Path,
    out_path: Path,
    max_records: int | None = None,
    model: str = "gpt-5-mini",
    provider: str = "openai",
    dry_run: bool = False,
) -> dict[str, Any]:
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows") or [] if isinstance(row, dict)]
    done = _existing_decision_question_ids(out_path)
    pending = [row for row in rows if str(row.get("question_id") or "") not in done]
    if max_records is not None:
        if max_records < 0:
            raise VisualTopicAuditError("max_records must be zero or greater.")
        pending = pending[:max_records]
    manifest = {
        "schema_name": "exam_bank.visual_topic_audit.runner_manifest",
        "provider": provider,
        "model": model,
        "prompt_version": VISUAL_AUDIT_PROMPT_VERSION,
        "dry_run": dry_run,
        "batch_path": str(batch_path),
        "pending_count": len(pending),
        "resumed_count": len(done),
        "created_at": _utc_now_iso(),
    }
    if dry_run:
        return manifest
    if provider != "openai":
        raise VisualTopicAuditError("visual topic audit runner supports provider=openai only.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise VisualTopicAuditError("visual topic audit runner requires OPENAI_API_KEY.")

    from openai import OpenAI

    client = OpenAI()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            try:
                decision = _request_visual_topic_audit_review(client=client, model=model, row=row)
            except Exception as exc:
                decision = _pending_error_decision(row, model=model, error=exc)
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return manifest


def import_visual_topic_audit_decisions(
    *,
    batch_path: str | Path,
    decisions_path: str | Path,
    base_overlap_review_path: str | Path | None = DEFAULT_BASE_OVERLAP_REVIEW,
    out_overlap_review_path: str | Path = DEFAULT_MERGED_OVERLAP_REVIEW,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    artifact_root: str | Path = Path("output"),
    dry_run: bool = False,
) -> dict[str, Any]:
    batch_path = Path(batch_path)
    decisions_path = Path(decisions_path)
    taxonomy_path = Path(taxonomy_path)
    artifact_root = Path(artifact_root)
    batch = _read_json(batch_path)
    taxonomy = load_packet_taxonomy(taxonomy_path)
    batch_rows = {str(row.get("question_id") or ""): row for row in batch.get("rows") or [] if isinstance(row, dict)}
    decisions = _read_decisions(decisions_path)

    errors: list[str] = []
    warnings: list[str] = []
    pending: list[dict[str, Any]] = []
    genuine_exceptions: list[dict[str, Any]] = []
    imported_records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            errors.append(f"decision:{index}:not_object")
            continue
        question_id = str(decision.get("question_id") or "").strip()
        if question_id in seen:
            errors.append(f"decision:{index}:{question_id or 'missing'}:duplicate_decision")
            continue
        seen[question_id] = decision
        validation_errors = validate_visual_topic_audit_decision(
            decision,
            batch_rows=batch_rows,
            taxonomy=taxonomy,
            artifact_root=artifact_root,
        )
        status = _decision_status(decision)
        if status == "pending":
            pending.append(_decision_report_row(decision, validation_errors))
            warnings.extend(f"decision:{index}:{question_id}:{error}" for error in validation_errors)
            continue
        if validation_errors:
            errors.extend(f"decision:{index}:{question_id}:{error}" for error in validation_errors)
            continue
        if status == "genuine_exception":
            genuine_exceptions.append(_decision_report_row(decision, []))
            continue
        imported_records.append(_overlap_record_from_decision(decision, batch_rows[question_id], taxonomy))

    base_payload = _read_optional_json(base_overlap_review_path) if base_overlap_review_path else None
    base_records = _overlap_records(base_payload)
    merged_records, superseded_count = _merge_overlap_records(base_records, imported_records)
    output_payload = _visual_topic_audit_overlap_payload(
        base_payload=base_payload,
        merged_records=merged_records,
        base_overlap_review_path=base_overlap_review_path,
        decisions_path=decisions_path,
        batch_path=batch_path,
        imported_count=len(imported_records),
        pending_count=len(pending),
        genuine_exception_count=len(genuine_exceptions),
        superseded_count=superseded_count,
    )
    report = {
        "schema_name": VISUAL_AUDIT_DECISION_IMPORT_SCHEMA,
        "schema_version": 1,
        "ok": not errors,
        "dry_run": dry_run,
        "batch_path": str(batch_path),
        "decisions_path": str(decisions_path),
        "base_overlap_review_path": str(base_overlap_review_path or ""),
        "out_overlap_review_path": str(out_overlap_review_path),
        "decision_count": len(decisions),
        "imported_count": len(imported_records),
        "pending_count": len(pending),
        "genuine_exception_count": len(genuine_exceptions),
        "superseded_count": superseded_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "pending": pending,
        "genuine_exceptions": genuine_exceptions,
        "status_counts": dict(sorted(Counter(_decision_status(decision) for decision in decisions if isinstance(decision, dict)).items())),
    }
    if not dry_run and not errors:
        write_atomic_json(output_payload, out_overlap_review_path, sort_keys=True)
    return report


def validate_visual_topic_audit_decision(
    decision: dict[str, Any],
    *,
    batch_rows: dict[str, dict[str, Any]],
    taxonomy: dict[str, Any],
    artifact_root: Path,
) -> list[str]:
    errors: list[str] = []
    if decision.get("decision_version") != VISUAL_AUDIT_DECISION_VERSION:
        errors.append("invalid_decision_version")
    question_id = str(decision.get("question_id") or "").strip()
    if not question_id:
        errors.append("missing_question_id")
    row = batch_rows.get(question_id)
    if not row:
        errors.append("unknown_question_id")
        row = {}
    status = _decision_status(decision)
    if status not in DECISION_STATUSES:
        errors.append("invalid_status")
    if str(decision.get("prompt_version") or "").strip() not in {"", VISUAL_AUDIT_PROMPT_VERSION}:
        errors.append("invalid_prompt_version")
    if str(decision.get("paper") or row.get("paper") or "") != str(row.get("paper") or ""):
        errors.append("paper_does_not_match_batch")
    family = normalize_paper_family(decision.get("paper_family") or row.get("paper_family"))
    if family != str(row.get("paper_family") or family):
        errors.append("paper_family_does_not_match_batch")
    if status == "pending":
        return errors

    if not str(decision.get("rationale") or decision.get("explanation") or "").strip():
        errors.append("missing_rationale")
    if not str(decision.get("source") or "").strip():
        errors.append("missing_source")
    errors.extend(_validate_image_evidence_refs(decision.get("evidence_refs"), artifact_root=artifact_root))

    if status == "exclude_current_syllabus":
        return errors
    if status == "genuine_exception":
        return errors

    primary = str(decision.get("primary_topic") or decision.get("primary") or "").strip()
    if not primary:
        errors.append("missing_primary_topic")
    elif _resolve_topic(primary, family, taxonomy) is None:
        errors.append(f"unknown_primary_topic:{primary}")

    secondary_topics = _topic_list(decision.get("secondary_topics", decision.get("secondary", [])))
    coverage_topics = _topic_list(decision.get("coverage_topics", []))
    if not coverage_topics:
        errors.append("missing_coverage_topics")
    for topic in secondary_topics:
        if _resolve_topic(topic, family, taxonomy) is None:
            errors.append(f"unknown_secondary_topic:{topic}")
    for topic in coverage_topics:
        if _resolve_topic(topic, family, taxonomy) is None:
            errors.append(f"unknown_coverage_topic:{topic}")
    if primary and coverage_topics and _resolve_topic(primary, family, taxonomy) not in {
        _resolve_topic(topic, family, taxonomy) for topic in coverage_topics
    }:
        errors.append("coverage_topics_missing_primary")
    return errors


def visual_topic_audit_decision_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "decision_version",
            "question_id",
            "paper",
            "paper_family",
            "status",
            "primary_topic",
            "secondary_topics",
            "coverage_topics",
            "rationale",
            "evidence_refs",
            "source",
        ],
        "properties": {
            "decision_version": {"const": VISUAL_AUDIT_DECISION_VERSION},
            "question_id": {"type": "string"},
            "paper": {"type": "string"},
            "paper_family": {"enum": ["p1", "p3", "p4", "p5"]},
            "status": {"enum": sorted(DECISION_STATUSES)},
            "primary_topic": {"type": "string"},
            "secondary_topics": {"type": "array", "items": {"type": "string"}},
            "coverage_topics": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
            "evidence_refs": {"type": "array"},
            "source": {"type": "string"},
            "prompt_version": {"const": VISUAL_AUDIT_PROMPT_VERSION},
        },
    }


def render_visual_topic_audit_batch_markdown(batch: dict[str, Any]) -> str:
    selection = batch["selection"]
    lines = [
        "# Visual Topic Audit Batch",
        "",
        f"- Queue: `{batch['queue']}`",
        f"- Selected papers: `{selection['selected_paper_count']}`",
        f"- Selected question rows: `{selection['selected_question_count']}`",
        f"- Skipped papers: `{selection['skipped_paper_count']}`",
        f"- Families: `{selection['row_family_counts']}`",
        "",
        "## Rows",
        "",
    ]
    for row in batch.get("rows", []):
        missing = ", ".join(row.get("missing_topics") or [])
        high = ", ".join(f"{topic}={count}" for topic, count in (row.get("high_count_topics") or {}).items())
        warning = f"; warning `{row['identity_warning']}`" if row.get("identity_warning") else ""
        lines.append(
            f"- `{row['question_id']}` `{row['paper_family']}/{row['paper']}` q{row.get('question_number')}: "
            f"current `{row.get('current_topic')}`; missing `{missing}`; high-count `{high}`{warning}"
        )
    lines.append("")
    return "\n".join(lines)


def _paper_anomalies_from_audit(payload: dict[str, Any], *, queue: str) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if queue in {"missing", "both"}:
        for row in payload.get("papers_lacking_at_least_one_topic") or []:
            if not isinstance(row, dict):
                continue
            key = (normalize_paper_family(row.get("paper_family")), str(row.get("paper") or ""))
            entry = merged.setdefault(
                key,
                {
                    "paper_family": key[0],
                    "paper": key[1],
                    "anomaly_types": [],
                    "missing_topics": [],
                    "high_count_topics": {},
                },
            )
            entry["anomaly_types"].append("missing_topic")
            entry["missing_topics"] = _dedupe([*entry["missing_topics"], *_topic_list(row.get("missing_topics"))])
    if queue in {"ge3", "both"}:
        for row in payload.get("papers_with_topic_count_ge_3") or []:
            if not isinstance(row, dict):
                continue
            key = (normalize_paper_family(row.get("paper_family")), str(row.get("paper") or ""))
            entry = merged.setdefault(
                key,
                {
                    "paper_family": key[0],
                    "paper": key[1],
                    "anomaly_types": [],
                    "missing_topics": [],
                    "high_count_topics": {},
                },
            )
            entry["anomaly_types"].append("topic_count_ge_3")
            for topic, count in (row.get("topics_count_ge_3") or {}).items():
                entry["high_count_topics"][str(topic)] = int(count)
    return sorted(
        merged.values(),
        key=lambda item: (item["paper_family"], item["paper"], ",".join(item["anomaly_types"])),
    )


def _coverage_by_paper(audit_payload: dict[str, Any], packet_summary: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, int]]:
    rows = audit_payload.get("paper_topic_counts")
    if not rows and packet_summary:
        rows = ((packet_summary.get("paper_topic_coverage_audit") or {}).get("papers") or [])
    coverage: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = (normalize_paper_family(row.get("paper_family")), str(row.get("paper") or ""))
        coverage[key] = {str(topic): int(count or 0) for topic, count in (row.get("topic_coverage_counts") or {}).items()}
    return coverage


def _records_by_packet_paper(
    records: Sequence[dict[str, Any]],
    *,
    taxonomy: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_paper: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        paper = str(record.get("paper") or "").strip()
        family = _packet_family_for_record(record, taxonomy)
        if not paper or family not in {"p1", "p3", "p4", "p5"}:
            continue
        by_paper.setdefault((family, paper), []).append(record)
    for rows in by_paper.values():
        rows.sort(key=lambda record: _question_sort_key(record))
    return by_paper


def _visual_audit_row(
    record: dict[str, Any],
    *,
    anomaly: dict[str, Any],
    coverage_counts: dict[str, int],
    taxonomy: dict[str, Any],
    artifact_root: Path,
    question_bank_root: Path,
    allowed_topics: list[dict[str, str]],
    existing_overlap: dict[str, Any] | None,
) -> dict[str, Any]:
    normalization = normalize_packet_topic(
        component_code=_source_paper_code(record),
        current_family=record.get("paper_family"),
        raw_topic=record.get("topic"),
        taxonomy=taxonomy,
    )
    source_component_family = _packet_family_for_component(normalization.source_component)
    target_family = anomaly["paper_family"]
    question_paths = _question_image_paths(record)
    mark_scheme_paths = _mark_scheme_image_paths(record)
    q_path = _first_existing_path(question_paths, artifact_root, question_bank_root)
    ms_path = _first_existing_path(mark_scheme_paths, artifact_root, question_bank_root)
    identity_warning = ""
    if source_component_family and source_component_family != target_family:
        identity_warning = "source_component_family_mismatch"
    return {
        "question_id": str(record.get("question_id") or ""),
        "paper": str(record.get("paper") or anomaly["paper"]),
        "paper_family": target_family,
        "raw_question_bank_family": normalize_paper_family(record.get("paper_family")),
        "source_component": normalization.source_component,
        "source_component_family": source_component_family,
        "identity_warning": identity_warning,
        "question_number": str(record.get("question_number") or ""),
        "anomaly_types": list(anomaly["anomaly_types"]),
        "missing_topics": list(anomaly["missing_topics"]),
        "high_count_topics": dict(anomaly["high_count_topics"]),
        "paper_topic_coverage_counts": dict(sorted(coverage_counts.items())),
        "raw_topic": str(record.get("topic") or ""),
        "current_topic": normalization.expected_topic if normalization.resolved else str(record.get("topic") or ""),
        "current_topic_family": normalization.expected_family if normalization.resolved else normalization.current_family,
        "topic_normalization_status": normalization.status,
        "topic_normalization_reason": normalization.reason,
        "question_text": str(record.get("question_text") or ""),
        "ocr_text": str(record.get("ocr_text") or ""),
        "mark_scheme_text": str(record.get("mark_scheme_text") or ""),
        "canonical_question_image_path": str(q_path) if q_path else "",
        "canonical_mark_scheme_image_path": str(ms_path) if ms_path else "",
        "image_evidence_available": bool(q_path and ms_path),
        "evidence_refs": _evidence_refs(q_path, ms_path),
        "allowed_topics": allowed_topics,
        "allowed_packet_topics": allowed_topics,
        "existing_overlap_review": existing_overlap or {},
        "recommended_review_focus": _recommended_review_focus(anomaly),
    }


def _recommended_review_focus(anomaly: dict[str, Any]) -> list[str]:
    focus: list[str] = []
    for topic in anomaly.get("missing_topics") or []:
        focus.append(f"Find a substantial mark-bearing {topic} placement or record a genuine exception.")
    for topic, count in (anomaly.get("high_count_topics") or {}).items():
        focus.append(f"Verify whether {topic} has {count} substantial placements; exact count 3 is only a watchlist signal.")
    return focus


def _request_visual_topic_audit_review(*, client: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                visual_topic_audit_prompt()
                + "\n\nDecision schema:\n"
                + json.dumps(visual_topic_audit_decision_schema(), indent=2, sort_keys=True)
                + "\n\nAudit row:\n"
                + json.dumps(row, indent=2, sort_keys=True)
            ),
        }
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
    parsed.setdefault("decision_version", VISUAL_AUDIT_DECISION_VERSION)
    parsed.setdefault("prompt_version", VISUAL_AUDIT_PROMPT_VERSION)
    parsed.setdefault("source", f"ai_assisted_visual_topic_audit:{model}")
    return parsed


def visual_topic_audit_prompt() -> str:
    return (
        "You are auditing CAIE 9709 topic packet routing from canonical question and mark-scheme images. "
        "Use the images as evidence, not extracted text alone. Choose exactly one status. "
        "Use keep only when the current primary topic and coverage topics are correct. "
        "Use relabel or add_secondary only when visual evidence clearly shows substantial mark-bearing assessment. "
        "Use exclude_current_syllabus only for material outside the current packet taxonomy. "
        "Use genuine_exception when a missing topic is genuinely absent after visual inspection. "
        "Use pending for ambiguous, mixed, or poor-image cases. Do not invent topic IDs."
    )


def _pending_error_decision(row: dict[str, Any], *, model: str, error: Exception) -> dict[str, Any]:
    return {
        "decision_version": VISUAL_AUDIT_DECISION_VERSION,
        "question_id": row.get("question_id"),
        "paper": row.get("paper"),
        "paper_family": row.get("paper_family"),
        "status": "pending",
        "primary_topic": "",
        "secondary_topics": [],
        "coverage_topics": [],
        "rationale": f"Provider call failed: {type(error).__name__}: {error}",
        "evidence_refs": [],
        "source": f"ai_assisted_visual_topic_audit:{model}",
        "prompt_version": VISUAL_AUDIT_PROMPT_VERSION,
    }


def _overlap_record_from_decision(
    decision: dict[str, Any],
    row: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    status = _decision_status(decision)
    family = normalize_paper_family(decision.get("paper_family") or row.get("paper_family"))
    if status == "exclude_current_syllabus":
        primary = "exclude_current_syllabus"
        secondary: list[str] = []
        coverage: list[str] = []
    else:
        primary = _resolve_topic(str(decision.get("primary_topic") or decision.get("primary") or ""), family, taxonomy) or ""
        secondary = [
            _resolve_topic(topic, family, taxonomy) or topic
            for topic in _topic_list(decision.get("secondary_topics", decision.get("secondary", [])))
        ]
        coverage = [
            _resolve_topic(topic, family, taxonomy) or topic
            for topic in _topic_list(decision.get("coverage_topics", []))
        ]
        if not coverage:
            coverage = _dedupe([primary, *secondary])
    return {
        "paper": str(decision.get("paper") or row.get("paper") or ""),
        "paper_family": family,
        "question_id": str(decision.get("question_id") or ""),
        "question_number": str(row.get("question_number") or ""),
        "current_topic": str(row.get("current_topic") or ""),
        "primary_topic": primary,
        "primary": primary,
        "secondary_topics": secondary,
        "secondary": secondary,
        "coverage_topics": coverage,
        "status": status,
        "rationale": str(decision.get("rationale") or decision.get("explanation") or ""),
        "source": str(decision.get("source") or ""),
        "evidence_refs": decision.get("evidence_refs") or [],
        "visual_audit": {
            "decision_version": VISUAL_AUDIT_DECISION_VERSION,
            "prompt_version": str(decision.get("prompt_version") or VISUAL_AUDIT_PROMPT_VERSION),
            "source_batch_id": row.get("batch_id"),
        },
    }


def _visual_topic_audit_overlap_payload(
    *,
    base_payload: dict[str, Any] | None,
    merged_records: list[dict[str, Any]],
    base_overlap_review_path: str | Path | None,
    decisions_path: Path,
    batch_path: Path,
    imported_count: int,
    pending_count: int,
    genuine_exception_count: int,
    superseded_count: int,
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status") or "") for record in merged_records)
    papers = {str(record.get("paper") or "") for record in merged_records if record.get("paper")}
    families = sorted({normalize_paper_family(record.get("paper_family")) for record in merged_records if record.get("paper_family")})
    source_sidecars = []
    if isinstance(base_payload, dict):
        source_sidecars.extend(str(path) for path in base_payload.get("source_sidecars") or [])
    if base_overlap_review_path:
        source_sidecars.append(str(base_overlap_review_path))
    source_sidecars.append(str(decisions_path))
    return {
        "schema_name": "exam_bank.topic_overlap_review_merged",
        "schema_version": "1.0",
        "generated_at": _utc_now_iso(),
        "paper_families": families,
        "source_sidecars": _dedupe(source_sidecars),
        "policy": (
            "Primary topic is the dominant assessed, mark-bearing topic; secondary topics count only for substantial "
            "mark-bearing overlap; coverage-only placement should not duplicate PDFs; visual topic audit decisions "
            "are imported only when canonical question and mark-scheme image evidence is present."
        ),
        "summary": {
            "records_reviewed": len(merged_records),
            "papers_reviewed": len(papers),
            "status_counts": dict(sorted(status_counts.items())),
            "visual_topic_audit_imported": imported_count,
            "visual_topic_audit_pending": pending_count,
            "visual_topic_audit_genuine_exceptions": genuine_exception_count,
            "visual_topic_audit_superseded_existing": superseded_count,
        },
        "visual_topic_audit": {
            "batch_path": str(batch_path),
            "decisions_path": str(decisions_path),
            "decision_version": VISUAL_AUDIT_DECISION_VERSION,
            "prompt_version": VISUAL_AUDIT_PROMPT_VERSION,
        },
        "records": merged_records,
    }


def _merge_overlap_records(
    base_records: list[dict[str, Any]],
    imported_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    merged: dict[str, dict[str, Any]] = {}
    order: dict[str, int] = {}
    for index, record in enumerate(base_records):
        question_id = str(record.get("question_id") or "")
        if not question_id:
            continue
        merged[question_id] = record
        order.setdefault(question_id, index)
    superseded = 0
    for record in imported_records:
        question_id = str(record.get("question_id") or "")
        if question_id in merged and merged[question_id] != record:
            superseded += 1
        merged[question_id] = record
        order.setdefault(question_id, len(order))
    rows = sorted(
        merged.values(),
        key=lambda record: (
            str(record.get("paper_family") or ""),
            str(record.get("paper") or ""),
            _question_number_key(record.get("question_number")),
            str(record.get("question_id") or ""),
            order.get(str(record.get("question_id") or ""), 0),
        ),
    )
    return rows, superseded


def _overlap_records(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    records = payload.get("records")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _existing_overlap_by_question(path: str | Path | None) -> dict[str, dict[str, Any]]:
    payload = _read_optional_json(path) if path else None
    return {
        str(record.get("question_id")): record
        for record in _overlap_records(payload)
        if str(record.get("question_id") or "")
    }


def _decision_status(decision: dict[str, Any]) -> str:
    return str(decision.get("status") or decision.get("decision_action") or "").strip()


def _decision_report_row(decision: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "question_id": str(decision.get("question_id") or ""),
        "paper": str(decision.get("paper") or ""),
        "paper_family": normalize_paper_family(decision.get("paper_family")),
        "status": _decision_status(decision),
        "errors": errors,
    }


def _validate_image_evidence_refs(value: Any, *, artifact_root: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["missing_evidence_refs"]
    errors: list[str] = []
    found_types: set[str] = set()
    for ref in value:
        if not isinstance(ref, dict):
            errors.append("evidence_ref_not_object")
            continue
        ref_type = str(ref.get("type") or "")
        if ref_type not in IMAGE_EVIDENCE_TYPES and ref_type != "syllabus_reference":
            errors.append(f"unsupported_evidence_ref_type:{ref_type}")
        if ref_type in IMAGE_EVIDENCE_TYPES:
            found_types.add(ref_type)
            raw_path = str(ref.get("path") or "")
            path = Path(raw_path)
            if not path.is_absolute():
                path = artifact_root / raw_path
            if not path.is_file():
                errors.append(f"evidence_path_not_found:{raw_path}")
    missing = IMAGE_EVIDENCE_TYPES - found_types
    for ref_type in sorted(missing):
        errors.append(f"missing_evidence_ref_type:{ref_type}")
    return errors


def _allowed_topics(taxonomy: dict[str, Any], family: str) -> list[dict[str, str]]:
    rows = []
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


def _resolve_topic(value: str, family: str, taxonomy: dict[str, Any]) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    canonical = taxonomy["canonical_topic_to_topic"].get(text)
    if canonical and canonical[0] == family:
        return canonical[1]
    slug = _slug(text)
    topic_id = taxonomy["topic_aliases"].get((family, slug), slug)
    return topic_id if (family, topic_id) in taxonomy["topics"] else None


def _packet_family_for_record(record: dict[str, Any], taxonomy: dict[str, Any]) -> str:
    normalization = normalize_packet_topic(
        component_code=_source_paper_code(record),
        current_family=record.get("paper_family"),
        raw_topic=record.get("topic"),
        taxonomy=taxonomy,
    )
    return normalization.expected_family if normalization.resolved else normalization.current_family


def _source_paper_code(record: dict[str, Any]) -> str:
    for value in (
        record.get("source_paper_code"),
        (record.get("notes") or {}).get("source_paper_code") if isinstance(record.get("notes"), dict) else "",
    ):
        code = _normalize_component_code(value)
        if code:
            return code
    for value in (record.get("paper"), record.get("question_id")):
        match = re.match(r"(\d+)", str(value or ""))
        if match:
            return _normalize_component_code(match.group(1))
    return ""


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


def _first_existing_path(paths: Sequence[str], artifact_root: Path, fallback_root: Path) -> Path | None:
    for raw in paths:
        path = Path(raw)
        candidates = [path] if path.is_absolute() else [artifact_root / path, fallback_root / path, path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _evidence_refs(q_path: Path | None, ms_path: Path | None) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if q_path:
        refs.append({"type": "canonical_question_image", "path": str(q_path)})
    if ms_path:
        refs.append({"type": "canonical_mark_scheme_image", "path": str(ms_path)})
    return refs


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VisualTopicAuditError(f"Expected JSON object: {path}")
    return payload


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return _read_json(candidate)


def _read_decisions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            if "decisions" in payload or "records" in payload:
                rows = payload.get("decisions", payload.get("records", []))
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
            return [payload]
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _existing_decision_question_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("question_id") or "").strip()
        for row in _read_decisions(path)
        if isinstance(row, dict) and str(row.get("question_id") or "").strip()
    }


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _topic_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _question_sort_key(record: dict[str, Any]) -> tuple[Any, str]:
    return (_question_number_key(record.get("question_number")), str(record.get("question_id") or ""))


def _question_number_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    match = re.search(r"\d+", text)
    return (int(match.group(0)) if match else 9999, text)


def _batch_id(rows: Sequence[dict[str, Any]]) -> str:
    import hashlib

    digest = json.dumps([row.get("question_id") for row in rows], sort_keys=True, separators=(",", ":"))
    return f"visual_topic_audit_{hashlib.sha256(digest.encode('utf-8')).hexdigest()[:12]}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
