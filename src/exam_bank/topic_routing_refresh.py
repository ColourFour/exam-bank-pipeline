from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .atomic_json import write_atomic_json
from .deepseek_enrich import load_question_bank
from .topic_packets import load_packet_taxonomy, normalize_packet_topic
from .topic_routing import (
    TOPIC_ROUTING_PROMPT_VERSION,
    TOPIC_ROUTING_SCHEMA_NAME,
    TOPIC_ROUTING_SCHEMA_VERSION,
    build_topic_routing_question_packet,
    build_topic_routing_sidecar,
    deterministic_review_reasons,
    hash_topic_routing_evidence_packet,
    load_topic_routing_sidecar_records,
)
from .topic_routing import _course_metadata_for_record
from .topic_routing_artifact import (
    DEFAULT_RELEASE_MANIFEST_PATH,
    build_topic_routing_release_manifest,
    file_sha256,
)


DEFAULT_QUESTION_BANK = Path("output/json/question_bank.json")
DEFAULT_TAXONOMY = Path("exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")
DEFAULT_CANONICAL_TAXONOMY_ROOT = Path("exam_bank_taxonomy/canonical")
DEFAULT_ROUTING = Path("data/topic_routing/question_bank.topic_routing.v1.json")
DEFAULT_REPORT_PREFIX = Path("reports/topic_routing_refresh_2026_06_27")
SCHEMA_NAME = "exam_bank.topic_routing_refresh"
SCHEMA_VERSION = 1
REFRESH_SOURCE = "deterministic_topic_packet_normalization"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh durable topic-routing sidecar from packet taxonomy normalization.")
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--canonical-taxonomy-root", type=Path, default=DEFAULT_CANONICAL_TAXONOMY_ROOT)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PREFIX)
    parser.add_argument("--write", action="store_true", help="Write refreshed sidecar and reports. Without this, dry-run only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = refresh_topic_routing(
        question_bank_path=args.question_bank,
        taxonomy_path=args.taxonomy,
        canonical_taxonomy_root=args.canonical_taxonomy_root,
        routing_path=args.routing,
        release_manifest_path=args.release_manifest,
        report_prefix=args.report,
        write=bool(args.write),
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def refresh_topic_routing(
    *,
    question_bank_path: str | Path = DEFAULT_QUESTION_BANK,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY,
    canonical_taxonomy_root: str | Path = DEFAULT_CANONICAL_TAXONOMY_ROOT,
    routing_path: str | Path = DEFAULT_ROUTING,
    release_manifest_path: str | Path | None = None,
    report_prefix: str | Path = DEFAULT_REPORT_PREFIX,
    write: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    taxonomy_path = Path(taxonomy_path)
    canonical_taxonomy_root = Path(canonical_taxonomy_root)
    routing_path = Path(routing_path)
    release_manifest_path = Path(release_manifest_path) if release_manifest_path is not None else _default_release_manifest_path(
        question_bank_path,
        routing_path,
    )
    report_prefix = Path(report_prefix)
    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    records = load_question_bank(question_bank_path)
    taxonomy = load_packet_taxonomy(taxonomy_path)
    existing_records = load_topic_routing_sidecar_records(routing_path) if routing_path.exists() else {}
    existing_aliases = build_existing_route_aliases(records, existing_records)
    claimed_existing_ids = {existing_id for existing_id, _route in existing_aliases.values()}

    refreshed: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    preserved_existing = 0
    preserved_reviewed = 0
    preserved_via_alias = 0
    hash_refreshed = 0
    new_entries = 0
    new_review_required = 0
    updated_entries = 0
    replaced_conflicts = 0
    identity_guard_rejections = 0

    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            exclusions.append({"reason": "missing_question_id", "paper": record.get("paper")})
            continue
        route_context = route_context_for_record(record, taxonomy, canonical_taxonomy_root)
        if route_context["normalization_status"] != "resolved":
            unresolved.append(_unresolved_report_row(record, route_context))
            route = build_deterministic_route_record(
                record,
                route_context,
                generated_at=generated_at,
                extra_review_reasons=[
                    f"topic_normalization_{route_context['normalization_status']}"
                ],
            )
            refreshed[question_id] = route
            new_entries += 1
            new_review_required += 1
            continue

        existing_key = question_id
        existing = None if question_id in claimed_existing_ids else existing_records.get(question_id)
        if existing is None and question_id in existing_aliases:
            existing_key, existing = existing_aliases[question_id]
        if existing is not None and not existing_route_identity_matches(existing, record):
            existing = None
            existing_key = question_id
            identity_guard_rejections += 1
        conflict = existing_route_conflict(existing, route_context)
        if conflict:
            conflicts.append(conflict_report_row(question_id, record, existing, route_context, conflict))
            extra_review_reasons = (
                []
                if conflict.get("type") == "missing_primary_topic"
                else ["existing_route_conflicts_with_normalized_topic"]
            )
            route = build_deterministic_route_record(
                record,
                route_context,
                generated_at=generated_at,
                extra_review_reasons=extra_review_reasons,
            )
            route["previous_route_conflict"] = conflict
            refreshed[question_id] = route
            replaced_conflicts += 1
            if route["review_required"] is True:
                new_review_required += 1
            continue

        if existing is not None:
            route = preserve_existing_route_record(
                existing,
                record,
                route_context,
                generated_at=generated_at,
                previous_question_id=existing_key if existing_key != question_id else "",
            )
            refreshed[question_id] = route
            preserved_existing += 1
            if existing.get("review_required") is True:
                preserved_reviewed += 1
            if existing_key != question_id:
                preserved_via_alias += 1
            if existing.get("evidence_packet_hash") != route_context["evidence_packet_hash"]:
                hash_refreshed += 1
            updated_entries += 1
            continue

        route = build_deterministic_route_record(record, route_context, generated_at=generated_at)
        refreshed[question_id] = route
        new_entries += 1
        if route["review_required"] is True:
            new_review_required += 1

    sidecar_metadata = {
        "input_path": str(question_bank_path),
        "taxonomy_path": str(taxonomy_path),
        "canonical_taxonomy_root": str(canonical_taxonomy_root),
        "output_path": str(routing_path),
        "refresh_source": REFRESH_SOURCE,
        "existing_record_count": len(existing_records),
        "question_bank_record_count": len(records),
        "sidecar_entry_count": len(refreshed),
        "explicit_exclusion_count": len(exclusions),
        "preserved_existing_entries": preserved_existing,
        "preserved_reviewed_entries": preserved_reviewed,
        "preserved_via_alias_entries": preserved_via_alias,
        "existing_hash_refreshed_count": hash_refreshed,
        "new_entries": new_entries,
        "new_review_required_entries": new_review_required,
        "conflicts_count": len(conflicts),
        "replaced_conflicts": replaced_conflicts,
        "identity_guard_rejections": identity_guard_rejections,
        "unresolved_count": len(unresolved),
    }
    sidecar = build_topic_routing_sidecar(
        refreshed,
        taxonomy_path=taxonomy_path,
        taxonomy_version_value=_packet_taxonomy_version(taxonomy),
        model="deterministic",
        prompt_version=TOPIC_ROUTING_PROMPT_VERSION,
        generated_at=generated_at,
        metadata=sidecar_metadata,
    )
    sidecar["metadata"]["run_summary"] = _routing_summary(sidecar)

    report = build_refresh_report(
        records=records,
        sidecar=sidecar,
        exclusions=exclusions,
        conflicts=conflicts,
        unresolved=unresolved,
        existing_record_count=len(existing_records),
        generated_at=generated_at,
        question_bank_path=question_bank_path,
        taxonomy_path=taxonomy_path,
        routing_path=routing_path,
        write=write,
    )

    if write:
        write_topic_routing_refresh_outputs(
            sidecar=sidecar,
            routing_path=routing_path,
            question_bank_path=question_bank_path,
            release_manifest_path=release_manifest_path,
            report=report,
            report_prefix=report_prefix,
        )
    return report


def route_context_for_record(
    record: dict[str, Any],
    taxonomy: dict[str, Any],
    canonical_taxonomy_root: Path,
) -> dict[str, Any]:
    component = source_component(record)
    normalization = normalize_packet_topic(
        component_code=component,
        current_family=record.get("paper_family"),
        raw_topic=record.get("topic"),
        taxonomy=taxonomy,
        year=record.get("year") or record.get("canonical_year_folder"),
        session=record.get("session"),
        paper=record.get("paper") or record.get("question_id"),
    )
    component_family = normalization.component_family
    packet_family = normalization.expected_family
    packet_topic_id = normalization.expected_topic
    normalization_status = normalization.status
    normalization_reason = normalization.reason
    # Packet normalization may identify a topic that is unique to another
    # syllabus paper.  That is useful anomaly evidence, but a release sidecar
    # must never use it to move a question into another course.  Keep the
    # source component authoritative and fail the route closed for review.
    if normalization.resolved and component_family and packet_family != component_family:
        packet_family = component_family
        packet_topic_id = ""
        normalization_status = "component_family_topic_mismatch"
        normalization_reason = "topic_resolves_only_outside_source_component_family"
    topic_ref = taxonomy["topics"].get((packet_family, packet_topic_id))
    canonical_topic_id = str((topic_ref or {}).get("canonical_topic_id") or "").strip()
    packet_record = normalized_record_for_packet(record, packet_family or normalization.current_family)
    packet = build_refresh_evidence_packet(packet_record, canonical_taxonomy_root)
    deterministic_reasons = deterministic_review_reasons(packet)
    return {
        "source_component": component,
        "identity_year": normalization.identity_year,
        "current_family": normalization.current_family,
        "packet_family": packet_family,
        "packet_topic_id": packet_topic_id,
        "canonical_topic_id": canonical_topic_id,
        "normalization_status": normalization_status,
        "normalization_reason": normalization_reason,
        "raw_topic": normalization.raw_topic,
        "packet": packet,
        "evidence_packet_hash": hash_topic_routing_evidence_packet(packet),
        "deterministic_review_reasons": deterministic_reasons,
    }


def build_refresh_evidence_packet(record: dict[str, Any], canonical_taxonomy_root: Path) -> dict[str, Any]:
    try:
        return build_topic_routing_question_packet(record, taxonomy_root=canonical_taxonomy_root).packet
    except Exception:
        return {
            "question_id": str(record.get("question_id") or ""),
            "paper_family": str(record.get("paper_family") or "").lower(),
            "paper": record.get("paper"),
            "question_number": record.get("question_number"),
            "visual_required": bool(record.get("visual_required")),
            "evidence": {},
            "available_evidence_fields": [],
            "evidence_sources": {},
            "question_text_source": "none",
            "ocr_text_source": "none",
            "allowed_topics": [],
        }


def build_deterministic_route_record(
    record: dict[str, Any],
    context: dict[str, Any],
    *,
    generated_at: str,
    extra_review_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    review_reasons = refresh_review_reasons(record, context, extra_review_reasons=extra_review_reasons)
    confidence = refresh_confidence(record, review_reasons)
    canonical_topic_id = context["canonical_topic_id"]
    route = {
        "primary_topic_id": canonical_topic_id,
        "topic_distribution": [{"topic_id": canonical_topic_id, "fit_percent": 100}] if canonical_topic_id else [],
        "confidence": confidence,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "evidence_used": list(context["packet"].get("available_evidence_fields") or []),
        "llm_provider": None,
        "llm_model": None,
        "llm_prompt_version": TOPIC_ROUTING_PROMPT_VERSION,
        "llm_run_timestamp": None,
        "routing_source": REFRESH_SOURCE,
        "routing_refreshed_at": generated_at,
        "paper": record.get("paper"),
        "paper_family": context["packet_family"],
        "question_number": record.get("question_number"),
        "evidence_packet_hash": context["evidence_packet_hash"],
        "source_record_hash": source_record_hash(record),
        "packet_family": context["packet_family"],
        "packet_topic_id": context["packet_topic_id"],
        "raw_topic": context["raw_topic"],
        "source_component": context["source_component"],
        "source_session_code": source_session_code_for_record(record),
        "normalization_status": context["normalization_status"],
        "normalization_reason": context["normalization_reason"],
        **route_course_metadata(record, context),
    }
    if not canonical_topic_id:
        route["review_required"] = True
        route["review_reasons"] = sorted(set(route["review_reasons"] + ["missing_canonical_topic_id"]))
    return route


def preserve_existing_route_record(
    existing: dict[str, Any],
    record: dict[str, Any],
    context: dict[str, Any],
    *,
    generated_at: str,
    previous_question_id: str = "",
) -> dict[str, Any]:
    route = dict(existing)
    route.update(
        {
            "paper": record.get("paper"),
            "paper_family": context["packet_family"],
            "question_number": record.get("question_number"),
            "evidence_packet_hash": context["evidence_packet_hash"],
            "source_record_hash": source_record_hash(record),
            "packet_family": context["packet_family"],
            "packet_topic_id": context["packet_topic_id"],
            "raw_topic": context["raw_topic"],
            "source_component": context["source_component"],
            "source_session_code": source_session_code_for_record(record),
            "normalization_status": context["normalization_status"],
            "normalization_reason": context["normalization_reason"],
            "routing_preserved_from_existing": True,
            "routing_refreshed_at": generated_at,
            **route_course_metadata(record, context),
        }
    )
    if previous_question_id:
        route["previous_question_id"] = previous_question_id
    if not route.get("evidence_used"):
        route["evidence_used"] = list(context["packet"].get("available_evidence_fields") or [])
    return route


def route_course_metadata(record: dict[str, Any], context: dict[str, Any]) -> dict[str, str | None]:
    return _course_metadata_for_record(
        {
            "paper_family": context.get("packet_family"),
            "paper": record.get("paper"),
            "source_paper_code": context.get("source_component"),
            "year": context.get("identity_year") or record.get("year") or record.get("canonical_year_folder"),
            "session": record.get("session"),
        }
    )


def existing_route_conflict(existing: dict[str, Any] | None, context: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(existing, dict):
        return None
    existing_topic = existing.get("primary_topic_id")
    expected_topic = context["canonical_topic_id"]
    if expected_topic and not existing_topic:
        return {
            "type": "missing_primary_topic",
            "existing_primary_topic_id": existing_topic,
            "normalized_primary_topic_id": expected_topic,
            "existing_review_required": existing.get("review_required"),
            "existing_confidence": existing.get("confidence"),
        }
    if existing_topic and expected_topic and existing_topic != expected_topic:
        return {
            "type": "primary_topic_conflict",
            "existing_primary_topic_id": existing_topic,
            "normalized_primary_topic_id": expected_topic,
            "existing_review_required": existing.get("review_required"),
            "existing_confidence": existing.get("confidence"),
        }
    return None


def build_existing_route_aliases(
    records: Sequence[dict[str, Any]],
    existing_records: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, dict[str, Any]]]:
    current_by_paper_qno: dict[tuple[str, str], str] = {}
    current_ids = {str(record.get("question_id") or "").strip() for record in records}
    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        paper = str(record.get("paper") or "").strip().lower()
        qno = normalize_question_number(record.get("question_number"))
        if question_id and paper and qno:
            current_by_paper_qno[(paper, qno)] = question_id

    aliases: dict[str, tuple[str, dict[str, Any]]] = {}

    # Before this contract distinguished March from May/June, March records
    # were keyed as ``summerYY``.  Claim that legacy route for the provenance-
    # matched March record before an independently admitted June record can
    # see the same old key.  A route already carrying June provenance, or a
    # source hash matching the current June record, is never claimed.
    current_hashes_by_id = {
        str(record.get("question_id") or "").strip(): source_record_hash(record)
        for record in records
        if str(record.get("question_id") or "").strip()
    }
    for record in records:
        current_id = str(record.get("question_id") or "").strip()
        source_session_code = source_session_code_for_record(record)
        if not current_id or not source_session_code.startswith("m") or "spring" not in current_id.lower():
            continue
        legacy_id = re.sub(r"(?<=^\d{2})spring(?=\d{2}_q)", "summer", current_id, count=1, flags=re.IGNORECASE)
        route = existing_records.get(legacy_id)
        if not isinstance(route, dict):
            continue
        route_session_code = str(route.get("source_session_code") or "").strip().lower()
        if route_session_code and route_session_code != source_session_code:
            continue
        route_hash = str(route.get("source_record_hash") or "").strip()
        if route_hash and route_hash == current_hashes_by_id.get(legacy_id):
            continue
        if current_id not in existing_records and current_id not in aliases:
            aliases[current_id] = (legacy_id, route)

    for existing_id, route in existing_records.items():
        if existing_id in current_ids:
            continue
        if not isinstance(route, dict):
            continue
        paper = str(route.get("paper") or "").strip().lower()
        if "autumn" not in paper:
            continue
        qno = normalize_question_number(route.get("question_number"))
        current_id = current_by_paper_qno.get((paper.replace("autumn", "winter"), qno))
        if current_id and current_id not in existing_records and current_id not in aliases:
            aliases[current_id] = (existing_id, route)
    return aliases


def existing_route_identity_matches(existing: dict[str, Any], record: dict[str, Any]) -> bool:
    current_session_code = source_session_code_for_record(record)
    route_session_code = str(existing.get("source_session_code") or "").strip().lower()
    if route_session_code and current_session_code and route_session_code != current_session_code:
        return False
    # A legacy ``summerYY`` route with no raw-session provenance may represent
    # the formerly collapsed March paper.  Never attach it to a June record if
    # its source-record hash does not identify that exact June record.
    if current_session_code.startswith("s") and not route_session_code:
        route_hash = str(existing.get("source_record_hash") or "").strip()
        if not route_hash or route_hash != source_record_hash(record):
            return False
    return True


def source_session_code_for_record(record: dict[str, Any]) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    candidates = [
        notes.get("source_pdf"),
        record.get("source_pdf"),
        record.get("question_image_path"),
        record.get("canonical_question_artifact"),
    ]
    for value in candidates:
        match = re.search(r"(?:^|_)(?P<code>[msw]\d{2})(?:_|$)", Path(str(value or "")).name.lower())
        if match:
            return match.group("code")
    return ""


def normalize_question_number(value: Any) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return str(int(text))
    return text


def refresh_review_reasons(
    record: dict[str, Any],
    context: dict[str, Any],
    *,
    extra_review_reasons: Sequence[str] = (),
) -> list[str]:
    reasons: list[str] = list(extra_review_reasons)
    for reason in context.get("deterministic_review_reasons") or []:
        reasons.append(str(reason))
    topic_confidence = status_value(record, "topic_confidence")
    if topic_confidence not in {"high", "medium"}:
        reasons.append("topic_confidence_not_high_or_medium")
    gates = [
        ("mapping_status", "pass", "mapping_status_not_pass"),
        ("validation_status", "pass", "validation_status_not_pass"),
        ("scope_quality_status", "clean", "scope_quality_status_not_clean"),
        ("question_crop_confidence", "high", "question_crop_confidence_not_high"),
        ("visual_curation_status", "ready", "visual_curation_status_not_ready"),
    ]
    for key, expected, reason in gates:
        if status_value(record, key) != expected:
            reasons.append(reason)
    if status_value(record, "text_only_status") == "fail":
        reasons.append("text_only_status_fail")
    return sorted(set(reasons), key=reasons.index)


def refresh_confidence(record: dict[str, Any], review_reasons: Sequence[str]) -> str:
    topic_confidence = status_value(record, "topic_confidence")
    if review_reasons:
        return "low" if topic_confidence not in {"high", "medium"} else topic_confidence
    return topic_confidence if topic_confidence in {"high", "medium"} else "low"


def build_refresh_report(
    *,
    records: Sequence[dict[str, Any]],
    sidecar: dict[str, Any],
    exclusions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    existing_record_count: int,
    generated_at: str,
    question_bank_path: Path,
    taxonomy_path: Path,
    routing_path: Path,
    write: bool,
) -> dict[str, Any]:
    rows = route_rows(sidecar)
    record_by_id = {str(record.get("question_id") or ""): record for record in records}
    combined_count = len(rows) + len(exclusions)
    summary = {
        "question_bank_records": len(records),
        "existing_sidecar_entries": existing_record_count,
        "sidecar_entries": len(rows),
        "explicit_exclusions": len(exclusions),
        "entries_plus_exclusions": combined_count,
        "coverage_complete": combined_count == len(records),
        "preserved_existing_entries": sidecar["metadata"].get("preserved_existing_entries", 0),
        "preserved_reviewed_entries": sidecar["metadata"].get("preserved_reviewed_entries", 0),
        "preserved_via_alias_entries": sidecar["metadata"].get("preserved_via_alias_entries", 0),
        "identity_guard_rejections": sidecar["metadata"].get("identity_guard_rejections", 0),
        "existing_hash_refreshed_count": sidecar["metadata"].get("existing_hash_refreshed_count", 0),
        "new_entries": sidecar["metadata"].get("new_entries", 0),
        "new_review_required_entries": sidecar["metadata"].get("new_review_required_entries", 0),
        "conflicts_count": len(conflicts),
        "unresolved_count": len(unresolved),
        "review_required_entries": sum(1 for row in rows if row.get("review_required") is True),
        "strict_filter_entries": sum(1 for row in rows if is_strict_route(row)),
        "write": write,
    }
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "inputs": {
            "question_bank": str(question_bank_path),
            "taxonomy": str(taxonomy_path),
            "routing": str(routing_path),
        },
        "summary": summary,
        "coverage_by_year": group_counts(rows, ("year",)),
        "coverage_by_family_component": group_counts(rows, ("paper_family", "source_component")),
        "coverage_by_year_family_component": group_counts(rows, ("year", "paper_family", "source_component")),
        "confidence_counts": counter_dict(row.get("confidence") for row in rows),
        "review_reason_counts": counter_dict(reason for row in rows for reason in (row.get("review_reasons") or [])),
        "routing_source_counts": counter_dict(row.get("routing_source") for row in rows),
        "exclusions": exclusions,
        "conflicts": conflicts,
        "unresolved": unresolved,
        "sample_review_required": [
            sample_report_row(row, record_by_id.get(str(row.get("question_id") or "")))
            for row in rows
            if row.get("review_required") is True
        ][:50],
    }


def write_topic_routing_refresh_outputs(
    *,
    sidecar: dict[str, Any],
    routing_path: Path,
    question_bank_path: Path,
    release_manifest_path: Path,
    report: dict[str, Any],
    report_prefix: Path,
) -> None:
    write_atomic_json(sidecar, routing_path)
    sha_path = routing_path.with_suffix(".sha256")
    sha_path.write_text(f"{file_sha256(routing_path)}  {routing_path.name}\n", encoding="utf-8")
    build_topic_routing_release_manifest(
        question_bank_path=question_bank_path,
        durable_sidecar_path=routing_path,
        release_manifest_path=release_manifest_path,
    )
    json_report, markdown_report = report_paths(report_prefix)
    write_atomic_json(report, json_report)
    markdown_report.parent.mkdir(parents=True, exist_ok=True)
    markdown_report.write_text(render_markdown(report), encoding="utf-8")


def _default_release_manifest_path(question_bank_path: Path, routing_path: Path) -> Path:
    if (
        question_bank_path.resolve() == DEFAULT_QUESTION_BANK.resolve()
        and routing_path.resolve() == DEFAULT_ROUTING.resolve()
    ):
        return DEFAULT_RELEASE_MANIFEST_PATH
    return routing_path.parent / "question_bank_release_manifest.v1.json"


def report_paths(prefix: Path) -> tuple[Path, Path]:
    if prefix.suffix == ".json":
        return prefix, prefix.with_suffix(".md")
    if prefix.suffix == ".md":
        return prefix.with_suffix(".json"), prefix
    return prefix.with_suffix(".json"), prefix.with_suffix(".md")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Topic Routing Refresh",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for key in [
        "question_bank_records",
        "existing_sidecar_entries",
        "sidecar_entries",
        "explicit_exclusions",
        "entries_plus_exclusions",
        "preserved_existing_entries",
        "preserved_reviewed_entries",
        "preserved_via_alias_entries",
        "identity_guard_rejections",
        "existing_hash_refreshed_count",
        "new_entries",
        "new_review_required_entries",
        "conflicts_count",
        "unresolved_count",
        "review_required_entries",
        "strict_filter_entries",
    ]:
        lines.append(f"| {key} | {summary[key]} |")
    lines.extend(["", "## Coverage By Year", "", "| Year | Count |", "|---:|---:|"])
    for row in report["coverage_by_year"]:
        lines.append(f"| {row['year']} | {row['count']} |")
    lines.extend(["", "## Coverage By Family/Component", "", "| Family | Component | Count |", "|---|---:|---:|"])
    for row in report["coverage_by_family_component"]:
        lines.append(f"| {row['paper_family']} | {row['source_component']} | {row['count']} |")
    lines.extend(["", "## Review Reason Counts", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in report["review_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Conflicts", "", f"Count: {summary['conflicts_count']}"])
    for row in report["conflicts"][:25]:
        lines.append(
            f"- `{row['question_id']}`: existing `{row['existing_primary_topic_id']}` vs normalized `{row['normalized_primary_topic_id']}`"
        )
    lines.extend(["", "## Exclusions", "", f"Count: {summary['explicit_exclusions']}"])
    for row in report["exclusions"][:25]:
        lines.append(f"- `{row.get('question_id', '')}`: {row.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


def normalized_record_for_packet(record: dict[str, Any], paper_family: str) -> dict[str, Any]:
    normalized = dict(record)
    normalized["paper_family"] = paper_family
    return normalized


def source_component(record: dict[str, Any]) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    for value in [notes.get("source_paper_code"), record.get("source_paper_code"), record.get("paper_code")]:
        text = str(value or "").strip()
        if text:
            return normalize_component(text)
    return normalize_component(str(record.get("paper") or ""))


def normalize_component(value: str) -> str:
    text = value.strip().lower().removeprefix("p")
    match = re.search(r"\d+", text)
    if not match:
        return ""
    code = match.group(0)
    return code.zfill(2) if len(code) == 1 else code


def status_value(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value in (None, "") and isinstance(record.get("notes"), dict):
        value = record["notes"].get(key)
    return str(value or "").strip().lower()


def source_record_hash(record: dict[str, Any]) -> str:
    stable = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def route_rows(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    records = sidecar.get("records") if isinstance(sidecar.get("records"), dict) else {}
    rows = []
    for question_id, row in records.items():
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["question_id"] = str(question_id)
        item["year"] = parse_year(str(question_id), str(row.get("paper") or ""))
        rows.append(item)
    return rows


def parse_year(question_id: str, paper: str) -> int | None:
    match = re.search(r"(\d{2})_q\d+", question_id)
    if match:
        return 2000 + int(match.group(1))
    match = re.search(r"(\d{2})$", paper)
    if match:
        return 2000 + int(match.group(1))
    return None


def is_strict_route(row: dict[str, Any]) -> bool:
    return (
        row.get("review_required") is False
        and row.get("confidence") in {"high", "medium"}
        and isinstance(row.get("primary_topic_id"), str)
        and bool(row.get("topic_distribution"))
    )


def group_counts(rows: Sequence[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    counts = Counter(tuple(row.get(field) for field in fields) for row in rows)
    result: list[dict[str, Any]] = []
    for key, count in sorted(counts.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        result.append({field: key[index] for index, field in enumerate(fields)} | {"count": count})
    return result


def counter_dict(values: Any) -> dict[str, int]:
    counts = Counter("" if value is None else str(value) for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def sample_report_row(row: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "paper": row.get("paper"),
        "paper_family": row.get("paper_family"),
        "source_component": row.get("source_component"),
        "raw_topic": row.get("raw_topic"),
        "primary_topic_id": row.get("primary_topic_id"),
        "confidence": row.get("confidence"),
        "review_reasons": row.get("review_reasons"),
        "mapping_status": status_value(record or {}, "mapping_status"),
        "validation_status": status_value(record or {}, "validation_status"),
    }


def _packet_taxonomy_version(taxonomy: dict[str, Any]) -> str | None:
    payload = taxonomy.get("payload") if isinstance(taxonomy.get("payload"), dict) else {}
    schema_version = payload.get("schema_version")
    generated_from = payload.get("generated_from")
    return f"packet_taxonomy_v{schema_version}" if schema_version else (str(generated_from) if generated_from else None)


def _routing_summary(sidecar: dict[str, Any]) -> dict[str, Any]:
    rows = route_rows(sidecar)
    return {
        "schema_name": TOPIC_ROUTING_SCHEMA_NAME,
        "schema_version": TOPIC_ROUTING_SCHEMA_VERSION,
        "generated_at": sidecar.get("generated_at"),
        "total_records": len(rows),
        "attempted_records": len(rows),
        "successful_records": len(rows),
        "failed_records": 0,
        "review_required_records": sum(1 for row in rows if row.get("review_required") is True),
        "provider_failure_records": 0,
        "strict_filter_records": sum(1 for row in rows if is_strict_route(row)),
        "failures_by_reason": {},
        "safe_for_strict_filters": any(is_strict_route(row) for row in rows),
    }


def _unresolved_report_row(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": record.get("question_id"),
        "paper": record.get("paper"),
        "paper_family": record.get("paper_family"),
        "raw_topic": record.get("topic"),
        "source_component": context.get("source_component"),
        "normalization_status": context.get("normalization_status"),
        "normalization_reason": context.get("normalization_reason"),
    }


def _exclusion_row(record: dict[str, Any], context: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return _unresolved_report_row(record, context) | {"reason": reason}


def conflict_report_row(
    question_id: str,
    record: dict[str, Any],
    existing: dict[str, Any] | None,
    context: dict[str, Any],
    conflict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "paper": record.get("paper"),
        "paper_family": record.get("paper_family"),
        "source_component": context.get("source_component"),
        "raw_topic": record.get("topic"),
        "packet_family": context.get("packet_family"),
        "packet_topic_id": context.get("packet_topic_id"),
        "existing_primary_topic_id": conflict.get("existing_primary_topic_id"),
        "normalized_primary_topic_id": conflict.get("normalized_primary_topic_id"),
        "existing_review_required": (existing or {}).get("review_required"),
        "existing_confidence": (existing or {}).get("confidence"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
