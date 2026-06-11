from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from openai import OpenAI

from .asterion_course_contract import (
    COURSE_IDS,
    component_name_for_course,
    course_id_for_record,
    course_registry,
)
from .atomic_json import write_atomic_json
from .deepseek_enrich import (
    AI_FAILURE_INVALID_JSON,
    AI_FAILURE_EMPTY_CONTENT,
    AI_FAILURE_PROVIDER_API_ERROR,
    AI_FAILURE_REASONING_CONTENT_ONLY,
    AI_FAILURE_SCHEMA_VALIDATION_ERROR,
    AI_FAILURE_TAXONOMY_VALIDATION_ERROR,
    ModelResponseError,
    StartupConfigurationError,
    _response_snapshot,
    canonical_component_for_paper_family,
    create_client,
    load_canonical_taxonomy,
    load_question_bank,
    response_text_candidates,
)
from .run_status import RunStatusTracker, default_status_root_for_output, resolve_run_id


TOPIC_ROUTING_SCHEMA_NAME = "exam_bank.topic_routing_sidecar"
TOPIC_ROUTING_SCHEMA_VERSION = 1
TOPIC_ROUTING_PROMPT_VERSION = "topic_routing_v1"
DEFAULT_INPUT_PATH = Path("output/json/question_bank.json")
DEFAULT_OUTPUT_PATH = Path("output/json/question_bank.topic_routing.v1.json")
DEFAULT_TAXONOMY_PATH = Path("exam_bank_taxonomy/canonical")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

TOPIC_ROUTE_RECORD_KEYS = {
    "primary_topic_id",
    "topic_distribution",
    "confidence",
    "review_required",
    "review_reasons",
    "evidence_used",
}
TOPIC_ROUTE_CONFIDENCE_VALUES = {"high", "medium", "low"}
EVIDENCE_FIELDS = {"question_text", "ocr_text", "mark_scheme_text"}
REVIEW_WEAK_EVIDENCE = "weak_or_missing_text_evidence"
REVIEW_VISUAL_INSUFFICIENT = "visual_required_without_sufficient_text_evidence"


class TopicRouteValidationError(ValueError):
    def __init__(self, message: str, *, error_type: str = AI_FAILURE_SCHEMA_VALIDATION_ERROR) -> None:
        super().__init__(message)
        self.error_type = error_type


def add_topic_routing_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to question_bank.json.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Canonical taxonomy root containing topic_filter_maps/.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the topic sidecar JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--component", choices=["p1", "p3", "p4", "p5", "m1", "s1"], default=None)
    parser.add_argument("--paper", default=None)
    parser.add_argument("--question-id", action="append", default=None, help="Question ID to route. Repeatable.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of filtered input records to route. Defaults to all filtered records.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing topic sidecar.")
    parser.add_argument("--force-rerun", action="store_true", help="With --resume, rerun existing records.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected packets without calling the provider or writing output.")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", dest="progress", action="store_true", default=True, help="Show terminal progress updates.")
    progress.add_argument("--no-progress", dest="progress", action="store_false", help="Disable terminal progress updates.")
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument("--batch-by-paper", dest="batch_by_paper", action="store_true", default=True)
    batch.add_argument("--no-batch-by-paper", dest="batch_by_paper", action="store_false")
    parser.add_argument("--failure-log", type=Path, default=None, help="Optional JSONL path for provider/validation failures.")
    parser.add_argument("--status-dir", type=Path, default=None, help="Directory for run_status files.")
    parser.add_argument("--run-id", default="", help="Optional stable run ID for status files and resume.")
    parser.add_argument("--prompt-version", default=TOPIC_ROUTING_PROMPT_VERSION, help=argparse.SUPPRESS)


def finalize_topic_routing_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.limit is not None and args.limit < 0:
        raise StartupConfigurationError("--limit must be zero or greater.")
    args.question_ids = _parse_question_ids(args.question_id)
    return args


def run_topic_routing(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict DeepSeek topic routing over question_bank.json.")
    add_topic_routing_cli_arguments(parser)
    args = finalize_topic_routing_args(parser.parse_args(argv))
    return run_topic_routing_from_args(args)


def run_topic_routing_from_args(args: argparse.Namespace) -> int:
    status_root = args.status_dir or default_status_root_for_output(args.output)
    run_id = resolve_run_id(
        status_root=status_root,
        run_type="topic_routing",
        requested_run_id=args.run_id,
        resume=bool(args.resume),
    )
    tracker = RunStatusTracker(
        run_id=run_id,
        run_type="topic_routing",
        status_root=status_root,
        command="topic-route-ai",
        input_paths=[args.input],
        output_paths=[args.output],
        config_paths=[args.taxonomy],
        provider="deepseek",
        model=args.model,
        prompt_version=args.prompt_version,
        progress=bool(args.progress),
    )
    failure_log = args.failure_log or args.output.with_suffix(".failures.jsonl")
    tracker.start(phase="loading_question_bank")
    try:
        output_existed = args.output.exists()
        if output_existed and not args.resume:
            raise StartupConfigurationError(
                "Output path already exists. Use --resume, --resume --force-rerun, or choose a new output path."
            )

        records = load_question_bank(args.input)
        selected = select_topic_routing_records(
            records,
            component=args.component,
            paper=args.paper,
            question_ids=args.question_ids,
            limit=args.limit,
        )
        if args.dry_run:
            packets = [
                build_topic_routing_question_packet(record, taxonomy_root=args.taxonomy).packet
                for record in selected
            ]
            print(
                json.dumps(
                    {
                        "selected_count": len(selected),
                        "would_call_network": False,
                        "packets": packets,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            tracker.finish("completed")
            return 0

        resume_records = load_topic_routing_sidecar_records(args.output) if args.resume and output_existed else {}
        resume_packet_hashes = (
            _evidence_packet_hashes_for_records(selected, taxonomy_root=args.taxonomy)
            if args.resume and not args.force_rerun
            else {}
        )
        if args.resume and not args.force_rerun:
            preserved_records = {
                question_id: value
                for question_id, value in resume_records.items()
                if question_id in {str(record.get("question_id", "")).strip() for record in selected}
                and _resume_record_is_current(
                    value,
                    model=args.model,
                    prompt_version=args.prompt_version,
                    evidence_packet_hash=resume_packet_hashes.get(question_id),
                )
            }
            records_to_route = [
                record
                for record in selected
                if not _resume_record_is_current(
                    resume_records.get(str(record.get("question_id", ""))),
                    model=args.model,
                    prompt_version=args.prompt_version,
                    evidence_packet_hash=resume_packet_hashes.get(str(record.get("question_id", "")).strip()),
                )
            ]
            stale_reasons = Counter(
                _resume_record_staleness_reason(
                    resume_records.get(str(record.get("question_id", ""))),
                    model=args.model,
                    prompt_version=args.prompt_version,
                    evidence_packet_hash=resume_packet_hashes.get(str(record.get("question_id", "")).strip()),
                )
                for record in selected
                if str(record.get("question_id", "")).strip() in resume_records
                and str(record.get("question_id", "")).strip() not in preserved_records
            )
        else:
            preserved_records = {}
            records_to_route = selected
            stale_reasons = Counter()

        route_batches = batch_topic_routing_records(records_to_route, batch_by_paper=args.batch_by_paper)
        tracker.set_totals(
            total_batches=len(route_batches) + (1 if preserved_records else 0),
            total_records=len(selected),
        )
        if preserved_records:
            tracker.update_phase("preserving_resume_records", force_render=True)
            tracker.skip_batch(
                batch_id="resume_preserved",
                paper=None,
                paper_family=None,
                record_count=len(preserved_records),
                skipped_records=len(preserved_records),
            )
            preserved_counts = topic_progress_counts(preserved_records.values())
            tracker.update_extra_counts(
                review_required_records=preserved_counts["review_required_records"],
                provider_failure_records=preserved_counts["provider_failure_records"],
                force_render=True,
            )

        run_timestamp = datetime.now(timezone.utc).isoformat()
        if records_to_route:
            tracker.update_phase("creating_provider_client", force_render=True)
            client = create_client(base_url=args.base_url)
            routed, manifest = route_topic_records(
                records_to_route,
                client=client,
                taxonomy_root=args.taxonomy,
                output_path=args.output,
                model=args.model,
                prompt_version=args.prompt_version,
                run_timestamp=run_timestamp,
                batch_by_paper=bool(args.batch_by_paper),
                failure_log_path=failure_log,
                progress=tracker,
            )
        else:
            routed = {}
            manifest = {
                "run_timestamp": run_timestamp,
                "model": args.model,
                "prompt_version": args.prompt_version,
                "batch_count": 0,
                "record_count": 0,
                "provider_batch_count": 0,
                "average_prompt_chars": 0,
                "estimated_prompt_tokens": 0,
                "batches": [],
            }
        selected_ids = {str(record.get("question_id", "")).strip() for record in selected}
        merged = (
            {
                question_id: value
                for question_id, value in resume_records.items()
                if question_id in selected_ids
                and _resume_record_is_current(
                    value,
                    model=args.model,
                    prompt_version=args.prompt_version,
                    evidence_packet_hash=resume_packet_hashes.get(question_id),
                )
            }
            if args.resume and not args.force_rerun
            else {}
        )
        merged.update(routed)

        metadata = {
            "input_path": str(args.input),
            "taxonomy_path": str(args.taxonomy),
            "taxonomy_version": taxonomy_version(args.taxonomy),
            "output_path": str(args.output),
            "model": args.model,
            "prompt_version": args.prompt_version,
            "selected_count": len(selected),
            "records_to_attempt_count": len(records_to_route),
            "resume": bool(args.resume),
            "force_rerun": bool(args.force_rerun),
            "resume_preserved_count": len(preserved_records),
            "resume_stale_count": sum(stale_reasons.values()),
            "resume_stale_reasons": dict(sorted(stale_reasons.items())),
            "batch_by_paper": bool(args.batch_by_paper),
            "run_manifest": manifest,
        }
        sidecar = build_topic_routing_sidecar(
            merged,
            taxonomy_path=args.taxonomy,
            taxonomy_version_value=metadata["taxonomy_version"],
            model=args.model,
            prompt_version=args.prompt_version,
            generated_at=run_timestamp,
            metadata=metadata,
        )
        metadata["run_summary"] = audit_topic_routing_sidecar_payload(sidecar)
        sidecar["metadata"] = metadata
        write_topic_routing_sidecar_payload(sidecar, args.output)
        summary = metadata["run_summary"]
        final_counts = topic_progress_counts(merged.values())
        tracker.update_extra_counts(
            review_required_records=final_counts["review_required_records"],
            provider_failure_records=final_counts["provider_failure_records"],
            force_render=True,
        )
        print(f"Wrote {summary['total_records']} topic-routing records to {args.output}")
        print(
            "Topic routing summary: "
            f"{summary['attempted_records']} attempted, "
            f"{summary['successful_records']} successful, "
            f"{summary['failed_records']} failed, "
            f"{summary['review_required_records']} review-required, "
            f"{summary['provider_failure_records']} provider failures."
        )
        tracker.update_phase("writing_final_sidecar", output_path=args.output, force_render=True)
        tracker.finish("completed")
        print(tracker.final_summary())
        return 0
    except Exception as exc:
        tracker.finish("failed", error_summary=str(exc))
        raise


def select_topic_routing_records(
    records: Sequence[dict[str, Any]],
    *,
    component: str | None = None,
    paper: str | None = None,
    question_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = list(records)
    if component:
        wanted_component = canonical_component_for_paper_family(component)
        selected = [
            record
            for record in selected
            if canonical_component_for_paper_family(record.get("paper_family")) == wanted_component
        ]
    if paper:
        wanted_paper = paper.strip().lower()
        selected = [record for record in selected if str(record.get("paper") or "").strip().lower() == wanted_paper]
    if question_ids:
        wanted_ids = {value.strip() for value in question_ids if value.strip()}
        selected = [record for record in selected if str(record.get("question_id") or "").strip() in wanted_ids]
    if limit is not None:
        selected = selected[:limit]
    return selected


def batch_topic_routing_records(records: Sequence[dict[str, Any]], *, batch_by_paper: bool = True) -> list[list[dict[str, Any]]]:
    if not batch_by_paper:
        return [[record] for record in records]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            canonical_component_for_paper_family(record.get("paper_family")),
            str(record.get("paper") or ""),
        )
        grouped[key].append(record)
    return [grouped[key] for key in sorted(grouped)]


class TopicQuestionPacket:
    def __init__(
        self,
        packet: dict[str, Any],
        *,
        evidence_packet_hash: str,
        review_record: dict[str, Any] | None = None,
    ) -> None:
        self.packet = packet
        self.evidence_packet_hash = evidence_packet_hash
        self.review_record = review_record


def build_topic_routing_question_packet(
    record: dict[str, Any],
    *,
    taxonomy_root: str | Path,
) -> TopicQuestionPacket:
    taxonomy = load_canonical_taxonomy(taxonomy_root, record.get("paper_family"))
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    evidence: dict[str, str] = {}
    question_text = _trusted_question_text(record, notes)
    if question_text:
        evidence["question_text"] = _truncate(question_text, 1600)
    ocr_text = _trusted_ocr_text(record, notes, question_text=question_text)
    if ocr_text:
        evidence["ocr_text"] = _truncate(ocr_text, 1000)
    mark_scheme_text = _clean_text(record.get("mark_scheme_text"))
    if mark_scheme_text:
        evidence["mark_scheme_text"] = _truncate(mark_scheme_text, 1600)

    visual_required = bool(record.get("visual_required") or notes.get("visual_required"))
    packet = {
        "question_id": str(record.get("question_id")),
        "paper_family": str(record.get("paper_family") or "").lower(),
        "paper": record.get("paper"),
        "question_number": record.get("question_number"),
        "visual_required": visual_required,
        "evidence": evidence,
        "allowed_topics": [
            {
                "topic_id": topic_id,
                "label": str(topic.get("topic_name") or topic.get("official_section_name") or topic_id),
            }
            for topic_id, topic in sorted(taxonomy.topics.items())
        ],
    }
    evidence_packet_hash = hash_topic_routing_evidence_packet(packet)
    review_reasons = deterministic_review_reasons(packet)
    if review_reasons:
        return TopicQuestionPacket(
            packet,
            evidence_packet_hash=evidence_packet_hash,
            review_record=build_deterministic_review_record(
                record,
                packet,
                evidence_packet_hash=evidence_packet_hash,
                review_reasons=review_reasons,
            ),
        )
    return TopicQuestionPacket(packet, evidence_packet_hash=evidence_packet_hash)


def hash_topic_routing_evidence_packet(packet: dict[str, Any]) -> str:
    stable_payload = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()


def deterministic_review_reasons(packet: dict[str, Any]) -> list[str]:
    evidence = packet.get("evidence") if isinstance(packet.get("evidence"), dict) else {}
    evidence_text = "\n".join(str(value) for value in evidence.values())
    if not evidence_text.strip():
        return [REVIEW_WEAK_EVIDENCE]
    if bool(packet.get("visual_required")) and _visual_evidence_insufficient(evidence):
        return [REVIEW_VISUAL_INSUFFICIENT]
    return []


def build_deterministic_review_record(
    record: dict[str, Any],
    packet: dict[str, Any],
    *,
    evidence_packet_hash: str | None = None,
    review_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "primary_topic_id": None,
        "topic_distribution": [],
        "confidence": "low",
        "review_required": True,
        "review_reasons": list(review_reasons),
        "evidence_used": sorted((packet.get("evidence") or {}).keys()),
        "llm_provider": None,
        "llm_model": None,
        "llm_prompt_version": TOPIC_ROUTING_PROMPT_VERSION,
        "routing_source": "deterministic_review_gate",
        "paper": record.get("paper"),
        "paper_family": str(record.get("paper_family") or "").lower(),
        "question_number": record.get("question_number"),
        "evidence_packet_hash": evidence_packet_hash or hash_topic_routing_evidence_packet(packet),
        **_course_metadata_for_record(record),
    }


def route_topic_records(
    records: Sequence[dict[str, Any]],
    *,
    client: OpenAI,
    taxonomy_root: str | Path,
    output_path: Path | None,
    model: str,
    prompt_version: str = TOPIC_ROUTING_PROMPT_VERSION,
    run_timestamp: str | None = None,
    batch_by_paper: bool = True,
    failure_log_path: Path | None = None,
    progress: RunStatusTracker | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_timestamp = run_timestamp or datetime.now(timezone.utc).isoformat()
    routed: dict[str, dict[str, Any]] = {}
    manifest_batches: list[dict[str, Any]] = []
    prompt_chars: list[int] = []
    all_topic_ids = load_all_topic_ids(taxonomy_root)
    batches = batch_topic_routing_records(records, batch_by_paper=batch_by_paper)
    if progress:
        status = progress.status_snapshot()
        if not int(status.get("total_records") or 0):
            progress.set_totals(total_batches=len(batches), total_records=len(records))

    for index, batch in enumerate(batches, start=1):
        paper = str(batch[0].get("paper") or "") if batch else ""
        paper_family = str(batch[0].get("paper_family") or "").lower() if batch else ""
        batch_id = _batch_id(batch, index=index)
        if progress:
            progress.start_batch(batch_id=batch_id, paper=paper, paper_family=paper_family, record_count=len(batch), phase="routing_topics")

        packets: list[dict[str, Any]] = []
        deterministic_reviews = 0
        for record in batch:
            question_id = str(record.get("question_id") or "")
            packet_result = build_topic_routing_question_packet(record, taxonomy_root=taxonomy_root)
            if packet_result.review_record is not None:
                routed[question_id] = packet_result.review_record
                deterministic_reviews += 1
            else:
                packets.append(packet_result.packet)

        status = "success"
        failed = 0
        batch_parse_metadata: dict[str, Any] = {}
        if packets:
            taxonomy = load_canonical_taxonomy(taxonomy_root, paper_family)
            payload = build_topic_routing_batch_payload(packets, taxonomy_path=taxonomy_root, prompt_version=prompt_version)
            messages = build_topic_routing_messages(payload, prompt_version=prompt_version)
            prompt_size = sum(len(message["content"]) for message in messages)
            prompt_chars.append(prompt_size)
            try:
                outputs, record_errors, parse_metadata, _raw_provider_output = request_topic_routing_batch(
                    client,
                    model=model,
                    payload=payload,
                    expected_packets=packets,
                    allowed_topic_ids=set(taxonomy.topics),
                    all_topic_ids=all_topic_ids,
                    prompt_version=prompt_version,
                )
                batch_parse_metadata = parse_metadata
                for packet in packets:
                    question_id = str(packet["question_id"])
                    if question_id in outputs:
                        routed[question_id] = build_topic_routing_success_record(
                            batch_record=batch,
                            question_id=question_id,
                            output=outputs[question_id],
                            model=model,
                            prompt_version=prompt_version,
                            run_timestamp=run_timestamp,
                            parse_metadata=parse_metadata,
                            packet=packet,
                        )
                    else:
                        error = record_errors.get(
                            question_id,
                            {
                                "type": AI_FAILURE_SCHEMA_VALIDATION_ERROR,
                                "message": f"records[{question_id}]: missing route record.",
                            },
                        )
                        routed[question_id] = build_topic_routing_error_record(
                            packet,
                            model=model,
                            prompt_version=prompt_version,
                            run_timestamp=run_timestamp,
                            error_type=str(error.get("type") or AI_FAILURE_SCHEMA_VALIDATION_ERROR),
                            message=str(error.get("message") or "Invalid topic-routing record."),
                        )
                failed = len(record_errors)
                if outputs and record_errors:
                    status = "partial_salvage"
            except ModelResponseError as exc:
                status = exc.error_type
                failed = len(packets)
                for packet in packets:
                    question_id = str(packet["question_id"])
                    routed[question_id] = build_topic_routing_error_record(
                        packet,
                        model=model,
                        prompt_version=prompt_version,
                        run_timestamp=run_timestamp,
                        error_type=exc.error_type,
                        message=str(exc),
                    )
                    append_topic_routing_failure_log(
                        failure_log_path,
                        question_id=question_id,
                        error_type=exc.error_type,
                        error_message=str(exc),
                        model=model,
                        run_timestamp=run_timestamp,
                        raw_provider_output=exc.raw_provider_output,
                        request_payload=payload,
                    )
            except Exception as exc:
                status = AI_FAILURE_PROVIDER_API_ERROR
                failed = len(packets)
                message = f"{exc.__class__.__name__}: {exc}"
                for packet in packets:
                    question_id = str(packet["question_id"])
                    routed[question_id] = build_topic_routing_error_record(
                        packet,
                        model=model,
                        prompt_version=prompt_version,
                        run_timestamp=run_timestamp,
                        error_type=AI_FAILURE_PROVIDER_API_ERROR,
                        message=message,
                    )
                    append_topic_routing_failure_log(
                        failure_log_path,
                        question_id=question_id,
                        error_type=AI_FAILURE_PROVIDER_API_ERROR,
                        error_message=message,
                        model=model,
                        run_timestamp=run_timestamp,
                        raw_provider_output=None,
                        request_payload=payload,
                    )

        successes = len(batch) - failed
        if progress:
            progress_counts = topic_progress_counts(routed.values())
            progress.update_extra_counts(
                review_required_records=progress_counts["review_required_records"],
                provider_failure_records=progress_counts["provider_failure_records"],
            )
            if failed:
                progress.fail_batch(
                    batch_id=batch_id,
                    paper=paper,
                    paper_family=paper_family,
                    record_count=len(batch),
                    successful_records=successes,
                    failed_records=failed,
                    error_message=status,
                )
            else:
                progress.complete_batch(
                    batch_id=batch_id,
                    paper=paper,
                    paper_family=paper_family,
                    record_count=len(batch),
                    successful_records=successes,
                )
        manifest_batches.append(
            {
                "batch_id": batch_id,
                "paper": paper,
                "paper_family": paper_family,
                "question_ids": [str(record.get("question_id")) for record in batch],
                "provider_question_ids": [str(packet.get("question_id")) for packet in packets],
                "deterministic_review_count": deterministic_reviews,
                "status": status,
                "record_count": len(batch),
                "provider_record_count": len(packets),
                "valid_records": int(batch_parse_metadata.get("valid_records") or len(packets) - failed),
                "invalid_records": int(batch_parse_metadata.get("invalid_records") or failed),
                "missing_records": int(batch_parse_metadata.get("missing_records") or 0),
                "duplicate_records": int(batch_parse_metadata.get("duplicate_records") or 0),
                "unknown_returned_records": int(batch_parse_metadata.get("unknown_returned_records") or 0),
                "batch_salvaged": bool(batch_parse_metadata.get("batch_salvaged") or (status == "partial_salvage")),
                "prompt_chars": prompt_chars[-1] if packets else 0,
            }
        )

    average_prompt_chars = int(sum(prompt_chars) / len(prompt_chars)) if prompt_chars else 0
    manifest = {
        "run_timestamp": run_timestamp,
        "model": model,
        "prompt_version": prompt_version,
        "batch_count": len(batches),
        "record_count": len(records),
        "provider_batch_count": len(prompt_chars),
        "average_prompt_chars": average_prompt_chars,
        "estimated_prompt_tokens": sum(prompt_chars) // 4,
        "batches": manifest_batches,
    }
    return routed, manifest


def build_topic_routing_batch_payload(
    packets: Sequence[dict[str, Any]],
    *,
    taxonomy_path: str | Path,
    prompt_version: str = TOPIC_ROUTING_PROMPT_VERSION,
) -> dict[str, Any]:
    return {
        "schema_name": "exam_bank.topic_routing_request",
        "prompt_version": prompt_version,
        "taxonomy_path": str(taxonomy_path),
        "questions": list(packets),
    }


def build_topic_routing_messages(payload: dict[str, Any], *, prompt_version: str = TOPIC_ROUTING_PROMPT_VERSION) -> list[dict[str, str]]:
    instructions = (
        "You are routing CAIE 9709 mathematics exam questions to canonical parent topic IDs only. "
        "Use only the allowed canonical topic IDs supplied with each question. Do not invent topic IDs. "
        "The project is image-first, but this request does not send images. Question text, OCR text, and mark-scheme text are advisory evidence only. "
        "Do not claim image evidence. Do not output image_reference unless an actual image was sent; no image is sent in this pass. "
        "Do not output difficulty, subtopics, skills, rationales, explanations, Content Lab metadata, Asterion readiness, or student-facing text. "
        "If evidence is weak or a visual-dependent question cannot be routed from the supplied text, set confidence to low and review_required to true. "
        "Return exactly one valid JSON object, no markdown, no prose, no comments, no trailing commas, and no reasoning text. "
        "The object must contain exactly one top-level key: records. records must map each question_id to exactly these keys: "
        + ", ".join(sorted(TOPIC_ROUTE_RECORD_KEYS))
        + ". primary_topic_id must be one allowed topic ID or null. topic_distribution must be an array of objects with topic_id and fit_percent. "
        "fit_percent values must be positive integers and must total exactly 100, unless topic_distribution is empty because no defensible topic can be chosen. "
        "confidence must be high, medium, or low. review_required must be a JSON boolean. review_reasons and evidence_used must be arrays of strings. "
        "evidence_used may contain only question_text, ocr_text, and mark_scheme_text, and only when that evidence field was supplied. "
        f"Prompt version: {prompt_version}."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def request_topic_routing_batch(
    client: OpenAI,
    *,
    model: str,
    payload: dict[str, Any],
    expected_packets: Sequence[dict[str, Any]],
    allowed_topic_ids: set[str],
    all_topic_ids: set[str],
    prompt_version: str = TOPIC_ROUTING_PROMPT_VERSION,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, Any], str]:
    response = client.chat.completions.create(
        model=model,
        messages=build_topic_routing_messages(payload, prompt_version=prompt_version),
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        parsed, parse_metadata = parse_topic_routing_response_with_recovery(
            response,
            lambda raw_text: parse_topic_routing_model_json(
                raw_text,
                expected_packets=expected_packets,
                allowed_topic_ids=allowed_topic_ids,
                all_topic_ids=all_topic_ids,
            ),
        )
    except ValueError as exc:
        raw_output = getattr(exc, "raw_provider_output", None)
        raise ModelResponseError(
            str(exc),
            raw_provider_output=raw_output or _response_snapshot(response),
            error_type=getattr(exc, "error_type", AI_FAILURE_SCHEMA_VALIDATION_ERROR),
        ) from exc
    response_metadata = parsed.get("response_metadata") if isinstance(parsed.get("response_metadata"), dict) else {}
    parse_metadata.update(response_metadata)
    return parsed["records"], parsed.get("record_errors", {}), parse_metadata, _response_snapshot(response)


def parse_topic_routing_response_with_recovery(
    response: Any,
    parser: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = response_text_candidates(response)
    final_content_sources = {"message.content", "output_text", "structured_response"}
    has_final_content = any(source in final_content_sources for source, _text in candidates)
    has_reasoning_content = any(source == "reasoning_content" for source, _text in candidates)
    if not has_final_content:
        error_type = AI_FAILURE_REASONING_CONTENT_ONLY if has_reasoning_content else AI_FAILURE_EMPTY_CONTENT
        message = (
            "Provider response only contained reasoning_content; final JSON content was empty."
            if has_reasoning_content
            else "Provider response did not contain final JSON content."
        )
        error = TopicRouteValidationError(message, error_type=error_type)
        setattr(error, "raw_provider_output", _response_snapshot(response))
        raise error

    first_error: ValueError | None = None
    first_text: str | None = None
    first_source: str | None = None
    for source, text in candidates:
        if source == "reasoning_content":
            continue
        for candidate_text in _topic_routing_candidate_json_texts(text):
            try:
                parsed = parser(candidate_text)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                    first_text = candidate_text
                    first_source = source
                continue
            return parsed, {
                "parse_recovered": source != "message.content" or candidate_text != text,
                "parse_recovery_source": source,
            }

    if first_error is not None:
        error = ValueError(str(first_error))
        setattr(error, "raw_provider_output", first_text)
        setattr(error, "parse_recovery_source", first_source)
        setattr(error, "error_type", getattr(first_error, "error_type", AI_FAILURE_SCHEMA_VALIDATION_ERROR))
        raise error
    error = TopicRouteValidationError(
        "Provider response did not contain parseable JSON content.",
        error_type=AI_FAILURE_INVALID_JSON,
    )
    setattr(error, "raw_provider_output", _response_snapshot(response))
    raise error


def _topic_routing_candidate_json_texts(text: str) -> list[str]:
    candidates = [text]
    extracted = _extract_topic_routing_json_object(text)
    if extracted and extracted != text:
        candidates.append(extracted)
    for escaped in re.findall(r'"(\{(?:[^"\\]|\\.)+\})"', text, flags=re.DOTALL):
        try:
            decoded = json.loads(f'"{escaped}"')
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, str) and decoded.strip():
            candidates.append(decoded)
            nested = _extract_topic_routing_json_object(decoded)
            if nested and nested != decoded:
                candidates.append(nested)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return unique


def _extract_topic_routing_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder(object_pairs_hook=_json_object_pairs)
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, _JsonObjectPairs):
            return text[start : start + end]
    return None


class _JsonObjectPairs(list[tuple[str, Any]]):
    pass


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> _JsonObjectPairs:
    return _JsonObjectPairs(pairs)


def parse_topic_routing_model_json(
    raw_text: str,
    *,
    expected_packets: Sequence[dict[str, Any]],
    allowed_topic_ids: set[str],
    all_topic_ids: set[str] | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text, object_pairs_hook=_json_object_pairs)
    except json.JSONDecodeError as exc:
        raise TopicRouteValidationError("Topic-routing model output was not valid JSON.", error_type=AI_FAILURE_INVALID_JSON) from exc
    if not isinstance(payload, _JsonObjectPairs) or [key for key, _value in payload] != ["records"]:
        raise TopicRouteValidationError(
            "Topic-routing model output must contain exactly one top-level key: records.",
            error_type=AI_FAILURE_SCHEMA_VALIDATION_ERROR,
        )
    records = payload[0][1]
    if not isinstance(records, _JsonObjectPairs):
        raise TopicRouteValidationError("records must be a JSON object.", error_type=AI_FAILURE_SCHEMA_VALIDATION_ERROR)

    expected_ids = {str(packet.get("question_id")) for packet in expected_packets}
    supplied_evidence = {
        str(packet.get("question_id")): set((packet.get("evidence") or {}).keys())
        for packet in expected_packets
    }
    validated: dict[str, dict[str, Any]] = {}
    record_errors: dict[str, dict[str, str]] = {}
    returned: dict[str, list[Any]] = defaultdict(list)
    unknown_returned_ids: list[str] = []
    for question_id, item in records:
        question_id = str(question_id)
        if question_id not in expected_ids:
            unknown_returned_ids.append(question_id)
            continue
        returned[question_id].append(item)

    for question_id in sorted(expected_ids):
        items = returned.get(question_id, [])
        if not items:
            record_errors[question_id] = {
                "type": AI_FAILURE_SCHEMA_VALIDATION_ERROR,
                "message": f"records[{question_id}]: missing route record.",
            }
            continue
        if len(items) > 1:
            record_errors[question_id] = {
                "type": AI_FAILURE_SCHEMA_VALIDATION_ERROR,
                "message": f"records[{question_id}]: duplicate route records returned.",
            }
            continue
        try:
            item = _json_pairs_to_plain_strict(items[0], context=f"records[{question_id}]")
            validated[question_id] = validate_topic_route_record(
                item,
                allowed_topic_ids=allowed_topic_ids,
                all_topic_ids=all_topic_ids or allowed_topic_ids,
                supplied_evidence=supplied_evidence[question_id],
            )
        except TopicRouteValidationError as exc:
            record_errors[question_id] = {
                "type": exc.error_type,
                "message": f"records[{question_id}]: {exc}",
            }
    duplicate_count = sum(1 for items in returned.values() if len(items) > 1)
    missing_count = sum(1 for question_id in expected_ids if not returned.get(question_id))
    return {
        "records": validated,
        "record_errors": record_errors,
        "response_metadata": {
            "valid_records": len(validated),
            "invalid_records": len(record_errors),
            "missing_records": missing_count,
            "duplicate_records": duplicate_count,
            "unknown_returned_records": len(unknown_returned_ids),
            "unknown_returned_question_ids": sorted(set(unknown_returned_ids)),
            "batch_salvaged": bool(validated and record_errors),
        },
    }


def _json_pairs_to_plain_strict(value: Any, *, context: str) -> Any:
    if isinstance(value, _JsonObjectPairs):
        payload: dict[str, Any] = {}
        for key, item in value:
            if key in payload:
                raise TopicRouteValidationError(f"{context} contained a duplicate key: {key}.")
            payload[key] = _json_pairs_to_plain_strict(item, context=f"{context}.{key}")
        return payload
    if isinstance(value, list):
        return [_json_pairs_to_plain_strict(item, context=f"{context}[]") for item in value]
    return value


def validate_topic_route_record(
    item: Any,
    *,
    allowed_topic_ids: set[str],
    all_topic_ids: set[str],
    supplied_evidence: set[str],
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TopicRouteValidationError("route record must be a JSON object.")
    actual_keys = set(item)
    if actual_keys != TOPIC_ROUTE_RECORD_KEYS:
        forbidden = sorted(actual_keys - TOPIC_ROUTE_RECORD_KEYS)
        if forbidden:
            raise TopicRouteValidationError(f"route record contains forbidden fields: {forbidden}.")
        raise TopicRouteValidationError(
            f"route record keys did not match required schema. Expected {sorted(TOPIC_ROUTE_RECORD_KEYS)}, got {sorted(actual_keys)}."
        )

    review_required = item["review_required"]
    if not isinstance(review_required, bool):
        raise TopicRouteValidationError("review_required must be a boolean.")

    confidence = item["confidence"]
    if not isinstance(confidence, str) or confidence not in TOPIC_ROUTE_CONFIDENCE_VALUES:
        raise TopicRouteValidationError("confidence must be high, medium, or low.")

    primary_topic_id = item["primary_topic_id"]
    if primary_topic_id is not None:
        primary_topic_id = _require_allowed_topic_id(primary_topic_id, allowed_topic_ids=allowed_topic_ids, all_topic_ids=all_topic_ids)
    elif not review_required:
        raise TopicRouteValidationError("primary_topic_id may be null only when review_required is true.")

    distribution = _validate_topic_distribution(
        item["topic_distribution"],
        allowed_topic_ids=allowed_topic_ids,
        all_topic_ids=all_topic_ids,
    )
    if primary_topic_id is None and distribution:
        raise TopicRouteValidationError("topic_distribution must be empty when primary_topic_id is null.")
    if primary_topic_id is not None and not distribution:
        raise TopicRouteValidationError("topic_distribution may be empty only when no defensible topic can be chosen.")
    if primary_topic_id is not None and primary_topic_id not in {entry["topic_id"] for entry in distribution}:
        raise TopicRouteValidationError("primary_topic_id must appear in topic_distribution.")

    review_reasons = _string_list(item["review_reasons"], "review_reasons")
    evidence_used = _string_list(item["evidence_used"], "evidence_used")
    evidence_used, evidence_repair = repair_evidence_used(
        evidence_used,
        supplied_evidence=supplied_evidence,
    )

    validated = {
        "primary_topic_id": primary_topic_id,
        "topic_distribution": distribution,
        "confidence": confidence,
        "review_required": review_required,
        "review_reasons": review_reasons,
        "evidence_used": evidence_used,
    }
    if evidence_repair:
        validated.update(evidence_repair)
    return validated


def repair_evidence_used(
    evidence_used: Sequence[str],
    *,
    supplied_evidence: set[str],
) -> tuple[list[str], dict[str, Any]]:
    available = sorted(value for value in supplied_evidence if value in EVIDENCE_FIELDS)
    supported = [value for value in evidence_used if value in available]
    dropped = [value for value in evidence_used if value not in available]
    if not dropped:
        return list(evidence_used), {}
    if supported:
        repaired = supported
    elif available:
        repaired = available
    else:
        raise TopicRouteValidationError(
            f"evidence_used contains evidence that was not supplied and no supplied evidence fallback exists: {dropped}."
        )
    return repaired, {
        "evidence_used_repaired": True,
        "evidence_used_original": list(evidence_used),
        "evidence_used_dropped": dropped,
    }


def _validate_topic_distribution(
    value: Any,
    *,
    allowed_topic_ids: set[str],
    all_topic_ids: set[str],
) -> list[dict[str, int | str]]:
    if not isinstance(value, list):
        raise TopicRouteValidationError("topic_distribution must be an array.")
    distribution: list[dict[str, int | str]] = []
    total = 0
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or set(entry) != {"topic_id", "fit_percent"}:
            raise TopicRouteValidationError(f"topic_distribution[{index}] must contain exactly topic_id and fit_percent.")
        topic_id = _require_allowed_topic_id(entry["topic_id"], allowed_topic_ids=allowed_topic_ids, all_topic_ids=all_topic_ids)
        if topic_id in seen:
            raise TopicRouteValidationError(f"topic_distribution contains duplicate topic_id: {topic_id}.")
        seen.add(topic_id)
        fit_percent = entry["fit_percent"]
        if isinstance(fit_percent, bool) or not isinstance(fit_percent, int) or fit_percent <= 0:
            raise TopicRouteValidationError("fit_percent values must be positive integers.")
        total += fit_percent
        distribution.append({"topic_id": topic_id, "fit_percent": fit_percent})
    if distribution and total != 100:
        raise TopicRouteValidationError("topic_distribution fit_percent values must total exactly 100.")
    return distribution


def _require_allowed_topic_id(value: Any, *, allowed_topic_ids: set[str], all_topic_ids: set[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopicRouteValidationError("topic_id must be a non-empty string.")
    topic_id = value.strip()
    if topic_id not in allowed_topic_ids:
        error_type = AI_FAILURE_TAXONOMY_VALIDATION_ERROR if topic_id not in all_topic_ids else "topic_outside_allowed_paper_family"
        raise TopicRouteValidationError(f"topic_id is not allowed for this paper family: {topic_id}", error_type=error_type)
    return topic_id


def build_topic_routing_success_record(
    *,
    batch_record: Sequence[dict[str, Any]],
    question_id: str,
    output: dict[str, Any],
    model: str,
    prompt_version: str,
    run_timestamp: str,
    parse_metadata: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    source = next((record for record in batch_record if str(record.get("question_id")) == question_id), {})
    record = dict(output)
    record.update(
        {
            "llm_provider": "deepseek",
            "llm_model": model,
            "llm_prompt_version": prompt_version,
            "llm_run_timestamp": run_timestamp,
            "routing_source": "deepseek_topic_routing",
            "paper": source.get("paper"),
            "paper_family": str(source.get("paper_family") or "").lower(),
            "question_number": source.get("question_number"),
            "evidence_packet_hash": hash_topic_routing_evidence_packet(packet),
            **_course_metadata_for_record(source),
        }
    )
    if parse_metadata.get("parse_recovered"):
        record["parse_recovered"] = True
        record["parse_recovery_source"] = parse_metadata.get("parse_recovery_source")
    return record


def build_topic_routing_error_record(
    packet: dict[str, Any],
    *,
    model: str,
    prompt_version: str,
    run_timestamp: str,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "primary_topic_id": None,
        "topic_distribution": [],
        "confidence": "low",
        "review_required": True,
        "review_reasons": [error_type],
        "evidence_used": sorted((packet.get("evidence") or {}).keys()),
        "error": {"type": error_type, "message": message},
        "llm_provider": "deepseek",
        "llm_model": model,
        "llm_prompt_version": prompt_version,
        "llm_run_timestamp": run_timestamp,
        "routing_source": "deepseek_topic_routing_error",
        "paper": packet.get("paper"),
        "paper_family": packet.get("paper_family"),
        "question_number": packet.get("question_number"),
        "evidence_packet_hash": hash_topic_routing_evidence_packet(packet),
        **_course_metadata_for_record(packet),
    }


def build_topic_routing_sidecar(
    records: dict[str, dict[str, Any]],
    *,
    taxonomy_path: str | Path,
    taxonomy_version_value: str | None,
    model: str,
    prompt_version: str,
    generated_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": TOPIC_ROUTING_SCHEMA_NAME,
        "schema_version": TOPIC_ROUTING_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_version": taxonomy_version_value,
        "model": model,
        "prompt_version": prompt_version,
        "course_contract": {
            "course_ids": list(COURSE_IDS),
            "courses": course_registry(),
            "routing_labels_are_advisory": True,
        },
        "records": dict(sorted(records.items())),
        "metadata": metadata or {},
    }


def _course_metadata_for_record(record: dict[str, Any]) -> dict[str, str | None]:
    course_id = course_id_for_record(record)
    return {
        "course_id": course_id,
        "component_name": component_name_for_course(course_id),
    }


def write_topic_routing_sidecar_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    return write_atomic_json(payload, path)


def load_topic_routing_sidecar_records(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_name") != TOPIC_ROUTING_SCHEMA_NAME:
        raise StartupConfigurationError(f"Existing sidecar is not a topic-routing sidecar: {path}")
    records = payload.get("records")
    if not isinstance(records, dict):
        return {}
    return {str(question_id): value for question_id, value in records.items() if isinstance(value, dict)}


def audit_topic_routing_sidecar(path: str | Path) -> dict[str, Any]:
    return audit_topic_routing_sidecar_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def audit_topic_routing_sidecar_payload(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records") if isinstance(payload.get("records"), dict) else {}
    counts = topic_progress_counts(records.values())
    failure_reasons = Counter(
        str(record["error"].get("type") or "unknown")
        for record in records.values()
        if isinstance(record, dict) and isinstance(record.get("error"), dict)
    )
    attempted = int(payload.get("metadata", {}).get("records_to_attempt_count") or counts["total_records"])
    return {
        "schema_name": payload.get("schema_name"),
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "total_records": counts["total_records"],
        "attempted_records": attempted,
        "successful_records": counts["successful_records"],
        "failed_records": counts["failed_records"],
        "review_required_records": counts["review_required_records"],
        "provider_failure_records": counts["provider_failure_records"],
        "strict_filter_records": counts["strict_filter_records"],
        "failures_by_reason": dict(sorted(failure_reasons.items())),
        "safe_for_strict_filters": counts["failed_records"] == 0 and counts["strict_filter_records"] > 0,
    }


def topic_progress_counts(records: Sequence[dict[str, Any]] | Any) -> Counter:
    counts: Counter = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        counts["total_records"] += 1
        if isinstance(record.get("error"), dict):
            counts["failed_records"] += 1
            if record["error"].get("type") == AI_FAILURE_PROVIDER_API_ERROR:
                counts["provider_failure_records"] += 1
        else:
            counts["successful_records"] += 1
        if record.get("review_required") is True:
            counts["review_required_records"] += 1
        if _topic_record_is_strict_filter_candidate(record):
            counts["strict_filter_records"] += 1
    return counts


def load_topic_routing_strict_filter_mappings(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = payload.get("records") if isinstance(payload.get("records"), dict) else {}
    strict: dict[str, list[dict[str, Any]]] = {}
    for question_id, record in records.items():
        if not isinstance(record, dict) or not _topic_record_is_strict_filter_candidate(record):
            continue
        strict[str(question_id)] = list(record.get("topic_distribution") or [])
    return strict


def _topic_record_is_strict_filter_candidate(record: dict[str, Any]) -> bool:
    return (
        not isinstance(record.get("error"), dict)
        and record.get("review_required") is False
        and record.get("confidence") in {"high", "medium"}
        and isinstance(record.get("primary_topic_id"), str)
        and bool(record.get("topic_distribution"))
    )


def append_topic_routing_failure_log(
    path: Path | None,
    *,
    question_id: str,
    error_type: str,
    error_message: str,
    model: str,
    run_timestamp: str,
    raw_provider_output: str | None,
    request_payload: dict[str, Any],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "question_id": question_id,
        "error_type": error_type,
        "error_message": error_message,
        "model": model,
        "run_timestamp": run_timestamp,
        "raw_provider_output": raw_provider_output,
        "request_payload": request_payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_all_topic_ids(taxonomy_root: str | Path) -> set[str]:
    topic_ids: set[str] = set()
    for component in ["p1", "p3", "m1", "s1"]:
        try:
            taxonomy = load_canonical_taxonomy(taxonomy_root, component)
        except StartupConfigurationError:
            continue
        topic_ids.update(taxonomy.topics)
    return topic_ids


def taxonomy_version(taxonomy_root: str | Path) -> str | None:
    root = Path(taxonomy_root)
    index_path = root / "indexes" / "topic_filter_map_index_v1.json"
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    generated_at = payload.get("generated_at")
    schema_version = payload.get("schema_version")
    if generated_at or schema_version:
        return f"topic_filter_map_index_v{schema_version or 'unknown'}:{generated_at or 'unknown'}"
    return None


def _evidence_packet_hashes_for_records(
    records: Sequence[dict[str, Any]],
    *,
    taxonomy_root: str | Path,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            continue
        hashes[question_id] = build_topic_routing_question_packet(
            record,
            taxonomy_root=taxonomy_root,
        ).evidence_packet_hash
    return hashes


def _resume_record_is_current(
    record: dict[str, Any] | None,
    *,
    model: str,
    prompt_version: str,
    evidence_packet_hash: str | None,
) -> bool:
    return (
        _resume_record_staleness_reason(
            record,
            model=model,
            prompt_version=prompt_version,
            evidence_packet_hash=evidence_packet_hash,
        )
        == "current"
    )


def _resume_record_staleness_reason(
    record: dict[str, Any] | None,
    *,
    model: str,
    prompt_version: str,
    evidence_packet_hash: str | None,
) -> str:
    if not isinstance(record, dict):
        return "missing_record"
    if isinstance(record.get("error"), dict):
        return "error_record"
    if record.get("routing_source") == "deterministic_review_gate":
        if record.get("llm_model") not in {None, model}:
            return "model_mismatch"
    elif record.get("llm_model") != model:
        return "model_mismatch"
    if record.get("llm_prompt_version") != prompt_version:
        return "prompt_version_mismatch"
    if not evidence_packet_hash:
        return "missing_expected_evidence_packet_hash"
    stored_hash = record.get("evidence_packet_hash")
    if not isinstance(stored_hash, str) or not stored_hash.strip():
        return "missing_evidence_packet_hash"
    if stored_hash != evidence_packet_hash:
        return "evidence_packet_hash_mismatch"
    return "current"


def _parse_question_ids(raw_values: Sequence[str] | None) -> list[str] | None:
    if not raw_values:
        return None
    question_ids: list[str] = []
    for value in raw_values:
        for part in value.split(","):
            cleaned = part.strip()
            if cleaned:
                question_ids.append(cleaned)
    return question_ids or None


def _batch_id(batch: Sequence[dict[str, Any]], *, index: int) -> str:
    if not batch:
        return f"empty_{index:04d}"
    paper = str(batch[0].get("paper") or "unknown")
    component = canonical_component_for_paper_family(batch[0].get("paper_family"))
    raw = f"{component}_{paper}_{index:04d}"
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw)
    digest = hashlib.sha256("|".join(str(record.get("question_id")) for record in batch).encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def _trusted_question_text(record: dict[str, Any], notes: dict[str, Any]) -> str:
    text_only_status = str(record.get("text_only_status") or notes.get("text_only_status") or "").strip().lower()
    trust = str(record.get("question_text_trust") or notes.get("question_text_trust") or "").strip().lower()
    role = str(record.get("question_text_role") or notes.get("question_text_role") or "").strip().lower()
    if text_only_status == "ready" and trust in {"high", "medium"} and role != "untrusted_math_text":
        return _clean_text(record.get("question_text"))
    return ""


def _trusted_ocr_text(record: dict[str, Any], notes: dict[str, Any], *, question_text: str) -> str:
    trust = str(record.get("ocr_text_trust") or notes.get("ocr_text_trust") or "").strip().lower()
    role = str(record.get("ocr_text_role") or notes.get("ocr_text_role") or "").strip().lower()
    text = _clean_text(record.get("ocr_text"))
    if trust not in {"high", "medium"} or role == "untrusted_math_text" or not text:
        return ""
    if question_text and _normalize_for_compare(text) == _normalize_for_compare(question_text):
        return ""
    return text


def _visual_evidence_insufficient(evidence: dict[str, Any]) -> bool:
    question_text = _clean_text(evidence.get("question_text") or evidence.get("ocr_text"))
    mark_scheme_text = _clean_text(evidence.get("mark_scheme_text"))
    combined = f"{question_text}\n{mark_scheme_text}".strip()
    if not combined:
        return True
    visual_cues = {"diagram", "figure", "sketch", "graph", "shown", "shaded", "region", "curve"}
    has_visual_cue = any(cue in combined.lower() for cue in visual_cues)
    has_math_topic_cue = any(
        cue in combined.lower()
        for cue in [
            "differentiat",
            "integrat",
            "binomial",
            "probability",
            "normal distribution",
            "vector",
            "complex",
            "momentum",
            "force",
            "trig",
            "quadratic",
        ]
    )
    return has_visual_cue and not has_math_topic_cue and len(combined) < 240


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _truncate(value: str, max_chars: int) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 20].rstrip() + " ... [truncated]"


def _normalize_for_compare(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TopicRouteValidationError(f"{field_name} must be an array.")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TopicRouteValidationError(f"{field_name}[{index}] must be a string.")
        cleaned = item.strip()
        if cleaned:
            strings.append(cleaned)
    return strings
