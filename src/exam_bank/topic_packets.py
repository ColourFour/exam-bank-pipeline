from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import fitz
from PIL import Image, ImageChops

from .atomic_json import write_atomic_json
from .deepseek_enrich import canonical_component_for_paper_family, load_question_bank


TOPIC_PACKET_SCHEMA_NAME = "exam_bank.topic_packets"
TOPIC_PACKET_SCHEMA_VERSION = 1
TOPIC_PACKET_SUMMARY_SCHEMA_NAME = "exam_bank.topic_packet_summary"
DEFAULT_QUESTION_BANK_PATH = Path("output/json/question_bank.json")
DEFAULT_TAXONOMY_PATH = Path("exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")
DEFAULT_CANONICAL_TAXONOMY_ROOT = Path("exam_bank_taxonomy/canonical")
DEFAULT_OUTPUT_ROOT = Path("output/topic_packets")
STRICT_REVIEW_STATUSES = {"reviewed", "high-confidence machine_candidate"}
STRICT_ASSIGNMENT_TYPES = {"primary_assessed", "secondary_assessed"}
REVIEWED_DECISION_ACTIONS = {"keep", "relabel", "exclude"}
REVIEWED_DECISION_MARKERS = {
    "keep": "Reviewed",
    "relabel": "Relabeled",
    "exclude": "Excluded from student version",
}
PDF_PROFILE_DEFAULTS: dict[str, dict[str, int | None]] = {
    "screen": {"image_dpi": 144, "jpeg_quality": 82, "max_image_width": 1600, "max_image_height": 2400},
    "print": {"image_dpi": 200, "jpeg_quality": 88, "max_image_width": 2200, "max_image_height": 3200},
    "archive": {"image_dpi": None, "jpeg_quality": 92, "max_image_width": None, "max_image_height": None},
}


class TopicPacketError(ValueError):
    pass


@dataclass(frozen=True)
class SubtopicRef:
    paper_family: str
    component_key: str
    topic_id: str
    topic_label: str
    subtopic_id: str
    subtopic_label: str
    packet_eligible: bool
    canonical_topic_id: str
    canonical_subtopic_id: str


@dataclass(frozen=True)
class Assignment:
    question_id: str
    paper_family: str
    topic_id: str
    topic_label: str
    subtopic_id: str
    subtopic_label: str
    source: str
    confidence: float | None
    trust_status: str
    strict_release_safe: bool
    review_reasons: tuple[str, ...]
    review_decision_action: str = ""
    review_status_marker: str = ""
    reviewed_topic: str = ""
    reviewed_subtopic: str = ""
    reviewed_skill: str = ""


@dataclass(frozen=True)
class TopicBankReviewDecision:
    question_id: str
    action: str
    reviewed_topic: str
    reviewed_subtopic: str
    reviewed_skill: str
    reason: str
    reviewer: str
    reviewed_at: str
    source: str


@dataclass(frozen=True)
class PacketKey:
    mode: str
    paper_family: str
    topic_id: str
    subtopic_id: str


@dataclass(frozen=True)
class PdfImageOptimizationOptions:
    enabled: bool
    profile: str
    image_dpi: int | None
    jpeg_quality: int
    max_image_width: int | None
    max_image_height: int | None


@dataclass(frozen=True)
class PdfLayoutOptions:
    page_size: str = "a4"
    orientation: str = "portrait"
    layout: str = "compact"
    answer_placement: str = "end"


@dataclass
class PdfWriteStats:
    path: str
    file_size_bytes: int
    image_count: int = 0
    original_source_image_bytes: int = 0
    optimized_embedded_image_bytes: int = 0
    downsampled_image_count: int = 0
    largest_source_images: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None
    layout_metadata: dict[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        ratio = None
        if self.original_source_image_bytes:
            ratio = round(self.optimized_embedded_image_bytes / self.original_source_image_bytes, 4)
        return {
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "image_count": self.image_count,
            "original_total_source_image_bytes": self.original_source_image_bytes,
            "optimized_embedded_image_bytes": self.optimized_embedded_image_bytes,
            "compression_ratio": ratio,
            "downsampled_image_count": self.downsampled_image_count,
            "largest_source_images": self.largest_source_images or [],
            "warnings": self.warnings or [],
        }


WARNING_TYPES = (
    "low_topic_confidence",
    "topic_uncertain",
    "degraded_text",
    "low_question_crop_confidence",
    "visual_review",
    "text_only_review",
    "text_only_fail",
    "missing_mark_scheme_image",
)


def add_topic_packet_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_QUESTION_BANK_PATH, help="Path to question_bank.json.")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH, help="CAIE 9709 packet taxonomy JSON.")
    parser.add_argument(
        "--canonical-taxonomy-root",
        type=Path,
        default=DEFAULT_CANONICAL_TAXONOMY_ROOT,
        help="Canonical taxonomy root used for reviewed/strict assignment sidecars.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Folder for generated packets.")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Root used to resolve relative image paths.")
    parser.add_argument(
        "--reviewed-decisions",
        type=Path,
        default=None,
        help="Optional reviewed topic-bank keep/relabel/exclude decision JSON.",
    )
    parser.add_argument("--paper-family", choices=["p1", "p3", "p4", "p5"], default=None)
    parser.add_argument("--topic", default=None, help="Packet taxonomy topic ID, e.g. integration.")
    parser.add_argument(
        "--subtopic",
        default=None,
        help="Experimental: packet taxonomy subtopic ID. Defaults to major-topic packet generation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum input records to scan after filters.")
    parser.add_argument("--dry-run", action="store_true", help="Write no PDFs or manifests; print/write summary only.")
    parser.add_argument("--strict-syllabus", action="store_true", help="Exclude and report records outside the packet taxonomy.")
    parser.add_argument("--include-review-required", action="store_true", help="Deprecated for major-topic packets; retained for compatibility.")
    parser.add_argument(
        "--enable-broad-topic-packets",
        action="store_true",
        help="Deprecated; major-topic packets are now the default.",
    )
    parser.add_argument("--major-topics-only", action="store_true", help="Generate major-topic packets only. This is the default.")
    parser.add_argument("--include-mapping-failures", action="store_true", help="Include records with mapping_status=fail.")
    parser.add_argument("--include-validation-failures", action="store_true", help="Include records with validation_status=fail.")
    parser.add_argument(
        "--pdf-profile",
        choices=["screen", "print", "archive"],
        default="print",
        help="Image optimization profile for generated packet PDFs.",
    )
    parser.add_argument("--page-size", choices=["a4", "letter"], default="a4", help="Page size for topic packet PDFs.")
    parser.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait", help="Page orientation.")
    parser.add_argument(
        "--layout",
        choices=["compact", "one-per-page"],
        default="compact",
        help="Topic packet layout. compact flows multiple blocks per page; one-per-page preserves the previous page-heavy behavior.",
    )
    parser.add_argument(
        "--answer-placement",
        choices=["end", "inline"],
        default="end",
        help="Place answers after all questions or inline after each problem.",
    )
    parser.add_argument("--image-dpi", type=int, default=None, help="Target image DPI metadata for optimized PDF copies.")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG quality for optimized PDF image copies.")
    parser.add_argument("--max-image-width", type=int, default=None, help="Maximum optimized PDF image width in pixels.")
    parser.add_argument("--max-image-height", type=int, default=None, help="Maximum optimized PDF image height in pixels.")
    parser.add_argument(
        "--no-image-optimization",
        action="store_true",
        help="Embed canonical images directly, preserving the previous topic packet PDF behavior.",
    )
    parser.add_argument(
        "--split-question-answer-pdfs",
        action="store_true",
        help="Also write legacy questions.pdf and answers.pdf files. The default output is topic_packet.pdf.",
    )


def run_topic_packets_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit < 0:
        raise TopicPacketError("--limit must be zero or greater.")
    return generate_topic_packets(
        question_bank_path=args.input,
        taxonomy_path=args.taxonomy,
        canonical_taxonomy_root=args.canonical_taxonomy_root,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        reviewed_decisions_path=args.reviewed_decisions,
        paper_family=args.paper_family,
        topic=args.topic,
        subtopic=args.subtopic,
        limit=args.limit,
        dry_run=bool(args.dry_run),
        strict_syllabus=bool(args.strict_syllabus),
        include_review_required=bool(args.include_review_required),
        enable_broad_topic_packets=bool(args.enable_broad_topic_packets),
        major_topics_only=bool(args.major_topics_only) or not bool(args.subtopic),
        include_mapping_failures=bool(args.include_mapping_failures),
        include_validation_failures=bool(args.include_validation_failures),
        pdf_profile=args.pdf_profile,
        page_size=args.page_size,
        orientation=args.orientation,
        layout=args.layout,
        answer_placement=args.answer_placement,
        image_dpi=args.image_dpi,
        jpeg_quality=args.jpeg_quality,
        max_image_width=args.max_image_width,
        max_image_height=args.max_image_height,
        image_optimization=not bool(args.no_image_optimization),
        split_question_answer_pdfs=bool(args.split_question_answer_pdfs),
    )


def generate_topic_packets(
    *,
    question_bank_path: str | Path,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    canonical_taxonomy_root: str | Path = DEFAULT_CANONICAL_TAXONOMY_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    artifact_root: str | Path | None = None,
    reviewed_decisions_path: str | Path | None = None,
    paper_family: str | None = None,
    topic: str | None = None,
    subtopic: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    strict_syllabus: bool = False,
    include_review_required: bool = False,
    enable_broad_topic_packets: bool = False,
    major_topics_only: bool = True,
    include_mapping_failures: bool = False,
    include_validation_failures: bool = False,
    pdf_profile: str = "print",
    page_size: str = "a4",
    orientation: str = "portrait",
    layout: str = "compact",
    answer_placement: str = "end",
    image_dpi: int | None = None,
    jpeg_quality: int | None = None,
    max_image_width: int | None = None,
    max_image_height: int | None = None,
    image_optimization: bool = True,
    split_question_answer_pdfs: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    taxonomy_path = Path(taxonomy_path)
    output_root = Path(output_root)
    records = load_question_bank(question_bank_path)
    all_records = list(records)
    taxonomy = load_packet_taxonomy(taxonomy_path)
    reviewed_decisions = load_topic_bank_reviewed_decisions(
        reviewed_decisions_path,
        records=all_records,
        taxonomy=taxonomy,
    )
    if paper_family:
        records = [r for r in records if _paper_family(r) == paper_family]
    if limit is not None:
        records = records[:limit]
    pdf_options = _pdf_image_optimization_options(
        profile=pdf_profile,
        image_dpi=image_dpi,
        jpeg_quality=jpeg_quality,
        max_image_width=max_image_width,
        max_image_height=max_image_height,
        enabled=image_optimization,
    )
    layout_options = _pdf_layout_options(
        page_size=page_size,
        orientation=orientation,
        layout=layout,
        answer_placement=answer_placement,
    )

    topic = _resolve_topic_filter(topic, paper_family, taxonomy)
    subtopic = _resolve_subtopic_filter(subtopic, paper_family, topic, taxonomy)
    if major_topics_only:
        subtopic = None
    assignments = load_assignment_index(canonical_taxonomy_root, taxonomy)
    artifact_root_path = Path(artifact_root) if artifact_root is not None else _default_artifact_root(question_bank_path)
    if _should_clean_output_root(
        dry_run=dry_run,
        paper_family=paper_family,
        topic=topic,
        subtopic=subtopic,
        limit=limit,
    ):
        shutil.rmtree(output_root, ignore_errors=True)
    elif _should_clean_paper_family_output(
        dry_run=dry_run,
        paper_family=paper_family,
        topic=topic,
        subtopic=subtopic,
        limit=limit,
    ):
        shutil.rmtree(output_root / str(paper_family), ignore_errors=True)
        shutil.rmtree(output_root / "review_required" / str(paper_family), ignore_errors=True)

    packets: dict[PacketKey, list[tuple[dict[str, Any], Assignment]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    missing_question_images: list[str] = []
    missing_answer_images: list[str] = []
    invalid_topics: list[str] = []
    mapping_failures: list[str] = []
    validation_failures: list[str] = []
    broad_topic_only: list[str] = []
    needs_precise_subtopic_review: list[str] = []
    quality_downgrades: list[dict[str, Any]] = []
    quality_downgrade_reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    applied_decision_counts: Counter[str] = Counter()

    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        decision = reviewed_decisions.get(question_id)
        record_family = _paper_family(record)
        if record_family not in {"p1", "p3", "p4", "p5"}:
            skipped.append(_skip(record, "invalid_paper_family"))
            continue
        reviewed_release_override = decision is not None and decision.action in {"keep", "relabel"}

        mapping_status = _status_value(record, "mapping_status")
        if mapping_status == "fail" and not include_mapping_failures and not reviewed_release_override:
            skipped.append(_skip(record, "mapping_status_fail"))
            mapping_failures.append(question_id)
            continue

        validation_status = _status_value(record, "validation_status")
        if validation_status == "fail" and not include_validation_failures and not reviewed_release_override:
            skipped.append(_skip(record, "validation_status_fail"))
            validation_failures.append(question_id)
            continue

        q_images = _question_image_paths(record)
        q_resolved = [
            _resolve_artifact_path(image_path, artifact_root_path, question_bank_path.parent)
            for image_path in q_images
        ]
        if not q_images or not all(path.is_file() for path in q_resolved):
            skipped.append(_skip(record, "missing_question_image", question_image_paths=q_images))
            missing_question_images.append(question_id)
            continue

        if major_topics_only:
            assignment, reasons = choose_major_topic_assignment(record, taxonomy)
        else:
            assignment, reasons = choose_assignment(
                record,
                assignments.get(question_id, []),
                taxonomy=taxonomy,
                include_review_required=include_review_required,
            )
        if assignment is None and decision is not None and decision.action == "relabel":
            assignment = _reviewed_relabel_seed_assignment(record, reasons)
        if assignment is None:
            skipped.append(_skip(record, reasons[0], reasons=reasons))
            if "invalid_major_topic" in reasons or "missing_topic" in reasons:
                invalid_topics.append(question_id)
            if "broad_topic_only" in reasons:
                broad_topic_only.append(question_id)
            if "needs_topic_review" in reasons or "needs_precise_subtopic_review" in reasons:
                needs_precise_subtopic_review.append(question_id)
            continue

        if decision is not None:
            assignment = apply_topic_bank_review_decision(
                assignment,
                decision,
                taxonomy=taxonomy,
                major_topics_only=major_topics_only,
            )
            applied_decision_counts.update([decision.action])

        if topic and assignment.topic_id != topic:
            skipped.append(
                _skip(
                    record,
                    "filtered_by_topic",
                    assigned_topic_id=assignment.topic_id,
                    requested_topic_id=topic,
                )
            )
            continue
        if subtopic and assignment.subtopic_id != subtopic:
            skipped.append(
                _skip(
                    record,
                    "filtered_by_subtopic",
                    assigned_topic_id=assignment.topic_id,
                    assigned_subtopic_id=assignment.subtopic_id,
                    requested_subtopic_id=subtopic,
                )
            )
            continue
        if not major_topics_only and not assignment.subtopic_id and not enable_broad_topic_packets:
            skipped.append(_skip(record, "broad_topic_only", reasons=["broad_topic_only", "broad_topic_packets_disabled"]))
            broad_topic_only.append(question_id)
            continue

        release_quality_reasons = _release_quality_reasons(record)
        if assignment.strict_release_safe and release_quality_reasons and not reviewed_release_override:
            assignment = _as_review_assignment(assignment, release_quality_reasons)
            quality_downgrades.append(
                _skip(
                    record,
                    "release_quality_gate_downgraded",
                    reasons=release_quality_reasons,
                    assigned_topic_id=assignment.topic_id,
                    assigned_subtopic_id=assignment.subtopic_id,
                )
            )
            quality_downgrade_reason_counts.update(release_quality_reasons)

        key = PacketKey(assignment_mode(assignment), assignment.paper_family, assignment.topic_id, assignment.subtopic_id)
        if not validate_packet_key(key, taxonomy):
            if strict_syllabus:
                raise TopicPacketError(f"Packet key outside allowed taxonomy: {key}")
            skipped.append(_skip(record, "invalid_taxonomy_path", topic_id=assignment.topic_id, subtopic_id=assignment.subtopic_id))
            invalid_topics.append(question_id)
            continue

        answer_paths = _answer_image_paths(record)
        answer_resolved = [
            _resolve_artifact_path(image_path, artifact_root_path, question_bank_path.parent)
            for image_path in answer_paths
        ]
        if not answer_paths or not all(path.is_file() for path in answer_resolved):
            missing_answer_images.append(question_id)
            warning_counts.update(["missing_mark_scheme_image"])
        warning_counts.update(_record_warnings(record))
        packets[key].append((record, assignment))

    generated: list[dict[str, Any]] = []
    generated_pdfs: list[str] = []
    empty_packets_skipped: list[dict[str, str]] = []

    for key in sorted(packets, key=lambda k: (k.mode, k.paper_family, k.topic_id, k.subtopic_id)):
        packet_records = sorted(packets[key], key=lambda item: _record_sort_key(item[0]))
        if not packet_records:
            empty_packets_skipped.append(key.__dict__)
            continue
        packet_dir = packet_output_dir(output_root, key)
        manifest = build_packet_manifest(
            key,
            packet_records,
            taxonomy=taxonomy,
            artifact_root=artifact_root_path,
            question_bank_path=question_bank_path,
            dry_run=dry_run,
            pdf_options=pdf_options,
            layout_options=layout_options,
        )
        if not dry_run:
            packet_dir.mkdir(parents=True, exist_ok=True)
            topic_pdf = packet_dir / "topic_packet.pdf"
            packet_stats = write_topic_packet_pdf(
                topic_pdf,
                packet_records,
                artifact_root_path,
                question_bank_path.parent,
                review=key.mode == "review",
                pdf_options=pdf_options,
                layout_options=layout_options,
            )
            page_count = _pdf_page_count(topic_pdf)
            manifest["pdf_path"] = str(topic_pdf)
            manifest["pdf_file_size_bytes"] = packet_stats.file_size_bytes
            manifest["pdf_profile"] = pdf_options.profile
            manifest["page_count"] = page_count
            _apply_layout_metadata_to_manifest(manifest, packet_stats.layout_metadata or {})
            manifest["pdf_outputs"] = {
                "topic_packet": packet_stats.to_manifest() | {"page_count": page_count},
            }
            generated_pdfs.append(str(topic_pdf))
            if split_question_answer_pdfs:
                question_pdf = packet_dir / "questions.pdf"
                answer_pdf = packet_dir / "answers.pdf"
                question_stats = write_packet_pdf(
                    question_pdf,
                    packet_records,
                    artifact_root_path,
                    question_bank_path.parent,
                    kind="questions",
                    review=key.mode == "review",
                    pdf_options=pdf_options,
                )
                answer_stats = write_packet_pdf(
                    answer_pdf,
                    packet_records,
                    artifact_root_path,
                    question_bank_path.parent,
                    kind="answers",
                    review=key.mode == "review",
                    pdf_options=pdf_options,
                )
                manifest["pdf_outputs"]["questions"] = question_stats.to_manifest()
                manifest["pdf_outputs"]["answers"] = answer_stats.to_manifest()
                generated_pdfs.extend([str(question_pdf), str(answer_pdf)])
            write_atomic_json(manifest, packet_dir / "manifest.json")
        else:
            packet_stats = None
            page_count = 0
        packet_summary = {
            "packet_mode": key.mode,
            "packet_level": "major_topic" if major_topics_only else "subtopic",
            "paper_family": key.paper_family,
            "topic_id": key.topic_id,
            "topic_label": manifest["topic_label"],
            "subtopic_id": key.subtopic_id,
            "problem_count": manifest["problem_count"],
            "question_count": manifest["question_count"],
            "answer_count": manifest["answer_count"],
            "missing_answer_count": manifest["missing_answer_count"],
            "output_dir": str(packet_dir),
            "pdf_path": manifest.get("pdf_path", str(packet_dir / "topic_packet.pdf")),
            "page_count": page_count,
            "pdf_image_optimization": manifest["pdf_image_optimization"],
            "page_size": manifest["page_size"],
            "orientation": manifest["orientation"],
            "layout": manifest["layout"],
            "answer_placement": manifest["answer_placement"],
            "questions_section_page_range": manifest.get("questions_section_page_range"),
            "answers_section_page_range": manifest.get("answers_section_page_range"),
            "problems_per_page_summary": manifest.get("problems_per_page_summary"),
            "average_problems_per_question_page": manifest.get("average_problems_per_question_page"),
            "average_answers_per_answer_page": manifest.get("average_answers_per_answer_page"),
            "oversized_block_warning_count": len(manifest.get("oversized_block_warnings") or []),
        }
        if packet_stats is not None:
            packet_summary.update(
                {
                    "pdf_file_size_bytes": packet_stats.file_size_bytes,
                    "pdf_compression_ratio": packet_stats.to_manifest()["compression_ratio"],
                }
            )
        generated.append(
            packet_summary
        )

    summary = build_summary(
        question_bank_path=question_bank_path,
        taxonomy_path=taxonomy_path,
        records_scanned=len(records),
        packets=generated,
        skipped=skipped,
        missing_question_images=missing_question_images,
        missing_answer_images=missing_answer_images,
        invalid_topics=invalid_topics,
        mapping_failures=mapping_failures,
        validation_failures=validation_failures,
        broad_topic_only=broad_topic_only,
        needs_precise_subtopic_review=needs_precise_subtopic_review,
        quality_downgrades=quality_downgrades,
        quality_downgrade_reason_counts=quality_downgrade_reason_counts,
        warning_counts=warning_counts,
        reviewed_decision_counts=Counter(decision.action for decision in reviewed_decisions.values()),
        applied_reviewed_decision_counts=applied_decision_counts,
        reviewed_decisions_path=Path(reviewed_decisions_path) if reviewed_decisions_path is not None else None,
        generated_pdfs=generated_pdfs,
        empty_packets_skipped=empty_packets_skipped,
        dry_run=dry_run,
    )
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        write_atomic_json(summary, output_root / "topic_packet_summary.json")
    return summary


def load_packet_taxonomy(path: str | Path) -> dict[str, Any]:
    taxonomy_path = Path(path)
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    components = payload.get("components")
    if not isinstance(components, list):
        raise TopicPacketError("Packet taxonomy must contain a components array.")
    subtopics: dict[tuple[str, str, str], SubtopicRef] = {}
    topics: dict[tuple[str, str], dict[str, Any]] = {}
    canonical_subtopic_to_ref: dict[str, SubtopicRef] = {}
    canonical_topic_to_topic: dict[str, tuple[str, str]] = {}
    topic_aliases: dict[tuple[str, str], str] = {}
    subtopic_aliases: dict[tuple[str, str], tuple[str, str]] = {}
    for component in components:
        family = str(component.get("paper_family", "")).strip().lower()
        component_key = str(component.get("component_key") or canonical_component_for_paper_family(family))
        if family not in {"p1", "p3", "p4", "p5"}:
            raise TopicPacketError(f"Unsupported paper family in taxonomy: {family}")
        for topic in component.get("topics", []):
            topic_id = str(topic.get("topic_id", "")).strip()
            topic_label = str(topic.get("topic_label", "")).strip()
            canonical_topic_id = str(topic.get("canonical_topic_id", "")).strip()
            if not topic_id or not topic_label:
                raise TopicPacketError(f"Invalid topic entry for {family}.")
            topics[(family, topic_id)] = {
                "paper_family": family,
                "component_key": component_key,
                "topic_id": topic_id,
                "topic_label": topic_label,
                "canonical_topic_id": canonical_topic_id,
            }
            topic_aliases[(family, topic_id)] = topic_id
            for alias in topic.get("aliases", []) + topic.get("deprecated_aliases", []):
                topic_aliases[(family, _slug(str(alias)))] = topic_id
            if canonical_topic_id:
                canonical_topic_to_topic[canonical_topic_id] = (family, topic_id)
            for subtopic in topic.get("subtopics", []):
                subtopic_id = str(subtopic.get("subtopic_id", "")).strip()
                subtopic_label = str(subtopic.get("subtopic_label", "")).strip()
                canonical_subtopic_id = str(subtopic.get("canonical_subtopic_id", "")).strip()
                if not subtopic_id or not subtopic_label:
                    raise TopicPacketError(f"Invalid subtopic entry for {family}/{topic_id}.")
                ref = SubtopicRef(
                    paper_family=family,
                    component_key=component_key,
                    topic_id=topic_id,
                    topic_label=topic_label,
                    subtopic_id=subtopic_id,
                    subtopic_label=subtopic_label,
                    packet_eligible=bool(subtopic.get("packet_eligible", True)),
                    canonical_topic_id=canonical_topic_id,
                    canonical_subtopic_id=canonical_subtopic_id,
                )
                subtopics[(family, topic_id, subtopic_id)] = ref
                subtopic_aliases[(family, subtopic_id)] = (topic_id, subtopic_id)
                for alias in subtopic.get("aliases", []):
                    subtopic_aliases[(family, _slug(str(alias)))] = (topic_id, subtopic_id)
                if canonical_subtopic_id:
                    canonical_subtopic_to_ref[canonical_subtopic_id] = ref
    return {
        "payload": payload,
        "topics": topics,
        "subtopics": subtopics,
        "canonical_subtopic_to_ref": canonical_subtopic_to_ref,
        "canonical_topic_to_topic": canonical_topic_to_topic,
        "topic_aliases": topic_aliases,
        "subtopic_aliases": subtopic_aliases,
    }


def load_assignment_index(canonical_root: str | Path, taxonomy: dict[str, Any]) -> dict[str, list[Assignment]]:
    root = Path(canonical_root)
    by_question: dict[str, list[Assignment]] = defaultdict(list)
    for component_key, paper_family in [("p1", "p1"), ("p3", "p3"), ("m1", "p4"), ("s1", "p5")]:
        path = root / "question_topic_assignments" / f"question_topic_assignments_9709_{component_key}_v1.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("assignments", []):
            question_id = str(item.get("question_id", "")).strip()
            for raw in item.get("topic_assignments", []):
                ref = taxonomy["canonical_subtopic_to_ref"].get(str(raw.get("subtopic_id", "")).strip())
                if ref is None:
                    continue
                strict_release_safe, review_reasons = _strict_assignment_status(raw)
                by_question[question_id].append(
                    Assignment(
                        question_id=question_id,
                        paper_family=paper_family,
                        topic_id=ref.topic_id,
                        topic_label=ref.topic_label,
                        subtopic_id=ref.subtopic_id,
                        subtopic_label=ref.subtopic_label,
                        source="canonical_question_topic_assignments",
                        confidence=_float_or_none(raw.get("confidence")),
                        trust_status=str(raw.get("review_status", "")).strip(),
                        strict_release_safe=strict_release_safe,
                        review_reasons=tuple(review_reasons),
                    )
                )
    return by_question


def load_topic_bank_reviewed_decisions(
    path: str | Path | None,
    *,
    records: Sequence[dict[str, Any]],
    taxonomy: dict[str, Any],
) -> dict[str, TopicBankReviewDecision]:
    if path is None:
        return {}
    decision_path = Path(path)
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        raise TopicPacketError("Reviewed topic-bank decisions must be a JSON object with records[] or a records array.")

    records_by_question = {str(record.get("question_id", "")).strip(): record for record in records}
    decisions: dict[str, TopicBankReviewDecision] = {}
    for index, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise TopicPacketError(f"Reviewed topic-bank decision #{index} must be an object.")
        question_id = str(raw.get("question_id", "")).strip()
        action = str(raw.get("action", "")).strip().lower()
        if not question_id:
            raise TopicPacketError(f"Reviewed topic-bank decision #{index} is missing question_id.")
        if question_id not in records_by_question:
            raise TopicPacketError(f"Reviewed topic-bank decision references unknown question_id: {question_id}")
        if question_id in decisions:
            raise TopicPacketError(f"Duplicate reviewed topic-bank decision for question_id: {question_id}")
        if action not in REVIEWED_DECISION_ACTIONS:
            raise TopicPacketError(f"Reviewed topic-bank decision for {question_id} has invalid action: {action}")
        source = str(raw.get("source", "")).strip()
        if source != "manual_review":
            raise TopicPacketError(f"Reviewed topic-bank decision for {question_id} must use source=manual_review.")

        reviewed_topic = str(raw.get("reviewed_topic", "")).strip()
        reviewed_subtopic = str(raw.get("reviewed_subtopic", "")).strip()
        if action == "relabel":
            if not reviewed_topic:
                raise TopicPacketError(f"Relabel decision for {question_id} is missing reviewed_topic.")
            family = _paper_family(records_by_question[question_id])
            topic_id = _resolve_reviewed_topic(reviewed_topic, family, taxonomy)
            if topic_id is None:
                raise TopicPacketError(f"Relabel decision for {question_id} uses unknown topic: {reviewed_topic}")
            if reviewed_subtopic and _resolve_reviewed_subtopic(reviewed_subtopic, family, topic_id, taxonomy) is None:
                raise TopicPacketError(
                    f"Relabel decision for {question_id} uses unknown subtopic for {topic_id}: {reviewed_subtopic}"
                )

        decisions[question_id] = TopicBankReviewDecision(
            question_id=question_id,
            action=action,
            reviewed_topic=reviewed_topic,
            reviewed_subtopic=str(raw.get("reviewed_subtopic", "")).strip(),
            reviewed_skill=str(raw.get("reviewed_skill", "")).strip(),
            reason=str(raw.get("reason", "")).strip(),
            reviewer=str(raw.get("reviewer", "")).strip(),
            reviewed_at=str(raw.get("reviewed_at", "")).strip(),
            source=source,
        )
    return decisions


def apply_topic_bank_review_decision(
    assignment: Assignment,
    decision: TopicBankReviewDecision,
    *,
    taxonomy: dict[str, Any],
    major_topics_only: bool = True,
) -> Assignment:
    marker = REVIEWED_DECISION_MARKERS[decision.action]
    if decision.action == "keep":
        return _with_review_decision(assignment, decision, marker)
    if decision.action == "exclude":
        return _with_review_decision(
            _as_review_assignment(assignment, ["topic_bank_reviewed_exclude"]),
            decision,
            marker,
        )

    topic_id = _resolve_reviewed_topic(decision.reviewed_topic, assignment.paper_family, taxonomy)
    if topic_id is None:
        raise TopicPacketError(f"Relabel decision for {decision.question_id} uses unknown topic: {decision.reviewed_topic}")
    topic_ref = taxonomy["topics"].get((assignment.paper_family, topic_id))
    if topic_ref is None:
        raise TopicPacketError(f"Relabel decision for {decision.question_id} uses unknown topic: {decision.reviewed_topic}")

    subtopic_id = ""
    subtopic_label = ""
    reviewed_subtopic = decision.reviewed_subtopic
    if reviewed_subtopic:
        subtopic_id = _resolve_reviewed_subtopic(reviewed_subtopic, assignment.paper_family, topic_id, taxonomy) or ""
        if not subtopic_id:
            raise TopicPacketError(
                f"Relabel decision for {decision.question_id} uses unknown subtopic for {topic_id}: {reviewed_subtopic}"
            )
        ref = taxonomy["subtopics"].get((assignment.paper_family, topic_id, subtopic_id))
        if ref is None:
            raise TopicPacketError(
                f"Relabel decision for {decision.question_id} uses unknown subtopic for {topic_id}: {reviewed_subtopic}"
            )
        subtopic_label = ref.subtopic_label
    if major_topics_only:
        subtopic_id = ""
        subtopic_label = ""

    return Assignment(
        question_id=assignment.question_id,
        paper_family=assignment.paper_family,
        topic_id=topic_id,
        topic_label=str(topic_ref["topic_label"]),
        subtopic_id=subtopic_id,
        subtopic_label=subtopic_label,
        source="reviewed_topic_bank_decision",
        confidence=1.0,
        trust_status="reviewed",
        strict_release_safe=True,
        review_reasons=tuple(reason for reason in assignment.review_reasons if reason not in {"needs_topic_review"}),
        review_decision_action=decision.action,
        review_status_marker=marker,
        reviewed_topic=decision.reviewed_topic,
        reviewed_subtopic=decision.reviewed_subtopic,
        reviewed_skill=decision.reviewed_skill,
    )


def choose_assignment(
    record: dict[str, Any],
    assignments: Sequence[Assignment],
    *,
    taxonomy: dict[str, Any],
    include_review_required: bool,
) -> tuple[Assignment | None, list[str]]:
    record_family = _paper_family(record)
    release = [a for a in assignments if a.paper_family == record_family and a.strict_release_safe]
    release_keys = {(a.topic_id, a.subtopic_id) for a in release}
    if len(release_keys) == 1:
        return sorted(release, key=lambda a: (a.confidence or 0), reverse=True)[0], []
    if len(release_keys) > 1:
        return None, ["ambiguous_topic_assignment", "needs_topic_review"]

    if include_review_required:
        review = [a for a in assignments if a.paper_family == record_family]
        review_keys = {(a.topic_id, a.subtopic_id) for a in review}
        if len(review_keys) == 1:
            candidate = sorted(review, key=lambda a: (a.confidence or 0), reverse=True)[0]
            return _as_review_assignment(candidate), []
        if len(review_keys) > 1:
            return None, ["ambiguous_topic_assignment", "needs_topic_review"]

    broad = _legacy_broad_topic_assignment(record, taxonomy)
    if broad is not None:
        return None, ["broad_topic_only", "needs_precise_subtopic_review"]
    return None, ["needs_topic_review"]


def choose_major_topic_assignment(record: dict[str, Any], taxonomy: dict[str, Any]) -> tuple[Assignment | None, list[str]]:
    family = _paper_family(record)
    raw_topic = str(record.get("topic") or "").strip()
    if not raw_topic:
        return None, ["missing_topic"]
    topic_id = _resolve_topic_for_family(raw_topic, family, taxonomy)
    topic_ref = taxonomy["topics"].get((family, topic_id))
    if topic_ref is None:
        return None, ["invalid_major_topic"]
    return (
        Assignment(
            question_id=str(record.get("question_id", "")).strip(),
            paper_family=family,
            topic_id=topic_id,
            topic_label=str(topic_ref["topic_label"]),
            subtopic_id="",
            subtopic_label="",
            source="question_bank_major_topic",
            confidence=_topic_confidence_score(record),
            trust_status=_status_value(record, "topic_trust_status"),
            strict_release_safe=True,
            review_reasons=tuple(_record_warnings(record)),
        ),
        [],
    )


def _reviewed_relabel_seed_assignment(record: dict[str, Any], reasons: Sequence[str]) -> Assignment:
    return Assignment(
        question_id=str(record.get("question_id", "")).strip(),
        paper_family=_paper_family(record),
        topic_id="",
        topic_label="",
        subtopic_id="",
        subtopic_label="",
        source="reviewed_topic_bank_decision_seed",
        confidence=_topic_confidence_score(record),
        trust_status="reviewed",
        strict_release_safe=True,
        review_reasons=tuple(reason for reason in reasons if reason != "needs_topic_review"),
    )


def validate_packet_key(key: PacketKey, taxonomy: dict[str, Any]) -> bool:
    if key.subtopic_id:
        ref = taxonomy["subtopics"].get((key.paper_family, key.topic_id, key.subtopic_id))
        return bool(ref and ref.packet_eligible)
    return (key.paper_family, key.topic_id) in taxonomy["topics"]


def _resolve_topic_filter(value: str | None, paper_family: str | None, taxonomy: dict[str, Any]) -> str | None:
    if not value:
        return None
    slug = _slug(value)
    families = [paper_family] if paper_family else ["p1", "p3", "p4", "p5"]
    matches = {
        taxonomy["topic_aliases"].get((family, slug))
        for family in families
        if taxonomy["topic_aliases"].get((family, slug))
    }
    if len(matches) == 1:
        return matches.pop()
    return slug


def _resolve_topic_for_family(value: str, paper_family: str, taxonomy: dict[str, Any]) -> str:
    slug = _slug(value)
    return taxonomy["topic_aliases"].get((paper_family, slug), slug)


def _resolve_reviewed_topic(value: str, paper_family: str, taxonomy: dict[str, Any]) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    canonical = taxonomy["canonical_topic_to_topic"].get(text)
    if canonical and canonical[0] == paper_family:
        return canonical[1]
    topic_id = _resolve_topic_for_family(text, paper_family, taxonomy)
    if (paper_family, topic_id) in taxonomy["topics"]:
        return topic_id
    return None


def _resolve_reviewed_subtopic(
    value: str,
    paper_family: str,
    topic_id: str,
    taxonomy: dict[str, Any],
) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    ref = taxonomy["canonical_subtopic_to_ref"].get(text)
    if ref and ref.paper_family == paper_family and ref.topic_id == topic_id:
        return ref.subtopic_id
    subtopic_id = _resolve_subtopic_filter(text, paper_family, topic_id, taxonomy)
    if (paper_family, topic_id, subtopic_id) in taxonomy["subtopics"]:
        return subtopic_id
    return None


def _resolve_subtopic_filter(
    value: str | None,
    paper_family: str | None,
    topic_id: str | None,
    taxonomy: dict[str, Any],
) -> str | None:
    if not value:
        return None
    slug = _slug(value)
    families = [paper_family] if paper_family else ["p1", "p3", "p4", "p5"]
    matches = {
        taxonomy["subtopic_aliases"].get((family, slug))
        for family in families
        if taxonomy["subtopic_aliases"].get((family, slug))
    }
    matches = {match for match in matches if match and (topic_id is None or match[0] == topic_id)}
    if len(matches) == 1:
        return matches.pop()[1]
    return slug


def build_packet_manifest(
    key: PacketKey,
    packet_records: Sequence[tuple[dict[str, Any], Assignment]],
    *,
    taxonomy: dict[str, Any],
    artifact_root: Path,
    question_bank_path: Path,
    dry_run: bool,
    pdf_options: PdfImageOptimizationOptions,
    layout_options: PdfLayoutOptions,
) -> dict[str, Any]:
    first_assignment = packet_records[0][1]
    included_ids = [str(record.get("question_id", "")) for record, _ in packet_records]
    included_records = [
        _included_record_manifest(
            index,
            record,
            assignment,
            artifact_root=artifact_root,
            fallback_root=question_bank_path.parent,
        )
        for index, (record, assignment) in enumerate(packet_records, start=1)
    ]
    missing_answer_ids = [str(item["question_id"]) for item in included_records if not item["answer_available"]]
    question_paths = [path for record, _ in packet_records for path in _question_image_paths(record)]
    answer_paths = [path for record, _ in packet_records for path in _answer_image_paths(record)]
    return {
        "schema_name": TOPIC_PACKET_SCHEMA_NAME,
        "schema_version": TOPIC_PACKET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "paper_family": key.paper_family,
        "topic_id": key.topic_id,
        "topic_label": first_assignment.topic_label,
        "packet_level": "major_topic" if not key.subtopic_id else "subtopic",
        "subtopic_id": key.subtopic_id or None,
        "subtopic_label": first_assignment.subtopic_label if key.subtopic_id else None,
        "packet_mode": key.mode,
        "problem_count": len(packet_records),
        "question_count": len(packet_records),
        "answer_count": len(packet_records) - len(missing_answer_ids),
        "missing_answer_count": len(missing_answer_ids),
        "skipped_count": 0,
        "pdf_path": "",
        "pdf_file_size_bytes": 0,
        "pdf_profile": pdf_options.profile,
        "page_size": layout_options.page_size,
        "orientation": layout_options.orientation,
        "layout": layout_options.layout,
        "answer_placement": layout_options.answer_placement,
        "page_count": 0,
        "questions_section_page_range": None,
        "answers_section_page_range": None,
        "problems_per_page_summary": {},
        "blocks_scaled_to_fit_count": 0,
        "oversized_block_warnings": [],
        "average_problems_per_question_page": 0,
        "average_answers_per_answer_page": 0,
        "included_records": included_records,
        "included_question_ids": included_ids,
        "skipped_question_ids": [],
        "skipped_questions": [],
        "skipped_records": [],
        "missing_answer_ids": missing_answer_ids,
        "source_image_paths": question_paths,
        "source_mark_scheme_image_paths": answer_paths,
        "source_mark_scheme_image_paths_present": [
            path
            for path in answer_paths
            if _resolve_artifact_path(path, artifact_root, question_bank_path.parent).is_file()
        ],
        "pdf_image_optimization": _pdf_options_manifest(pdf_options),
        "pdf_outputs": {},
        "warning_counts": _counts(warning for record, _ in packet_records for warning in _record_warnings(record)),
        "topic_assignment_source": _counts(a.source for _, a in packet_records),
        "topic_assignment_confidence_trust_status": [
            {
                "question_id": str(record.get("question_id", "")),
                "source": assignment.source,
                "confidence": assignment.confidence,
                "trust_status": assignment.trust_status,
                "strict_release_safe": assignment.strict_release_safe,
                "review_reasons": list(assignment.review_reasons),
            }
            for record, assignment in packet_records
        ],
        "warnings": ["review_only_watermark"] if key.mode == "review" else [],
    }


def _included_record_manifest(
    problem_number: int,
    record: dict[str, Any],
    assignment: Assignment,
    *,
    artifact_root: Path,
    fallback_root: Path,
) -> dict[str, Any]:
    question_paths = _question_image_paths(record)
    answer_paths = _answer_image_paths(record)
    answer_available = bool(answer_paths) and all(
        _resolve_artifact_path(path, artifact_root, fallback_root).is_file() for path in answer_paths
    )
    warnings = _record_warnings(record)
    if not answer_available:
        warnings = sorted(set(warnings + ["missing_mark_scheme_image"]))
    return {
        "problem_number": problem_number,
        "question_id": str(record.get("question_id", "")),
        "source_label": _source_label(record),
        "paper": str(record.get("paper") or ""),
        "source_paper_code": _source_paper_code(record),
        "question_number": str(record.get("question_number") or ""),
        "marks": _marks(record),
        "question_image_paths": question_paths,
        "mark_scheme_image_paths": answer_paths,
        "answer_available": answer_available,
        "warnings": warnings,
        "review_reasons": list(assignment.review_reasons),
        "review_status_marker": assignment.review_status_marker,
        "review_decision_action": assignment.review_decision_action,
        "reviewed_topic": assignment.reviewed_topic,
        "reviewed_subtopic": assignment.reviewed_subtopic,
        "reviewed_skill": assignment.reviewed_skill,
    }


def write_packet_pdf(
    output_path: Path,
    packet_records: Sequence[tuple[dict[str, Any], Assignment]],
    artifact_root: Path,
    fallback_root: Path,
    *,
    kind: str,
    review: bool,
    pdf_options: PdfImageOptimizationOptions | None = None,
) -> PdfWriteStats:
    pdf_options = pdf_options or _pdf_image_optimization_options(enabled=False)
    doc = fitz.open()
    image_stats: list[dict[str, Any]] = []
    warnings: list[str] = []
    for record, assignment in packet_records:
        image_value = _question_image_path(record) if kind == "questions" else _answer_image_path(record)
        image_path = _resolve_artifact_path(image_value, artifact_root, fallback_root)
        if kind == "answers" and not image_path.is_file():
            continue
        if not image_path.is_file():
            continue
        image_info = _prepare_pdf_image(image_path, pdf_options)
        image_stats.append(image_info)
        warnings.extend(image_info["warnings"])
        width = max(612.0, float(image_info["original_width"]) * 0.75 + 72.0)
        height = max(792.0, float(image_info["original_height"]) * 0.75 + 104.0)
        page = doc.new_page(width=width, height=height)
        header = _page_header(record, assignment)
        page.insert_text((36, 24), header, fontsize=8, color=(0.15, 0.15, 0.15))
        rect = fitz.Rect(36, 46, width - 36, height - 46)
        if image_info["stream"] is None:
            page.insert_image(rect, filename=str(image_path), keep_proportion=True)
        else:
            page.insert_image(rect, stream=image_info["stream"], keep_proportion=True)
        footer = f"{kind[:-1].title()} crop source: {Path(image_value).name if image_value else 'missing'}"
        page.insert_text((36, height - 24), footer, fontsize=7, color=(0.35, 0.35, 0.35))
        if review:
            page.insert_text(
                (width * 0.32, height * 0.52),
                "REVIEW ONLY",
                fontsize=46,
                color=(0.85, 0.1, 0.1),
                fill_opacity=0.22,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    actual_embedded_bytes = _pdf_embedded_image_stream_bytes(output_path)
    optimized_bytes = sum(int(item["optimized_bytes"]) for item in image_stats)
    stats = PdfWriteStats(
        path=str(output_path),
        file_size_bytes=output_path.stat().st_size,
        image_count=len(image_stats),
        original_source_image_bytes=sum(int(item["source_bytes"]) for item in image_stats),
        optimized_embedded_image_bytes=optimized_bytes if pdf_options.enabled else actual_embedded_bytes,
        downsampled_image_count=sum(1 for item in image_stats if item["downsampled"]),
        largest_source_images=_largest_source_images(image_stats),
        warnings=sorted(set(warnings)),
    )
    return stats


def write_topic_packet_pdf(
    output_path: Path,
    packet_records: Sequence[tuple[dict[str, Any], Assignment]],
    artifact_root: Path,
    fallback_root: Path,
    *,
    review: bool,
    pdf_options: PdfImageOptimizationOptions | None = None,
    layout_options: PdfLayoutOptions | None = None,
) -> PdfWriteStats:
    pdf_options = pdf_options or _pdf_image_optimization_options(enabled=False)
    layout_options = layout_options or PdfLayoutOptions()
    doc = fitz.open()
    image_stats: list[dict[str, Any]] = []
    warnings: list[str] = []

    layout_state = _write_flow_topic_packet_pdf(
        doc,
        packet_records,
        artifact_root,
        fallback_root,
        review=review,
        pdf_options=pdf_options,
        layout_options=layout_options,
        image_stats=image_stats,
        warnings=warnings,
    )

    _add_page_headers_and_numbers(doc, layout_state["page_sections"], layout_state["title"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    actual_embedded_bytes = _pdf_embedded_image_stream_bytes(output_path)
    optimized_bytes = sum(int(item["optimized_bytes"]) for item in image_stats)
    return PdfWriteStats(
        path=str(output_path),
        file_size_bytes=output_path.stat().st_size,
        image_count=len(image_stats),
        original_source_image_bytes=sum(int(item["source_bytes"]) for item in image_stats),
        optimized_embedded_image_bytes=optimized_bytes if pdf_options.enabled else actual_embedded_bytes,
        downsampled_image_count=sum(1 for item in image_stats if item["downsampled"]),
        largest_source_images=_largest_source_images(image_stats),
        warnings=sorted(set(warnings)),
        layout_metadata=layout_state,
    )


def _write_flow_topic_packet_pdf(
    doc: fitz.Document,
    packet_records: Sequence[tuple[dict[str, Any], Assignment]],
    artifact_root: Path,
    fallback_root: Path,
    *,
    review: bool,
    pdf_options: PdfImageOptimizationOptions,
    layout_options: PdfLayoutOptions,
    image_stats: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    width, height = _page_dimensions(layout_options)
    metrics = _layout_metrics(width, height)
    title = _packet_title(packet_records)
    page_sections: list[str] = []
    records_metadata: dict[int, dict[str, Any]] = defaultdict(dict)
    question_counts_by_page: Counter[int] = Counter()
    answer_counts_by_page: Counter[int] = Counter()
    oversized_warnings: list[str] = []
    blocks_scaled_to_fit_count = 0
    current_y = 0.0

    def new_page(section: str, *, heading: str | None = None) -> fitz.Page:
        nonlocal current_y
        page = doc.new_page(width=width, height=height)
        page_sections.append(section)
        current_y = metrics["content_top"]
        if heading:
            page.insert_text((metrics["left"], current_y + 2), heading, fontsize=15, color=(0.1, 0.1, 0.1))
            current_y += 30
        if review:
            page.insert_text(
                (width * 0.32, height * 0.52),
                "REVIEW ONLY",
                fontsize=46,
                color=(0.85, 0.1, 0.1),
                fill_opacity=0.22,
            )
        return page

    page = new_page("Questions", heading="Questions")

    question_blocks: list[dict[str, Any]] = []
    answer_blocks: list[dict[str, Any]] = []
    for problem_number, (record, assignment) in enumerate(packet_records, start=1):
        source_label = _source_label(record)
        marks = _marks(record)
        mark_text = f" - {marks} marks" if marks not in ("", None) else ""
        question_blocks.append(
            {
                "kind": "question",
                "problem_number": problem_number,
                "record": record,
                "assignment": assignment,
                "header": _problem_header(problem_number, source_label, mark_text, assignment),
                "image_values": _question_image_paths(record),
                "note": "",
            }
        )
        answer_blocks.append(
            {
                "kind": "answer",
                "problem_number": problem_number,
                "record": record,
                "assignment": assignment,
                "header": f"Answer to Problem {problem_number} - {source_label}",
                "image_values": _answer_image_paths(record),
                "note": "Answer unavailable: missing mark-scheme image",
            }
        )

    ordered_blocks = question_blocks
    if layout_options.answer_placement == "inline":
        ordered_blocks = [block for pair in zip(question_blocks, answer_blocks) for block in pair]

    for block in ordered_blocks:
        if block["kind"] == "answer" and layout_options.answer_placement == "end":
            continue
        page, current_y, scaled = _place_flow_block(
            doc,
            page,
            current_y,
            block,
            artifact_root,
            fallback_root,
            page_sections,
            metrics,
            layout_options,
            pdf_options,
            image_stats,
            warnings,
            oversized_warnings,
        )
        blocks_scaled_to_fit_count += int(scaled)
        start_page = page.number + 1
        problem_number = int(block["problem_number"])
        if block["kind"] == "question":
            records_metadata[problem_number]["question_start_page"] = start_page
            records_metadata[problem_number]["question_block_height_estimate"] = round(float(block.get("height_estimate") or 0), 2)
            question_counts_by_page[start_page] += 1
        else:
            records_metadata[problem_number]["answer_start_page"] = start_page
            records_metadata[problem_number]["answer_block_height_estimate"] = round(float(block.get("height_estimate") or 0), 2)
            answer_counts_by_page[start_page] += 1

    answers_start_page: int | None = None
    if layout_options.answer_placement == "end":
        page = new_page("Answers / Mark Schemes", heading="Answers / Mark Schemes")
        answers_start_page = page.number + 1
        for block in answer_blocks:
            page, current_y, scaled = _place_flow_block(
                doc,
                page,
                current_y,
                block,
                artifact_root,
                fallback_root,
                page_sections,
                metrics,
                layout_options,
                pdf_options,
                image_stats,
                warnings,
                oversized_warnings,
            )
            blocks_scaled_to_fit_count += int(scaled)
            start_page = page.number + 1
            problem_number = int(block["problem_number"])
            records_metadata[problem_number]["answer_start_page"] = start_page
            records_metadata[problem_number]["answer_block_height_estimate"] = round(float(block.get("height_estimate") or 0), 2)
            answer_counts_by_page[start_page] += 1

    question_pages = sorted(question_counts_by_page)
    answer_pages = sorted(answer_counts_by_page)
    questions_range = [min(question_pages), max(question_pages)] if question_pages else None
    if layout_options.answer_placement == "end" and answers_start_page is not None and answer_pages:
        answers_range = [answers_start_page, max(answer_pages)]
    elif answer_pages:
        answers_range = [min(answer_pages), max(answer_pages)]
    else:
        answers_range = None
    return {
        "title": title,
        "page_size": layout_options.page_size,
        "orientation": layout_options.orientation,
        "layout": layout_options.layout,
        "answer_placement": layout_options.answer_placement,
        "page_count": doc.page_count,
        "questions_section_page_range": questions_range,
        "answers_section_page_range": answers_range,
        "problems_per_page_summary": {
            "questions": {str(page): question_counts_by_page[page] for page in sorted(question_counts_by_page)},
            "answers": {str(page): answer_counts_by_page[page] for page in sorted(answer_counts_by_page)},
        },
        "average_problems_per_question_page": _rounded_average(question_counts_by_page.values()),
        "average_answers_per_answer_page": _rounded_average(answer_counts_by_page.values()),
        "blocks_scaled_to_fit_count": blocks_scaled_to_fit_count,
        "oversized_block_warnings": oversized_warnings,
        "records": dict(records_metadata),
        "page_sections": page_sections,
    }


def _append_image_page(
    doc: fitz.Document,
    *,
    image_path: Path,
    image_value: str,
    header: str,
    review: bool,
    pdf_options: PdfImageOptimizationOptions,
    image_stats: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    image_info = _prepare_pdf_image(image_path, pdf_options)
    image_stats.append(image_info)
    warnings.extend(image_info["warnings"])
    width = max(612.0, float(image_info["original_width"]) * 0.75 + 72.0)
    height = max(792.0, float(image_info["original_height"]) * 0.75 + 120.0)
    page = doc.new_page(width=width, height=height)
    page.insert_text((36, 28), header, fontsize=10, color=(0.1, 0.1, 0.1))
    rect = fitz.Rect(36, 56, width - 36, height - 58)
    if image_info["stream"] is None:
        page.insert_image(rect, filename=str(image_path), keep_proportion=True)
    else:
        page.insert_image(rect, stream=image_info["stream"], keep_proportion=True)
    page.insert_text(
        (36, height - 36),
        f"Crop source: {Path(image_value).name if image_value else 'missing'}",
        fontsize=7,
        color=(0.45, 0.45, 0.45),
    )
    if review:
        page.insert_text(
            (width * 0.32, height * 0.52),
            "REVIEW ONLY",
            fontsize=46,
            color=(0.85, 0.1, 0.1),
            fill_opacity=0.22,
        )


def _place_flow_block(
    doc: fitz.Document,
    page: fitz.Page,
    current_y: float,
    block: dict[str, Any],
    artifact_root: Path,
    fallback_root: Path,
    page_sections: list[str],
    metrics: dict[str, float],
    layout_options: PdfLayoutOptions,
    pdf_options: PdfImageOptimizationOptions,
    image_stats: list[dict[str, Any]],
    warnings: list[str],
    oversized_warnings: list[str],
) -> tuple[fitz.Page, float, bool]:
    section = "Questions" if block["kind"] == "question" else "Answers / Mark Schemes"
    prepared_images: list[dict[str, Any]] = []
    answer_available = True
    for image_value in block["image_values"]:
        image_path = _resolve_artifact_path(image_value, artifact_root, fallback_root)
        if not image_path.is_file():
            answer_available = False
            continue
        image_info = _prepare_pdf_image(image_path, pdf_options)
        image_info["image_value"] = image_value
        image_stats.append(image_info)
        warnings.extend(image_info["warnings"])
        prepared_images.append(image_info)

    if block["kind"] == "answer" and (not block["image_values"] or not answer_available or not prepared_images):
        warnings.append("missing_mark_scheme_image")
        prepared_images = []

    content_width = metrics["right"] - metrics["left"]
    natural_height = _flow_block_height(block, prepared_images, content_width, metrics)
    block["height_estimate"] = natural_height
    available_full = metrics["content_bottom"] - metrics["content_top"]
    starts_new_page = layout_options.layout == "one-per-page" and current_y > metrics["content_top"] + 35
    if not starts_new_page and current_y > metrics["content_top"] and natural_height > metrics["content_bottom"] - current_y:
        starts_new_page = True
    if starts_new_page:
        page = doc.new_page(width=page.rect.width, height=page.rect.height)
        page_sections.append(section)
        current_y = metrics["content_top"]

    render_scale = 1.0
    scaled = False
    if natural_height > available_full:
        fixed = metrics["block_header_height"] + metrics["block_bottom_gap"]
        image_gap_total = max(0, len(prepared_images) - 1) * metrics["image_gap"]
        image_height = sum(_rendered_image_height(item, content_width) for item in prepared_images)
        note_height = metrics["note_height"] if not prepared_images else 0
        scalable = max(0.0, image_height)
        target_scalable = max(1.0, available_full - fixed - image_gap_total - note_height)
        render_scale = min(1.0, target_scalable / scalable) if scalable else 1.0
        if render_scale < 1.0:
            scaled = True
        if render_scale < 0.55:
            warning = (
                f"oversized_block_scaled_below_legibility:"
                f"{block['kind']}:{block['problem_number']}:scale={render_scale:.2f}"
            )
            oversized_warnings.append(warning)
            warnings.append(warning)

    page.insert_textbox(
        fitz.Rect(metrics["left"], current_y, metrics["right"], current_y + metrics["block_header_height"]),
        str(block["header"]),
        fontsize=8.5,
        color=(0.1, 0.1, 0.1),
    )
    y = current_y + metrics["block_header_height"]
    if prepared_images:
        for image_info in prepared_images:
            render_width = _rendered_image_width(image_info, content_width) * render_scale
            natural_image_height = _rendered_image_height(image_info, content_width)
            render_height = natural_image_height * render_scale
            rect = fitz.Rect(metrics["left"], y, metrics["left"] + render_width, y + render_height)
            if image_info["stream"] is None:
                page.insert_image(rect, filename=str(image_info["path"]), keep_proportion=True)
            else:
                page.insert_image(rect, stream=image_info["stream"], keep_proportion=True)
            y += render_height + metrics["image_gap"]
    else:
        page.insert_textbox(
            fitz.Rect(metrics["left"], y + 4, metrics["right"], y + metrics["note_height"]),
            str(block["note"]),
            fontsize=9,
            color=(0.35, 0.15, 0.15),
        )
        y += metrics["note_height"]
    y += metrics["block_bottom_gap"]
    if layout_options.layout == "one-per-page":
        y = metrics["content_bottom"]
    return page, y, scaled


def _flow_block_height(
    block: dict[str, Any],
    prepared_images: Sequence[dict[str, Any]],
    content_width: float,
    metrics: dict[str, float],
) -> float:
    image_height = sum(_rendered_image_height(item, content_width) for item in prepared_images)
    image_gaps = max(0, len(prepared_images) - 1) * metrics["image_gap"]
    note_height = metrics["note_height"] if block["kind"] == "answer" and not prepared_images else 0
    return metrics["block_header_height"] + image_height + image_gaps + note_height + metrics["block_bottom_gap"]


def _rendered_image_height(image_info: dict[str, Any], content_width: float) -> float:
    width = max(1.0, float(image_info["original_width"]))
    height = max(1.0, float(image_info["original_height"]))
    return _rendered_image_width(image_info, content_width) * height / width


def _rendered_image_width(image_info: dict[str, Any], content_width: float) -> float:
    return min(content_width, max(1.0, float(image_info["original_width"])) * 0.75)


def _append_note_page(doc: fitz.Document, *, header: str, note: str, review: bool) -> None:
    width = 612.0
    height = 792.0
    page = doc.new_page(width=width, height=height)
    page.insert_text((36, 28), header, fontsize=10, color=(0.1, 0.1, 0.1))
    page.insert_textbox(
        fitz.Rect(36, 78, width - 36, 180),
        note,
        fontsize=12,
        color=(0.35, 0.15, 0.15),
    )
    if review:
        page.insert_text(
            (width * 0.32, height * 0.52),
            "REVIEW ONLY",
            fontsize=46,
            color=(0.85, 0.1, 0.1),
            fill_opacity=0.22,
        )


def _add_page_numbers(doc: fitz.Document) -> None:
    total = doc.page_count
    for index, page in enumerate(doc, start=1):
        footer = f"Page {index} of {total}"
        rect = page.rect
        page.insert_textbox(
            fitz.Rect(36, rect.height - 26, rect.width - 36, rect.height - 10),
            footer,
            fontsize=8,
            align=fitz.TEXT_ALIGN_CENTER,
            color=(0.35, 0.35, 0.35),
        )


def _add_page_headers_and_numbers(doc: fitz.Document, page_sections: Sequence[str], title: str) -> None:
    total = doc.page_count
    for index, page in enumerate(doc, start=1):
        rect = page.rect
        section = page_sections[index - 1] if index - 1 < len(page_sections) else title
        page.insert_textbox(
            fitz.Rect(36, 16, rect.width - 36, 30),
            f"{title} - {section}",
            fontsize=7.5,
            align=fitz.TEXT_ALIGN_CENTER,
            color=(0.35, 0.35, 0.35),
        )
        page.insert_textbox(
            fitz.Rect(36, rect.height - 28, rect.width - 36, rect.height - 12),
            f"Page {index} of {total}",
            fontsize=8,
            align=fitz.TEXT_ALIGN_CENTER,
            color=(0.35, 0.35, 0.35),
        )


def _pdf_page_count(path: Path) -> int:
    doc = fitz.open(path)
    try:
        return doc.page_count
    finally:
        doc.close()


def _pdf_image_optimization_options(
    *,
    profile: str = "print",
    image_dpi: int | None = None,
    jpeg_quality: int | None = None,
    max_image_width: int | None = None,
    max_image_height: int | None = None,
    enabled: bool = True,
) -> PdfImageOptimizationOptions:
    if profile not in PDF_PROFILE_DEFAULTS:
        raise TopicPacketError(f"Unsupported --pdf-profile: {profile}")
    defaults = PDF_PROFILE_DEFAULTS[profile]
    quality = int(jpeg_quality if jpeg_quality is not None else defaults["jpeg_quality"] or 88)
    if not 1 <= quality <= 95:
        raise TopicPacketError("--jpeg-quality must be between 1 and 95.")
    dpi = int(image_dpi) if image_dpi is not None else defaults["image_dpi"]
    if dpi is not None and dpi <= 0:
        raise TopicPacketError("--image-dpi must be greater than zero.")
    width = int(max_image_width) if max_image_width is not None else defaults["max_image_width"]
    height = int(max_image_height) if max_image_height is not None else defaults["max_image_height"]
    if width is not None and width <= 0:
        raise TopicPacketError("--max-image-width must be greater than zero.")
    if height is not None and height <= 0:
        raise TopicPacketError("--max-image-height must be greater than zero.")
    return PdfImageOptimizationOptions(
        enabled=enabled,
        profile=profile,
        image_dpi=dpi,
        jpeg_quality=quality,
        max_image_width=width,
        max_image_height=height,
    )


def _pdf_layout_options(
    *,
    page_size: str = "a4",
    orientation: str = "portrait",
    layout: str = "compact",
    answer_placement: str = "end",
) -> PdfLayoutOptions:
    if page_size not in {"a4", "letter"}:
        raise TopicPacketError(f"Unsupported --page-size: {page_size}")
    if orientation not in {"portrait", "landscape"}:
        raise TopicPacketError(f"Unsupported --orientation: {orientation}")
    if layout not in {"compact", "one-per-page"}:
        raise TopicPacketError(f"Unsupported --layout: {layout}")
    if answer_placement not in {"end", "inline"}:
        raise TopicPacketError(f"Unsupported --answer-placement: {answer_placement}")
    return PdfLayoutOptions(
        page_size=page_size,
        orientation=orientation,
        layout=layout,
        answer_placement=answer_placement,
    )


def _page_dimensions(options: PdfLayoutOptions) -> tuple[float, float]:
    sizes = {
        "a4": (595.2756, 841.8898),
        "letter": (612.0, 792.0),
    }
    width, height = sizes[options.page_size]
    if options.orientation == "landscape":
        return height, width
    return width, height


def _layout_metrics(width: float, height: float) -> dict[str, float]:
    left = 40.0
    right = width - 40.0
    return {
        "left": left,
        "right": right,
        "content_top": 46.0,
        "content_bottom": height - 46.0,
        "block_header_height": 18.0,
        "image_gap": 5.0,
        "block_bottom_gap": 11.0,
        "note_height": 32.0,
    }


def _packet_title(packet_records: Sequence[tuple[dict[str, Any], Assignment]]) -> str:
    if not packet_records:
        return "Topic Packet"
    assignment = packet_records[0][1]
    return f"{assignment.paper_family.upper()} {assignment.topic_label}"


def _problem_header(problem_number: int, source_label: str, mark_text: str, assignment: Assignment) -> str:
    marker = f" - {assignment.review_status_marker}" if assignment.review_status_marker else ""
    return f"Problem {problem_number} - {source_label}{mark_text} - {assignment.topic_label}{marker}"


def _rounded_average(values: Iterable[int]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return round(sum(values_list) / len(values_list), 2)


def _apply_layout_metadata_to_manifest(manifest: dict[str, Any], metadata: dict[str, Any]) -> None:
    for key in [
        "page_size",
        "orientation",
        "layout",
        "answer_placement",
        "page_count",
        "questions_section_page_range",
        "answers_section_page_range",
        "problems_per_page_summary",
        "blocks_scaled_to_fit_count",
        "oversized_block_warnings",
        "average_problems_per_question_page",
        "average_answers_per_answer_page",
    ]:
        if key in metadata:
            manifest[key] = metadata[key]
    record_metadata = metadata.get("records") if isinstance(metadata.get("records"), dict) else {}
    for item in manifest.get("included_records", []):
        problem_number = item.get("problem_number")
        extra = record_metadata.get(problem_number) or record_metadata.get(str(problem_number))
        if isinstance(extra, dict):
            item.update(extra)


def _pdf_options_manifest(options: PdfImageOptimizationOptions) -> dict[str, Any]:
    return {
        "enabled": options.enabled,
        "profile": options.profile,
        "image_dpi": options.image_dpi,
        "jpeg_quality": options.jpeg_quality,
        "max_image_width": options.max_image_width,
        "max_image_height": options.max_image_height,
    }


def _prepare_pdf_image(image_path: Path, options: PdfImageOptimizationOptions) -> dict[str, Any]:
    source_bytes = image_path.stat().st_size
    with Image.open(image_path) as original:
        original.load()
        original_width, original_height = original.size
        if not options.enabled:
            return {
                "path": str(image_path),
                "source_bytes": source_bytes,
                "optimized_bytes": source_bytes,
                "original_width": original_width,
                "original_height": original_height,
                "embedded_width": original_width,
                "embedded_height": original_height,
                "downsampled": False,
                "warnings": [],
                "stream": None,
            }
        image = _flatten_for_pdf(original)
        if _is_grayscale_image(image):
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        image, downsampled, scale = _downsample_for_pdf(image, options)
        stream = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": options.jpeg_quality, "optimize": True}
        if options.image_dpi is not None:
            save_kwargs["dpi"] = (options.image_dpi, options.image_dpi)
        image.save(stream, **save_kwargs)
        stream_bytes = stream.getvalue()
    warnings = []
    if downsampled and scale < 0.65:
        warnings.append(
            f"image_downsampled_heavily:{image_path.name}:{original_width}x{original_height}->{image.width}x{image.height}"
        )
    return {
        "path": str(image_path),
        "source_bytes": source_bytes,
        "optimized_bytes": len(stream_bytes),
        "original_width": original_width,
        "original_height": original_height,
        "embedded_width": image.width,
        "embedded_height": image.height,
        "downsampled": downsampled,
        "warnings": warnings,
        "stream": stream_bytes,
    }


def _flatten_for_pdf(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.copy()


def _is_grayscale_image(image: Image.Image) -> bool:
    if image.mode in {"1", "L"}:
        return True
    rgb = image.convert("RGB")
    sample = rgb.copy()
    sample.thumbnail((256, 256), Image.Resampling.LANCZOS)
    red, green, blue = sample.split()
    return not ImageChops.difference(red, green).getbbox() and not ImageChops.difference(red, blue).getbbox()


def _downsample_for_pdf(image: Image.Image, options: PdfImageOptimizationOptions) -> tuple[Image.Image, bool, float]:
    max_width = options.max_image_width
    max_height = options.max_image_height
    if max_width is None and max_height is None:
        return image, False, 1.0
    width_scale = (max_width / image.width) if max_width is not None and image.width > max_width else 1.0
    height_scale = (max_height / image.height) if max_height is not None and image.height > max_height else 1.0
    scale = min(width_scale, height_scale)
    if scale >= 1.0:
        return image, False, 1.0
    target_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(target_size, Image.Resampling.LANCZOS), True, scale


def _largest_source_images(image_stats: Sequence[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    largest = sorted(image_stats, key=lambda item: int(item["source_bytes"]), reverse=True)[:limit]
    return [
        {
            "path": item["path"],
            "source_bytes": item["source_bytes"],
            "original_width": item["original_width"],
            "original_height": item["original_height"],
            "embedded_width": item["embedded_width"],
            "embedded_height": item["embedded_height"],
            "downsampled": item["downsampled"],
        }
        for item in largest
    ]


def _pdf_embedded_image_stream_bytes(path: Path) -> int:
    doc = fitz.open(path)
    try:
        total = 0
        seen: set[int] = set()
        for page in doc:
            for image in page.get_images(full=True):
                xref = int(image[0])
                if xref in seen:
                    continue
                seen.add(xref)
                value = doc.xref_get_key(xref, "Length")[1]
                total += int(value or 0)
        return total
    finally:
        doc.close()


def build_summary(
    *,
    question_bank_path: Path,
    taxonomy_path: Path,
    records_scanned: int,
    packets: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    missing_question_images: Sequence[str],
    missing_answer_images: Sequence[str],
    invalid_topics: Sequence[str],
    mapping_failures: Sequence[str],
    validation_failures: Sequence[str],
    broad_topic_only: Sequence[str],
    needs_precise_subtopic_review: Sequence[str],
    quality_downgrades: Sequence[dict[str, Any]],
    quality_downgrade_reason_counts: Counter[str],
    warning_counts: Counter[str],
    reviewed_decision_counts: Counter[str],
    applied_reviewed_decision_counts: Counter[str],
    reviewed_decisions_path: Path | None,
    generated_pdfs: Sequence[str],
    empty_packets_skipped: Sequence[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    included_release = sum(p["question_count"] for p in packets if p["packet_mode"] == "release")
    included_review = sum(p["question_count"] for p in packets if p["packet_mode"] == "review")
    total_included = included_release + included_review
    skipped_by_reason = _counts(item.get("reason") for item in skipped)
    largest_pdfs = sorted(
        [
            {
                "pdf_path": p.get("pdf_path"),
                "file_size_bytes": int(p.get("pdf_file_size_bytes") or 0),
                "paper_family": p.get("paper_family"),
                "topic_id": p.get("topic_id"),
            }
            for p in packets
            if p.get("pdf_file_size_bytes")
        ],
        key=lambda item: item["file_size_bytes"],
        reverse=True,
    )[:10]
    largest_by_pages = max(packets, key=lambda p: int(p.get("page_count") or 0), default=None)
    largest_by_bytes = max(packets, key=lambda p: int(p.get("pdf_file_size_bytes") or 0), default=None)
    total_question_pages = sum(len((p.get("problems_per_page_summary") or {}).get("questions") or {}) for p in packets)
    total_answer_pages = sum(len((p.get("problems_per_page_summary") or {}).get("answers") or {}) for p in packets)
    return {
        "schema_name": TOPIC_PACKET_SUMMARY_SCHEMA_NAME,
        "schema_version": TOPIC_PACKET_SCHEMA_VERSION,
        "source_question_bank_path": str(question_bank_path),
        "source_taxonomy_path": str(taxonomy_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "total_records_scanned": records_scanned,
        "total_included": total_included,
        "total_included_in_release_packets": included_release,
        "total_included_in_review_packets": included_review,
        "total_skipped": len(skipped),
        "total_major_topic_packets_generated": sum(1 for p in packets if p.get("packet_level") == "major_topic"),
        "total_topic_pdfs_generated": len([p for p in packets if p.get("pdf_path")]),
        "total_problems_included": total_included,
        "total_pages_generated": sum(int(p.get("page_count") or 0) for p in packets),
        "total_pdf_bytes": sum(int(p.get("pdf_file_size_bytes") or 0) for p in packets),
        "average_problems_per_question_page": round(total_included / total_question_pages, 2) if total_question_pages else 0.0,
        "average_answers_per_answer_page": round(
            sum(int(p.get("answer_count") or 0) for p in packets) / total_answer_pages,
            2,
        )
        if total_answer_pages
        else 0.0,
        "largest_packet_by_pages": _summary_packet_ref(largest_by_pages),
        "largest_packet_by_bytes": _summary_packet_ref(largest_by_bytes),
        "oversized_block_warning_count": sum(int(p.get("oversized_block_warning_count") or 0) for p in packets),
        "largest_pdfs": largest_pdfs,
        "missing_answer_count": len(missing_answer_images),
        "packets_generated": list(packets),
        "empty_packets_skipped": list(empty_packets_skipped),
        "per_paper_family_included_counts": _counts(p["paper_family"] for p in packets for _ in range(int(p["question_count"]))),
        "per_topic_included_counts": _counts(
            f"{p['paper_family']}/{p['topic_id']}" for p in packets for _ in range(int(p["question_count"]))
        ),
        "skipped_by_reason": skipped_by_reason,
        "skipped_records_by_reason": skipped_by_reason,
        "warnings_by_type": {key: int(warning_counts.get(key, 0)) for key in WARNING_TYPES},
        "warning_counts_by_type": {key: int(warning_counts.get(key, 0)) for key in WARNING_TYPES},
        "missing_question_images": list(missing_question_images),
        "missing_mark_scheme_images": list(missing_answer_images),
        "records_with_missing_mark_schemes": list(missing_answer_images),
        "records_with_invalid_topics": list(invalid_topics),
        "records_with_mapping_failures": list(mapping_failures),
        "records_with_validation_failures": list(validation_failures),
        "records_with_unsafe_topic_assignment": [],
        "records_with_broad_topic_only_assignment": list(broad_topic_only),
        "records_needing_precise_subtopic_review": list(needs_precise_subtopic_review),
        "records_downgraded_to_review": list(quality_downgrades),
        "release_quality_downgrade_count": len(quality_downgrades),
        "release_quality_downgrade_reason_counts": dict(sorted(quality_downgrade_reason_counts.items())),
        "reviewed_decisions_path": str(reviewed_decisions_path) if reviewed_decisions_path else "",
        "reviewed_decisions_loaded": sum(reviewed_decision_counts.values()),
        "reviewed_decision_counts": {
            action: int(reviewed_decision_counts.get(action, 0))
            for action in sorted(REVIEWED_DECISION_ACTIONS)
        },
        "reviewed_decisions_applied": sum(applied_reviewed_decision_counts.values()),
        "applied_reviewed_decision_counts": {
            action: int(applied_reviewed_decision_counts.get(action, 0))
            for action in sorted(REVIEWED_DECISION_ACTIONS)
        },
        "per_paper_family_counts": _counts(p["paper_family"] for p in packets for _ in range(int(p["question_count"]))),
        "per_topic_counts": _counts(p["topic_id"] for p in packets for _ in range(int(p["question_count"]))),
        "per_subtopic_counts": _counts(p["subtopic_id"] or "broad_topic" for p in packets for _ in range(int(p["question_count"]))),
        "generated_pdfs": list(generated_pdfs),
        "skipped_records": list(skipped),
    }


def packet_output_dir(output_root: Path, key: PacketKey) -> Path:
    if key.mode == "review":
        base = output_root / "review_required" / key.paper_family / key.topic_id
    else:
        base = output_root / key.paper_family / key.topic_id
    return base / key.subtopic_id if key.subtopic_id else base


def _summary_packet_ref(packet: dict[str, Any] | None) -> dict[str, Any] | None:
    if packet is None:
        return None
    return {
        "pdf_path": packet.get("pdf_path"),
        "paper_family": packet.get("paper_family"),
        "topic_id": packet.get("topic_id"),
        "page_count": packet.get("page_count"),
        "file_size_bytes": packet.get("pdf_file_size_bytes"),
    }


def assignment_mode(assignment: Assignment) -> str:
    return "release" if assignment.strict_release_safe else "review"


def _strict_assignment_status(raw: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(raw.get("strict_filter_eligible")):
        reasons.append("strict_filter_eligible_false")
    if str(raw.get("assignment_type", "")).strip() not in STRICT_ASSIGNMENT_TYPES:
        reasons.append("assignment_type_not_release_safe")
    confidence = _float_or_none(raw.get("confidence")) or 0.0
    if confidence < 0.85:
        reasons.append("confidence_below_0_85")
    if str(raw.get("review_status", "")).strip() not in STRICT_REVIEW_STATUSES:
        reasons.append("review_status_not_release_safe")
    return not reasons, reasons


def _release_quality_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status_expectations = [
        ("mapping_status", "pass", "mapping_status_not_pass"),
        ("validation_status", "pass", "validation_status_not_pass"),
        ("scope_quality_status", "clean", "scope_quality_status_not_clean"),
        ("question_crop_confidence", "high", "question_crop_confidence_not_high"),
        ("visual_curation_status", "ready", "visual_curation_status_not_ready"),
    ]
    for key, expected, reason in status_expectations:
        if _status_value(record, key) != expected:
            reasons.append(reason)
    if _status_value(record, "text_only_status") == "fail":
        reasons.append("text_only_status_fail")
    return reasons


def _should_clean_output_root(
    *,
    dry_run: bool,
    paper_family: str | None,
    topic: str | None,
    subtopic: str | None,
    limit: int | None,
) -> bool:
    return not dry_run and paper_family is None and topic is None and subtopic is None and limit is None


def _should_clean_paper_family_output(
    *,
    dry_run: bool,
    paper_family: str | None,
    topic: str | None,
    subtopic: str | None,
    limit: int | None,
) -> bool:
    return not dry_run and paper_family is not None and topic is None and subtopic is None and limit is None


def _as_review_assignment(assignment: Assignment, review_reasons: Sequence[str] | None = None) -> Assignment:
    reasons = list(assignment.review_reasons)
    for reason in review_reasons or ("review_required",):
        if reason not in reasons:
            reasons.append(reason)
    return Assignment(
        question_id=assignment.question_id,
        paper_family=assignment.paper_family,
        topic_id=assignment.topic_id,
        topic_label=assignment.topic_label,
        subtopic_id=assignment.subtopic_id,
        subtopic_label=assignment.subtopic_label,
        source=assignment.source,
        confidence=assignment.confidence,
        trust_status=assignment.trust_status,
        strict_release_safe=False,
        review_reasons=tuple(reasons),
        review_decision_action=assignment.review_decision_action,
        review_status_marker=assignment.review_status_marker,
        reviewed_topic=assignment.reviewed_topic,
        reviewed_subtopic=assignment.reviewed_subtopic,
        reviewed_skill=assignment.reviewed_skill,
    )


def _with_review_decision(
    assignment: Assignment,
    decision: TopicBankReviewDecision,
    marker: str,
) -> Assignment:
    return Assignment(
        question_id=assignment.question_id,
        paper_family=assignment.paper_family,
        topic_id=assignment.topic_id,
        topic_label=assignment.topic_label,
        subtopic_id=assignment.subtopic_id,
        subtopic_label=assignment.subtopic_label,
        source=assignment.source,
        confidence=assignment.confidence,
        trust_status=assignment.trust_status,
        strict_release_safe=assignment.strict_release_safe,
        review_reasons=assignment.review_reasons,
        review_decision_action=decision.action,
        review_status_marker=marker,
        reviewed_topic=decision.reviewed_topic,
        reviewed_subtopic=decision.reviewed_subtopic,
        reviewed_skill=decision.reviewed_skill,
    )


def _legacy_broad_topic_assignment(record: dict[str, Any], taxonomy: dict[str, Any]) -> tuple[str, str] | None:
    family = _paper_family(record)
    legacy = _slug(str(record.get("topic", "")))
    if not legacy:
        return None
    for (topic_family, topic_id), _topic in taxonomy["topics"].items():
        if topic_family == family and topic_id == legacy:
            return topic_family, topic_id
    return None


def _question_image_path(record: dict[str, Any]) -> str:
    return _question_image_paths(record)[0] if _question_image_paths(record) else ""


def _answer_image_path(record: dict[str, Any]) -> str:
    return _answer_image_paths(record)[0] if _answer_image_paths(record) else ""


def _question_image_paths(record: dict[str, Any]) -> list[str]:
    return _unique_paths(
        _list_paths(record.get("question_image_paths"))
        + _list_paths(record.get("canonical_question_artifact"))
        + _list_paths(record.get("question_image_path"))
    )


def _answer_image_paths(record: dict[str, Any]) -> list[str]:
    return _unique_paths(_list_paths(record.get("mark_scheme_image_paths")) + _list_paths(record.get("mark_scheme_image_path")))


def _list_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _unique_paths(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None


def _paper_family(record: dict[str, Any]) -> str:
    family = str(record.get("paper_family", "")).strip().lower()
    if family in {"m1", "p4"}:
        return "p4"
    if family in {"s1", "p5"}:
        return "p5"
    return family


def _resolve_artifact_path(value: str, artifact_root: Path, fallback_root: Path) -> Path:
    if not value:
        return Path("")
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = artifact_root / path
    if candidate.exists():
        return candidate
    return fallback_root / path


def _default_artifact_root(question_bank_path: Path) -> Path:
    try:
        payload = json.loads(question_bank_path.read_text(encoding="utf-8"))
    except Exception:
        return question_bank_path.parent.parent
    manifest = payload.get("run_manifest") if isinstance(payload, dict) else None
    artifact_root = manifest.get("artifact_root") if isinstance(manifest, dict) else None
    return Path(artifact_root) if artifact_root else question_bank_path.parent.parent


def _page_header(record: dict[str, Any], assignment: Assignment) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    source = notes.get("source_pdf") or record.get("paper") or ""
    qno = record.get("question_number") or ""
    marks = record.get("question_solution_marks") or ""
    topic = f"{assignment.paper_family.upper()} / {assignment.topic_label}"
    if assignment.subtopic_label:
        topic = f"{topic} / {assignment.subtopic_label}"
    marker = f" | {assignment.review_status_marker}" if assignment.review_status_marker else ""
    return (
        f"{record.get('question_id', '')} | {source} | Q{qno} | "
        f"{marks} marks | {topic}{marker}"
    )


def _source_label(record: dict[str, Any]) -> str:
    year, month, source_paper_code = _source_label_parts(record)
    question_number = str(record.get("question_number") or "").strip()
    parts = []
    if year:
        parts.append(year)
    if month:
        parts.append(month)
    if source_paper_code:
        parts.append(f"P{source_paper_code}")
    label = " ".join(parts) or str(record.get("paper") or "Unknown source")
    if question_number:
        label = f"{label} Question {question_number}"
    return label


def _source_label_parts(record: dict[str, Any]) -> tuple[str, str, str]:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    paper = str(record.get("paper") or "").strip()
    source_paper_code = _source_paper_code(record)
    haystack = " ".join(
        str(value or "")
        for value in [
            paper,
            notes.get("source_pdf") if isinstance(notes, dict) else "",
            notes.get("source_paper") if isinstance(notes, dict) else "",
        ]
    )
    year = _source_year(haystack)
    month = _source_month(haystack)
    return year, month, source_paper_code


def _source_paper_code(record: dict[str, Any]) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    for value in [
        notes.get("source_paper_code") if isinstance(notes, dict) else "",
        record.get("source_paper_code"),
        record.get("paper_code"),
    ]:
        text = str(value or "").strip()
        if text:
            return text.upper().removeprefix("P")
    paper = str(record.get("paper") or "")
    match = re.search(r"\b(?:p)?([1-9]\d?)\s*(?:spring|summer|autumn|winter)\s*\d{2,4}\b", paper, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:9709[_\s-]?)?([1-9]\d?)\s*(?:m|s|w|n)\s*\d{2}\b", paper, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _source_year(text: str) -> str:
    match = re.search(r"(?:spring|summer|autumn|winter)\s*(\d{2,4})", text, re.IGNORECASE)
    if not match:
        match = re.search(r"\b(?:m|s|w|n)\s*(\d{2})\b", text, re.IGNORECASE)
    if not match:
        match = re.search(r"\b(20\d{2})\b", text)
    if not match:
        return ""
    year = int(match.group(1))
    if year < 100:
        year += 2000
    return str(year)


def _source_month(text: str) -> str:
    lowered = text.lower()
    season_months = {
        "spring": "March",
        "march": "March",
        "summer": "June",
        "june": "June",
        "autumn": "November",
        "winter": "November",
        "november": "November",
    }
    for token, month in season_months.items():
        if token in lowered:
            return month
    compact = re.search(r"\b(?:[1-9]\d?|9709[_\s-]?[1-9]\d?)\s*([mswn])\s*\d{2}\b", lowered)
    if compact:
        return {"m": "March", "s": "June", "w": "November", "n": "November"}[compact.group(1)]
    return ""


def _marks(record: dict[str, Any]) -> Any:
    for key in ["marks", "question_solution_marks", "total_marks", "question_marks"]:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return ""


def _skip(record: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "question_id": str(record.get("question_id", "")),
        "paper_family": _paper_family(record),
        "reason": reason,
    }
    payload.update(extra)
    return payload


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_value(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value in (None, ""):
        notes = record.get("notes")
        if isinstance(notes, dict):
            value = notes.get(key)
    return str(value or "").strip().lower()


def _topic_confidence_score(record: dict[str, Any]) -> float | None:
    value = _status_value(record, "topic_confidence")
    return {"high": 1.0, "medium": 0.65, "low": 0.3}.get(value, _float_or_none(value))


def _record_warnings(record: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _status_value(record, "topic_confidence") == "low":
        warnings.append("low_topic_confidence")
    if _bool_status(record, "topic_uncertain"):
        warnings.append("topic_uncertain")
    if _status_value(record, "topic_trust_status") in {"degraded_text", "review_required"}:
        warnings.append("degraded_text")
    if _status_value(record, "text_fidelity_status") == "degraded":
        warnings.append("degraded_text")
    if _status_value(record, "question_crop_confidence") == "low":
        warnings.append("low_question_crop_confidence")
    if _status_value(record, "visual_curation_status") == "review":
        warnings.append("visual_review")
    text_only_status = _status_value(record, "text_only_status")
    if text_only_status == "review":
        warnings.append("text_only_review")
    if text_only_status == "fail":
        warnings.append("text_only_fail")
    return sorted(set(warnings), key=warnings.index)


def _bool_status(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if value in (None, ""):
        notes = record.get("notes")
        if isinstance(notes, dict):
            value = notes.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    paper = str(record.get("paper") or "")
    return (
        _paper_family(record),
        _paper_chronology_key(paper),
        paper,
        _question_number_key(record.get("question_number")),
        str(record.get("question_id", "")),
    )


def _paper_chronology_key(paper: str) -> tuple[int, int, str]:
    match = re.search(r"(spring|summer|autumn|winter)\s*(\d{2,4})", paper.lower())
    if not match:
        compact = re.search(r"\b(?:[1-9]\d?|9709[_\s-]?[1-9]\d?)\s*([mswn])\s*(\d{2})\b", paper.lower())
        if compact:
            season = {"m": "spring", "s": "summer", "w": "autumn", "n": "autumn"}[compact.group(1)]
            year = int(compact.group(2)) + 2000
            season_order = {"spring": 1, "summer": 2, "autumn": 3}.get(season, 9)
            return (year, season_order, paper)
    if not match:
        return (9999, 99, paper)
    season = match.group(1) or ""
    year_text = match.group(2)
    year = int(year_text)
    if year < 100:
        year += 2000
    season_order = {"spring": 1, "summer": 2, "autumn": 3, "winter": 3}.get(season, 9)
    return (year, season_order, paper)


def _question_number_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    match = re.search(r"\d+", text)
    return (int(match.group(0)) if match else 9999, text)


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(value or "unknown") for value in values)
    return {key: counter[key] for key in sorted(counter)}


def _slug(value: str) -> str:
    value = re.sub(r"^9709_[a-z0-9]+_(topic|subtopic)_", "", value.strip().lower())
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")
