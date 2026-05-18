from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import re
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

    taxonomy = load_packet_taxonomy(taxonomy_path)
    topic = _resolve_topic_filter(topic, paper_family, taxonomy)
    subtopic = _resolve_subtopic_filter(subtopic, paper_family, topic, taxonomy)
    if major_topics_only:
        subtopic = None
    assignments = load_assignment_index(canonical_taxonomy_root, taxonomy)
    artifact_root_path = Path(artifact_root) if artifact_root is not None else _default_artifact_root(question_bank_path)

    packets: dict[PacketKey, list[tuple[dict[str, Any], Assignment]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    missing_question_images: list[str] = []
    missing_answer_images: list[str] = []
    invalid_topics: list[str] = []
    mapping_failures: list[str] = []
    validation_failures: list[str] = []
    broad_topic_only: list[str] = []
    needs_precise_subtopic_review: list[str] = []
    warning_counts: Counter[str] = Counter()

    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        record_family = _paper_family(record)
        if record_family not in {"p1", "p3", "p4", "p5"}:
            skipped.append(_skip(record, "invalid_paper_family"))
            continue

        mapping_status = _status_value(record, "mapping_status")
        if mapping_status == "fail" and not include_mapping_failures:
            skipped.append(_skip(record, "mapping_status_fail"))
            mapping_failures.append(question_id)
            continue

        validation_status = _status_value(record, "validation_status")
        if validation_status == "fail" and not include_validation_failures:
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
        if assignment is None:
            skipped.append(_skip(record, reasons[0], reasons=reasons))
            if "invalid_major_topic" in reasons or "missing_topic" in reasons:
                invalid_topics.append(question_id)
            if "broad_topic_only" in reasons:
                broad_topic_only.append(question_id)
            if "needs_topic_review" in reasons or "needs_precise_subtopic_review" in reasons:
                needs_precise_subtopic_review.append(question_id)
            continue

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
            )
            page_count = _pdf_page_count(topic_pdf)
            manifest["pdf_path"] = str(topic_pdf)
            manifest["pdf_file_size_bytes"] = packet_stats.file_size_bytes
            manifest["pdf_profile"] = pdf_options.profile
            manifest["page_count"] = page_count
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
        warning_counts=warning_counts,
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
        "page_count": 0,
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
) -> PdfWriteStats:
    pdf_options = pdf_options or _pdf_image_optimization_options(enabled=False)
    doc = fitz.open()
    image_stats: list[dict[str, Any]] = []
    warnings: list[str] = []

    for problem_number, (record, assignment) in enumerate(packet_records, start=1):
        source_label = _source_label(record)
        marks = _marks(record)
        mark_text = f" - {marks} marks" if marks not in ("", None) else ""
        problem_header = f"Problem {problem_number} - {source_label}{mark_text} - {assignment.topic_label}"
        for image_index, image_value in enumerate(_question_image_paths(record), start=1):
            image_path = _resolve_artifact_path(image_value, artifact_root, fallback_root)
            if not image_path.is_file():
                continue
            header = problem_header if image_index == 1 else f"{problem_header} (continued)"
            _append_image_page(
                doc,
                image_path=image_path,
                image_value=image_value,
                header=header,
                review=review,
                pdf_options=pdf_options,
                image_stats=image_stats,
                warnings=warnings,
            )

        answer_header = f"Answer to Problem {problem_number} - {source_label}"
        answer_paths = _answer_image_paths(record)
        answer_available = bool(answer_paths)
        for image_index, image_value in enumerate(answer_paths, start=1):
            image_path = _resolve_artifact_path(image_value, artifact_root, fallback_root)
            if not image_path.is_file():
                answer_available = False
                continue
            header = answer_header if image_index == 1 else f"{answer_header} (continued)"
            _append_image_page(
                doc,
                image_path=image_path,
                image_value=image_value,
                header=header,
                review=review,
                pdf_options=pdf_options,
                image_stats=image_stats,
                warnings=warnings,
            )
        if not answer_available:
            warnings.append("missing_mark_scheme_image")
            _append_note_page(
                doc,
                header=answer_header,
                note="Answer unavailable: missing mark-scheme image",
                review=review,
            )

    _add_page_numbers(doc)
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
    )


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
    warning_counts: Counter[str],
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


def _as_review_assignment(assignment: Assignment) -> Assignment:
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
        review_reasons=assignment.review_reasons or ("review_required",),
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
    return (
        f"{record.get('question_id', '')} | {source} | Q{qno} | "
        f"{marks} marks | {topic}"
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
