from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import fitz

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
    parser.add_argument("--subtopic", default=None, help="Packet taxonomy subtopic ID, e.g. integration_by_parts.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum input records to scan after filters.")
    parser.add_argument("--dry-run", action="store_true", help="Write no PDFs or manifests; print/write summary only.")
    parser.add_argument("--strict-syllabus", action="store_true", help="Fail if a packet key is outside the taxonomy.")
    parser.add_argument("--include-review-required", action="store_true", help="Also write watermarked review-only packets.")
    parser.add_argument(
        "--enable-broad-topic-packets",
        action="store_true",
        help="Allow broad topic packets when a record has no safe precise subtopic.",
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
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    taxonomy_path = Path(taxonomy_path)
    output_root = Path(output_root)
    records = load_question_bank(question_bank_path)
    if paper_family:
        records = [r for r in records if _paper_family(r) == paper_family]
    if limit is not None:
        records = records[:limit]

    taxonomy = load_packet_taxonomy(taxonomy_path)
    topic = _resolve_topic_filter(topic, paper_family, taxonomy)
    subtopic = _resolve_subtopic_filter(subtopic, paper_family, topic, taxonomy)
    assignments = load_assignment_index(canonical_taxonomy_root, taxonomy)
    artifact_root_path = Path(artifact_root) if artifact_root is not None else _default_artifact_root(question_bank_path)

    packets: dict[PacketKey, list[tuple[dict[str, Any], Assignment]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    missing_question_images: list[str] = []
    missing_answer_images: list[str] = []
    unsafe_topic_assignment: list[str] = []
    broad_topic_only: list[str] = []
    needs_precise_subtopic_review: list[str] = []

    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        q_image = _question_image_path(record)
        q_resolved = _resolve_artifact_path(q_image, artifact_root_path, question_bank_path.parent)
        if not q_image or not q_resolved.is_file():
            skipped.append(_skip(record, "missing_question_image", question_image_path=q_image))
            missing_question_images.append(question_id)
            continue

        assignment, reasons = choose_assignment(
            record,
            assignments.get(question_id, []),
            taxonomy=taxonomy,
            include_review_required=include_review_required,
        )
        if assignment is None:
            skipped.append(_skip(record, reasons[0], reasons=reasons))
            if "broad_topic_only" in reasons:
                broad_topic_only.append(question_id)
            if "needs_topic_review" in reasons or "needs_precise_subtopic_review" in reasons:
                needs_precise_subtopic_review.append(question_id)
            if any(reason.startswith("unsafe") or reason == "ambiguous_topic_assignment" for reason in reasons):
                unsafe_topic_assignment.append(question_id)
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
        if not assignment.subtopic_id and not enable_broad_topic_packets:
            skipped.append(_skip(record, "broad_topic_only", reasons=["broad_topic_only", "broad_topic_packets_disabled"]))
            broad_topic_only.append(question_id)
            continue

        key = PacketKey(assignment_mode(assignment), assignment.paper_family, assignment.topic_id, assignment.subtopic_id)
        if not validate_packet_key(key, taxonomy):
            if strict_syllabus:
                raise TopicPacketError(f"Packet key outside allowed taxonomy: {key}")
            skipped.append(_skip(record, "invalid_taxonomy_path", topic_id=assignment.topic_id, subtopic_id=assignment.subtopic_id))
            continue

        answer_path = _answer_image_path(record)
        answer_resolved = _resolve_artifact_path(answer_path, artifact_root_path, question_bank_path.parent)
        if not answer_path or not answer_resolved.is_file():
            missing_answer_images.append(question_id)
        packets[key].append((record, assignment))

    generated: list[dict[str, Any]] = []
    generated_pdfs: list[str] = []
    empty_packets_skipped: list[dict[str, str]] = []

    for key in sorted(packets, key=lambda k: (k.mode, k.paper_family, k.topic_id, k.subtopic_id)):
        packet_records = packets[key]
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
        )
        if not dry_run:
            packet_dir.mkdir(parents=True, exist_ok=True)
            question_pdf = packet_dir / "questions.pdf"
            answer_pdf = packet_dir / "answers.pdf"
            write_packet_pdf(question_pdf, packet_records, artifact_root_path, question_bank_path.parent, kind="questions", review=key.mode == "review")
            write_packet_pdf(answer_pdf, packet_records, artifact_root_path, question_bank_path.parent, kind="answers", review=key.mode == "review")
            write_atomic_json(manifest, packet_dir / "manifest.json")
            generated_pdfs.extend([str(question_pdf), str(answer_pdf)])
        generated.append(
            {
                "packet_mode": key.mode,
                "paper_family": key.paper_family,
                "topic_id": key.topic_id,
                "subtopic_id": key.subtopic_id,
                "question_count": manifest["question_count"],
                "answer_count": manifest["answer_count"],
                "output_dir": str(packet_dir),
            }
        )

    summary = build_summary(
        question_bank_path=question_bank_path,
        taxonomy_path=taxonomy_path,
        records_scanned=len(records),
        packets=generated,
        skipped=skipped,
        missing_question_images=missing_question_images,
        missing_answer_images=missing_answer_images,
        unsafe_topic_assignment=unsafe_topic_assignment,
        broad_topic_only=broad_topic_only,
        needs_precise_subtopic_review=needs_precise_subtopic_review,
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
) -> dict[str, Any]:
    first_assignment = packet_records[0][1]
    included_ids = [str(record.get("question_id", "")) for record, _ in packet_records]
    missing_answer_ids = [
        str(record.get("question_id", ""))
        for record, _ in packet_records
        if not _resolve_artifact_path(_answer_image_path(record), artifact_root, question_bank_path.parent).is_file()
    ]
    question_paths = [_question_image_path(record) for record, _ in packet_records]
    answer_paths = [_answer_image_path(record) for record, _ in packet_records]
    return {
        "schema_name": TOPIC_PACKET_SCHEMA_NAME,
        "schema_version": TOPIC_PACKET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "paper_family": key.paper_family,
        "topic_id": key.topic_id,
        "topic_label": first_assignment.topic_label,
        "subtopic_id": key.subtopic_id or None,
        "subtopic_label": first_assignment.subtopic_label if key.subtopic_id else None,
        "packet_mode": key.mode,
        "question_count": len(packet_records),
        "answer_count": len(packet_records) - len(missing_answer_ids),
        "skipped_count": 0,
        "included_question_ids": included_ids,
        "missing_answer_ids": missing_answer_ids,
        "source_image_paths": question_paths,
        "source_mark_scheme_image_paths": answer_paths,
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


def write_packet_pdf(
    output_path: Path,
    packet_records: Sequence[tuple[dict[str, Any], Assignment]],
    artifact_root: Path,
    fallback_root: Path,
    *,
    kind: str,
    review: bool,
) -> None:
    doc = fitz.open()
    for record, assignment in packet_records:
        image_value = _question_image_path(record) if kind == "questions" else _answer_image_path(record)
        image_path = _resolve_artifact_path(image_value, artifact_root, fallback_root)
        if kind == "answers" and not image_path.is_file():
            continue
        if not image_path.is_file():
            continue
        pix = fitz.Pixmap(str(image_path))
        width = max(612.0, float(pix.width) * 0.75 + 72.0)
        height = max(792.0, float(pix.height) * 0.75 + 104.0)
        page = doc.new_page(width=width, height=height)
        header = _page_header(record, assignment)
        page.insert_text((36, 24), header, fontsize=8, color=(0.15, 0.15, 0.15))
        rect = fitz.Rect(36, 46, width - 36, height - 46)
        page.insert_image(rect, filename=str(image_path), keep_proportion=True)
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


def build_summary(
    *,
    question_bank_path: Path,
    taxonomy_path: Path,
    records_scanned: int,
    packets: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    missing_question_images: Sequence[str],
    missing_answer_images: Sequence[str],
    unsafe_topic_assignment: Sequence[str],
    broad_topic_only: Sequence[str],
    needs_precise_subtopic_review: Sequence[str],
    generated_pdfs: Sequence[str],
    empty_packets_skipped: Sequence[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    included_release = sum(p["question_count"] for p in packets if p["packet_mode"] == "release")
    included_review = sum(p["question_count"] for p in packets if p["packet_mode"] == "review")
    return {
        "schema_name": TOPIC_PACKET_SUMMARY_SCHEMA_NAME,
        "schema_version": TOPIC_PACKET_SCHEMA_VERSION,
        "source_question_bank_path": str(question_bank_path),
        "source_taxonomy_path": str(taxonomy_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "total_records_scanned": records_scanned,
        "total_included_in_release_packets": included_release,
        "total_included_in_review_packets": included_review,
        "total_skipped": len(skipped),
        "packets_generated": list(packets),
        "empty_packets_skipped": list(empty_packets_skipped),
        "missing_question_images": list(missing_question_images),
        "missing_mark_scheme_images": list(missing_answer_images),
        "records_with_unsafe_topic_assignment": list(unsafe_topic_assignment),
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
    return base / key.subtopic_id if key.subtopic_id else base / "_broad_topic"


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
    return str(record.get("question_image_path") or record.get("canonical_question_artifact") or _first(record.get("question_image_paths")) or "")


def _answer_image_path(record: dict[str, Any]) -> str:
    return str(record.get("mark_scheme_image_path") or _first(record.get("mark_scheme_image_paths")) or "")


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
    return (
        f"{record.get('question_id', '')} | {source} | Q{qno} | "
        f"{marks} marks | {assignment.paper_family.upper()} / {assignment.topic_label} / {assignment.subtopic_label}"
    )


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


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(value or "unknown") for value in values)
    return {key: counter[key] for key in sorted(counter)}


def _slug(value: str) -> str:
    value = re.sub(r"^9709_[a-z0-9]+_(topic|subtopic)_", "", value.strip().lower())
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")
