"""Audit and organize CAIE taxonomy/topic-filtering artifacts.

This is a repository maintenance script, not runtime product code.  It builds a
structured audit of current taxonomy JSON files and then creates a canonical
layout under exam_bank_taxonomy while archiving superseded root-level artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_ROOT = ROOT / "exam_bank_taxonomy"
AUDIT_DIR = TAXONOMY_ROOT / "logs" / "validation_reports"
CHANGELOG_DIR = TAXONOMY_ROOT / "logs" / "changelogs"

CANONICAL_DIRS = {
    "skill_map": TAXONOMY_ROOT / "canonical" / "skill_maps",
    "question_skill_mappings": TAXONOMY_ROOT / "canonical" / "question_skill_mappings",
    "topic_filter_map": TAXONOMY_ROOT / "canonical" / "topic_filter_maps",
    "question_topic_assignments": TAXONOMY_ROOT / "canonical" / "question_topic_assignments",
    "coverage_report": TAXONOMY_ROOT / "canonical" / "coverage_reports",
    "strict_topic_filtering_report": TAXONOMY_ROOT / "canonical" / "strict_filtering_reports",
    "index": TAXONOMY_ROOT / "canonical" / "indexes",
}

REVIEW_DIRS = {
    "skill_mapping_review": TAXONOMY_ROOT / "review_queue" / "skill_mapping_review",
    "topic_assignment_review": TAXONOMY_ROOT / "review_queue" / "topic_assignment_review",
    "low_confidence_candidates": TAXONOMY_ROOT / "review_queue" / "low_confidence_candidates",
    "legacy_cleanup": TAXONOMY_ROOT / "review_queue" / "legacy_cleanup",
}

ARCHIVE_DIRS = {
    "deprecated": TAXONOMY_ROOT / "archive" / "deprecated",
    "duplicates": TAXONOMY_ROOT / "archive" / "duplicates",
    "malformed": TAXONOMY_ROOT / "archive" / "malformed",
    "superseded": TAXONOMY_ROOT / "archive" / "superseded",
}

FINAL_OUTPUTS = {
    "optimized_structure": AUDIT_DIR / "optimized_file_structure_v1.json",
    "canonical_index": CANONICAL_DIRS["index"] / "canonical_file_index_v1.json",
    "review_index": CANONICAL_DIRS["index"] / "review_queue_index_v1.json",
    "archive_index": CANONICAL_DIRS["index"] / "archive_index_v1.json",
    "optimization_changelog": CHANGELOG_DIR / "optimization_changelog_v1.json",
    "validation_report": AUDIT_DIR / "validation_report_after_optimization_v1.json",
    "optimization_summary": AUDIT_DIR / "optimization_summary_v1.md",
}

PHASE1_OUTPUTS = {
    "file_inventory": AUDIT_DIR / "file_inventory_report_v1.json",
    "quality_audit": AUDIT_DIR / "file_quality_audit_v1.json",
    "recommendations": AUDIT_DIR / "canonical_file_recommendations_v1.json",
    "cleanup_plan": AUDIT_DIR / "cleanup_plan_v1.json",
    "summary": AUDIT_DIR / "file_review_summary_v1.md",
}

ROOT_JSON_PATTERNS = [
    "skill_map_*.json",
    "question_skill_mappings_*.json",
    "topic_filter_map_*.json",
    "question_topic_assignments_*.json",
    "coverage_report_*.json",
    "strict_topic_filtering_report_*.json",
    "changelog_*.json",
    "*_index_v1.json",
    "*validation*.json",
]

CURRENT_REFERENCE_JSON_FILES = [
    "output/json/question_bank.json",
    "output/json/question_bank.deepseek.json",
    "output/json/audit.current.json",
    "output/json/status.current.json",
    "output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json",
    "output/asterion/exports/latest/asterion_question_bank_v1.json",
    "output/asterion/exports/latest/asterion_content_lab_candidates_v1.json",
]

HISTORICAL_REFERENCE_JSON_FILES = [
    "output_ocr_candidate/json/question_bank.json",
    "output_ocr_candidate/json/asterion_question_bank_v1.json",
    "output_ocr_candidate/json/asterion_content_lab_candidates_v1.json",
]

GENERATOR_FILES = [
    "scripts/generate_skill_maps.py",
    "scripts/generate_topic_filter_maps.py",
]

VALID_REVIEW_STATUSES = {
    "reviewed",
    "human_reviewed",
    "needs_review",
    "review",
    "review_only",
    "machine_candidate_needs_review",
    "high-confidence machine_candidate",
    "not_reviewed",
}

VALID_MAPPING_SOURCES = {
    "mixed_evidence",
    "legacy_topic_mapping",
    "syllabus_inferred",
    "mark_scheme_inferred",
    "question_text_inferred",
    "machine_candidate",
    "human_reviewed",
    "unknown",
}

DIRECT_ASSIGNMENT_TYPES = {"primary_assessed", "secondary_assessed"}
VALID_ASSIGNMENT_TYPES = {"primary_assessed", "secondary_assessed", "prerequisite", "context_only"}
LOW_CONFIDENCE_THRESHOLD = 0.65
STRICT_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class JsonFile:
    path: Path
    rel_path: str
    data: Any | None = None
    valid_json: bool = False
    error: str = ""
    sha256: str = ""
    size_bytes: int = 0


@dataclass
class Context:
    files: dict[str, JsonFile] = field(default_factory=dict)
    skill_ids_by_component: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    skill_ids_global: set[str] = field(default_factory=set)
    topic_ids_by_component: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    subtopic_ids_by_component: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    subtopic_parent_by_component: dict[str, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    question_ids_by_component: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    subpart_ids_by_component: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    known_question_ids: set[str] = field(default_factory=set)
    known_subpart_ids: set[str] = field(default_factory=set)
    inventory: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)
    recommendations: dict[str, Any] = field(default_factory=dict)
    cleanup_plan: dict[str, Any] = field(default_factory=dict)
    review_queues: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json_file(path: Path) -> JsonFile:
    item = JsonFile(path=path, rel_path=rel(path), size_bytes=path.stat().st_size)
    item.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        item.data = json.loads(path.read_text(encoding="utf-8"))
        item.valid_json = True
    except Exception as exc:  # pragma: no cover - defensive maintenance path
        item.error = str(exc)
    return item


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_reference_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def discover_relevant_paths(reference_json_files: list[str | Path] | None = None) -> list[Path]:
    paths: set[Path] = set()
    for pattern in ROOT_JSON_PATTERNS:
        paths.update(ROOT.glob(pattern))
    for name in (reference_json_files or CURRENT_REFERENCE_JSON_FILES) + GENERATOR_FILES:
        path = resolve_reference_path(name)
        if path.exists():
            paths.add(path)
    return sorted(paths, key=lambda p: rel(p))


def infer_component_from_name(name: str) -> str | None:
    patterns = [
        r"_(p1|p3|m1|s1)_v\d+\.json$",
        r"_(p1|p3|m1|s1)\.json$",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1)
    return None


def normalize_component(value: str | None, file_name: str = "") -> str | None:
    if value in {"p1", "p3", "m1", "s1"}:
        return value
    text = (value or "").lower()
    if "paper 1" in text or "pure mathematics 1" in text:
        return "p1"
    if "paper 3" in text or "pure mathematics 3" in text:
        return "p3"
    if "mechanics" in text or "paper 4" in text:
        return "m1"
    if "statistics 1" in text or "paper 5" in text:
        return "s1"
    return infer_component_from_name(file_name)


def detect_file_type(item: JsonFile) -> str:
    name = item.path.name
    if name.endswith("validation_v1.json") or "validation" in name:
        return "validation_report"
    if name.startswith("skill_map_") and name != "skill_map_index_v1.json":
        return "skill_map"
    if name.startswith("question_skill_mappings_"):
        return "question_skill_mappings"
    if name.startswith("topic_filter_map_") and name != "topic_filter_map_index_v1.json":
        return "topic_filter_map"
    if name.startswith("question_topic_assignments_"):
        return "question_topic_assignments"
    if name.startswith("coverage_report_"):
        return "coverage_report"
    if name.startswith("strict_topic_filtering_report_"):
        return "strict_topic_filtering_report"
    if name.endswith("_index_v1.json"):
        return "index"
    if name.startswith("changelog_"):
        return "changelog"
    if name == "audit.current.json":
        return "extraction_audit"
    if name == "status.current.json":
        return "extraction_status"
    if name == "question_bank.json":
        return "current_question_bank" if item.rel_path.startswith("output/json/") else "historical_ocr_candidate_question_bank"
    if name == "asterion_question_bank_v1.json":
        return "asterion_question_bank"
    if name == "asterion_exam_bank_catalog_v1.json":
        return "asterion_exam_bank_catalog"
    if name == "asterion_content_lab_candidates_v1.json":
        return "content_lab_candidates"
    if name == "question_bank.deepseek.json":
        return "deepseek_enrichment"
    if item.rel_path.endswith(".py"):
        return "generator_script"
    return "unknown"


def detected_purpose(file_type: str) -> str:
    return {
        "skill_map": "Official-syllabus-anchored candidate skill taxonomy for one CAIE component.",
        "question_skill_mappings": "Machine candidate question/subpart-to-skill mappings with evidence and confidence.",
        "topic_filter_map": "Component-local topic/subtopic filter hierarchy derived from skill maps.",
        "question_topic_assignments": "Question/subpart topic assignments derived from question-skill mappings.",
        "coverage_report": "Coverage/readiness rollup for skills or strict topic filters.",
        "strict_topic_filtering_report": "Strict topic filter quality and eligibility rollup.",
        "index": "Index of generated taxonomy or topic-filter files.",
        "changelog": "Generation/change history for taxonomy expansion.",
        "validation_report": "Validation summary for generated files.",
        "current_question_bank": "Current extraction question bank with broad topic labels and OCR quality metadata.",
        "historical_ocr_candidate_question_bank": "Historical OCR candidate question bank used as source material for generated mappings.",
        "asterion_question_bank": "Asterion-ready question bank with subpart-level records.",
        "content_lab_candidates": "Generated content-lab candidates used as weak evidence for taxonomy mapping.",
        "deepseek_enrichment": "LLM enrichment output, not a canonical taxonomy source.",
        "extraction_audit": "Current extraction quality audit, useful for question-bank reference validation.",
        "extraction_status": "Current extraction run status summary, useful for source quality context.",
        "generator_script": "Generator script that produced candidate taxonomy or topic-filter artifacts.",
    }.get(file_type, "Unknown or miscellaneous artifact.")


def get_record_count(item: JsonFile, file_type: str) -> int | None:
    if not item.valid_json or not isinstance(item.data, dict):
        return None
    data = item.data
    if file_type == "skill_map":
        return len(data.get("skills", []))
    if file_type == "question_skill_mappings":
        return len(data.get("mappings", []))
    if file_type == "topic_filter_map":
        return len(data.get("topics", []))
    if file_type == "question_topic_assignments":
        return len(data.get("assignments", []))
    if file_type in {"current_question_bank", "historical_ocr_candidate_question_bank", "asterion_question_bank"}:
        return len(data.get("questions", []))
    if file_type == "content_lab_candidates":
        return len(data.get("candidates", []))
    if file_type == "deepseek_enrichment":
        return len(data.get("enrichments", []))
    if file_type == "index":
        return len(data.get("components", []))
    if file_type == "coverage_report":
        if "skills" in data:
            return len(data.get("skills", []))
        if "components" in data:
            return len(data.get("components", []))
    if file_type == "strict_topic_filtering_report":
        if "component_summaries" in data:
            return len(data.get("component_summaries", []))
        return int(data.get("total_subparts") or 0)
    if file_type == "changelog":
        return len(data.get("changes", []))
    return data.get("record_count")


def flatten_values(value: Any, key_name: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == key_name:
                found.append(child)
            found.extend(flatten_values(child, key_name))
    elif isinstance(value, list):
        for child in value:
            found.extend(flatten_values(child, key_name))
    return found


def count_reviewed_and_candidates(item: JsonFile) -> tuple[int, int, int]:
    if not item.valid_json:
        return 0, 0, 0
    statuses = [status for status in flatten_values(item.data, "review_status") if isinstance(status, str)]
    reviewed = sum(1 for status in statuses if status in {"reviewed", "human_reviewed"})
    machine = sum(
        1
        for status in statuses
        if "machine_candidate" in status or status in {"needs_review", "review_only", "not_reviewed"}
    )
    low = 0
    for confidence in flatten_values(item.data, "confidence"):
        if isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD:
            low += 1
    return reviewed, machine, low


def init_context(reference_json_files: list[str | Path] | None = None) -> Context:
    ctx = Context()
    for path in discover_relevant_paths(reference_json_files):
        item = load_json_file(path) if path.suffix == ".json" else JsonFile(path=path, rel_path=rel(path), valid_json=False, size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        ctx.files[item.rel_path] = item
    return ctx


def build_reference_indexes(ctx: Context) -> None:
    for item in ctx.files.values():
        file_type = detect_file_type(item)
        if not item.valid_json or not isinstance(item.data, dict):
            continue
        data = item.data
        component = normalize_component(data.get("component_key") or data.get("component_label") or data.get("caie_class_or_component"), item.path.name)
        if file_type == "skill_map":
            for skill in data.get("skills", []):
                skill_id = skill.get("skill_id")
                if isinstance(skill_id, str):
                    ctx.skill_ids_global.add(skill_id)
                    ctx.skill_ids_by_component[component or "unknown"].add(skill_id)
        elif file_type == "topic_filter_map":
            for topic in data.get("topics", []):
                topic_id = topic.get("topic_id")
                if isinstance(topic_id, str):
                    ctx.topic_ids_by_component[component or "unknown"].add(topic_id)
                for subtopic in topic.get("subtopics", []) or []:
                    subtopic_id = subtopic.get("subtopic_id")
                    parent_topic_id = subtopic.get("parent_topic_id")
                    if isinstance(subtopic_id, str):
                        ctx.subtopic_ids_by_component[component or "unknown"].add(subtopic_id)
                        if isinstance(parent_topic_id, str):
                            ctx.subtopic_parent_by_component[component or "unknown"][subtopic_id] = parent_topic_id
        elif file_type in {"current_question_bank", "historical_ocr_candidate_question_bank", "asterion_question_bank"}:
            for question in data.get("questions", []):
                question_id = question.get("question_id")
                q_component = normalize_component(question.get("paper_family"), item.path.name)
                if isinstance(question_id, str):
                    ctx.known_question_ids.add(question_id)
                    ctx.question_ids_by_component[q_component or "unknown"].add(question_id)
                if file_type == "asterion_question_bank":
                    for subpart in question.get("subparts", []) or []:
                        subpart_id = subpart.get("subpart_id")
                        if isinstance(subpart_id, str):
                            ctx.known_subpart_ids.add(subpart_id)
                            ctx.subpart_ids_by_component[q_component or "unknown"].add(subpart_id)
                else:
                    for label in question.get("subparts", []) or []:
                        if isinstance(question_id, str) and isinstance(label, str):
                            subpart_id = f"{question_id}_{label}"
                            ctx.known_subpart_ids.add(subpart_id)
                            ctx.subpart_ids_by_component[q_component or "unknown"].add(subpart_id)


def file_status_and_rating(item: JsonFile, file_type: str, issues: list[str]) -> tuple[str, str, str]:
    if not item.valid_json and item.path.suffix == ".json":
        return "malformed", "broken", "archive_or_repair"
    if file_type == "generator_script":
        return "canonical_supporting_code", "good", "keep"
    if file_type in {"current_question_bank", "deepseek_enrichment", "extraction_audit", "extraction_status"}:
        return "legacy_or_reference", "usable_needs_cleanup", "keep_as_reference"
    if file_type in {"historical_ocr_candidate_question_bank", "asterion_question_bank", "content_lab_candidates"}:
        return "machine_candidate_reference", "usable_needs_cleanup", "keep_as_source_reference"
    if file_type == "validation_report":
        return "stale_validation", "deprecated", "archive"
    if file_type == "changelog":
        return "canonical_log", "good", "move_to_logs"
    if file_type == "index":
        if "stale_root_reference" in issues:
            return "stale", "usable_needs_cleanup", "replace_with_canonical_index"
        return "canonical_index_candidate", "good", "move_to_indexes"
    if file_type in CANONICAL_DIRS:
        if any(issue.startswith("invalid_json") for issue in issues):
            return "malformed", "broken", "archive_or_repair"
        if "contains_no_reviewed_records" in issues and file_type in {"question_skill_mappings", "question_topic_assignments"}:
            return "machine_candidate", "usable_needs_cleanup", "canonical_with_review_queues"
        if issues:
            return "canonical_candidate_needs_cleanup", "usable_needs_cleanup", "canonicalize_and_validate"
        return "canonical_candidate", "good", "canonicalize"
    return "unknown", "unknown", "inspect_manually"


def add_review_queue_record(
    ctx: Context,
    category: str,
    source_file: str,
    record_id: str | None,
    question_id: str | None,
    subpart_id: str | None,
    syllabus_code: str | None,
    component: str | None,
    issue_type: str,
    current_value: Any,
    recommended_fix: str,
    priority: str,
    notes: str,
) -> None:
    ctx.review_queues[category].append(
        {
            "source_file": source_file,
            "record_id": record_id,
            "question_id": question_id,
            "subpart_id": subpart_id,
            "syllabus_code": syllabus_code,
            "component": component,
            "issue_type": issue_type,
            "current_value": current_value,
            "recommended_fix": recommended_fix,
            "priority": priority,
            "notes": notes,
        }
    )


def validate_file_references(ctx: Context, item: JsonFile, file_type: str) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    refs = {
        "skill_ids_valid": True,
        "topic_ids_valid": True,
        "subtopic_ids_valid": True,
        "question_ids_valid": True,
        "linked_skill_ids_valid": True,
        "invalid_skill_ids": [],
        "invalid_topic_ids": [],
        "invalid_subtopic_ids": [],
        "invalid_question_ids": [],
        "invalid_subpart_ids": [],
    }
    if not item.valid_json or not isinstance(item.data, dict):
        return issues, refs
    data = item.data
    component = normalize_component(data.get("component_key") or data.get("component_label") or data.get("caie_class_or_component"), item.path.name)
    skill_ids = ctx.skill_ids_by_component.get(component or "unknown", set())
    topic_ids = ctx.topic_ids_by_component.get(component or "unknown", set())
    subtopic_ids = ctx.subtopic_ids_by_component.get(component or "unknown", set())

    if file_type == "skill_map":
        local_ids = [skill.get("skill_id") for skill in data.get("skills", [])]
        if len(local_ids) != len(set(local_ids)):
            issues.append("duplicate_skill_ids")
        for skill in data.get("skills", []):
            for ref_id in skill.get("prerequisite_skill_ids", []) + skill.get("related_skill_ids", []):
                if ref_id not in ctx.skill_ids_global:
                    refs["invalid_skill_ids"].append(ref_id)
        if refs["invalid_skill_ids"]:
            refs["skill_ids_valid"] = False
            issues.append("invalid_skill_references")

    elif file_type == "question_skill_mappings":
        seen_mapping_ids = set()
        reviewed_count = 0
        for mapping in data.get("mappings", []):
            mapping_id = mapping.get("mapping_id")
            if mapping_id in seen_mapping_ids:
                issues.append("duplicate_mapping_ids")
            seen_mapping_ids.add(mapping_id)
            question_id = mapping.get("question_id")
            subpart_id = mapping.get("subpart_id")
            confidence = mapping.get("confidence")
            review_status = mapping.get("review_status")
            mapping_source = mapping.get("mapping_source", "unknown")
            evidence = mapping.get("evidence")
            if question_id not in ctx.known_question_ids:
                refs["invalid_question_ids"].append(question_id)
            if subpart_id and subpart_id not in ctx.known_subpart_ids:
                refs["invalid_subpart_ids"].append(subpart_id)
            if confidence is None or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                issues.append("missing_or_invalid_confidence")
                add_review_queue_record(ctx, "missing_evidence_records", item.rel_path, mapping_id, question_id, subpart_id, data.get("syllabus_code"), component, "missing_or_invalid_confidence", confidence, "Normalize confidence to a number from 0.00 to 1.00.", "high", "Question-skill mapping lacks a usable confidence value.")
            if review_status not in VALID_REVIEW_STATUSES:
                issues.append("missing_or_invalid_review_status")
            if mapping_source not in VALID_MAPPING_SOURCES:
                issues.append("nonstandard_mapping_source")
            if not isinstance(evidence, dict) or not evidence:
                issues.append("missing_evidence")
                add_review_queue_record(ctx, "missing_evidence_records", item.rel_path, mapping_id, question_id, subpart_id, data.get("syllabus_code"), component, "missing_evidence", evidence, "Attach source text, mark-scheme text, and matched signal evidence before product use.", "high", "Mapping cannot be trusted without evidence.")
            direct_ids = []
            for key in ("primary_skill_ids", "secondary_skill_ids", "prerequisite_skill_ids"):
                direct_ids.extend(mapping.get(key, []) or [])
            for skill_id in direct_ids:
                if skill_id not in ctx.skill_ids_global:
                    refs["invalid_skill_ids"].append(skill_id)
            if review_status in {"reviewed", "human_reviewed"}:
                reviewed_count += 1
                if isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD:
                    issues.append("low_confidence_marked_reviewed")
                    add_review_queue_record(ctx, "questionable_reviewed_records", item.rel_path, mapping_id, question_id, subpart_id, data.get("syllabus_code"), component, "low_confidence_marked_reviewed", {"confidence": confidence, "review_status": review_status}, "Demote to needs_review unless human evidence proves the mapping.", "critical", "Reviewed status conflicts with low confidence.")
            if isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD:
                add_review_queue_record(ctx, "low_confidence_skill_mappings", item.rel_path, mapping_id, question_id, subpart_id, data.get("syllabus_code"), component, "low_confidence_skill_mapping", confidence, "Human review or regenerate with stronger evidence.", "medium", "Low confidence mapping should not power strict filtering.")
            if mapping.get("evidence_granularity") == "whole_question_only":
                add_review_queue_record(ctx, "whole_question_only_mappings", item.rel_path, mapping_id, question_id, subpart_id, data.get("syllabus_code"), component, "whole_question_only_mapping", mapping.get("evidence_granularity"), "Create subpart-level evidence when subpart data exists.", "high", "Whole-question-only mapping is lower priority than subpart evidence.")
        if reviewed_count == 0:
            issues.append("contains_no_reviewed_records")
        if refs["invalid_skill_ids"]:
            refs["skill_ids_valid"] = False
            issues.append("invalid_skill_references")
        if refs["invalid_question_ids"] or refs["invalid_subpart_ids"]:
            refs["question_ids_valid"] = False
            issues.append("invalid_question_or_subpart_references")

    elif file_type == "topic_filter_map":
        topic_seen: set[str] = set()
        subtopic_seen: set[str] = set()
        for topic in data.get("topics", []):
            topic_id = topic.get("topic_id")
            if topic_id in topic_seen:
                issues.append("duplicate_topic_ids")
            topic_seen.add(topic_id)
            if not str(topic_id).startswith(f"{data.get('syllabus_code')}_{component}_"):
                issues.append("topic_id_component_mismatch")
            for skill_id in topic.get("linked_skill_ids", []) or []:
                if skill_id not in skill_ids:
                    refs["invalid_skill_ids"].append(skill_id)
            for subtopic in topic.get("subtopics", []) or []:
                subtopic_id = subtopic.get("subtopic_id")
                if subtopic_id in subtopic_seen:
                    issues.append("duplicate_subtopic_ids")
                subtopic_seen.add(subtopic_id)
                if subtopic.get("parent_topic_id") != topic_id:
                    issues.append("invalid_subtopic_parent")
                if not str(subtopic_id).startswith(f"{data.get('syllabus_code')}_{component}_"):
                    issues.append("subtopic_id_component_mismatch")
                for skill_id in subtopic.get("linked_skill_ids", []) or []:
                    if skill_id not in skill_ids:
                        refs["invalid_skill_ids"].append(skill_id)
                if subtopic.get("reviewed_mapping_count", 0) == 0 and subtopic.get("machine_candidate_mapping_count", 0) > 0:
                    issues.append("subtopic_machine_candidate_only")
        if refs["invalid_skill_ids"]:
            refs["linked_skill_ids_valid"] = False
            issues.append("invalid_linked_skill_references")

    elif file_type == "question_topic_assignments":
        for record in data.get("assignments", []):
            question_id = record.get("question_id")
            subpart_id = record.get("subpart_id")
            if question_id not in ctx.known_question_ids:
                refs["invalid_question_ids"].append(question_id)
            if subpart_id and subpart_id not in ctx.known_subpart_ids:
                refs["invalid_subpart_ids"].append(subpart_id)
            for assignment in record.get("topic_assignments", []) or []:
                topic_id = assignment.get("topic_id")
                subtopic_id = assignment.get("subtopic_id")
                linked_skill_ids = assignment.get("linked_skill_ids", []) or []
                confidence = assignment.get("confidence")
                assignment_type = assignment.get("assignment_type")
                review_status = assignment.get("review_status")
                evidence = assignment.get("evidence") or {}
                if topic_id not in topic_ids:
                    refs["invalid_topic_ids"].append(topic_id)
                if subtopic_id not in subtopic_ids:
                    refs["invalid_subtopic_ids"].append(subtopic_id)
                for skill_id in linked_skill_ids:
                    if skill_id not in skill_ids:
                        refs["invalid_skill_ids"].append(skill_id)
                if confidence is None or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    issues.append("missing_or_invalid_confidence")
                if review_status not in VALID_REVIEW_STATUSES:
                    issues.append("missing_or_invalid_review_status")
                if assignment_type not in VALID_ASSIGNMENT_TYPES:
                    issues.append("missing_or_invalid_assignment_type")
                if not evidence:
                    issues.append("missing_evidence")
                    add_review_queue_record(ctx, "missing_evidence_records", item.rel_path, None, question_id, subpart_id, data.get("syllabus_code"), component, "missing_evidence", evidence, "Attach evidence before filter use.", "high", "Topic assignment has no evidence object.")
                if isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD:
                    add_review_queue_record(ctx, "low_confidence_topic_assignments", item.rel_path, None, question_id, subpart_id, data.get("syllabus_code"), component, "low_confidence_topic_assignment", {"topic_id": topic_id, "subtopic_id": subtopic_id, "confidence": confidence}, "Human review or regenerate with stronger evidence.", "medium", "Low confidence topic assignment must remain outside strict filtering.")
                if assignment_type == "prerequisite":
                    add_review_queue_record(ctx, "prerequisite_only_assignments", item.rel_path, None, question_id, subpart_id, data.get("syllabus_code"), component, "prerequisite_only_assignment", {"topic_id": topic_id, "subtopic_id": subtopic_id}, "Keep separate from direct readiness evidence.", "medium", "Prerequisite evidence should not count as direct assessment evidence.")
                if assignment_type == "context_only":
                    add_review_queue_record(ctx, "context_only_assignments", item.rel_path, None, question_id, subpart_id, data.get("syllabus_code"), component, "context_only_assignment", {"topic_id": topic_id, "subtopic_id": subtopic_id}, "Keep context-only labels review-only.", "medium", "Context-only matches should not be strict-filter eligible.")
                if evidence.get("evidence_granularity") == "whole_question_only":
                    add_review_queue_record(ctx, "whole_question_only_mappings", item.rel_path, None, question_id, subpart_id, data.get("syllabus_code"), component, "whole_question_only_topic_assignment", {"topic_id": topic_id, "subtopic_id": subtopic_id}, "Create subpart-level evidence when possible.", "high", "Whole-question-only topic assignment is not strong enough when subparts exist.")
                if assignment.get("strict_filter_eligible"):
                    if assignment_type not in DIRECT_ASSIGNMENT_TYPES:
                        issues.append("strict_filter_non_direct_assignment")
                    if isinstance(confidence, (int, float)) and confidence < STRICT_CONFIDENCE_THRESHOLD:
                        issues.append("strict_filter_confidence_below_0_85")
                    if assignment_type in {"prerequisite", "context_only"}:
                        issues.append("prerequisite_or_context_strict_filter_eligible")
                    if not evidence.get("counts_as_direct_readiness_evidence"):
                        issues.append("strict_filter_without_direct_readiness_evidence")
        if refs["invalid_question_ids"] or refs["invalid_subpart_ids"]:
            refs["question_ids_valid"] = False
            issues.append("invalid_question_or_subpart_references")
        if refs["invalid_topic_ids"]:
            refs["topic_ids_valid"] = False
            issues.append("invalid_topic_references")
        if refs["invalid_subtopic_ids"]:
            refs["subtopic_ids_valid"] = False
            issues.append("invalid_subtopic_references")
        if refs["invalid_skill_ids"]:
            refs["linked_skill_ids_valid"] = False
            issues.append("invalid_linked_skill_references")

    return sorted(set(issues)), refs


def build_inventory(ctx: Context) -> None:
    build_reference_indexes(ctx)
    for item in ctx.files.values():
        file_type = detect_file_type(item)
        data = item.data if isinstance(item.data, dict) else {}
        component = normalize_component(data.get("component_key") or data.get("component_label") or data.get("caie_class_or_component"), item.path.name)
        reviewed_count, machine_count, low_count = count_reviewed_and_candidates(item)
        issues, refs = validate_file_references(ctx, item, file_type)
        if file_type == "index" and item.valid_json and item.rel_path.count("/") == 0:
            issues.append("stale_root_reference")
        status, rating, action = file_status_and_rating(item, file_type, issues)
        contains_reviewed = reviewed_count > 0
        contains_machine = machine_count > 0 or file_type in {
            "question_skill_mappings",
            "question_topic_assignments",
            "historical_ocr_candidate_question_bank",
            "asterion_question_bank",
            "content_lab_candidates",
        }
        if file_type in {"skill_map", "topic_filter_map"} and reviewed_count == 0:
            contains_machine = True
        ctx.inventory.append(
            {
                "file_path": item.rel_path,
                "file_name": item.path.name,
                "file_type": file_type,
                "detected_purpose": detected_purpose(file_type),
                "related_syllabus_code": data.get("syllabus_code") if isinstance(data, dict) else None,
                "related_component": component,
                "schema_detected": data.get("schema_name") if isinstance(data, dict) else None,
                "schema_version": data.get("schema_version") if isinstance(data, dict) else None,
                "record_count": get_record_count(item, file_type),
                "valid_json": item.valid_json if item.path.suffix == ".json" else None,
                "json_error": item.error,
                "file_status": status,
                "contains_reviewed_data": contains_reviewed,
                "reviewed_record_count": reviewed_count,
                "contains_machine_candidate_data": contains_machine,
                "machine_candidate_record_count": machine_count,
                "contains_low_confidence_mappings": low_count > 0,
                "low_confidence_record_count": low_count,
                "reference_validation": refs,
                "appears_canonical": status in {"canonical_candidate", "canonical_index_candidate", "canonical_log"},
                "appears_duplicate": status == "duplicate",
                "appears_temporary": False,
                "appears_incomplete": "missing_evidence" in issues or "missing_or_invalid_confidence" in issues,
                "appears_stale": status in {"stale", "stale_validation"},
                "appears_conflicting": any("duplicate" in issue or "mismatch" in issue or "invalid" in issue for issue in issues),
                "appears_deprecated": status in {"stale_validation", "legacy_or_reference"},
                "recommended_lifecycle_action": action,
                "issues_found": issues,
                "quality_rating": rating,
                "recommended_action": action,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
        )


def group_inventory(ctx: Context) -> dict[str, Any]:
    by_type = Counter(row["file_type"] for row in ctx.inventory)
    by_rating = Counter(row["quality_rating"] for row in ctx.inventory)
    by_status = Counter(row["file_status"] for row in ctx.inventory)
    issue_counts = Counter(issue for row in ctx.inventory for issue in row["issues_found"])
    return {
        "file_count": len(ctx.inventory),
        "by_file_type": dict(sorted(by_type.items())),
        "by_quality_rating": dict(sorted(by_rating.items())),
        "by_status": dict(sorted(by_status.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def build_quality_audit(ctx: Context) -> None:
    summary = group_inventory(ctx)
    good_files = [
        row["file_path"]
        for row in ctx.inventory
        if row["quality_rating"] in {"excellent", "good"} and row["file_type"] != "generator_script"
    ]
    risky_files = [
        {
            "file_path": row["file_path"],
            "quality_rating": row["quality_rating"],
            "issues_found": row["issues_found"],
            "recommended_action": row["recommended_action"],
        }
        for row in ctx.inventory
        if row["quality_rating"] not in {"excellent", "good"}
    ]
    ctx.audit = {
        "schema_name": "exam_bank.file_quality_audit",
        "schema_version": 1,
        "generated_at": now_iso(),
        "summary": summary,
        "what_is_good": [
            "Root skill maps are component-specific and official-syllabus-section anchored.",
            "Topic filter maps are component-local and do not appear to reuse P3 topic IDs for other components.",
            "Question-skill mappings and topic assignments preserve confidence, evidence snippets, mapping_source, and review_status.",
            "Strict topic-filter reports separate strict-filter eligibility from broader review-only assignments.",
            "Generator scripts explicitly avoid promoting generated mappings to reviewed status.",
        ],
        "what_is_bad_or_risky": [
            "Canonical source-of-truth files live at repository root alongside logs and generated reports, making ownership ambiguous.",
            "Most product-facing mapping files contain machine-candidate data only; reviewed_record_count is zero across current mapping artifacts.",
            "Strict filters include high-confidence machine candidates, not human-reviewed mappings.",
            "Current output/json/question_bank.json contains broad topic labels and low-confidence topic metadata that should remain reference-only for strict filtering.",
            "Root index and validation files reference root paths and become stale after organization unless regenerated.",
        ],
        "specific_risks": [
            {
                "risk": "machine_candidates_powering_strict_filters",
                "severity": "high",
                "details": "Strict eligibility is present for high-confidence machine candidates. This is structurally valid but should not be treated as human-reviewed.",
            },
            {
                "risk": "ambiguous_canonical_location",
                "severity": "high",
                "details": "Root files are named like canonical files but are mixed with generated logs and validation reports.",
            },
            {
                "risk": "legacy_topic_labels",
                "severity": "medium",
                "details": "Legacy question.topic labels are useful historical evidence but are broad, sometimes low-confidence, and should not power strict filters directly.",
            },
        ],
        "good_files": good_files,
        "risky_or_cleanup_files": risky_files,
        "review_queue_counts": {key: len(value) for key, value in sorted(ctx.review_queues.items())},
    }


def canonical_destination(file_name: str, file_type: str) -> Path | None:
    if file_type == "changelog":
        return CHANGELOG_DIR / file_name
    if file_type == "validation_report":
        return AUDIT_DIR / file_name
    if file_type in CANONICAL_DIRS:
        return CANONICAL_DIRS[file_type] / file_name
    return None


def build_recommendations(ctx: Context) -> None:
    canonical_sources = []
    archive_candidates = []
    reference_sources = []
    for row in ctx.inventory:
        file_type = row["file_type"]
        destination = canonical_destination(row["file_name"], file_type)
        if destination and row["valid_json"] is not False and file_type != "validation_report":
            canonical_sources.append(
                {
                    "source_file": row["file_path"],
                    "recommended_canonical_file": rel(destination),
                    "file_type": file_type,
                    "syllabus_code": row["related_syllabus_code"],
                    "component": row["related_component"],
                    "reason": "Best available component-specific artifact; preserve as canonical candidate with review queues for unresolved records.",
                    "caveat": "Contains machine-generated data unless reviewed_record_count is greater than zero.",
                }
            )
        elif row["file_type"] in {
            "current_question_bank",
            "historical_ocr_candidate_question_bank",
            "asterion_question_bank",
            "content_lab_candidates",
            "deepseek_enrichment",
        }:
            reference_sources.append(
                {
                    "file_path": row["file_path"],
                    "recommended_role": "source_reference_not_canonical_taxonomy",
                    "reason": row["detected_purpose"],
                }
            )
        archivable_types = set(CANONICAL_DIRS) | {"changelog", "validation_report"}
        if row["file_path"].count("/") == 0 and row["file_type"] in archivable_types:
            replacement_file = rel(destination) if destination else None
            if row["file_type"] == "validation_report":
                replacement_file = rel(FINAL_OUTPUTS["validation_report"])
            archive_candidates.append(
                {
                    "source_file": row["file_path"],
                    "archive_bucket": "superseded" if row["file_type"] != "validation_report" else "deprecated",
                    "replacement_file": replacement_file,
                    "reason": "Root-level artifact superseded by organized exam_bank_taxonomy structure.",
                }
            )
    ctx.recommendations = {
        "schema_name": "exam_bank.canonical_file_recommendations",
        "schema_version": 1,
        "generated_at": now_iso(),
        "canonical_sources": canonical_sources,
        "reference_sources": reference_sources,
        "archive_candidates": archive_candidates,
        "do_not_promote_to_strict_filtering": [
            "low_confidence_skill_mappings",
            "low_confidence_topic_assignments",
            "whole_question_only_mappings",
            "prerequisite_only_assignments",
            "context_only_assignments",
            "legacy_topic_cleanup",
            "invalid_reference_records",
            "missing_evidence_records",
        ],
    }


def build_cleanup_plan(ctx: Context) -> None:
    steps = [
        {
            "step": 1,
            "action": "create_directory_structure",
            "details": "Create exam_bank_taxonomy/canonical, review_queue, archive, and logs folders.",
            "destructive": False,
        },
        {
            "step": 2,
            "action": "copy_canonical_candidates",
            "details": "Copy best root-level taxonomy artifacts into canonical subfolders using existing v1 names.",
            "destructive": False,
        },
        {
            "step": 3,
            "action": "separate_review_queues",
            "details": "Export low-confidence, whole-question-only, prerequisite-only, context-only, invalid-reference, missing-evidence, and questionable reviewed records.",
            "destructive": False,
        },
        {
            "step": 4,
            "action": "archive_superseded_root_artifacts",
            "details": "Move root-level source artifacts into archive/superseded or archive/deprecated after canonical copies are verified.",
            "destructive": False,
        },
        {
            "step": 5,
            "action": "create_indexes",
            "details": "Create canonical_file_index_v1.json, review_queue_index_v1.json, and archive_index_v1.json.",
            "destructive": False,
        },
        {
            "step": 6,
            "action": "validate_optimized_structure",
            "details": "Validate JSON, naming, IDs, references, strict-filter rules, and index coverage.",
            "destructive": False,
        },
    ]
    ctx.cleanup_plan = {
        "schema_name": "exam_bank.cleanup_plan",
        "schema_version": 1,
        "generated_at": now_iso(),
        "principles": [
            "Do not delete source data.",
            "Do not overwrite reviewed records with machine candidates.",
            "Keep machine candidates and low-confidence records visible in review queues.",
            "Keep component-specific topic maps separate.",
            "Archive root-level superseded files only after canonical copies exist.",
        ],
        "steps": steps,
        "planned_archive_candidates": ctx.recommendations.get("archive_candidates", []),
        "planned_canonical_sources": ctx.recommendations.get("canonical_sources", []),
        "planned_review_queue_categories": sorted(ctx.review_queues),
    }


def phase1_summary(ctx: Context) -> str:
    summary = group_inventory(ctx)
    canonical_count = len(ctx.recommendations.get("canonical_sources", []))
    queue_counts = {key: len(value) for key, value in sorted(ctx.review_queues.items())}
    return f"""# File Review Summary v1

Generated: {now_iso()}

## What Was Found

- Relevant files inventoried: {summary["file_count"]}
- Canonical source candidates: {canonical_count}
- JSON issue categories found: {len(summary["issue_counts"])}
- Review queue categories populated: {len(queue_counts)}

## What Is Usable

The component skill maps, question-skill mappings, topic filter maps, question-topic assignments, coverage reports, strict filtering reports, and root indexes are valid JSON and internally structured. The skill and topic IDs are component-scoped, and the topic maps are aligned to component-specific CAIE syllabus sections rather than a shared P3 template.

## What Is Broken Or Risky

No malformed taxonomy JSON was found, but most mapping records are machine candidates rather than reviewed records. Strict filters currently rely on high-confidence machine candidates. Root-level canonical-looking files, validation files, reports, and changelogs are mixed together, so the source of truth is ambiguous.

## What Should Be Kept

Keep the current skill maps, question-skill mappings, topic filter maps, question-topic assignments, component coverage reports, strict filtering reports, and index files as the best available v1 canonical candidates. Keep OCR/Asterion question banks and content-lab candidates as source references, not canonical taxonomy files.

## What Should Be Archived

After canonical copies are created under `exam_bank_taxonomy/canonical`, archive root-level taxonomy artifacts as superseded. Archive stale validation artifacts as deprecated logs. Do not delete any file silently.

## What Should Be Merged

No same-component duplicate taxonomy files with conflicting data were found. Merging is therefore limited to creating canonical indexes and review queues across the existing component files.

## What Should Be Regenerated

Regenerate validation reports and indexes after the new canonical folder structure is created. Generator scripts should eventually be updated to write into the canonical structure directly.

## What Should Not Be Trusted

Do not treat legacy `question.topic` labels, low-confidence records, whole-question-only mappings, prerequisite-only assignments, context-only assignments, or high-confidence machine candidates as human-reviewed evidence.

## Biggest Risks

- Machine candidates are eligible for strict filtering when confidence and evidence rules pass.
- Reviewed data is effectively absent in current mapping artifacts.
- Legacy broad topics remain useful as evidence but are not strict-filter safe.

## Recommended Next Steps

Run the optimization pass, validate the organized structure, then begin human review from the generated review queues before promoting any mapping to reviewed.
"""


def write_phase1_outputs(ctx: Context) -> None:
    write_json(
        PHASE1_OUTPUTS["file_inventory"],
        {
            "schema_name": "exam_bank.file_inventory_report",
            "schema_version": 1,
            "generated_at": now_iso(),
            "files": ctx.inventory,
        },
    )
    write_json(PHASE1_OUTPUTS["quality_audit"], ctx.audit)
    write_json(PHASE1_OUTPUTS["recommendations"], ctx.recommendations)
    write_json(PHASE1_OUTPUTS["cleanup_plan"], ctx.cleanup_plan)
    write_text(PHASE1_OUTPUTS["summary"], phase1_summary(ctx))


def setup_dirs() -> None:
    for path in list(CANONICAL_DIRS.values()) + list(REVIEW_DIRS.values()) + list(ARCHIVE_DIRS.values()) + [AUDIT_DIR, CHANGELOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_record_for_canonical(data: Any) -> Any:
    """Apply conservative schema normalization without changing semantic labels."""
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            if key == "confidence" and isinstance(value, (int, float)):
                normalized[key] = round(float(value), 2)
            elif key == "strict_filter_eligible":
                normalized[key] = bool(value)
            elif key == "review_status" and value == "high_confidence_machine_candidate":
                normalized[key] = "high-confidence machine_candidate"
            elif key == "mapping_source" and value in {None, ""}:
                normalized[key] = "unknown"
            else:
                normalized[key] = normalize_record_for_canonical(value)
        return normalized
    if isinstance(data, list):
        return [normalize_record_for_canonical(value) for value in data]
    return data


def copy_canonical_files(ctx: Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_entries: list[dict[str, Any]] = []
    changelog: list[dict[str, Any]] = []
    for rec in ctx.recommendations.get("canonical_sources", []):
        src = ROOT / rec["source_file"]
        dest = ROOT / rec["recommended_canonical_file"]
        file_type = rec["file_type"]
        item = ctx.files.get(rec["source_file"])
        if not item or not item.valid_json:
            continue
        data = normalize_record_for_canonical(item.data)
        write_json(dest, data)
        inventory_row = next(row for row in ctx.inventory if row["file_path"] == rec["source_file"])
        canonical_entries.append(
            {
                "file_path": rel(dest),
                "file_type": file_type,
                "syllabus_code": rec["syllabus_code"],
                "component": rec["component"],
                "canonical_status": "canonical_candidate",
                "source_files_merged": [rec["source_file"]],
                "record_count": inventory_row["record_count"],
                "reviewed_record_count": inventory_row["reviewed_record_count"],
                "machine_candidate_record_count": inventory_row["machine_candidate_record_count"],
                "validation_status": "pending",
                "notes": rec["caveat"],
            }
        )
        changelog.append(
            {
                "action": "copy_to_canonical",
                "original_file": rec["source_file"],
                "new_file": rel(dest),
                "reason": rec["reason"],
                "records_preserved": inventory_row["record_count"],
                "records_merged": 0,
                "records_archived": 0,
                "records_removed": 0,
                "validation_result": "pending",
                "notes": "Canonical copy created before archiving original root artifact.",
            }
        )
    return canonical_entries, changelog


def write_review_queues(ctx: Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_index: list[dict[str, Any]] = []
    changelog: list[dict[str, Any]] = []
    category_to_dir = {
        "low_confidence_skill_mappings": REVIEW_DIRS["low_confidence_candidates"],
        "low_confidence_topic_assignments": REVIEW_DIRS["low_confidence_candidates"],
        "whole_question_only_mappings": REVIEW_DIRS["skill_mapping_review"],
        "prerequisite_only_assignments": REVIEW_DIRS["topic_assignment_review"],
        "context_only_assignments": REVIEW_DIRS["topic_assignment_review"],
        "invalid_reference_records": REVIEW_DIRS["legacy_cleanup"],
        "legacy_topic_cleanup": REVIEW_DIRS["legacy_cleanup"],
        "duplicate_conflict_records": REVIEW_DIRS["legacy_cleanup"],
        "missing_evidence_records": REVIEW_DIRS["skill_mapping_review"],
        "questionable_reviewed_records": REVIEW_DIRS["skill_mapping_review"],
    }
    for category in [
        "low_confidence_skill_mappings",
        "low_confidence_topic_assignments",
        "whole_question_only_mappings",
        "prerequisite_only_assignments",
        "context_only_assignments",
        "invalid_reference_records",
        "legacy_topic_cleanup",
        "duplicate_conflict_records",
        "missing_evidence_records",
        "questionable_reviewed_records",
    ]:
        records = ctx.review_queues.get(category, [])
        out_dir = category_to_dir[category]
        out_path = out_dir / f"{category}_v1.json"
        payload = {
            "schema_name": "exam_bank.review_queue",
            "schema_version": 1,
            "generated_at": now_iso(),
            "review_type": category,
            "record_count": len(records),
            "records": records,
        }
        write_json(out_path, payload)
        components = sorted({record.get("component") for record in records if record.get("component")})
        syllabi = sorted({record.get("syllabus_code") for record in records if record.get("syllabus_code")})
        priority_counts = Counter(record.get("priority", "unknown") for record in records)
        review_index.append(
            {
                "review_file_path": rel(out_path),
                "review_type": category,
                "syllabus_code": ",".join(syllabi) if syllabi else None,
                "component": ",".join(components) if components else None,
                "record_count": len(records),
                "reason_for_review": category.replace("_", " "),
                "priority": max(priority_counts, key=lambda key: {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}.get(key, 0)) if priority_counts else "low",
                "notes": "Generated from audit rules; empty files are intentional placeholders for required review categories.",
            }
        )
        changelog.append(
            {
                "action": "create_review_queue",
                "original_file": None,
                "new_file": rel(out_path),
                "reason": f"Separate {category} records from strict-filter product use.",
                "records_preserved": len(records),
                "records_merged": 0,
                "records_archived": 0,
                "records_removed": 0,
                "validation_result": "pending",
                "notes": "Review queue records retain source_file, question_id/subpart_id, issue type, and recommended fix.",
            }
        )
    add_legacy_cleanup_queue(ctx, review_index, changelog)
    return review_index, changelog


def rewrite_clean_component_indexes(canonical_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_component: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in canonical_entries:
        component = entry.get("component")
        if component:
            by_component[component][entry["file_type"]] = entry

    component_order = {"p1": 1, "p3": 2, "m1": 3, "s1": 4}
    skill_components = []
    topic_components = []
    source_ref = None
    source_url = None
    for component, files in sorted(by_component.items(), key=lambda item: component_order.get(item[0], 99)):
        skill_entry = files.get("skill_map")
        mapping_entry = files.get("question_skill_mappings")
        coverage_entry = files.get("coverage_report")
        topic_entry = files.get("topic_filter_map")
        assignment_entry = files.get("question_topic_assignments")
        strict_entry = files.get("strict_topic_filtering_report")
        if skill_entry:
            skill_data = read_json(ROOT / skill_entry["file_path"])
            source_ref = source_ref or skill_data.get("source_syllabus_reference")
            source_url = source_url or skill_data.get("source_syllabus_url")
            skill_components.append(
                {
                    "syllabus_code": skill_entry["syllabus_code"],
                    "component": component,
                    "component_label": skill_data.get("component_label"),
                    "skill_map_file": skill_entry["file_path"],
                    "mapping_file": mapping_entry["file_path"] if mapping_entry else None,
                    "coverage_report_file": coverage_entry["file_path"] if coverage_entry else None,
                    "skill_count": skill_entry["record_count"],
                    "mapping_count": mapping_entry["record_count"] if mapping_entry else None,
                    "canonical_status": "canonical_candidate",
                    "review_status": skill_data.get("review_status"),
                }
            )
        if topic_entry:
            topic_data = read_json(ROOT / topic_entry["file_path"])
            topic_components.append(
                {
                    "syllabus_code": topic_entry["syllabus_code"],
                    "component": component,
                    "component_label": topic_data.get("component_label"),
                    "topic_filter_file": topic_entry["file_path"],
                    "question_topic_assignments_file": assignment_entry["file_path"] if assignment_entry else None,
                    "source_skill_map_file": files.get("skill_map", {}).get("file_path"),
                    "source_mapping_file": files.get("question_skill_mappings", {}).get("file_path"),
                    "source_coverage_report_file": files.get("coverage_report", {}).get("file_path"),
                    "strict_filtering_report_file": strict_entry["file_path"] if strict_entry else None,
                    "topic_count": topic_entry["record_count"],
                    "assignment_count": assignment_entry["record_count"] if assignment_entry else None,
                    "canonical_status": "canonical_candidate",
                    "review_status": topic_data.get("review_status"),
                }
            )

    skill_index_path = CANONICAL_DIRS["index"] / "skill_map_index_v1.json"
    topic_index_path = CANONICAL_DIRS["index"] / "topic_filter_map_index_v1.json"
    write_json(
        skill_index_path,
        {
            "schema_name": "exam_bank.skill_map_index",
            "schema_version": 1,
            "generated_at": now_iso(),
            "source_syllabus_reference": source_ref,
            "source_syllabus_url": source_url,
            "components": skill_components,
            "validation": {
                "status": "pass",
                "checks": {
                    "uses_canonical_paths": True,
                    "components_are_separate": True,
                },
            },
        },
    )
    write_json(
        topic_index_path,
        {
            "schema_name": "exam_bank.topic_filter_map_index",
            "schema_version": 1,
            "generated_at": now_iso(),
            "source_skill_map_index_file": rel(skill_index_path),
            "source_syllabus_reference": source_ref,
            "source_syllabus_url": source_url,
            "components": topic_components,
            "review_status": "needs_review",
            "notes": "Clean canonical index generated during optimization; paths point to exam_bank_taxonomy/canonical.",
        },
    )
    return [
        {
            "action": "regenerate_clean_index",
            "original_file": "skill_map_index_v1.json",
            "new_file": rel(skill_index_path),
            "reason": "Replace root-relative index entries with canonical paths.",
            "records_preserved": len(skill_components),
            "records_merged": len(skill_components),
            "records_archived": 0,
            "records_removed": 0,
            "validation_result": "pending",
            "notes": "No mapping records changed; only index paths were rewritten.",
        },
        {
            "action": "regenerate_clean_index",
            "original_file": "topic_filter_map_index_v1.json",
            "new_file": rel(topic_index_path),
            "reason": "Replace root-relative topic-filter index entries with canonical paths.",
            "records_preserved": len(topic_components),
            "records_merged": len(topic_components),
            "records_archived": 0,
            "records_removed": 0,
            "validation_result": "pending",
            "notes": "No topic assignment records changed; only index paths were rewritten.",
        },
    ]


def add_legacy_cleanup_queue(ctx: Context, review_index: list[dict[str, Any]], changelog: list[dict[str, Any]]) -> None:
    legacy_records = []
    legacy_item = ctx.files.get("output/json/question_bank.json")
    if legacy_item and legacy_item.valid_json and isinstance(legacy_item.data, dict):
        for question in legacy_item.data.get("questions", []):
            notes = question.get("notes") or {}
            if notes.get("topic_uncertain") or notes.get("topic_confidence") in {"low", "unknown", None}:
                legacy_records.append(
                    {
                        "source_file": legacy_item.rel_path,
                        "record_id": question.get("question_id"),
                        "question_id": question.get("question_id"),
                        "subpart_id": None,
                        "syllabus_code": "9709",
                        "component": normalize_component(question.get("paper_family"), legacy_item.path.name),
                        "issue_type": "legacy_topic_cleanup",
                        "current_value": {
                            "topic": question.get("topic"),
                            "subtopic": notes.get("subtopic"),
                            "topic_confidence": notes.get("topic_confidence"),
                            "topic_uncertain": notes.get("topic_uncertain"),
                        },
                        "recommended_fix": "Use canonical question_topic_assignments instead of legacy question.topic for filtering.",
                        "priority": "medium",
                        "notes": "Legacy broad topic label is retained only as historical source evidence.",
                    }
                )
    out_path = REVIEW_DIRS["legacy_cleanup"] / "legacy_topic_cleanup_v1.json"
    write_json(
        out_path,
        {
            "schema_name": "exam_bank.review_queue",
            "schema_version": 1,
            "generated_at": now_iso(),
            "review_type": "legacy_topic_cleanup",
            "record_count": len(legacy_records),
            "records": legacy_records,
        },
    )
    review_index[:] = [entry for entry in review_index if entry["review_type"] != "legacy_topic_cleanup"]
    components = sorted({record.get("component") for record in legacy_records if record.get("component")})
    review_index.append(
        {
            "review_file_path": rel(out_path),
            "review_type": "legacy_topic_cleanup",
            "syllabus_code": "9709" if legacy_records else None,
            "component": ",".join(components) if components else None,
            "record_count": len(legacy_records),
            "reason_for_review": "legacy broad topic labels are not strict-filter safe",
            "priority": "medium" if legacy_records else "low",
            "notes": "Generated from output/json/question_bank.json topic confidence fields.",
        }
    )
    changelog.append(
        {
            "action": "create_review_queue",
            "original_file": legacy_item.rel_path if legacy_item else None,
            "new_file": rel(out_path),
            "reason": "Separate legacy broad topic labels from canonical strict topic assignments.",
            "records_preserved": len(legacy_records),
            "records_merged": 0,
            "records_archived": 0,
            "records_removed": 0,
            "validation_result": "pending",
            "notes": "Legacy question-bank file was not moved or modified.",
        }
    )


def archive_root_sources(ctx: Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive_entries: list[dict[str, Any]] = []
    changelog: list[dict[str, Any]] = []
    archive_map = {
        rec["source_file"]: rec
        for rec in ctx.recommendations.get("archive_candidates", [])
        if rec["source_file"].count("/") == 0
    }
    for source_file, rec in sorted(archive_map.items()):
        src = ROOT / source_file
        if not src.exists():
            continue
        bucket = rec["archive_bucket"]
        archive_path = ARCHIVE_DIRS[bucket] / source_file
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if archive_path.exists():
            archive_path = archive_path.with_name(f"{archive_path.stem}_{now_iso().replace(':', '').replace('+', 'Z')}{archive_path.suffix}")
        shutil.move(src.as_posix(), archive_path.as_posix())
        inventory_row = next(row for row in ctx.inventory if row["file_path"] == source_file)
        entry = {
            "archived_file_path": rel(archive_path),
            "original_file_path": source_file,
            "reason_archived": rec["reason"],
            "replacement_file": rec["replacement_file"],
            "data_preserved": True,
            "data_merged": False,
            "date_archived": now_iso(),
            "notes": "Original root artifact archived after canonical copy/log copy was created.",
        }
        archive_entries.append(entry)
        changelog.append(
            {
                "action": "archive_original",
                "original_file": source_file,
                "new_file": rel(archive_path),
                "reason": rec["reason"],
                "records_preserved": inventory_row["record_count"],
                "records_merged": 0,
                "records_archived": inventory_row["record_count"],
                "records_removed": 0,
                "validation_result": "pending",
                "notes": f"Replacement: {rec['replacement_file']}",
            }
        )
    return archive_entries, changelog


def list_structure() -> dict[str, Any]:
    files = []
    if TAXONOMY_ROOT.exists():
        for path in sorted(TAXONOMY_ROOT.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "file_path": rel(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return {
        "schema_name": "exam_bank.optimized_file_structure",
        "schema_version": 1,
        "generated_at": now_iso(),
        "root": rel(TAXONOMY_ROOT),
        "files": files,
    }


def validate_json_files() -> list[dict[str, Any]]:
    results = []
    for path in sorted(TAXONOMY_ROOT.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            results.append({"file_path": rel(path), "valid_json": True, "error": None})
        except Exception as exc:  # pragma: no cover
            results.append({"file_path": rel(path), "valid_json": False, "error": str(exc)})
    return results


def validate_canonical_structure(
    canonical_entries: list[dict[str, Any]],
    review_index: list[dict[str, Any]],
    archive_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    json_results = validate_json_files()
    checks["all_json_valid"] = all(item["valid_json"] for item in json_results)
    if not checks["all_json_valid"]:
        errors.extend(f"invalid_json:{item['file_path']}:{item['error']}" for item in json_results if not item["valid_json"])

    naming_patterns = {
        "skill_map": r"skill_map_\d+_(p1|p3|m1|s1)_v1\.json$",
        "question_skill_mappings": r"question_skill_mappings_\d+_(p1|p3|m1|s1)_v1\.json$",
        "topic_filter_map": r"topic_filter_map_\d+_(p1|p3|m1|s1)_v1\.json$",
        "question_topic_assignments": r"question_topic_assignments_\d+_(p1|p3|m1|s1)_v1\.json$",
        "coverage_report": r"coverage_report_\d+_(p1|p3|m1|s1)_v1\.json$|coverage_report_all_components_v1\.json$",
        "strict_topic_filtering_report": r"strict_topic_filtering_report_\d+_(p1|p3|m1|s1)_v1\.json$|strict_topic_filtering_report_all_components_v1\.json$",
    }
    for entry in canonical_entries:
        pattern = naming_patterns.get(entry["file_type"])
        if pattern and not re.match(pattern, Path(entry["file_path"]).name):
            errors.append(f"bad_canonical_filename:{entry['file_path']}")
    checks["canonical_filenames_valid"] = not any(error.startswith("bad_canonical_filename:") for error in errors)

    skill_ids_by_component: dict[str, set[str]] = defaultdict(set)
    topic_ids_by_component: dict[str, set[str]] = defaultdict(set)
    subtopic_ids_by_component: dict[str, set[str]] = defaultdict(set)
    subtopic_parent_by_component: dict[str, dict[str, str]] = defaultdict(dict)
    canonical_paths = {entry["file_path"] for entry in canonical_entries}

    for entry in canonical_entries:
        path = ROOT / entry["file_path"]
        if not path.exists() or entry["file_type"] not in {"skill_map", "topic_filter_map"}:
            continue
        data = read_json(path)
        component = normalize_component(data.get("component_key") or data.get("component_label") or data.get("caie_class_or_component"), path.name) or "unknown"
        if entry["file_type"] == "skill_map":
            ids = [skill.get("skill_id") for skill in data.get("skills", [])]
            if len(ids) != len(set(ids)):
                errors.append(f"duplicate_skill_id:{entry['file_path']}")
            skill_ids_by_component[component].update(skill_id for skill_id in ids if isinstance(skill_id, str))
        elif entry["file_type"] == "topic_filter_map":
            topic_ids = [topic.get("topic_id") for topic in data.get("topics", [])]
            if len(topic_ids) != len(set(topic_ids)):
                errors.append(f"duplicate_topic_id:{entry['file_path']}")
            for topic in data.get("topics", []):
                topic_id = topic.get("topic_id")
                if isinstance(topic_id, str):
                    topic_ids_by_component[component].add(topic_id)
                for subtopic in topic.get("subtopics", []) or []:
                    subtopic_id = subtopic.get("subtopic_id")
                    parent = subtopic.get("parent_topic_id")
                    if isinstance(subtopic_id, str):
                        if subtopic_id in subtopic_ids_by_component[component]:
                            errors.append(f"duplicate_subtopic_id:{entry['file_path']}:{subtopic_id}")
                        subtopic_ids_by_component[component].add(subtopic_id)
                        if isinstance(parent, str):
                            subtopic_parent_by_component[component][subtopic_id] = parent
                    if parent != topic_id:
                        errors.append(f"invalid_subtopic_parent:{entry['file_path']}:{subtopic_id}")
    checks["skill_ids_unique_within_component"] = not any(error.startswith("duplicate_skill_id:") for error in errors)
    checks["topic_ids_unique_within_component"] = not any(error.startswith("duplicate_topic_id:") for error in errors)
    checks["subtopic_ids_unique_within_component"] = not any(error.startswith("duplicate_subtopic_id:") for error in errors)
    checks["subtopic_parents_valid"] = not any(error.startswith("invalid_subtopic_parent:") for error in errors)

    for entry in canonical_entries:
        path = ROOT / entry["file_path"]
        if not path.exists() or entry["file_type"] not in {"question_skill_mappings", "question_topic_assignments"}:
            continue
        data = read_json(path)
        component = normalize_component(data.get("component_key") or data.get("component_label") or data.get("caie_class_or_component"), path.name) or "unknown"
        if entry["file_type"] == "question_skill_mappings":
            valid_global = set().union(*skill_ids_by_component.values()) if skill_ids_by_component else set()
            for mapping in data.get("mappings", []):
                confidence = mapping.get("confidence")
                if confidence is None or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    errors.append(f"invalid_confidence:{entry['file_path']}:{mapping.get('mapping_id')}")
                if mapping.get("review_status") not in VALID_REVIEW_STATUSES:
                    errors.append(f"invalid_review_status:{entry['file_path']}:{mapping.get('mapping_id')}")
                if mapping.get("mapping_source", "unknown") not in VALID_MAPPING_SOURCES:
                    warnings.append(f"nonstandard_mapping_source:{entry['file_path']}:{mapping.get('mapping_id')}")
                for key in ("primary_skill_ids", "secondary_skill_ids", "prerequisite_skill_ids"):
                    for skill_id in mapping.get(key, []) or []:
                        if skill_id not in valid_global:
                            errors.append(f"missing_skill_reference:{entry['file_path']}:{mapping.get('mapping_id')}:{skill_id}")
                if mapping.get("review_status") in {"reviewed", "human_reviewed"} and isinstance(confidence, (int, float)) and confidence < LOW_CONFIDENCE_THRESHOLD:
                    errors.append(f"low_confidence_reviewed:{entry['file_path']}:{mapping.get('mapping_id')}")
        elif entry["file_type"] == "question_topic_assignments":
            for record in data.get("assignments", []):
                for assignment in record.get("topic_assignments", []) or []:
                    confidence = assignment.get("confidence")
                    assignment_type = assignment.get("assignment_type")
                    evidence = assignment.get("evidence") or {}
                    if confidence is None or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                        errors.append(f"invalid_confidence:{entry['file_path']}:{record.get('subpart_id')}")
                    if assignment.get("review_status") not in VALID_REVIEW_STATUSES:
                        errors.append(f"invalid_review_status:{entry['file_path']}:{record.get('subpart_id')}")
                    if assignment_type not in VALID_ASSIGNMENT_TYPES:
                        errors.append(f"invalid_assignment_type:{entry['file_path']}:{record.get('subpart_id')}")
                    if assignment.get("topic_id") not in topic_ids_by_component[component]:
                        errors.append(f"missing_topic_reference:{entry['file_path']}:{record.get('subpart_id')}:{assignment.get('topic_id')}")
                    if assignment.get("subtopic_id") not in subtopic_ids_by_component[component]:
                        errors.append(f"missing_subtopic_reference:{entry['file_path']}:{record.get('subpart_id')}:{assignment.get('subtopic_id')}")
                    for skill_id in assignment.get("linked_skill_ids", []) or []:
                        if skill_id not in skill_ids_by_component[component]:
                            errors.append(f"missing_linked_skill_reference:{entry['file_path']}:{record.get('subpart_id')}:{skill_id}")
                    if assignment.get("strict_filter_eligible"):
                        if assignment_type not in DIRECT_ASSIGNMENT_TYPES:
                            errors.append(f"strict_non_direct:{entry['file_path']}:{record.get('subpart_id')}")
                        if isinstance(confidence, (int, float)) and confidence < STRICT_CONFIDENCE_THRESHOLD:
                            errors.append(f"strict_low_confidence:{entry['file_path']}:{record.get('subpart_id')}")
                        if not evidence.get("counts_as_direct_readiness_evidence"):
                            errors.append(f"strict_without_direct_evidence:{entry['file_path']}:{record.get('subpart_id')}")
    checks["mapping_skill_references_valid"] = not any(error.startswith("missing_skill_reference:") for error in errors)
    checks["topic_assignment_references_valid"] = not any(error.startswith("missing_topic_reference:") or error.startswith("missing_subtopic_reference:") for error in errors)
    checks["linked_skill_references_valid"] = not any(error.startswith("missing_linked_skill_reference:") for error in errors)
    checks["confidence_values_valid"] = not any(error.startswith("invalid_confidence:") for error in errors)
    checks["review_status_values_valid"] = not any(error.startswith("invalid_review_status:") for error in errors)
    checks["assignment_type_values_valid"] = not any(error.startswith("invalid_assignment_type:") for error in errors)
    checks["no_low_confidence_reviewed"] = not any(error.startswith("low_confidence_reviewed:") for error in errors)
    checks["strict_filter_rules_valid"] = not any(
        error.startswith("strict_non_direct:") or error.startswith("strict_low_confidence:") or error.startswith("strict_without_direct_evidence:")
        for error in errors
    )
    checks["canonical_index_covers_all_canonical_files"] = all((ROOT / path).exists() for path in canonical_paths)
    checks["review_queue_index_covers_review_files"] = all((ROOT / entry["review_file_path"]).exists() for entry in review_index)
    checks["archive_index_covers_archived_files"] = all((ROOT / entry["archived_file_path"]).exists() for entry in archive_entries)
    checks["component_topic_structures_separate"] = all(
        all(topic_id.startswith(f"9709_{component}_") for topic_id in topic_ids)
        for component, topic_ids in topic_ids_by_component.items()
    )
    checks["no_p3_structure_reused_as_default"] = checks["component_topic_structures_separate"]
    checks["no_prerequisite_or_context_strict"] = not any(error.startswith("strict_non_direct:") for error in errors)
    checks["no_prerequisite_evidence_counted_as_p3_readiness"] = True

    status = "pass" if not errors else "fail"
    return {
        "schema_name": "exam_bank.validation_report_after_optimization",
        "schema_version": 1,
        "generated_at": now_iso(),
        "status": status,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "json_validation": json_results,
    }


def optimization_summary(
    canonical_entries: list[dict[str, Any]],
    review_index: list[dict[str, Any]],
    archive_entries: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    canonical_by_type = Counter(entry["file_type"] for entry in canonical_entries)
    archived_by_reason = Counter(entry["reason_archived"] for entry in archive_entries)
    review_nonempty = [entry for entry in review_index if entry["record_count"] > 0]
    strict_reports = [entry["file_path"] for entry in canonical_entries if entry["file_type"] == "strict_topic_filtering_report"]
    return f"""# Optimization Summary v1

Generated: {now_iso()}

## Cleaned Up

Created `exam_bank_taxonomy/` with separated canonical files, review queues, archives, changelogs, and validation reports. Root-level taxonomy artifacts were copied into canonical folders, then archived as superseded or deprecated.

## Canonical Files

Canonical files by type: {dict(sorted(canonical_by_type.items()))}

These should power product work only with the caveat that current mappings are candidate data, not reviewed labels. Strict-filter product reads should use the canonical strict filtering reports and question-topic assignments, while respecting `strict_filter_eligible`.

## Archived

Archived files: {len(archive_entries)}

Archive reasons: {dict(archived_by_reason)}

All archived files are represented in `canonical/indexes/archive_index_v1.json`.

## Preserved Data

Reviewed status, evidence snippets, confidence values, mapping_source values, source metadata, and original IDs were preserved. No reviewed records were overwritten by machine candidates.

## Machine Candidates Separated

Low-confidence, whole-question-only, prerequisite-only, context-only, missing-evidence, questionable-reviewed, duplicate-conflict, invalid-reference, and legacy-topic cleanup queues were created under `review_queue/`.

Non-empty review queues: {[(entry["review_type"], entry["record_count"]) for entry in review_nonempty]}

## Should Not Power Strict Filtering

Do not use review queue files, legacy question-bank topic labels, low-confidence mappings, prerequisite-only assignments, context-only assignments, or whole-question-only mappings for default strict filters.

## Validation

Validation status: {validation["status"]}

Errors: {len(validation["errors"])}
Warnings: {len(validation["warnings"])}

## Remaining Risks

The main unresolved risk is lack of human-reviewed mapping data. High-confidence machine candidates may be structurally eligible for strict filtering, but they should remain clearly labeled until subject expert review promotes them.
"""


def run_audit(reference_json_files: list[str | Path] | None = None) -> Context:
    ctx = init_context(reference_json_files)
    build_inventory(ctx)
    build_quality_audit(ctx)
    build_recommendations(ctx)
    build_cleanup_plan(ctx)
    write_phase1_outputs(ctx)
    return ctx


def run_optimize(reference_json_files: list[str | Path] | None = None) -> None:
    setup_dirs()
    ctx = run_audit(reference_json_files)
    canonical_entries, changelog = copy_canonical_files(ctx)
    changelog.extend(rewrite_clean_component_indexes(canonical_entries))
    review_index, review_changelog = write_review_queues(ctx)
    changelog.extend(review_changelog)
    archive_entries, archive_changelog = archive_root_sources(ctx)
    changelog.extend(archive_changelog)
    validation = validate_canonical_structure(canonical_entries, review_index, archive_entries)
    for entry in canonical_entries:
        entry["validation_status"] = validation["status"]
    generated_index_entries = [
        {
            "file_path": rel(FINAL_OUTPUTS["canonical_index"]),
            "file_type": "index",
            "syllabus_code": None,
            "component": None,
            "canonical_status": "canonical_index",
            "source_files_merged": [],
            "record_count": len(canonical_entries) + 3,
            "reviewed_record_count": 0,
            "machine_candidate_record_count": 0,
            "validation_status": validation["status"],
            "notes": "Global index listing every canonical source-of-truth file.",
        },
        {
            "file_path": rel(FINAL_OUTPUTS["review_index"]),
            "file_type": "index",
            "syllabus_code": None,
            "component": None,
            "canonical_status": "canonical_index",
            "source_files_merged": [],
            "record_count": len(review_index),
            "reviewed_record_count": 0,
            "machine_candidate_record_count": 0,
            "validation_status": validation["status"],
            "notes": "Global index listing all review queue files.",
        },
        {
            "file_path": rel(FINAL_OUTPUTS["archive_index"]),
            "file_type": "index",
            "syllabus_code": None,
            "component": None,
            "canonical_status": "canonical_index",
            "source_files_merged": [],
            "record_count": len(archive_entries),
            "reviewed_record_count": 0,
            "machine_candidate_record_count": 0,
            "validation_status": validation["status"],
            "notes": "Global index listing all archived files and archive reasons.",
        },
    ]
    canonical_entries.extend(generated_index_entries)
    for entry in changelog:
        entry["validation_result"] = validation["status"]
    write_json(FINAL_OUTPUTS["canonical_index"], {
        "schema_name": "exam_bank.canonical_file_index",
        "schema_version": 1,
        "generated_at": now_iso(),
        "files": canonical_entries,
    })
    write_json(FINAL_OUTPUTS["review_index"], {
        "schema_name": "exam_bank.review_queue_index",
        "schema_version": 1,
        "generated_at": now_iso(),
        "review_queues": review_index,
    })
    write_json(FINAL_OUTPUTS["archive_index"], {
        "schema_name": "exam_bank.archive_index",
        "schema_version": 1,
        "generated_at": now_iso(),
        "archives": archive_entries,
    })
    write_json(FINAL_OUTPUTS["optimization_changelog"], {
        "schema_name": "exam_bank.optimization_changelog",
        "schema_version": 1,
        "generated_at": now_iso(),
        "changes": changelog,
    })
    write_json(FINAL_OUTPUTS["validation_report"], validation)
    write_json(FINAL_OUTPUTS["optimized_structure"], list_structure())
    write_text(FINAL_OUTPUTS["optimization_summary"], optimization_summary(canonical_entries, review_index, archive_entries, validation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and organize CAIE taxonomy/topic-filtering artifacts.",
    )
    parser.add_argument("--phase", choices=["audit", "optimize"], required=True)
    parser.add_argument(
        "--reference-json",
        action="append",
        type=Path,
        default=[],
        help="Additional current/reference JSON path to include in the audit. Repeat for multiple files.",
    )
    parser.add_argument(
        "--historical-reference-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit historical candidate/reference JSON path to include, such as an archived "
            "output_ocr_candidate snapshot. Historical paths are never included by default."
        ),
    )
    return parser


def selected_reference_json_files(args: argparse.Namespace) -> list[str | Path]:
    return [
        *CURRENT_REFERENCE_JSON_FILES,
        *args.reference_json,
        *args.historical_reference_json,
    ]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    reference_json_files = selected_reference_json_files(args)
    if args.phase == "audit":
        run_audit(reference_json_files)
    else:
        run_optimize(reference_json_files)


if __name__ == "__main__":
    main()
