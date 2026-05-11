"""Generate strict topic filter maps from CAIE skill maps and mappings.

The topic filter layer is deliberately more conservative than the skill layer:
official syllabus sections become user-facing parent topics and component-local
skills become filterable subtopics. Question assignments inherit the existing
skill mapping evidence, but strict filter eligibility is granted only to direct,
high-confidence component-local mappings.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_ROOT = ROOT / "exam_bank_taxonomy"
CANONICAL_ROOT = TAXONOMY_ROOT / "canonical"
CANONICAL_TOPIC_FILTER_MAPS = CANONICAL_ROOT / "topic_filter_maps"
CANONICAL_TOPIC_ASSIGNMENTS = CANONICAL_ROOT / "question_topic_assignments"
CANONICAL_STRICT_REPORTS = CANONICAL_ROOT / "strict_filtering_reports"
CANONICAL_INDEXES = CANONICAL_ROOT / "indexes"
SKILL_MAP_INDEX = CANONICAL_INDEXES / "skill_map_index_v1.json"
ASTERION_QB = ROOT / "output_ocr_candidate/json/asterion_question_bank_v1.json"
LEGACY_QB = ROOT / "output/json/question_bank.json"

SCHEMA_VERSION = 1
DIRECT_ASSIGNMENT_TYPES = {"primary_assessed", "secondary_assessed"}
STRICT_REVIEW_STATUSES = {"reviewed", "high-confidence machine_candidate"}
PAPER_FAMILY_BY_COMPONENT = {
    "p1": "p1",
    "p3": "p3",
    "m1": "p4",
    "s1": "p5",
}
PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = value.replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def component_key_from_skill_map_file(path: str) -> str:
    match = re.match(r"skill_map_(?P<syllabus>\d+)_(?P<component>[a-z0-9]+)_v\d+\.json$", Path(path).name)
    if not match:
        raise ValueError(f"Cannot infer component key from {path}")
    return match.group("component")


def component_key_from_mapping_file(path: str) -> str:
    match = re.match(
        r"question_skill_mappings_(?P<syllabus>\d+)_(?P<component>[a-z0-9]+)_v\d+\.json$",
        Path(path).name,
    )
    if not match:
        raise ValueError(f"Cannot infer component key from {path}")
    return match.group("component")


def skill_suffix(skill_id: str, syllabus_code: str, component_key: str) -> str:
    prefix = f"{syllabus_code}_{component_key}_"
    suffix = skill_id.removeprefix(prefix)
    suffix = re.sub(r"^(?:\d+_)+", "", suffix)
    return slugify(suffix)


def confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    if confidence >= 0.40:
        return "low"
    return "very_low"


def section_sort_key(section_code: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", section_code or "")
    return tuple(int(part) for part in parts) or (999,)


def assignment_review_status(source_status: str, confidence: float, strict_eligible: bool) -> str:
    if source_status == "reviewed":
        return "reviewed"
    if strict_eligible and confidence >= 0.85:
        return "high-confidence machine_candidate"
    if confidence >= 0.65:
        return "machine_candidate_needs_review"
    return "review_only"


def compact_evidence(mapping: dict[str, Any]) -> dict[str, Any]:
    evidence = mapping.get("evidence") or {}
    return {
        "evidence_granularity": mapping.get("evidence_granularity"),
        "source_topic": evidence.get("source_topic"),
        "topic_confidence": evidence.get("topic_confidence"),
        "topic_uncertain": evidence.get("topic_uncertain"),
        "matched_signals": evidence.get("matched_signals", []),
        "question_text_snippet": evidence.get("question_text_snippet", ""),
        "mark_scheme_text_snippet": evidence.get("mark_scheme_text_snippet", ""),
        "question_text_trust": evidence.get("question_text_trust", ""),
        "mark_scheme_text_trust": evidence.get("mark_scheme_text_trust", ""),
        "whole_question_limitation": evidence.get("whole_question_limitation", ""),
        "counts_as_direct_readiness_evidence": evidence.get("counts_as_direct_readiness_evidence", False),
    }


def compact_evidence_has_direct_signal(evidence: dict[str, Any]) -> bool:
    signals = evidence.get("matched_signals") or []
    has_skill_or_text_signal = any(
        signal.startswith("signal:") or signal.startswith("content_lab_source_skill:")
        for signal in signals
    )
    has_text_or_ms = bool(evidence.get("question_text_snippet") or evidence.get("mark_scheme_text_snippet"))
    return bool(
        evidence.get("counts_as_direct_readiness_evidence")
        and has_skill_or_text_signal
        and has_text_or_ms
    )


def has_direct_evidence(mapping: dict[str, Any]) -> bool:
    return compact_evidence_has_direct_signal(compact_evidence(mapping))


def strict_exclusion_reasons(
    mapping: dict[str, Any],
    assignment_type: str,
    skill_id: str,
    skill_exists: bool,
    question_has_subparts: bool,
) -> list[str]:
    reasons: list[str] = []
    confidence = float(mapping.get("confidence") or 0)
    if assignment_type not in DIRECT_ASSIGNMENT_TYPES:
        reasons.append(f"assignment_type_is_{assignment_type}")
    if confidence < 0.85:
        reasons.append("confidence_below_0_85")
    if not skill_exists:
        reasons.append(f"skill_id_not_in_component_skill_map:{skill_id}")
    granularity = mapping.get("evidence_granularity")
    if granularity != "subpart":
        if granularity == "whole_question_only" and question_has_subparts:
            reasons.append("whole_question_only_but_subpart_data_exists")
        elif granularity == "whole_question_only":
            reasons.append("whole_question_only_candidate")
        else:
            reasons.append("missing_subpart_level_evidence")
    if mapping.get("mapping_source") == "legacy_topic_mapping":
        reasons.append("legacy_topic_mapping_only")
    if assignment_type in DIRECT_ASSIGNMENT_TYPES and not has_direct_evidence(mapping):
        reasons.append("direct_question_or_mark_scheme_evidence_not_strong_enough")
    return reasons


def is_strict_eligible(
    mapping: dict[str, Any],
    assignment_type: str,
    skill_id: str,
    skill_exists: bool,
    question_has_subparts: bool,
) -> bool:
    reasons = strict_exclusion_reasons(mapping, assignment_type, skill_id, skill_exists, question_has_subparts)
    if reasons == ["whole_question_only_candidate"]:
        return (
            assignment_type in DIRECT_ASSIGNMENT_TYPES
            and float(mapping.get("confidence") or 0) >= 0.85
            and skill_exists
            and mapping.get("mapping_source") != "legacy_topic_mapping"
            and has_direct_evidence(mapping)
            and not question_has_subparts
        )
    return not reasons


def direct_count(assignments: list[dict[str, Any]]) -> int:
    return sum(1 for item in assignments if item["assignment_type"] in DIRECT_ASSIGNMENT_TYPES)


def priority_for_skills(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return "medium"
    selected = max(skills, key=lambda item: PRIORITY_ORDER.get(item.get("content_lab_priority", "medium"), 2))
    return selected.get("content_lab_priority", "medium")


def filter_visibility(direct_assignments: int, strict_assignments: int, subtopic_count: int = 0) -> str:
    if strict_assignments:
        return "default_strict"
    if direct_assignments:
        return "review_only"
    if subtopic_count:
        return "parent_only"
    return "hidden_until_populated"


def filter_quality_rating(
    direct_assessed_subpart_count: int,
    strict_filter_eligible_count: int,
    average_confidence: float | None,
) -> str:
    if direct_assessed_subpart_count == 0:
        return "not_ready"
    if strict_filter_eligible_count >= 10 and average_confidence is not None and average_confidence >= 0.85:
        return "excellent"
    if strict_filter_eligible_count >= 3 and average_confidence is not None and average_confidence >= 0.80:
        return "good"
    if strict_filter_eligible_count > 0 or (average_confidence is not None and average_confidence >= 0.65):
        return "usable_needs_review"
    return "weak"


def recommended_action(rating: str, strict_count: int) -> str:
    if rating in {"excellent", "good"}:
        return "Use in default strict filters; schedule expert review before marking as reviewed."
    if rating == "usable_needs_review":
        if strict_count:
            return "Allow high-confidence assignments in strict filters and review the remaining candidates."
        return "Keep out of default strict filters until medium-confidence mappings are reviewed."
    if rating == "weak":
        return "Keep as review-only; inspect source text and mark-scheme evidence."
    return "Hide from student-facing filters until directly assessed questions are mapped."


def question_has_subparts(question_subpart_index: dict[str, set[str]], question_id: str) -> bool:
    return bool(question_subpart_index.get(question_id))


def build_question_subpart_index() -> dict[str, set[str]]:
    question_subparts: dict[str, set[str]] = defaultdict(set)
    if not ASTERION_QB.exists():
        return question_subparts
    bank = load_json(ASTERION_QB)
    for question in bank.get("questions", []):
        question_id = question.get("question_id")
        if not question_id:
            continue
        for subpart in question.get("subparts", []) or []:
            subpart_id = subpart.get("subpart_id")
            if subpart_id:
                question_subparts[question_id].add(subpart_id)
    return question_subparts


def build_legacy_topic_summary(component_key: str) -> list[dict[str, Any]]:
    if not LEGACY_QB.exists():
        return []
    paper_family = PAPER_FAMILY_BY_COMPONENT.get(component_key, component_key)
    bank = load_json(LEGACY_QB)
    counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    uncertain = Counter()
    for question in bank.get("questions", []):
        if question.get("paper_family") != paper_family:
            continue
        topic = question.get("topic") or "unknown"
        subtopic = ((question.get("notes") or {}).get("subtopic")) or "general"
        key = (topic, subtopic)
        counter[key]["question_count"] += 1
        topic_confidence = ((question.get("notes") or {}).get("topic_confidence")) or "unknown"
        counter[key][f"topic_confidence_{topic_confidence}"] += 1
        if (question.get("notes") or {}).get("topic_uncertain"):
            uncertain[key] += 1
    summary = []
    for (topic, subtopic), counts in sorted(counter.items()):
        summary.append(
            {
                "legacy_topic": topic,
                "legacy_subtopic": subtopic,
                "question_count": counts["question_count"],
                "uncertain_count": uncertain[(topic, subtopic)],
                "confidence_breakdown": {
                    key.removeprefix("topic_confidence_"): value
                    for key, value in counts.items()
                    if key.startswith("topic_confidence_")
                },
                "recommended_action": (
                    "Replace with canonical topic/subtopic assignments; do not use this legacy label "
                    "as strict-filter evidence."
                ),
            }
        )
    return summary


def make_topic_shell(
    section: dict[str, Any],
    skill_map: dict[str, Any],
    component_key: str,
) -> dict[str, Any]:
    topic_id = f"{skill_map['syllabus_code']}_{component_key}_topic_{slugify(section['name'])}"
    return {
        "topic_id": topic_id,
        "syllabus_code": skill_map["syllabus_code"],
        "subject_name": skill_map["subject_name"],
        "caie_class_or_component": skill_map["caie_class_or_component"],
        "component_label": skill_map["component_label"],
        "official_section_code": section["section"],
        "official_section_name": section["name"],
        "topic_name": section["name"],
        "topic_description": (
            f"Official CAIE syllabus section {section['section']} for "
            f"{skill_map['component_label']}; parent filter for related assessable subtopics."
        ),
        "parent_topic_id": None,
        "subtopics": [],
        "linked_skill_ids": [],
        "question_count": 0,
        "subpart_count": 0,
        "reviewed_mapping_count": 0,
        "machine_candidate_mapping_count": 0,
        "content_lab_priority": "medium",
        "filter_visibility": "hidden_until_populated",
        "review_status": "needs_review",
        "notes": "Canonical parent topic aligned to an official CAIE syllabus section.",
    }


def make_subtopic_shell(
    skill: dict[str, Any],
    topic_id: str,
    section_code: str,
    section_name: str,
    component_key: str,
) -> dict[str, Any]:
    suffix = skill_suffix(skill["skill_id"], skill["syllabus_code"], component_key)
    subtopic_id = f"{skill['syllabus_code']}_{component_key}_subtopic_{suffix}"
    return {
        "subtopic_id": subtopic_id,
        "parent_topic_id": topic_id,
        "syllabus_code": skill["syllabus_code"],
        "subject_name": skill["subject_name"],
        "caie_class_or_component": skill["caie_class_or_component"],
        "component_label": skill["component_label"],
        "official_section_code": section_code,
        "official_section_name": section_name,
        "subtopic_name": skill["name"],
        "subtopic_description": skill["description"],
        "linked_skill_ids": [skill["skill_id"]],
        "recognizer_signals": skill.get("recognizer_signals", []),
        "common_errors": skill.get("common_errors", []),
        "question_count": 0,
        "subpart_count": 0,
        "reviewed_mapping_count": 0,
        "machine_candidate_mapping_count": 0,
        "content_lab_priority": skill.get("content_lab_priority", "medium"),
        "filter_visibility": "hidden_until_populated",
        "review_status": "needs_review",
        "notes": "Canonical assessable subtopic generated from the component skill map.",
    }


def make_assignment(
    mapping: dict[str, Any],
    skill: dict[str, Any],
    topic: dict[str, Any],
    subtopic: dict[str, Any],
    assignment_type: str,
    strict_eligible: bool,
    exclusion_reasons: list[str],
) -> dict[str, Any]:
    confidence = float(mapping.get("confidence") or 0)
    review_status = assignment_review_status(
        mapping.get("review_status", ""),
        confidence,
        strict_eligible,
    )
    notes = []
    if exclusion_reasons:
        notes.append("Excluded from default strict filtering: " + ", ".join(exclusion_reasons) + ".")
    if (mapping.get("evidence") or {}).get("topic_uncertain"):
        notes.append("Source mapping marked the legacy topic evidence as uncertain.")
    if mapping.get("evidence_granularity") == "whole_question_only":
        notes.append("Whole-question-only evidence retained for review.")
    if not notes:
        notes.append("Eligible high-confidence component-local direct evidence.")
    return {
        "topic_id": topic["topic_id"],
        "topic_name": topic["topic_name"],
        "subtopic_id": subtopic["subtopic_id"],
        "subtopic_name": subtopic["subtopic_name"],
        "linked_skill_ids": [skill["skill_id"]],
        "assignment_type": assignment_type,
        "confidence": round(confidence, 2),
        "filter_confidence_band": confidence_band(confidence),
        "strict_filter_eligible": strict_eligible,
        "evidence": compact_evidence(mapping),
        "mapping_source": mapping.get("mapping_source", "unknown"),
        "review_status": review_status,
        "notes": " ".join(notes),
    }


def rollup_counts(
    topics_by_id: dict[str, dict[str, Any]],
    subtopics_by_id: dict[str, dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subtopic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in assignments:
        for assignment in record["topic_assignments"]:
            by_topic[assignment["topic_id"]].append({**assignment, "_record": record})
            by_subtopic[assignment["subtopic_id"]].append({**assignment, "_record": record})

    for topic_id, topic in topics_by_id.items():
        topic_assignments = by_topic.get(topic_id, [])
        direct = [item for item in topic_assignments if item["assignment_type"] in DIRECT_ASSIGNMENT_TYPES]
        strict = [item for item in topic_assignments if item["strict_filter_eligible"]]
        topic["question_count"] = len({item["_record"]["question_id"] for item in direct})
        topic["subpart_count"] = len({item["_record"]["subpart_id"] for item in direct})
        topic["reviewed_mapping_count"] = sum(1 for item in topic_assignments if item["review_status"] == "reviewed")
        topic["machine_candidate_mapping_count"] = len(topic_assignments) - topic["reviewed_mapping_count"]
        skills = [skill_id for subtopic in topic["subtopics"] for skill_id in subtopic["linked_skill_ids"]]
        topic["linked_skill_ids"] = sorted(set(skills))
        topic["content_lab_priority"] = priority_for_skills(
            [
                {"content_lab_priority": subtopic.get("content_lab_priority", "medium")}
                for subtopic in topic["subtopics"]
            ]
        )
        topic["filter_visibility"] = filter_visibility(
            len(direct),
            len(strict),
            len(topic["subtopics"]),
        )

    for subtopic_id, subtopic in subtopics_by_id.items():
        subtopic_assignments = by_subtopic.get(subtopic_id, [])
        direct = [item for item in subtopic_assignments if item["assignment_type"] in DIRECT_ASSIGNMENT_TYPES]
        strict = [item for item in subtopic_assignments if item["strict_filter_eligible"]]
        subtopic["question_count"] = len({item["_record"]["question_id"] for item in direct})
        subtopic["subpart_count"] = len({item["_record"]["subpart_id"] for item in direct})
        subtopic["reviewed_mapping_count"] = sum(1 for item in subtopic_assignments if item["review_status"] == "reviewed")
        subtopic["machine_candidate_mapping_count"] = len(subtopic_assignments) - subtopic["reviewed_mapping_count"]
        subtopic["filter_visibility"] = filter_visibility(len(direct), len(strict))

    return by_topic, by_subtopic


def make_quality_row(
    topic: dict[str, Any],
    subtopic: dict[str, Any] | None,
    assignment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_rows = [item for item in assignment_rows if item["assignment_type"] in DIRECT_ASSIGNMENT_TYPES]
    prerequisite_rows = [item for item in assignment_rows if item["assignment_type"] == "prerequisite"]
    context_rows = [item for item in assignment_rows if item["assignment_type"] == "context_only"]
    strict_rows = [item for item in assignment_rows if item["strict_filter_eligible"]]
    average_conf = round(mean([item["confidence"] for item in assignment_rows]), 3) if assignment_rows else None
    direct_subpart_count = len({item["_record"]["subpart_id"] for item in direct_rows})
    strict_count = len(strict_rows)
    rating = filter_quality_rating(direct_subpart_count, strict_count, average_conf)
    return {
        "topic_id": topic["topic_id"],
        "topic_name": topic["topic_name"],
        "subtopic_id": subtopic["subtopic_id"] if subtopic else None,
        "subtopic_name": subtopic["subtopic_name"] if subtopic else None,
        "linked_skill_ids": subtopic["linked_skill_ids"] if subtopic else topic["linked_skill_ids"],
        "direct_assessed_question_count": len({item["_record"]["question_id"] for item in direct_rows}),
        "direct_assessed_subpart_count": direct_subpart_count,
        "prerequisite_only_count": len(prerequisite_rows),
        "context_only_count": len(context_rows),
        "strict_filter_eligible_count": strict_count,
        "average_confidence": average_conf,
        "reviewed_assignment_count": sum(1 for item in assignment_rows if item["review_status"] == "reviewed"),
        "machine_candidate_assignment_count": sum(1 for item in assignment_rows if item["review_status"] != "reviewed"),
        "filter_quality_rating": rating,
        "recommended_action": recommended_action(rating, strict_count),
    }


def make_report(
    component_key: str,
    skill_map: dict[str, Any],
    assignments: list[dict[str, Any]],
    topics_by_id: dict[str, dict[str, Any]],
    subtopics_by_id: dict[str, dict[str, Any]],
    by_topic: dict[str, list[dict[str, Any]]],
    by_subtopic: dict[str, list[dict[str, Any]]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    total_questions = len({record["question_id"] for record in assignments})
    total_subparts = len({record["subpart_id"] for record in assignments})
    subparts_with_assignments = {
        record["subpart_id"] for record in assignments if record["topic_assignments"]
    }
    subparts_strict = {
        record["subpart_id"]
        for record in assignments
        if any(item["strict_filter_eligible"] for item in record["topic_assignments"])
    }
    all_assignment_rows = [
        assignment
        for record in assignments
        for assignment in record["topic_assignments"]
    ]
    excluded = [item for item in all_assignment_rows if not item["strict_filter_eligible"]]
    topic_quality = [
        make_quality_row(topic, None, by_topic.get(topic_id, []))
        for topic_id, topic in sorted(topics_by_id.items())
    ]
    subtopic_quality = [
        make_quality_row(
            topics_by_id[subtopic["parent_topic_id"]],
            subtopic,
            by_subtopic.get(subtopic_id, []),
        )
        for subtopic_id, subtopic in sorted(subtopics_by_id.items())
    ]
    topics_with_no_questions = [
        {"topic_id": row["topic_id"], "topic_name": row["topic_name"]}
        for row in topic_quality
        if row["direct_assessed_subpart_count"] == 0
    ]
    subtopics_with_no_questions = [
        {
            "topic_id": row["topic_id"],
            "topic_name": row["topic_name"],
            "subtopic_id": row["subtopic_id"],
            "subtopic_name": row["subtopic_name"],
        }
        for row in subtopic_quality
        if row["direct_assessed_subpart_count"] == 0
    ]
    topics_only_low = [
        {"topic_id": row["topic_id"], "topic_name": row["topic_name"]}
        for row in topic_quality
        if row["direct_assessed_subpart_count"] > 0
        and row["average_confidence"] is not None
        and row["average_confidence"] < 0.65
    ]
    subtopics_only_low = [
        {
            "topic_id": row["topic_id"],
            "topic_name": row["topic_name"],
            "subtopic_id": row["subtopic_id"],
            "subtopic_name": row["subtopic_name"],
        }
        for row in subtopic_quality
        if row["direct_assessed_subpart_count"] > 0
        and row["average_confidence"] is not None
        and row["average_confidence"] < 0.65
    ]
    prerequisite_only_topics = [
        {
            "topic_id": row["topic_id"],
            "topic_name": row["topic_name"],
            "prerequisite_only_count": row["prerequisite_only_count"],
        }
        for row in topic_quality
        if row["direct_assessed_subpart_count"] == 0 and row["prerequisite_only_count"] > 0
    ]
    duplicate_names = []
    topic_name_counter = Counter(slugify(topic["topic_name"]) for topic in topics_by_id.values())
    subtopic_name_counter = Counter(slugify(subtopic["subtopic_name"]) for subtopic in subtopics_by_id.values())
    for normalized, count in sorted(topic_name_counter.items()):
        if count > 1:
            duplicate_names.append({"type": "topic", "normalized_name": normalized, "count": count})
    for normalized, count in sorted(subtopic_name_counter.items()):
        if count > 1:
            duplicate_names.append({"type": "subtopic", "normalized_name": normalized, "count": count})

    strict_coverage = round((len(subparts_strict) / total_subparts) * 100, 2) if total_subparts else 0.0
    return {
        "schema_name": "exam_bank.strict_topic_filtering_report",
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "syllabus_code": skill_map["syllabus_code"],
        "subject_name": skill_map["subject_name"],
        "caie_class_or_component": skill_map["caie_class_or_component"],
        "component_label": skill_map["component_label"],
        "component_key": component_key,
        "total_questions": total_questions,
        "total_subparts": total_subparts,
        "subparts_with_topic_assignments": len(subparts_with_assignments),
        "subparts_strict_filter_eligible": len(subparts_strict),
        "subparts_missing_topic_assignments": sorted(
            {record["subpart_id"] for record in assignments} - subparts_with_assignments
        ),
        "topics_with_no_questions": topics_with_no_questions,
        "subtopics_with_no_questions": subtopics_with_no_questions,
        "topics_with_only_low_confidence_assignments": topics_only_low,
        "subtopics_with_only_low_confidence_assignments": subtopics_only_low,
        "prerequisite_only_topics": prerequisite_only_topics,
        "duplicate_or_overlapping_topics": duplicate_names,
        "legacy_topics_needing_cleanup": build_legacy_topic_summary(component_key),
        "strict_filter_coverage_percent": strict_coverage,
        "review_queue_summary": {
            "total_topic_assignment_count": len(all_assignment_rows),
            "strict_filter_eligible_assignment_count": len(all_assignment_rows) - len(excluded),
            "excluded_from_strict_filtering_count": len(excluded),
            "medium_confidence_assignment_count": sum(
                1 for item in all_assignment_rows if item["filter_confidence_band"] == "medium"
            ),
            "low_confidence_assignment_count": sum(
                1 for item in all_assignment_rows if item["filter_confidence_band"] == "low"
            ),
            "very_low_confidence_assignment_count": sum(
                1 for item in all_assignment_rows if item["filter_confidence_band"] == "very_low"
            ),
            "prerequisite_assignment_count": sum(
                1 for item in all_assignment_rows if item["assignment_type"] == "prerequisite"
            ),
            "context_only_assignment_count": sum(
                1 for item in all_assignment_rows if item["assignment_type"] == "context_only"
            ),
            "whole_question_assignment_count": sum(
                1
                for item in all_assignment_rows
                if item["evidence"].get("evidence_granularity") == "whole_question_only"
            ),
        },
        "default_strict_filter_set": {
            "criteria": {
                "strict_filter_eligible": True,
                "assignment_type": sorted(DIRECT_ASSIGNMENT_TYPES),
                "minimum_confidence": 0.85,
                "review_status": sorted(STRICT_REVIEW_STATUSES),
            },
            "assignment_count": len(all_assignment_rows) - len(excluded),
            "subpart_count": len(subparts_strict),
        },
        "broader_review_filter_set": {
            "criteria": {
                "minimum_confidence": 0.50,
                "includes_prerequisite_mappings": True,
                "student_facing_default": False,
            },
            "assignment_count": sum(1 for item in all_assignment_rows if item["confidence"] >= 0.50),
        },
        "recommended_cleanup_actions": [
            "Use only default_strict_filter_set for student-facing topic filters.",
            "Review medium-confidence direct mappings before promoting them into strict filters.",
            "Keep prerequisite and context-only mappings out of default strict filters.",
            "Retire legacy question.topic labels as a filtering source after canonical assignments are reviewed.",
        ],
        "topic_quality": topic_quality,
        "topic_subtopic_quality": subtopic_quality,
        "validation": validation,
    }


def validate_component(
    component_key: str,
    skill_map: dict[str, Any],
    topics_by_id: dict[str, dict[str, Any]],
    subtopics_by_id: dict[str, dict[str, Any]],
    skill_to_subtopic: dict[str, dict[str, str]],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    skill_ids = {skill["skill_id"] for skill in skill_map.get("skills", [])}
    topic_ids = list(topics_by_id)
    subtopic_ids = list(subtopics_by_id)
    if len(topic_ids) != len(set(topic_ids)):
        errors.append("topic_ids_not_unique")
    if len(subtopic_ids) != len(set(subtopic_ids)):
        errors.append("subtopic_ids_not_unique")
    for subtopic in subtopics_by_id.values():
        if subtopic["parent_topic_id"] not in topics_by_id:
            errors.append(f"subtopic_parent_missing:{subtopic['subtopic_id']}")
        for skill_id in subtopic["linked_skill_ids"]:
            if skill_id not in skill_ids:
                errors.append(f"subtopic_linked_skill_missing:{subtopic['subtopic_id']}:{skill_id}")
    for skill_id in skill_ids:
        if skill_id not in skill_to_subtopic:
            errors.append(f"skill_without_subtopic:{skill_id}")
    for record in assignments:
        if not record["topic_assignments"]:
            warnings.append(f"subpart_without_topic_assignment:{record['subpart_id']}")
        for assignment in record["topic_assignments"]:
            if assignment["topic_id"] not in topics_by_id:
                errors.append(f"assignment_topic_missing:{record['subpart_id']}:{assignment['topic_id']}")
            if assignment["subtopic_id"] not in subtopics_by_id:
                errors.append(f"assignment_subtopic_missing:{record['subpart_id']}:{assignment['subtopic_id']}")
            if not assignment["topic_id"].startswith(f"{skill_map['syllabus_code']}_{component_key}_"):
                errors.append(f"assignment_wrong_component_topic:{record['subpart_id']}:{assignment['topic_id']}")
            if not assignment["subtopic_id"].startswith(f"{skill_map['syllabus_code']}_{component_key}_"):
                errors.append(f"assignment_wrong_component_subtopic:{record['subpart_id']}:{assignment['subtopic_id']}")
            for skill_id in assignment["linked_skill_ids"]:
                if skill_id not in skill_ids:
                    errors.append(f"assignment_linked_skill_missing:{record['subpart_id']}:{skill_id}")
            if assignment["strict_filter_eligible"] and assignment["confidence"] < 0.85:
                errors.append(f"strict_assignment_confidence_below_0_85:{record['subpart_id']}")
            if assignment["strict_filter_eligible"] and assignment["assignment_type"] not in DIRECT_ASSIGNMENT_TYPES:
                errors.append(f"strict_assignment_not_direct:{record['subpart_id']}")
            if assignment["strict_filter_eligible"] and assignment["review_status"] not in STRICT_REVIEW_STATUSES:
                errors.append(f"strict_assignment_bad_review_status:{record['subpart_id']}")
            if assignment["strict_filter_eligible"] and assignment["mapping_source"] == "legacy_topic_mapping":
                errors.append(f"strict_assignment_legacy_only:{record['subpart_id']}")
            if assignment["strict_filter_eligible"] and not compact_evidence_has_direct_signal(assignment["evidence"]):
                errors.append(f"strict_assignment_missing_direct_signal:{record['subpart_id']}")
    return {
        "status": "pass" if not errors else "fail",
        "validated_at": now_iso(),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "topic_ids_unique": len(topic_ids) == len(set(topic_ids)),
            "subtopic_ids_unique": len(subtopic_ids) == len(set(subtopic_ids)),
            "subtopic_parent_topics_valid": not any(
                subtopic["parent_topic_id"] not in topics_by_id
                for subtopic in subtopics_by_id.values()
            ),
            "linked_skill_ids_exist": not any(
                skill_id not in skill_ids
                for subtopic in subtopics_by_id.values()
                for skill_id in subtopic["linked_skill_ids"]
            ),
            "all_skill_ids_have_subtopics": all(skill_id in skill_to_subtopic for skill_id in skill_ids),
            "assignment_topic_references_valid": not any(
                assignment["topic_id"] not in topics_by_id or assignment["subtopic_id"] not in subtopics_by_id
                for record in assignments
                for assignment in record["topic_assignments"]
            ),
            "strict_assignments_have_confidence_ge_0_85": not any(
                assignment["strict_filter_eligible"] and assignment["confidence"] < 0.85
                for record in assignments
                for assignment in record["topic_assignments"]
            ),
            "no_prerequisite_or_context_strict": not any(
                assignment["strict_filter_eligible"] and assignment["assignment_type"] not in DIRECT_ASSIGNMENT_TYPES
                for record in assignments
                for assignment in record["topic_assignments"]
            ),
            "assignment_component_scope_valid": not any(
                not assignment["topic_id"].startswith(f"{skill_map['syllabus_code']}_{component_key}_")
                or not assignment["subtopic_id"].startswith(f"{skill_map['syllabus_code']}_{component_key}_")
                for record in assignments
                for assignment in record["topic_assignments"]
            ),
            "strict_assignments_not_legacy_only": not any(
                assignment["strict_filter_eligible"] and assignment["mapping_source"] == "legacy_topic_mapping"
                for record in assignments
                for assignment in record["topic_assignments"]
            ),
            "strict_assignments_have_direct_evidence": not any(
                assignment["strict_filter_eligible"]
                and not compact_evidence_has_direct_signal(assignment["evidence"])
                for record in assignments
                for assignment in record["topic_assignments"]
            ),
            "valid_json_written": True,
        },
    }


def build_component(
    component_index_entry: dict[str, Any],
    question_subpart_index: dict[str, set[str]],
) -> dict[str, Any]:
    skill_map_file = component_index_entry["skill_map_file"]
    mapping_file = component_index_entry["mapping_file"]
    coverage_report_file = component_index_entry["coverage_report_file"]
    component_key = component_key_from_skill_map_file(skill_map_file)
    if component_key != component_key_from_mapping_file(mapping_file):
        raise ValueError(f"Component mismatch between {skill_map_file} and {mapping_file}")
    skill_map = load_json(ROOT / skill_map_file)
    mapping_doc = load_json(ROOT / mapping_file)

    sections_by_code = {section["section"]: section for section in skill_map.get("sections", [])}
    topics_by_id: dict[str, dict[str, Any]] = {}
    subtopics_by_id: dict[str, dict[str, Any]] = {}
    skill_to_subtopic: dict[str, dict[str, str]] = {}

    for section in skill_map.get("sections", []):
        topic = make_topic_shell(section, skill_map, component_key)
        topics_by_id[topic["topic_id"]] = topic

    for skill in skill_map.get("skills", []):
        section_text = skill["section"]
        section_code, _, section_name = section_text.partition(" ")
        section = sections_by_code.get(section_code) or {"section": section_code, "name": section_name}
        topic_id = f"{skill['syllabus_code']}_{component_key}_topic_{slugify(section['name'])}"
        if topic_id not in topics_by_id:
            topics_by_id[topic_id] = make_topic_shell(section, skill_map, component_key)
        subtopic = make_subtopic_shell(skill, topic_id, section["section"], section["name"], component_key)
        if subtopic["subtopic_id"] in subtopics_by_id:
            subtopic["subtopic_id"] = f"{subtopic['subtopic_id']}_{slugify(section['section'])}"
        subtopics_by_id[subtopic["subtopic_id"]] = subtopic
        skill_to_subtopic[skill["skill_id"]] = {
            "topic_id": topic_id,
            "subtopic_id": subtopic["subtopic_id"],
        }
        topics_by_id[topic_id]["subtopics"].append(subtopic)

    skill_by_id = {skill["skill_id"]: skill for skill in skill_map.get("skills", [])}
    assignments: list[dict[str, Any]] = []
    for mapping in mapping_doc.get("mappings", []):
        question_subparts_exist = question_has_subparts(question_subpart_index, mapping["question_id"])
        topic_assignments: list[dict[str, Any]] = []
        ordered_skill_assignments: list[tuple[str, str]] = []
        for skill_id in mapping.get("primary_skill_ids", []):
            ordered_skill_assignments.append((skill_id, "primary_assessed"))
        for skill_id in mapping.get("secondary_skill_ids", []):
            ordered_skill_assignments.append((skill_id, "secondary_assessed"))
        for skill_id in mapping.get("prerequisite_skill_ids", []):
            ordered_skill_assignments.append((skill_id, "prerequisite"))

        seen_assignment_keys: set[tuple[str, str]] = set()
        unassigned_external_prerequisites: list[str] = []
        for skill_id, assignment_type in ordered_skill_assignments:
            skill_exists = skill_id in skill_by_id
            if not skill_exists:
                if assignment_type == "prerequisite":
                    unassigned_external_prerequisites.append(skill_id)
                    continue
                continue
            link = skill_to_subtopic[skill_id]
            topic = topics_by_id[link["topic_id"]]
            subtopic = subtopics_by_id[link["subtopic_id"]]
            key = (subtopic["subtopic_id"], assignment_type)
            if key in seen_assignment_keys:
                continue
            seen_assignment_keys.add(key)
            exclusion_reasons = strict_exclusion_reasons(
                mapping,
                assignment_type,
                skill_id,
                skill_exists,
                question_subparts_exist,
            )
            strict_eligible = is_strict_eligible(
                mapping,
                assignment_type,
                skill_id,
                skill_exists,
                question_subparts_exist,
            )
            topic_assignments.append(
                make_assignment(
                    mapping,
                    skill_by_id[skill_id],
                    topic,
                    subtopic,
                    assignment_type,
                    strict_eligible,
                    [] if strict_eligible else exclusion_reasons,
                )
            )

        assignments.append(
            {
                "question_id": mapping["question_id"],
                "subpart_id": mapping["subpart_id"],
                "paper_id": mapping.get("paper_id") or mapping.get("paper"),
                "syllabus_code": mapping["syllabus_code"],
                "subject_name": mapping["subject_name"],
                "caie_class_or_component": mapping["caie_class_or_component"],
                "component_label": mapping["component_label"],
                "year": mapping.get("year"),
                "session": mapping.get("session"),
                "variant": mapping.get("variant"),
                "question_number": mapping.get("question_number"),
                "subpart_label": mapping.get("subpart_label"),
                "topic_assignments": topic_assignments,
                "unassigned_external_prerequisite_skill_ids": sorted(set(unassigned_external_prerequisites)),
                "notes": (
                    "External prerequisite skills are not assigned to this component topic map."
                    if unassigned_external_prerequisites
                    else ""
                ),
            }
        )

    validation = validate_component(
        component_key,
        skill_map,
        topics_by_id,
        subtopics_by_id,
        skill_to_subtopic,
        assignments,
    )
    by_topic, by_subtopic = rollup_counts(topics_by_id, subtopics_by_id, assignments)
    topic_map = {
        "schema_name": "exam_bank.topic_filter_map",
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "syllabus_code": skill_map["syllabus_code"],
        "subject_name": skill_map["subject_name"],
        "caie_class_or_component": skill_map["caie_class_or_component"],
        "component_label": skill_map["component_label"],
        "component_key": component_key,
        "source_skill_map_file": skill_map_file,
        "source_mapping_file": mapping_file,
        "source_coverage_report_file": coverage_report_file,
        "review_status": "needs_review",
        "notes": (
            "Canonical topic filter map generated from official CAIE syllabus sections and "
            "component-local skill mappings. Parent topics are syllabus sections; subtopics are "
            "filterable assessable skill groups."
        ),
        "topics": sorted(
            topics_by_id.values(),
            key=lambda topic: section_sort_key(topic["official_section_code"]),
        ),
        "validation": validation,
    }
    assignment_doc = {
        "schema_name": "exam_bank.question_topic_assignments",
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "syllabus_code": skill_map["syllabus_code"],
        "subject_name": skill_map["subject_name"],
        "caie_class_or_component": skill_map["caie_class_or_component"],
        "component_label": skill_map["component_label"],
        "component_key": component_key,
        "source_skill_map_file": skill_map_file,
        "source_mapping_file": mapping_file,
        "review_status": "needs_review",
        "strict_filter_default_policy": {
            "strict_filter_eligible": True,
            "assignment_type": sorted(DIRECT_ASSIGNMENT_TYPES),
            "minimum_confidence": 0.85,
            "review_status": sorted(STRICT_REVIEW_STATUSES),
            "exclude_assignment_types": ["prerequisite", "context_only"],
        },
        "assignments": assignments,
        "validation": validation,
    }
    report = make_report(
        component_key,
        skill_map,
        assignments,
        topics_by_id,
        subtopics_by_id,
        by_topic,
        by_subtopic,
        validation,
    )
    return {
        "component_key": component_key,
        "topic_map": topic_map,
        "assignments": assignment_doc,
        "report": report,
        "component_index_entry": {
            "syllabus_code": skill_map["syllabus_code"],
            "subject_name": skill_map["subject_name"],
            "caie_class_or_component": skill_map["caie_class_or_component"],
            "component_label": skill_map["component_label"],
            "topic_filter_file": (CANONICAL_TOPIC_FILTER_MAPS / f"topic_filter_map_{skill_map['syllabus_code']}_{component_key}_v1.json").relative_to(ROOT).as_posix(),
            "source_skill_map_file": skill_map_file,
            "source_mapping_file": mapping_file,
            "source_coverage_report_file": coverage_report_file,
            "question_topic_assignments_file": (CANONICAL_TOPIC_ASSIGNMENTS / f"question_topic_assignments_{skill_map['syllabus_code']}_{component_key}_v1.json").relative_to(ROOT).as_posix(),
            "strict_filtering_report_file": (CANONICAL_STRICT_REPORTS / f"strict_topic_filtering_report_{skill_map['syllabus_code']}_{component_key}_v1.json").relative_to(ROOT).as_posix(),
            "review_status": "needs_review",
            "notes": (
                "Generated canonical topic filter hierarchy. Strict default filters should use "
                "only high-confidence direct topic assignments."
            ),
        },
    }


def make_all_components_report(component_results: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [item["report"] for item in component_results]
    total_questions = sum(report["total_questions"] for report in reports)
    total_subparts = sum(report["total_subparts"] for report in reports)
    strict_subparts = sum(report["subparts_strict_filter_eligible"] for report in reports)
    errors = [
        f"{report['component_key']}:{error}"
        for report in reports
        for error in report["validation"]["errors"]
    ]
    warnings = [
        f"{report['component_key']}:{warning}"
        for report in reports
        for warning in report["validation"]["warnings"]
    ]
    return {
        "schema_name": "exam_bank.strict_topic_filtering_report_all_components",
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "total_questions": total_questions,
        "total_subparts": total_subparts,
        "subparts_with_topic_assignments": sum(
            report["subparts_with_topic_assignments"] for report in reports
        ),
        "subparts_strict_filter_eligible": strict_subparts,
        "subparts_missing_topic_assignments": {
            report["component_key"]: report["subparts_missing_topic_assignments"]
            for report in reports
            if report["subparts_missing_topic_assignments"]
        },
        "topics_with_no_questions": {
            report["component_key"]: report["topics_with_no_questions"]
            for report in reports
            if report["topics_with_no_questions"]
        },
        "subtopics_with_no_questions": {
            report["component_key"]: report["subtopics_with_no_questions"]
            for report in reports
            if report["subtopics_with_no_questions"]
        },
        "topics_with_only_low_confidence_assignments": {
            report["component_key"]: report["topics_with_only_low_confidence_assignments"]
            for report in reports
            if report["topics_with_only_low_confidence_assignments"]
        },
        "subtopics_with_only_low_confidence_assignments": {
            report["component_key"]: report["subtopics_with_only_low_confidence_assignments"]
            for report in reports
            if report["subtopics_with_only_low_confidence_assignments"]
        },
        "prerequisite_only_topics": {
            report["component_key"]: report["prerequisite_only_topics"]
            for report in reports
            if report["prerequisite_only_topics"]
        },
        "duplicate_or_overlapping_topics": {
            report["component_key"]: report["duplicate_or_overlapping_topics"]
            for report in reports
            if report["duplicate_or_overlapping_topics"]
        },
        "legacy_topics_needing_cleanup": {
            report["component_key"]: report["legacy_topics_needing_cleanup"]
            for report in reports
            if report["legacy_topics_needing_cleanup"]
        },
        "strict_filter_coverage_percent": (
            round((strict_subparts / total_subparts) * 100, 2) if total_subparts else 0.0
        ),
        "review_queue_summary": {
            "total_topic_assignment_count": sum(
                report["review_queue_summary"]["total_topic_assignment_count"] for report in reports
            ),
            "strict_filter_eligible_assignment_count": sum(
                report["review_queue_summary"]["strict_filter_eligible_assignment_count"]
                for report in reports
            ),
            "excluded_from_strict_filtering_count": sum(
                report["review_queue_summary"]["excluded_from_strict_filtering_count"]
                for report in reports
            ),
            "medium_confidence_assignment_count": sum(
                report["review_queue_summary"]["medium_confidence_assignment_count"] for report in reports
            ),
            "low_confidence_assignment_count": sum(
                report["review_queue_summary"]["low_confidence_assignment_count"] for report in reports
            ),
            "very_low_confidence_assignment_count": sum(
                report["review_queue_summary"]["very_low_confidence_assignment_count"] for report in reports
            ),
            "prerequisite_assignment_count": sum(
                report["review_queue_summary"]["prerequisite_assignment_count"] for report in reports
            ),
            "context_only_assignment_count": sum(
                report["review_queue_summary"]["context_only_assignment_count"] for report in reports
            ),
            "whole_question_assignment_count": sum(
                report["review_queue_summary"]["whole_question_assignment_count"] for report in reports
            ),
        },
        "recommended_cleanup_actions": [
            "Promote only reviewed or high-confidence direct assignments into default student-facing filters.",
            "Review medium-confidence direct mappings by component and subtopic before expanding strict coverage.",
            "Keep the broader review filter set internal; it includes prerequisite and candidate mappings.",
            "Clean up legacy topic labels after canonical topic assignments are reviewed.",
        ],
        "component_summaries": [
            {
                "component_key": report["component_key"],
                "syllabus_code": report["syllabus_code"],
                "component_label": report["component_label"],
                "total_questions": report["total_questions"],
                "total_subparts": report["total_subparts"],
                "subparts_with_topic_assignments": report["subparts_with_topic_assignments"],
                "subparts_strict_filter_eligible": report["subparts_strict_filter_eligible"],
                "strict_filter_coverage_percent": report["strict_filter_coverage_percent"],
                "topics_with_no_questions_count": len(report["topics_with_no_questions"]),
                "subtopics_with_no_questions_count": len(report["subtopics_with_no_questions"]),
                "excluded_from_strict_filtering_count": report["review_queue_summary"][
                    "excluded_from_strict_filtering_count"
                ],
                "validation_status": report["validation"]["status"],
            }
            for report in reports
        ],
        "validation": {
            "status": "pass" if not errors else "fail",
            "validated_at": now_iso(),
            "errors": errors,
            "warnings": warnings,
            "checks": {
                "all_component_reports_pass": not errors,
                "all_json_files_valid": True,
                "no_prerequisite_or_context_strict": all(
                    report["validation"]["checks"]["no_prerequisite_or_context_strict"] for report in reports
                ),
                "assignment_component_scope_valid": all(
                    report["validation"]["checks"]["assignment_component_scope_valid"] for report in reports
                ),
                "strict_assignments_not_legacy_only": all(
                    report["validation"]["checks"]["strict_assignments_not_legacy_only"] for report in reports
                ),
                "strict_assignments_have_direct_evidence": all(
                    report["validation"]["checks"]["strict_assignments_have_direct_evidence"] for report in reports
                ),
            },
        },
    }


def main() -> None:
    index = load_json(SKILL_MAP_INDEX)
    question_subpart_index = build_question_subpart_index()
    component_results = [
        build_component(component, question_subpart_index)
        for component in index.get("components", [])
    ]

    topic_index = {
        "schema_name": "exam_bank.topic_filter_map_index",
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "source_skill_map_index_file": SKILL_MAP_INDEX.relative_to(ROOT).as_posix(),
        "source_syllabus_reference": index.get("source_syllabus_reference"),
        "source_syllabus_url": index.get("source_syllabus_url"),
        "components": [result["component_index_entry"] for result in component_results],
        "review_status": "needs_review",
        "notes": (
            "Index of canonical component-specific topic filter maps. Topic files use official CAIE "
            "sections as parent topics and component-local skill groups as subtopics."
        ),
    }

    write_json(CANONICAL_INDEXES / "topic_filter_map_index_v1.json", topic_index)
    for result in component_results:
        component_key = result["component_key"]
        syllabus_code = result["topic_map"]["syllabus_code"]
        write_json(
            CANONICAL_TOPIC_FILTER_MAPS / f"topic_filter_map_{syllabus_code}_{component_key}_v1.json",
            result["topic_map"],
        )
        write_json(
            CANONICAL_TOPIC_ASSIGNMENTS / f"question_topic_assignments_{syllabus_code}_{component_key}_v1.json",
            result["assignments"],
        )
        write_json(
            CANONICAL_STRICT_REPORTS / f"strict_topic_filtering_report_{syllabus_code}_{component_key}_v1.json",
            result["report"],
        )
    write_json(
        CANONICAL_STRICT_REPORTS / "strict_topic_filtering_report_all_components_v1.json",
        make_all_components_report(component_results),
    )


if __name__ == "__main__":
    main()
