from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from exam_bank.content_lab_auto_review import (
    automated_decisions_for_audit,
    automated_mark_event_ids,
    automated_source_records,
)


AUDIT_VERSION = "asterion_content_lab_readiness_audit_v1"
DEFAULT_SAMPLE_SEED = "asterion-content-lab-p3-20260527"
TARGET_PASS_RATE = 0.70

P3_REGION_ORDER = [
    "Algebra Vault",
    "Logarithm Observatory",
    "Trigonometry Spire",
    "Calculus Cliffs",
    "Integral Terraces",
    "Vectors Gate",
    "Differential Shrine",
    "Iteration Forge",
    "Argand Atrium",
    "Unresolved P3 Skill Region",
]

SECTION_REGION_NAMES = {
    "3.1": "Algebra Vault",
    "3.2": "Logarithm Observatory",
    "3.3": "Trigonometry Spire",
    "3.4": "Calculus Cliffs",
    "3.5": "Integral Terraces",
    "3.6": "Iteration Forge",
    "3.7": "Vectors Gate",
    "3.8": "Differential Shrine",
    "3.9": "Argand Atrium",
}

REGION_KEYWORDS = {
    "Algebra Vault": ("algebra", "modulus", "polynomial", "partial_fraction", "binomial", "series"),
    "Logarithm Observatory": ("log", "exponential", "exponent"),
    "Trigonometry Spire": ("trig", "sine", "cosine", "tan", "compound", "double_angle"),
    "Calculus Cliffs": ("differentiat", "derivative", "implicit", "parametric", "normal", "tangent"),
    "Integral Terraces": ("integrat", "area_under_curve", "volume"),
    "Vectors Gate": ("vector", "line", "plane", "scalar_product"),
    "Differential Shrine": ("differential_equation", "differential equation"),
    "Iteration Forge": ("iteration", "numerical", "fixed_point", "newton"),
    "Argand Atrium": ("complex", "argand", "modulus_argument", "polar"),
}

REPAIRABLE_BLOCKERS = {
    "blocked_schema_mismatch",
    "blocked_artifact_path",
}
HUMAN_REVIEW_BLOCKERS = {
    "blocked_skill_mapping",
    "blocked_mapping_review_gate",
    "blocked_mark_events",
    "blocked_quarantine_ambiguous",
    "review_required",
}
ASTERION_SIDE_BLOCKERS = {
    "blocked_schema_mismatch",
}


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    question_id: str
    subpart_id: str
    region: str
    region_source: str
    sample_bucket: str
    status: str
    passed: bool
    blocker_classes: tuple[str, ...]
    blocker_reasons: tuple[str, ...]
    generation_ready: bool
    student_runtime_safe: bool
    app_safe_after_reviewed_gates: bool
    legacy_schema_mismatch: bool
    mapping_validation_contradiction: bool


def run_audit(
    *,
    candidates_path: Path,
    question_bank_path: Path | None,
    topic_routing_path: Path | None,
    asterion_bank_path: Path | None,
    artifact_root: Path,
    out_dir: Path,
    sample_size: int = 100,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
    skill_map_path: Path | None = Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    question_skill_mappings_path: Path | None = Path(
        "exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"
    ),
    reviewed_source_skills_path: Path | None = Path("data/review/p3_exact_skill_reviewed_decisions.v1.json"),
    reviewed_mark_events_path: Path | None = Path("data/review/p3_exact_skill_reviewed_mark_events.v1.json"),
    reviewed_decisions_path: Path | None = None,
    target_pass_rate: float = TARGET_PASS_RATE,
    full_pool_report: bool = False,
) -> dict[str, Any]:
    candidates_payload = _read_json(candidates_path)
    candidates = [item for item in candidates_payload.get("candidates", []) if isinstance(item, dict)]
    asterion_questions = _questions_by_id(_read_json(asterion_bank_path) if asterion_bank_path else {})
    question_bank = _questions_by_id(_read_json(question_bank_path) if question_bank_path else {})
    topic_routing = _topic_routing_by_question(_read_json(topic_routing_path) if topic_routing_path else {})
    skill_regions = _skill_regions(_read_json(skill_map_path) if skill_map_path and skill_map_path.exists() else {})
    reviewed_source_payload = (
        _read_json(reviewed_source_skills_path) if reviewed_source_skills_path and reviewed_source_skills_path.exists() else {}
    )
    reviewed_mark_events_payload = (
        _read_json(reviewed_mark_events_path) if reviewed_mark_events_path and reviewed_mark_events_path.exists() else {}
    )
    automated_review_payload = _read_json(reviewed_decisions_path) if reviewed_decisions_path and reviewed_decisions_path.exists() else {}
    clean_source_records = _clean_source_skill_records_by_subpart(reviewed_source_payload)
    approved_mark_event_ids = _approved_mark_event_ids(reviewed_mark_events_payload)
    for subpart_id, records in automated_source_records(automated_review_payload).items():
        clean_source_records[subpart_id].extend(records)
    approved_mark_event_ids.update(automated_mark_event_ids(automated_review_payload))
    automated_review_decisions = automated_decisions_for_audit(automated_review_payload, artifact_root=artifact_root)
    mapping_regions = _question_skill_mapping_regions(
        _read_json(question_skill_mappings_path)
        if question_skill_mappings_path and question_skill_mappings_path.exists()
        else {},
        skill_regions=skill_regions,
    )

    audited = [
        audit_candidate(
            candidate,
            asterion_questions=asterion_questions,
            question_bank=question_bank,
            topic_routing=topic_routing,
            skill_regions=skill_regions,
            mapping_regions=mapping_regions,
            artifact_root=artifact_root,
            clean_source_records=clean_source_records,
            approved_mark_event_ids=approved_mark_event_ids,
            automated_review_decision=automated_review_decisions.get(str(candidate.get("candidate_id") or "")),
        )
        for candidate in candidates
        if str(candidate.get("paper_family") or "").lower() == "p3"
    ]
    audited.sort(key=lambda row: (row.region, row.question_id, row.subpart_id, row.candidate_id))
    sample = stratified_sample(audited, sample_size=sample_size, seed=sample_seed)

    summary = _summary(
        candidates=candidates,
        audited=audited,
        sample=sample,
        sample_size_requested=sample_size,
        sample_seed=sample_seed,
        target_pass_rate=target_pass_rate,
        candidates_path=candidates_path,
        question_bank_path=question_bank_path,
        topic_routing_path=topic_routing_path,
        asterion_bank_path=asterion_bank_path,
    )
    coverage = reviewed_evidence_coverage(
        candidates=[candidate for candidate in candidates if str(candidate.get("paper_family") or "").lower() == "p3"],
        audited=audited,
        reviewed_source_skills_path=reviewed_source_skills_path,
        reviewed_mark_events_path=reviewed_mark_events_path,
        reviewed_decisions_path=reviewed_decisions_path,
        source_payload=reviewed_source_payload,
        mark_payload=reviewed_mark_events_payload,
        automated_review_payload=automated_review_payload,
    )
    summary["reviewed_evidence_coverage"] = coverage["summary"]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "audit_summary.json", summary)
    _write_markdown(out_dir / "audit_summary.md", summary)
    _write_inventory_csv(out_dir / "p3_candidate_inventory.csv", audited)
    _write_inventory_csv(out_dir / "p3_sample_frame.csv", sample)
    _write_inventory_csv(out_dir / "p3_sample_results.csv", sample)
    _write_inventory_csv(out_dir / "sample_results.csv", sample)
    _write_blocker_breakdown(out_dir / "blocker_breakdown.csv", sample)
    _write_blocker_breakdown(out_dir / "blocker_summary.csv", sample)
    _write_blocker_by_region(out_dir / "blocker_by_region.csv", sample)
    _write_filtered_report(
        out_dir / "schema_gate_mismatch_report.csv",
        audited,
        lambda row: row.legacy_schema_mismatch or "blocked_schema_mismatch" in row.blocker_classes,
    )
    _write_filtered_report(out_dir / "skill_mapping_gap_report.csv", audited, lambda row: "blocked_skill_mapping" in row.blocker_classes)
    _write_filtered_report(out_dir / "mark_evidence_gap_report.csv", audited, lambda row: "blocked_mark_events" in row.blocker_classes)
    _write_filtered_report(out_dir / "artifact_path_gap_report.csv", audited, lambda row: "blocked_artifact_path" in row.blocker_classes)
    _write_filtered_report(out_dir / "promotion_candidates.csv", audited, lambda row: row.passed)
    _write_filtered_report(out_dir / "still_blocked_candidates.csv", audited, lambda row: not row.passed)
    _write_filtered_report(
        out_dir / "repair_candidates.csv",
        audited,
        lambda row: any(blocker in REPAIRABLE_BLOCKERS for blocker in row.blocker_classes) or row.legacy_schema_mismatch,
    )
    _write_json(out_dir / "reviewed_evidence_coverage.json", coverage["summary"])
    _write_coverage_markdown(out_dir / "reviewed_evidence_coverage.md", coverage["summary"])
    _write_dict_csv(out_dir / "candidate_evidence_join.csv", coverage["candidate_rows"])
    _write_dict_csv(out_dir / "human_review_queue.csv", coverage["human_review_rows"])
    if full_pool_report:
        _write_full_pool_reports(out_dir, audited, sample, summary)
    _write_recommendations(out_dir / "next_iteration_recommendations.md", summary)
    return summary


def reviewed_evidence_coverage(
    *,
    candidates: list[dict[str, Any]],
    audited: list[CandidateAudit],
    reviewed_source_skills_path: Path | None,
    reviewed_mark_events_path: Path | None,
    reviewed_decisions_path: Path | None = None,
    source_payload: dict[str, Any] | None = None,
    mark_payload: dict[str, Any] | None = None,
    automated_review_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_payload = (
        source_payload
        if source_payload is not None
        else _read_json(reviewed_source_skills_path)
        if reviewed_source_skills_path and reviewed_source_skills_path.exists()
        else {}
    )
    mark_payload = (
        mark_payload
        if mark_payload is not None
        else _read_json(reviewed_mark_events_path)
        if reviewed_mark_events_path and reviewed_mark_events_path.exists()
        else {}
    )
    source_records = [record for record in source_payload.get("records", []) if isinstance(record, dict)]
    mark_decisions = [decision for decision in mark_payload.get("decisions", []) if isinstance(decision, dict)]
    automated_review_payload = automated_review_payload or {}
    automated_source_records_list = [
        record
        for records in automated_source_records(automated_review_payload).values()
        for record in records
        if isinstance(record, dict)
    ]
    automated_mark_decisions = [
        decision
        for decision in automated_review_payload.get("mark_event_decisions", [])
        if isinstance(decision, dict)
    ]
    source_records.extend(automated_source_records_list)
    mark_decisions.extend(automated_mark_decisions)
    source_by_subpart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        subpart_id = str(record.get("subpart_id") or "").strip()
        if subpart_id:
            source_by_subpart[subpart_id].append(record)
    approved_mark_events = {
        str(decision.get("event_id") or "").strip(): decision
        for decision in mark_decisions
        if _mark_event_decision_generation_satisfying(decision)
    }

    audits_by_id = {row.candidate_id: row for row in audited}
    candidate_rows: list[dict[str, Any]] = []
    human_review_rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: str(item.get("candidate_id") or "")):
        row = audits_by_id.get(str(candidate.get("candidate_id") or ""))
        if row is None:
            continue
        source_matches = source_by_subpart.get(row.subpart_id, [])
        clean_source_matches = [record for record in source_matches if _source_skill_record_generation_satisfying(record)]
        non_generation_source_matches = [record for record in source_matches if not _source_skill_record_generation_satisfying(record)]
        source_event_ids = _strings(candidate.get("source_mark_event_ids"))
        approved_source_event_ids = [event_id for event_id in source_event_ids if event_id in approved_mark_events]
        all_mark_events_reviewed = bool(source_event_ids) and len(approved_source_event_ids) == len(source_event_ids)
        clean_source_surfaced = bool(candidate.get("source_skill_review_satisfied"))
        mark_events_surfaced = (candidate.get("mark_event_review_gate") or {}).get("status") == "allow"
        mapping_surfaced = bool(candidate.get("mapping_review_satisfied"))
        evidence_complete = bool(clean_source_matches) and all_mark_events_reviewed
        evidence_not_surfaced = evidence_complete and (
            not clean_source_surfaced or not mark_events_surfaced or not mapping_surfaced
        )
        missing_source = not clean_source_matches
        missing_mark_events = not all_mark_events_reviewed
        blocked = not row.passed
        queue_reasons = []
        if missing_source:
            queue_reasons.append("review_exact_skill_source_skill_mapping")
        if missing_mark_events:
            queue_reasons.append("review_mark_events_for_candidate_part")
        if non_generation_source_matches:
            queue_reasons.append("resolve_non_generation_satisfying_exact_skill_record")
        if row.legacy_schema_mismatch:
            queue_reasons.append("asterion_legacy_validator_contract_update")

        candidate_row = {
            "candidate_id": row.candidate_id,
            "question_id": row.question_id,
            "subpart_id": row.subpart_id,
            "region": row.region,
            "passed": row.passed,
            "blocker_classes": "|".join(row.blocker_classes),
            "clean_reviewed_exact_skill_record_count": len(clean_source_matches),
            "any_reviewed_exact_skill_record_count": len(source_matches),
            "non_generation_satisfying_exact_skill_record_count": len(non_generation_source_matches),
            "clean_source_skill_evidence_exists": bool(clean_source_matches),
            "clean_source_skill_evidence_surfaced": clean_source_surfaced,
            "source_mark_event_count": len(source_event_ids),
            "approved_reviewed_mark_event_count": len(approved_source_event_ids),
            "all_mark_events_reviewed": all_mark_events_reviewed,
            "mark_event_evidence_surfaced": mark_events_surfaced,
            "mapping_review_surfaced": mapping_surfaced,
            "evidence_complete_but_not_surfaced": evidence_not_surfaced,
            "safe_compatibility_bridge_candidate": row.legacy_schema_mismatch,
            "requires_human_review": blocked and not evidence_not_surfaced,
            "human_review_reasons": "|".join(_dedupe(queue_reasons)),
            "reviewed_exact_skill_decision_ids": "|".join(str(record.get("evidence_id") or "") for record in clean_source_matches),
            "approved_mark_event_ids": "|".join(approved_source_event_ids),
        }
        candidate_rows.append(candidate_row)
        if candidate_row["requires_human_review"]:
            human_review_rows.append(candidate_row)

    row_count = len(candidate_rows)
    passed_count = sum(1 for row in candidate_rows if row["passed"])
    summary = {
        "reviewed_source_skills_path": str(reviewed_source_skills_path) if reviewed_source_skills_path else None,
        "reviewed_mark_events_path": str(reviewed_mark_events_path) if reviewed_mark_events_path else None,
        "reviewed_decisions_path": str(reviewed_decisions_path) if reviewed_decisions_path else None,
        "p3_candidate_count": row_count,
        "existing_reviewed_exact_skill_evidence_count": sum(row["clean_source_skill_evidence_exists"] for row in candidate_rows),
        "existing_reviewed_source_skill_evidence_count": sum(row["clean_source_skill_evidence_exists"] for row in candidate_rows),
        "existing_reviewed_mark_event_subpart_evidence_count": sum(row["all_mark_events_reviewed"] for row in candidate_rows),
        "fails_only_because_reviewed_evidence_not_surfaced_count": sum(
            row["evidence_complete_but_not_surfaced"] and not row["passed"] for row in candidate_rows
        ),
        "fails_because_reviewed_evidence_absent_count": sum(
            (not row["passed"]) and (not row["clean_source_skill_evidence_exists"] or not row["all_mark_events_reviewed"])
            for row in candidate_rows
        ),
        "fails_with_ambiguous_or_quarantined_evidence_count": sum(
            (not row["passed"]) and row["non_generation_satisfying_exact_skill_record_count"] > 0 for row in candidate_rows
        ),
        "legacy_schema_mismatch_count": sum(row["safe_compatibility_bridge_candidate"] for row in candidate_rows),
        "safe_compatibility_export_bridge_candidate_count": sum(row["safe_compatibility_bridge_candidate"] for row in candidate_rows),
        "requires_human_review_no_matter_code_count": sum(row["requires_human_review"] for row in candidate_rows),
        "currently_passed_count": passed_count,
        "current_pass_rate": round(passed_count / row_count, 4) if row_count else 0.0,
        "source_review_record_count": len(source_records),
        "source_review_generation_satisfying_record_count": sum(_source_skill_record_generation_satisfying(record) for record in source_records),
        "reviewed_mark_event_decision_count": len(mark_decisions),
        "reviewed_mark_event_generation_satisfying_decision_count": len(approved_mark_events),
        "automated_reviewed_decision_count": len(automated_review_payload.get("records", []))
        if isinstance(automated_review_payload.get("records"), list)
        else 0,
    }
    return {"summary": summary, "candidate_rows": candidate_rows, "human_review_rows": human_review_rows}


def audit_candidate(
    candidate: dict[str, Any],
    *,
    asterion_questions: dict[str, dict[str, Any]],
    question_bank: dict[str, dict[str, Any]],
    topic_routing: dict[str, dict[str, Any]],
    skill_regions: dict[str, str],
    mapping_regions: dict[str, tuple[str, str]],
    artifact_root: Path,
    clean_source_records: dict[str, list[dict[str, Any]]] | None = None,
    approved_mark_event_ids: set[str] | None = None,
    automated_review_decision: dict[str, Any] | None = None,
) -> CandidateAudit:
    candidate_id = str(candidate.get("candidate_id") or "")
    question_id = str(candidate.get("question_id") or "")
    subpart_id = str(candidate.get("subpart_id") or "")
    question = asterion_questions.get(question_id) or question_bank.get(question_id) or {}
    region, region_source = _candidate_region(candidate, question, topic_routing.get(question_id, {}), skill_regions, mapping_regions)

    blocker_classes: list[str] = []
    blocker_reasons: list[str] = []
    generation_gate = candidate.get("generation_gate") if isinstance(candidate.get("generation_gate"), dict) else {}
    mark_event_gate = candidate.get("mark_event_review_gate") if isinstance(candidate.get("mark_event_review_gate"), dict) else {}
    mapping_gate = candidate.get("mapping_review_gate") if isinstance(candidate.get("mapping_review_gate"), dict) else {}
    source_skill_gate = candidate.get("source_skill_review_gate") if isinstance(candidate.get("source_skill_review_gate"), dict) else {}
    clean_source_records = clean_source_records or {}
    approved_mark_event_ids = approved_mark_event_ids or set()
    source_skill_backed_by_review_file = _candidate_source_skill_backed_by_review_file(
        candidate,
        clean_source_records.get(subpart_id, []),
    )
    mark_events_backed_by_review_file = _candidate_mark_events_backed_by_review_file(candidate, approved_mark_event_ids)
    automated_review_satisfies_generation = automated_review_decision is not None
    automated_skill_ids = _strings((automated_review_decision or {}).get("approved_source_skill_ids")) or _strings(
        (automated_review_decision or {}).get("approved_exact_skill_ids")
    )

    if not question_id or not question:
        blocker_classes.append("blocked_missing_evidence")
        blocker_reasons.append("question_record_missing")

    if str(candidate.get("paper_family") or "").lower() != "p3":
        blocker_classes.append("fail")
        blocker_reasons.append("not_p3_candidate")

    _add_artifact_blockers(candidate, question, artifact_root, blocker_classes, blocker_reasons)

    mapping_status = _status_from_question(question, "mapping_status")
    validation_status = _status_from_question(question, "validation_status")
    if mapping_status == "fail":
        blocker_classes.append("blocked_skill_mapping")
        blocker_reasons.append("mapping_status_fail")
    if validation_status == "fail":
        blocker_classes.append("fail")
        blocker_reasons.append("validation_status_fail")
    mapping_validation_contradiction = mapping_status == "fail" and validation_status == "pass"
    if mapping_validation_contradiction:
        blocker_classes.append("fail")
        blocker_reasons.append("mapping_fail_validation_pass_contradiction")

    source_skill_ids = _strings(candidate.get("source_skill_ids")) or automated_skill_ids
    reviewed_skill_ids = _strings(candidate.get("reviewed_source_skill_ids")) or automated_skill_ids
    if not source_skill_ids:
        blocker_classes.append("blocked_skill_mapping")
        blocker_reasons.append("missing_source_skill_ids")
    if not reviewed_skill_ids or (not candidate.get("source_skill_review_satisfied") and not automated_review_satisfies_generation):
        blocker_classes.append("blocked_skill_mapping")
        blocker_reasons.extend(_strings(source_skill_gate.get("block_reasons")) or ["reviewed_source_skill_gate_not_satisfied"])
    elif not source_skill_backed_by_review_file:
        blocker_classes.append("blocked_skill_mapping")
        blocker_reasons.append("reviewed_source_skill_gate_unbacked_by_reviewed_source_file")
    if not candidate.get("mapping_review_satisfied") and not automated_review_satisfies_generation:
        blocker_classes.append("blocked_mapping_review_gate")
        blocker_reasons.extend(_strings(mapping_gate.get("block_reasons")) or _strings(candidate.get("mapping_review_blocked_reasons")))
    elif (candidate.get("source_skill_review_satisfied") or automated_review_satisfies_generation) and not source_skill_backed_by_review_file:
        blocker_classes.append("blocked_mapping_review_gate")
        blocker_reasons.append("mapping_review_gate_unbacked_by_reviewed_source_file")

    if str(mark_event_gate.get("status") or "") != "allow" and not automated_review_satisfies_generation:
        blocker_classes.append("blocked_mark_events")
        blocker_reasons.extend(_strings(mark_event_gate.get("block_reasons")) or ["reviewed_mark_event_gate_not_satisfied"])
    elif not mark_events_backed_by_review_file:
        blocker_classes.append("blocked_mark_events")
        blocker_reasons.append("mark_event_gate_unbacked_by_reviewed_source_file")
    if not candidate.get("source_mark_event_count"):
        blocker_classes.append("blocked_mark_events")
        blocker_reasons.append("missing_source_mark_events")

    generation_reasons = _strings(generation_gate.get("block_reasons"))
    if str(generation_gate.get("status") or "") != "allow" and not automated_review_satisfies_generation:
        blocker_classes.append("review_required")
        blocker_reasons.extend(generation_reasons or ["generation_gate_not_allow"])
    if any("quarantin" in reason for reason in generation_reasons):
        blocker_classes.append("blocked_quarantine_ambiguous")
    if _unsafe_generated_content_present(candidate):
        blocker_classes.append("fail")
        blocker_reasons.append("generated_content_present_in_candidate_export")

    legacy_schema_mismatch = (
        str(generation_gate.get("status") or "") == "allow"
        and candidate.get("mapping_review_satisfied") is True
        and candidate.get("source_skill_review_satisfied") is True
        and not bool((candidate.get("candidate_selection") or {}).get("reviewed_or_approved_subpart"))
    )

    blocker_classes = _dedupe(blocker_classes)
    blocker_reasons = _dedupe(blocker_reasons)
    passed = not blocker_classes
    generation_ready = str(generation_gate.get("status") or "") == "allow" and not generation_reasons
    sample_bucket = "generation_allow" if generation_ready else str(candidate.get("review_status") or "blocked")
    return CandidateAudit(
        candidate_id=candidate_id,
        question_id=question_id,
        subpart_id=subpart_id,
        region=region,
        region_source=region_source,
        sample_bucket=sample_bucket,
        status="pass" if passed else "fail",
        passed=passed,
        blocker_classes=tuple(blocker_classes),
        blocker_reasons=tuple(blocker_reasons),
        generation_ready=generation_ready,
        student_runtime_safe=False,
        app_safe_after_reviewed_gates=passed,
        legacy_schema_mismatch=legacy_schema_mismatch,
        mapping_validation_contradiction=mapping_validation_contradiction,
    )


def stratified_sample(rows: list[CandidateAudit], *, sample_size: int, seed: str) -> list[CandidateAudit]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return list(rows)

    by_region: dict[str, list[CandidateAudit]] = defaultdict(list)
    for row in rows:
        by_region[row.region].append(row)

    selected: dict[str, CandidateAudit] = {}
    for region in _ordered_regions(by_region):
        region_rows = by_region[region]
        by_bucket: dict[str, list[CandidateAudit]] = defaultdict(list)
        for row in region_rows:
            by_bucket[row.sample_bucket].append(row)
        for bucket in sorted(by_bucket):
            if len(selected) >= sample_size:
                break
            chosen = sorted(by_bucket[bucket], key=lambda row: _sample_sort_key(row, seed))[0]
            selected[chosen.candidate_id] = chosen

    region_count = len(by_region)
    allocations = _region_allocations(by_region, sample_size, preselected=selected)
    for region in _ordered_regions(by_region):
        region_rows = sorted(by_region[region], key=lambda row: _sample_sort_key(row, seed))
        target = allocations.get(region, 0)
        current = sum(1 for row in selected.values() if row.region == region)
        for row in region_rows:
            if len(selected) >= sample_size or current >= target:
                break
            if row.candidate_id in selected:
                continue
            selected[row.candidate_id] = row
            current += 1

    if len(selected) < sample_size:
        for row in sorted(rows, key=lambda item: _sample_sort_key(item, seed)):
            if len(selected) >= sample_size:
                break
            selected.setdefault(row.candidate_id, row)

    return sorted(selected.values(), key=lambda row: (_region_rank(row.region), row.question_id, row.subpart_id, row.candidate_id))[
        :sample_size
    ]


def _region_allocations(
    by_region: dict[str, list[CandidateAudit]], sample_size: int, *, preselected: dict[str, CandidateAudit]
) -> dict[str, int]:
    total = sum(len(rows) for rows in by_region.values())
    allocations: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for region, rows in by_region.items():
        minimum = 1 if rows else 0
        already = sum(1 for row in preselected.values() if row.region == region)
        raw = sample_size * (len(rows) / total)
        allocation = max(minimum, already, min(len(rows), int(raw)))
        allocations[region] = allocation
        fractions.append((raw - int(raw), region))
    while sum(allocations.values()) < sample_size:
        progressed = False
        for _, region in sorted(fractions, reverse=True):
            if allocations[region] < len(by_region[region]):
                allocations[region] += 1
                progressed = True
                if sum(allocations.values()) >= sample_size:
                    break
        if not progressed:
            break
    while sum(allocations.values()) > sample_size:
        for region in sorted(allocations, key=lambda item: (allocations[item], -len(by_region[item])), reverse=True):
            floor = sum(1 for row in preselected.values() if row.region == region) or 1
            if allocations[region] > floor:
                allocations[region] -= 1
                break
        else:
            break
    return allocations


def _summary(
    *,
    candidates: list[dict[str, Any]],
    audited: list[CandidateAudit],
    sample: list[CandidateAudit],
    sample_size_requested: int,
    sample_seed: str,
    target_pass_rate: float,
    candidates_path: Path,
    question_bank_path: Path | None,
    topic_routing_path: Path | None,
    asterion_bank_path: Path | None,
) -> dict[str, Any]:
    sample_passed = sum(1 for row in sample if row.passed)
    sample_failed = len(sample) - sample_passed
    sample_blockers = Counter(blocker for row in sample for blocker in row.blocker_classes)
    all_blockers = Counter(blocker for row in audited for blocker in row.blocker_classes)
    sample_reasons = Counter(reason for row in sample for reason in row.blocker_reasons)
    return {
        "audit_version": AUDIT_VERSION,
        "source_files": {
            "candidates": str(candidates_path),
            "question_bank": str(question_bank_path) if question_bank_path else None,
            "topic_routing": str(topic_routing_path) if topic_routing_path else None,
            "asterion_bank": str(asterion_bank_path) if asterion_bank_path else None,
        },
        "total_candidates": len(candidates),
        "p3_candidates": len(audited),
        "sample_size_requested": sample_size_requested,
        "sample_size": len(sample),
        "sample_seed": sample_seed,
        "sample_method": (
            "deterministic stratified sample by P3 Asterion region with one deterministic row per "
            "region/status bucket before proportional fill; stable SHA-256 sort over seed and candidate_id"
        ),
        "sample_passed": sample_passed,
        "sample_failed": sample_failed,
        "sample_pass_rate": round(sample_passed / len(sample), 4) if sample else 0.0,
        "target_pass_rate": target_pass_rate,
        "target_met": bool(sample and sample_passed / len(sample) >= target_pass_rate),
        "top_blockers": _counter_items(sample_blockers),
        "top_blocker_reasons": _counter_items(sample_reasons),
        "full_p3_top_blockers": _counter_items(all_blockers),
        "regions_covered": sorted({row.region for row in sample}, key=_region_rank),
        "region_counts_in_sample": dict(Counter(row.region for row in sample)),
        "region_counts_in_p3_inventory": dict(Counter(row.region for row in audited)),
        "generation_ready_not_student_runtime_safe": sum(1 for row in audited if row.generation_ready and not row.student_runtime_safe),
        "app_safe_only_after_reviewed_mapping_evidence_gates": sum(row.app_safe_after_reviewed_gates for row in audited),
        "legacy_schema_mismatch_count": sum(row.legacy_schema_mismatch for row in audited),
        "mapping_validation_contradiction_count": sum(row.mapping_validation_contradiction for row in audited),
        "blocker_classes_code_data_repairable": sorted(set(all_blockers) & REPAIRABLE_BLOCKERS),
        "blocker_classes_requiring_human_review": sorted(set(all_blockers) & HUMAN_REVIEW_BLOCKERS),
        "blocker_classes_requiring_asterion_side_schema_or_validator_change": sorted(
            set(all_blockers) & ASTERION_SIDE_BLOCKERS
        )
        + (["legacy_candidate_selection_review_flag_missing"] if any(row.legacy_schema_mismatch for row in audited) else []),
        "student_runtime_changed": False,
        "trust_gates_weakened": False,
    }


def _candidate_region(
    candidate: dict[str, Any],
    question: dict[str, Any],
    topic_route: dict[str, Any],
    skill_regions: dict[str, str],
    mapping_regions: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    for field in ("reviewed_source_skill_ids", "source_skill_ids"):
        for skill_id in _strings(candidate.get(field)):
            if skill_id in skill_regions:
                return skill_regions[skill_id], field

    subpart_id = str(candidate.get("subpart_id") or "")
    question_id = str(candidate.get("question_id") or "")
    for key in (subpart_id, f"{question_id}_whole", question_id):
        if key in mapping_regions:
            return mapping_regions[key]

    text = " ".join(
        _strings(
            [
                question.get("topic"),
                (question.get("notes") or {}).get("subtopic") if isinstance(question.get("notes"), dict) else None,
                topic_route.get("primary_topic_id"),
            ]
        )
    ).lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return region, "topic_keyword"
    return "Unresolved P3 Skill Region", "unresolved"


def _skill_regions(payload: dict[str, Any]) -> dict[str, str]:
    result = {}
    for skill in payload.get("skills", []):
        if not isinstance(skill, dict):
            continue
        skill_id = str(skill.get("skill_id") or "")
        section = str(skill.get("section") or "")
        region = _region_from_section(section)
        if skill_id and region:
            result[skill_id] = region
    return result


def _question_skill_mapping_regions(payload: dict[str, Any], *, skill_regions: dict[str, str]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for mapping in payload.get("mappings", []):
        if not isinstance(mapping, dict):
            continue
        skill_ids = (
            _strings(mapping.get("primary_skill_ids"))
            + _strings(mapping.get("secondary_skill_ids"))
            + _strings(mapping.get("prerequisite_skill_ids"))
        )
        region = next((skill_regions[skill_id] for skill_id in skill_ids if skill_id in skill_regions), None)
        if not region:
            continue
        source = "question_skill_mapping_for_stratification"
        for key in _strings([mapping.get("subpart_id"), mapping.get("question_id")]):
            result.setdefault(key, (region, source))
    return result


def _clean_source_skill_records_by_subpart(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict) or not _source_skill_record_generation_satisfying(record):
            continue
        subpart_id = str(record.get("subpart_id") or "").strip()
        if subpart_id:
            result[subpart_id].append(record)
    return result


def _approved_mark_event_ids(payload: dict[str, Any]) -> set[str]:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return set()
    return {
        str(decision.get("event_id") or "").strip()
        for decision in decisions
        if isinstance(decision, dict)
        and str(decision.get("event_id") or "").strip()
        and _mark_event_decision_generation_satisfying(decision)
    }


def _candidate_source_skill_backed_by_review_file(candidate: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    candidate_skill_ids = set(_strings(candidate.get("reviewed_source_skill_ids")) or _strings(candidate.get("source_skill_ids")))
    candidate_event_ids = set(_strings(candidate.get("source_mark_event_ids")))
    for record in records:
        reviewed_skill_ids = set(_strings(record.get("reviewed_source_skill_ids")))
        reviewed_event_ids = {
            str(ref.get("event_id") or "").strip()
            for ref in record.get("mark_event_refs") or []
            if isinstance(ref, dict) and str(ref.get("event_id") or "").strip()
        }
        if candidate_skill_ids and reviewed_skill_ids and not candidate_skill_ids.intersection(reviewed_skill_ids):
            continue
        if candidate_event_ids and reviewed_event_ids and not candidate_event_ids.issubset(reviewed_event_ids):
            continue
        return True
    return False


def _candidate_mark_events_backed_by_review_file(candidate: dict[str, Any], approved_mark_event_ids: set[str]) -> bool:
    candidate_event_ids = set(_strings(candidate.get("source_mark_event_ids")))
    return bool(candidate_event_ids) and candidate_event_ids.issubset(approved_mark_event_ids)


def _region_from_section(section: str) -> str | None:
    for prefix, region in SECTION_REGION_NAMES.items():
        if section.startswith(prefix) or f" {prefix} " in section:
            return region
    return None


def _add_artifact_blockers(
    candidate: dict[str, Any],
    question: dict[str, Any],
    artifact_root: Path,
    blocker_classes: list[str],
    blocker_reasons: list[str],
) -> None:
    artifacts = candidate.get("source_artifacts") if isinstance(candidate.get("source_artifacts"), dict) else {}
    question_path = artifacts.get("question_crop_path") or question.get("canonical_question_artifact") or question.get("question_image_path")
    mark_scheme_path = (
        artifacts.get("mark_scheme_crop_path") or question.get("canonical_mark_scheme_artifact") or question.get("mark_scheme_image_path")
    )
    if not question_path:
        blocker_classes.append("blocked_artifact_path")
        blocker_reasons.append("missing_canonical_question_image_path")
    elif not _artifact_exists(question_path, artifact_root):
        blocker_classes.append("blocked_artifact_path")
        blocker_reasons.append("missing_canonical_question_image_file")
    if not mark_scheme_path:
        blocker_classes.append("blocked_mark_scheme")
        blocker_reasons.append("missing_canonical_mark_scheme_image_path")
    elif not _artifact_exists(mark_scheme_path, artifact_root):
        blocker_classes.append("blocked_mark_scheme")
        blocker_reasons.append("missing_canonical_mark_scheme_image_file")


def _artifact_exists(path_value: Any, artifact_root: Path) -> bool:
    path = Path(str(path_value))
    if path.is_absolute():
        return path.exists()
    candidates = [artifact_root / path, Path.cwd() / path]
    if str(path).startswith("output/"):
        candidates.append(Path.cwd() / path)
    return any(path.exists() for path in candidates)


def _status_from_question(question: dict[str, Any], field: str) -> str:
    notes = question.get("notes") if isinstance(question.get("notes"), dict) else {}
    return str(question.get(field) or notes.get(field) or "").strip().lower()


def _unsafe_generated_content_present(candidate: dict[str, Any]) -> bool:
    return "generated_content" in candidate or "student_facing_text" in candidate


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _questions_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    questions = payload.get("questions")
    if not isinstance(questions, list):
        return {}
    return {str(record.get("question_id") or ""): record for record in questions if isinstance(record, dict)}


def _topic_routing_by_question(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if isinstance(records, dict):
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}
    if isinstance(records, list):
        return {str(record.get("question_id") or ""): record for record in records if isinstance(record, dict)}
    return {}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _sample_sort_key(row: CandidateAudit, seed: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{row.candidate_id}".encode("utf-8")).hexdigest()
    return digest, row.candidate_id


def _ordered_regions(by_region: dict[str, list[CandidateAudit]]) -> list[str]:
    return sorted(by_region, key=_region_rank)


def _region_rank(region: str) -> tuple[int, str]:
    try:
        return (P3_REGION_ORDER.index(region), region)
    except ValueError:
        return (len(P3_REGION_ORDER), region)


def _counter_items(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"class": key, "count": count} for key, count in counter.most_common()]


def _row_dict(row: CandidateAudit) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "question_id": row.question_id,
        "subpart_id": row.subpart_id,
        "region": row.region,
        "region_source": row.region_source,
        "sample_bucket": row.sample_bucket,
        "status": row.status,
        "passed": row.passed,
        "blocker_classes": "|".join(row.blocker_classes),
        "blocker_reasons": "|".join(row.blocker_reasons),
        "generation_ready": row.generation_ready,
        "student_runtime_safe": row.student_runtime_safe,
        "app_safe_after_reviewed_gates": row.app_safe_after_reviewed_gates,
        "legacy_schema_mismatch": row.legacy_schema_mismatch,
        "mapping_validation_contradiction": row.mapping_validation_contradiction,
    }


def _write_inventory_csv(path: Path, rows: list[CandidateAudit]) -> None:
    fieldnames = list(_row_dict(rows[0]).keys()) if rows else list(_row_dict(_empty_row()).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_dict(row))


def _write_filtered_report(path: Path, rows: list[CandidateAudit], predicate: Any) -> None:
    _write_inventory_csv(path, [row for row in rows if predicate(row)])


def _write_full_pool_reports(
    out_dir: Path,
    audited: list[CandidateAudit],
    sample: list[CandidateAudit],
    summary: dict[str, Any],
) -> None:
    total = len(audited)
    passed = sum(row.passed for row in audited)
    sample_passed = sum(row.passed for row in sample)
    region_rows = []
    for region in sorted({row.region for row in audited}, key=_region_rank):
        rows = [row for row in audited if row.region == region]
        region_passed = sum(row.passed for row in rows)
        region_rows.append(
            {
                "region": region,
                "total_candidates": len(rows),
                "pass_count": region_passed,
                "fail_count": len(rows) - region_passed,
                "pass_rate": round(region_passed / len(rows), 4) if rows else 0.0,
            }
        )
    blocker_rows = []
    for region in sorted({row.region for row in audited}, key=_region_rank):
        counter = Counter(blocker for row in audited if row.region == region for blocker in row.blocker_classes)
        for blocker, count in counter.most_common():
            blocker_rows.append({"region": region, "blocker_class": blocker, "count": count})
    report = {
        "audit_version": AUDIT_VERSION,
        "total_p3_candidates": total,
        "pass_count": passed,
        "fail_count": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "deterministic_sample_guard": {
            "sample_size": len(sample),
            "pass_count": sample_passed,
            "fail_count": len(sample) - sample_passed,
            "pass_rate": round(sample_passed / len(sample), 4) if sample else 0.0,
            "sample_seed": summary["sample_seed"],
        },
        "additional_approvals_needed": {
            "for_70_percent": max(0, int((0.70 * total + 0.999999)) - passed),
            "for_90_percent": max(0, int((0.90 * total + 0.999999)) - passed),
        },
        "pass_rate_by_region": region_rows,
        "blocker_classes_by_region": blocker_rows,
        "reviewed_decisions_path": summary.get("reviewed_evidence_coverage", {}).get("reviewed_decisions_path"),
    }
    _write_json(out_dir / "full_pool_current.json", report)
    _write_full_pool_markdown(out_dir / "full_pool_current.md", report)
    baseline_path = out_dir / "full_pool_baseline.json"
    if not baseline_path.exists():
        _write_inventory_csv(out_dir / "full_pool_candidate_results.csv", audited)
        _write_dict_csv(out_dir / "full_pool_region_summary.csv", region_rows)
        _write_dict_csv(out_dir / "full_pool_blocker_summary.csv", blocker_rows)
        _write_json(baseline_path, report)
        _write_full_pool_markdown(out_dir / "full_pool_baseline.md", report)
    else:
        _write_inventory_csv(out_dir / "full_pool_final_candidate_results.csv", audited)
        _write_dict_csv(out_dir / "full_pool_final_region_summary.csv", region_rows)
        _write_dict_csv(out_dir / "full_pool_final_blocker_summary.csv", blocker_rows)


def _write_full_pool_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Full Pool P3 Content Lab Readiness",
        "",
        f"- Total P3 candidates: `{report['total_p3_candidates']}`",
        f"- Passed: `{report['pass_count']}`",
        f"- Pass rate: `{report['pass_rate']:.2%}`",
        f"- Deterministic sample guard: `{report['deterministic_sample_guard']['pass_count']}` / "
        f"`{report['deterministic_sample_guard']['sample_size']}` "
        f"(`{report['deterministic_sample_guard']['pass_rate']:.2%}`)",
        f"- Additional approvals needed for 70%: `{report['additional_approvals_needed']['for_70_percent']}`",
        f"- Additional approvals needed for 90%: `{report['additional_approvals_needed']['for_90_percent']}`",
        "",
        "## Region Pass Rates",
        "",
    ]
    for row in report["pass_rate_by_region"]:
        lines.append(f"- {row['region']}: `{row['pass_count']}` / `{row['total_candidates']}` (`{row['pass_rate']:.2%}`)")
    lines.extend(["", "## Top Blockers By Region", ""])
    for row in report["blocker_classes_by_region"][:30]:
        lines.append(f"- {row['region']} `{row['blocker_class']}`: `{row['count']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_blocker_breakdown(path: Path, rows: list[CandidateAudit]) -> None:
    counter = Counter(blocker for row in rows for blocker in row.blocker_classes)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blocker_class", "count"])
        writer.writeheader()
        for blocker, count in counter.most_common():
            writer.writerow({"blocker_class": blocker, "count": count})


def _write_blocker_by_region(path: Path, rows: list[CandidateAudit]) -> None:
    counter = Counter((row.region, blocker) for row in rows for blocker in row.blocker_classes)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region", "blocker_class", "count"])
        writer.writeheader()
        for (region, blocker), count in sorted(counter.items(), key=lambda item: (_region_rank(item[0][0]), item[0][1])):
            writer.writerow({"region": region, "blocker_class": blocker, "count": count})


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Asterion Content Lab P3 Readiness Audit",
        "",
        f"- Audit version: `{summary['audit_version']}`",
        f"- P3 candidates: `{summary['p3_candidates']}`",
        f"- Sample: `{summary['sample_passed']}` / `{summary['sample_size']}` passed",
        f"- Pass rate: `{summary['sample_pass_rate']:.2%}`",
        f"- Target met: `{summary['target_met']}`",
        f"- Sample seed: `{summary['sample_seed']}`",
        f"- Sample method: {summary['sample_method']}",
        f"- Student runtime changed: `{summary['student_runtime_changed']}`",
        f"- Trust gates weakened: `{summary['trust_gates_weakened']}`",
        "",
        "## Top Sample Blockers",
        "",
    ]
    for item in summary["top_blockers"][:12]:
        lines.append(f"- `{item['class']}`: {item['count']}")
    lines.extend(["", "## Regions Covered", ""])
    for region in summary["regions_covered"]:
        lines.append(f"- {region}: {summary['region_counts_in_sample'].get(region, 0)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_coverage_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Reviewed Evidence Coverage",
        "",
        f"- P3 candidates: `{summary['p3_candidate_count']}`",
        f"- Existing reviewed exact-skill evidence: `{summary['existing_reviewed_exact_skill_evidence_count']}`",
        f"- Existing reviewed source-skill evidence: `{summary['existing_reviewed_source_skill_evidence_count']}`",
        f"- Existing reviewed mark-event/subpart evidence: `{summary['existing_reviewed_mark_event_subpart_evidence_count']}`",
        f"- Fails only because reviewed evidence is not surfaced: `{summary['fails_only_because_reviewed_evidence_not_surfaced_count']}`",
        f"- Fails because reviewed evidence is absent: `{summary['fails_because_reviewed_evidence_absent_count']}`",
        f"- Fails with ambiguous/quarantined evidence: `{summary['fails_with_ambiguous_or_quarantined_evidence_count']}`",
        f"- Legacy schema mismatch count: `{summary['legacy_schema_mismatch_count']}`",
        f"- Safe compatibility/export bridge candidates: `{summary['safe_compatibility_export_bridge_candidate_count']}`",
        f"- Requires human review no matter code: `{summary['requires_human_review_no_matter_code_count']}`",
        "",
        "## Evidence Sources",
        "",
        f"- Source-skill decisions: `{summary['reviewed_source_skills_path']}`",
        f"- Mark-event decisions: `{summary['reviewed_mark_events_path']}`",
        f"- Source review records: `{summary['source_review_record_count']}`",
        f"- Generation-satisfying source review records: `{summary['source_review_generation_satisfying_record_count']}`",
        f"- Mark-event decisions: `{summary['reviewed_mark_event_decision_count']}`",
        f"- Generation-satisfying mark-event decisions: `{summary['reviewed_mark_event_generation_satisfying_decision_count']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommendations(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Next Iteration Recommendations",
        "",
        f"Target met: `{summary['target_met']}`.",
        "",
        "Largest sample blocker classes:",
    ]
    for item in summary["top_blockers"][:8]:
        lines.append(f"- `{item['class']}`: {item['count']}")
    lines.extend(
        [
            "",
            "Repairability classification:",
            f"- Code/data repairable: `{summary['blocker_classes_code_data_repairable']}`",
            f"- Human review required: `{summary['blocker_classes_requiring_human_review']}`",
            f"- Asterion-side schema/validator: `{summary['blocker_classes_requiring_asterion_side_schema_or_validator_change']}`",
            "",
            "Recommended next loop: populate reviewed mark-event and exact-skill/source-skill evidence only from human-reviewed records. Do not weaken gates or infer skills from topics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _empty_row() -> CandidateAudit:
    return CandidateAudit("", "", "", "", "", "", "", False, (), (), False, False, False, False, False)


def _source_skill_record_generation_satisfying(record: dict[str, Any]) -> bool:
    reviewer = record.get("reviewer") if isinstance(record.get("reviewer"), dict) else {}
    return (
        str(record.get("route_status") or "").strip().lower() == "clean"
        and str(reviewer.get("review_status") or "").strip().lower() in {"approved", "reviewed"}
        and bool(_strings(record.get("reviewed_source_skill_ids")))
        and not bool(record.get("blockers"))
    )


def _mark_event_decision_generation_satisfying(decision: dict[str, Any]) -> bool:
    return (
        str(decision.get("status") or "").strip().lower() in {"approved", "reviewed"}
        and bool(decision.get("satisfies_generation_gate"))
    )
