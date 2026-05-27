from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from exam_bank.atomic_json import write_atomic_json
from exam_bank.content_lab_auto_review import (
    AUTO_REVIEW_SOURCE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    automated_mark_event_ids,
    automated_source_records,
)


RUNTIME_SAFE_SCHEMA = "exam_bank.asterion.student_runtime_safe"
RUNTIME_SAFE_SCHEMA_VERSION = 1
PROMOTION_DECISIONS_SCHEMA = "exam_bank.asterion.student_runtime_safe_decisions"
PROMOTION_DECISIONS_SCHEMA_VERSION = 1
RUNTIME_SAFE_CANDIDATES_SCHEMA = "asterion.student_runtime_safe_candidates"
RUNTIME_SAFE_CONTRACT_VERSION = "asterion_student_runtime_safe_contract_v1"

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

AMBIGUITY_TERMS = {
    "ambiguous",
    "ambiguity",
    "quarantine",
    "quarantined",
    "advisory",
    "advisory_text_only",
    "ocr_only",
    "missing_canonical_image",
    "missing_mark_scheme",
    "reviewer_verifier_disagreement",
    "hard_block",
}

CONTRACT = [
    "Canonical question image exists.",
    "Canonical mark-scheme image exists where mark-scheme evidence is required.",
    "Candidate has validated question/question-part identity.",
    "Candidate has reviewed exact/source skill mapping.",
    "Candidate has reviewed mark-event/subpart evidence where required.",
    "Candidate is not ambiguous or quarantined.",
    "Candidate is not generation-seed-only.",
    "Candidate is not teacher/dev-preview-only.",
    "Candidate region/topic route is valid for Asterion.",
    "Candidate does not require unsafe advisory text to be student-usable.",
    "Candidate has no mapping/validation contradictions.",
    "Candidate has no unresolved artifact path blocker.",
    "Candidate passes the current Asterion export/runtime schema contract.",
]


@dataclass(frozen=True)
class RuntimeSafetyResult:
    candidate_id: str
    question_id: str
    subpart_id: str
    region: str
    current_student_runtime_safe: bool
    contract_safe: bool
    final_student_runtime_safe: bool
    promotion_decision_valid: bool
    promotion_eligible: bool
    runtime_role_projection_applied: bool
    classification: str
    blocker_classes: tuple[str, ...]
    blocker_reasons: tuple[str, ...]
    reviewed_decision_refs: tuple[str, ...]
    exact_skill_evidence_refs: tuple[str, ...]
    mark_event_evidence_refs: tuple[str, ...]
    canonical_question_image_path: str
    canonical_mark_scheme_image_path: str
    risk_flags: tuple[str, ...]
    in_deterministic_sample: bool
    content_lab_reviewed_approval: bool
    current_runtime_role: str
    current_candidate_category: str


def run_runtime_safe_audit(
    *,
    candidates_path: Path,
    question_bank_path: Path | None,
    topic_routing_path: Path | None,
    asterion_bank_path: Path,
    mark_events_path: Path | None,
    reviewed_decisions_path: Path | None,
    artifact_root: Path,
    out_dir: Path,
    target_pass_rate: float = 0.50,
    reviewed_source_skills_path: Path | None = Path("data/review/p3_exact_skill_reviewed_decisions.v1.json"),
    reviewed_mark_events_path: Path | None = Path("data/review/p3_exact_skill_reviewed_mark_events.v1.json"),
    skill_map_path: Path | None = Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    question_skill_mappings_path: Path | None = Path(
        "exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"
    ),
    deterministic_sample_path: Path | None = Path("output/audits/asterion_content_lab_loop/iteration_005/sample_results.csv"),
    regeneration_backlog_path: Path | None = Path("output/review/content_lab_p3_auto_loop_005/candidate_regeneration_backlog.json"),
    promotion_decisions_path: Path | None = None,
    write_promotion_decisions_path: Path | None = None,
    write_export_decisions_path: Path | None = None,
    write_export_candidates_path: Path | None = None,
) -> dict[str, Any]:
    candidates_payload = _read_json(candidates_path)
    candidates = [item for item in candidates_payload.get("candidates", []) if isinstance(item, dict)]
    p3_candidates = [item for item in candidates if _text(item.get("paper_family")).lower() == "p3"]
    asterion_questions = _questions_by_id(_read_json(asterion_bank_path))
    source_questions = _questions_by_id(_read_json(question_bank_path) if question_bank_path else {})
    topic_routing = _topic_routing_by_question(_read_json(topic_routing_path) if topic_routing_path else {})
    mark_events = _mark_event_index(_read_json(mark_events_path) if mark_events_path else {})
    skill_regions = _skill_regions(_read_json(skill_map_path) if skill_map_path and skill_map_path.exists() else {})
    mapping_regions = _question_skill_mapping_regions(
        _read_json(question_skill_mappings_path)
        if question_skill_mappings_path and question_skill_mappings_path.exists()
        else {},
        skill_regions=skill_regions,
    )
    reviewed_payload = _read_json(reviewed_decisions_path) if reviewed_decisions_path and reviewed_decisions_path.exists() else {}
    source_review_payload = (
        _read_json(reviewed_source_skills_path)
        if reviewed_source_skills_path and reviewed_source_skills_path.exists()
        else {}
    )
    mark_review_payload = (
        _read_json(reviewed_mark_events_path)
        if reviewed_mark_events_path and reviewed_mark_events_path.exists()
        else {}
    )
    evidence = _review_evidence(
        reviewed_payload=reviewed_payload,
        source_review_payload=source_review_payload,
        mark_review_payload=mark_review_payload,
        artifact_root=artifact_root,
    )
    sample_ids = _candidate_ids_from_csv(deterministic_sample_path) if deterministic_sample_path else set()
    regeneration_ids = _regeneration_backlog_ids(regeneration_backlog_path)

    preliminary = [
        _evaluate_candidate(
            candidate,
            asterion_questions=asterion_questions,
            source_questions=source_questions,
            topic_routing=topic_routing,
            mark_events=mark_events,
            skill_regions=skill_regions,
            mapping_regions=mapping_regions,
            artifact_root=artifact_root,
            evidence=evidence,
            sample_ids=sample_ids,
            regeneration_ids=regeneration_ids,
            valid_promotions={},
        )
        for candidate in p3_candidates
    ]
    promotions_to_write = [_promotion_decision(row, evidence=evidence) for row in preliminary if row.promotion_eligible]
    if write_promotion_decisions_path:
        _write_promotion_payload(write_promotion_decisions_path, promotions_to_write, reviewed_decisions_path=reviewed_decisions_path)

    promotion_payload = {}
    if promotion_decisions_path and promotion_decisions_path.exists():
        promotion_payload = _read_json(promotion_decisions_path)
    elif write_promotion_decisions_path:
        promotion_payload = {
            "schema": PROMOTION_DECISIONS_SCHEMA,
            "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
            "decisions": promotions_to_write,
            "decision_count": len(promotions_to_write),
        }
    promotion_errors, valid_promotions = validate_promotion_decisions_payload(
        promotion_payload,
        candidates_by_id={_text(candidate.get("candidate_id")): candidate for candidate in p3_candidates},
        artifact_root=artifact_root,
        evidence=evidence,
    )

    results = [
        _evaluate_candidate(
            candidate,
            asterion_questions=asterion_questions,
            source_questions=source_questions,
            topic_routing=topic_routing,
            mark_events=mark_events,
            skill_regions=skill_regions,
            mapping_regions=mapping_regions,
            artifact_root=artifact_root,
            evidence=evidence,
            sample_ids=sample_ids,
            regeneration_ids=regeneration_ids,
            valid_promotions=valid_promotions,
        )
        for candidate in p3_candidates
    ]
    results.sort(key=lambda row: (_region_rank(row.region), row.question_id, row.subpart_id, row.candidate_id))

    summary = _summary(
        results,
        candidates_path=candidates_path,
        question_bank_path=question_bank_path,
        topic_routing_path=topic_routing_path,
        asterion_bank_path=asterion_bank_path,
        mark_events_path=mark_events_path,
        reviewed_decisions_path=reviewed_decisions_path,
        promotion_decisions_path=promotion_decisions_path or write_promotion_decisions_path,
        target_pass_rate=target_pass_rate,
        promotion_validation_errors=promotion_errors,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    final_run = bool(promotion_decisions_path or write_promotion_decisions_path)
    baseline_exists = (out_dir / "runtime_safe_baseline.json").exists()
    if not final_run or not baseline_exists:
        _write_json(out_dir / "runtime_safe_baseline.json", summary)
        _write_baseline_markdown(out_dir / "runtime_safe_baseline.md", summary)
        _write_result_csv(out_dir / "runtime_safe_candidate_results.csv", results)
        _write_dict_csv(out_dir / "runtime_safe_region_summary.csv", _region_rows(results))
        _write_dict_csv(out_dir / "runtime_safe_blocker_summary.csv", _blocker_rows(results))
    else:
        _write_json(out_dir / "runtime_safe_final.json", summary)
        _write_baseline_markdown(out_dir / "runtime_safe_final.md", summary)
        _write_result_csv(out_dir / "runtime_safe_final_candidate_results.csv", results)
        _write_dict_csv(out_dir / "runtime_safe_final_region_summary.csv", _region_rows(results))
        _write_dict_csv(out_dir / "runtime_safe_final_blocker_summary.csv", _blocker_rows(results))
    classification_payload = _classification_payload(results, summary=summary)
    _write_json(out_dir / "runtime_safe_classification.json", classification_payload)
    _write_result_csv(out_dir / "runtime_safe_classification.csv", results)
    _write_classification_markdown(out_dir / "runtime_safe_classification.md", classification_payload)
    if write_export_decisions_path:
        _write_promotion_payload(write_export_decisions_path, list(valid_promotions.values()), reviewed_decisions_path=reviewed_decisions_path)
    if write_export_candidates_path:
        _write_student_runtime_safe_candidates(write_export_candidates_path, results)
    _write_artifact_schema_diagnosis(out_dir, results)
    return summary


def validate_promotion_decisions_payload(
    payload: Any,
    *,
    candidates_by_id: dict[str, dict[str, Any]],
    artifact_root: Path,
    evidence: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not payload:
        return [], {}
    errors: list[str] = []
    valid: dict[str, dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return ["promotion_payload_not_object"], {}
    if payload.get("schema") != PROMOTION_DECISIONS_SCHEMA:
        errors.append(f"promotion_schema_mismatch:{payload.get('schema')}")
    if int(payload.get("schema_version") or 0) != PROMOTION_DECISIONS_SCHEMA_VERSION:
        errors.append(f"promotion_schema_version_mismatch:{payload.get('schema_version')}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return errors + ["promotion_decisions_not_list"], {}
    if payload.get("decision_count") not in (None, len(decisions)):
        errors.append(f"promotion_decision_count_mismatch:{payload.get('decision_count')}:{len(decisions)}")

    seen_by_candidate: dict[str, dict[str, Any]] = {}
    seen_decision_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"promotion_decision_not_object:{index}")
            continue
        decision_errors = _validate_promotion_decision(
            index,
            decision,
            candidates_by_id=candidates_by_id,
            artifact_root=artifact_root,
            seen_decision_ids=seen_decision_ids,
            seen_by_candidate=seen_by_candidate,
            evidence=evidence,
        )
        errors.extend(decision_errors)
        if not decision_errors:
            valid[_text(decision.get("candidate_id"))] = decision
    return errors, valid


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    asterion_questions: dict[str, dict[str, Any]],
    source_questions: dict[str, dict[str, Any]],
    topic_routing: dict[str, dict[str, Any]],
    mark_events: dict[str, dict[str, Any]],
    skill_regions: dict[str, str],
    mapping_regions: dict[str, tuple[str, str]],
    artifact_root: Path,
    evidence: dict[str, Any],
    sample_ids: set[str],
    regeneration_ids: set[str],
    valid_promotions: dict[str, dict[str, Any]],
) -> RuntimeSafetyResult:
    candidate_id = _text(candidate.get("candidate_id"))
    question_id = _text(candidate.get("question_id"))
    subpart_id = _text(candidate.get("subpart_id"))
    question = asterion_questions.get(question_id) or source_questions.get(question_id) or {}
    source_question = source_questions.get(question_id) or question
    region, _ = _candidate_region(candidate, question, topic_routing.get(question_id, {}), skill_regions, mapping_regions)
    question_image = _canonical_path(
        (candidate.get("source_artifacts") or {}).get("question_crop_path")
        or _subpart_by_id(question, subpart_id).get("question_crop_path")
        or question.get("canonical_question_artifact")
        or source_question.get("canonical_question_artifact"),
        artifact_root=artifact_root,
    )
    mark_scheme_image = _canonical_path(
        (candidate.get("source_artifacts") or {}).get("mark_scheme_crop_path")
        or _subpart_by_id(question, subpart_id).get("mark_scheme_crop_path")
        or question.get("canonical_mark_scheme_artifact")
        or source_question.get("canonical_mark_scheme_artifact"),
        artifact_root=artifact_root,
    )

    blockers: list[str] = []
    reasons: list[str] = []
    risk_flags: list[str] = []
    roles = question.get("usage_roles") if isinstance(question.get("usage_roles"), dict) else {}
    quality_gate = question.get("quality_gate") if isinstance(question.get("quality_gate"), dict) else {}
    subpart = _subpart_by_id(question, subpart_id)
    event_ids = _strings(candidate.get("source_mark_event_ids"))
    mark_scheme_required = bool(event_ids or candidate.get("source_mark_event_count") or candidate.get("marks"))
    source_skill_ids = _strings(candidate.get("reviewed_source_skill_ids")) or _strings(candidate.get("source_skill_ids"))
    review = evidence["by_candidate"].get(candidate_id)
    reviewed_refs: list[str] = []
    exact_refs: list[str] = []
    mark_refs: list[str] = []
    content_lab_reviewed = False

    if review:
        content_lab_reviewed = True
        reviewed_refs.append(_text(review.get("decision_id")))
        exact_refs.extend(_strings(review.get("approved_source_skill_ids")) or _strings(review.get("approved_exact_skill_ids")))
        mark_refs.extend(_text(ref.get("event_id")) for ref in review.get("approved_mark_event_refs", []) if isinstance(ref, dict))
        risk_flags.extend(_strings(review.get("risk_flags")))
    else:
        exact_record = _matching_source_review(candidate, evidence["source_records_by_subpart"].get(subpart_id, []))
        if exact_record:
            exact_refs.append(_text(exact_record.get("evidence_id")))
            reviewed_refs.append(_text(exact_record.get("evidence_id")))
        mark_refs.extend(event_id for event_id in event_ids if event_id in evidence["approved_mark_event_ids"])
        if exact_record or mark_refs:
            content_lab_reviewed = True

    if not question_id or not question:
        blockers.append("question_identity")
        reasons.append("question_record_missing")
    if not subpart:
        blockers.append("question_identity")
        reasons.append("subpart_record_missing")
    if not question_image:
        blockers.append("canonical_question_image")
        reasons.append("missing_canonical_question_image")
    if mark_scheme_required and not mark_scheme_image:
        blockers.append("canonical_mark_scheme_image")
        reasons.append("missing_canonical_mark_scheme_image")
    if _status(source_question, "mapping_status") != "pass":
        blockers.append("validated_identity")
        reasons.append(f"mapping_status_{_status(source_question, 'mapping_status') or 'missing'}")
    if _status(source_question, "validation_status") != "pass":
        blockers.append("validated_identity")
        reasons.append(f"validation_status_{_status(source_question, 'validation_status') or 'missing'}")
    if _status(source_question, "mapping_status") == "fail" and _status(source_question, "validation_status") == "pass":
        blockers.append("mapping_validation_contradiction")
        reasons.append("mapping_fail_validation_pass_contradiction")
    if not source_skill_ids and not exact_refs:
        blockers.append("reviewed_exact_skill_mapping")
        reasons.append("missing_source_skill_ids")
    if not exact_refs:
        blockers.append("reviewed_exact_skill_mapping")
        reasons.append("reviewed_exact_skill_evidence_missing")
    if mark_scheme_required and (not event_ids or len(set(event_ids) - set(mark_refs)) > 0):
        blockers.append("reviewed_mark_event_evidence")
        reasons.append("reviewed_mark_event_evidence_missing_or_incomplete")
    missing_canonical_events = [event_id for event_id in event_ids if event_id not in mark_events]
    if missing_canonical_events:
        blockers.append("reviewed_mark_event_evidence")
        reasons.append("source_mark_event_missing_from_canonical_mark_events")
    if _any_ambiguity(_strings(candidate.get("mapping_review_blocked_reasons"))):
        blockers.append("ambiguous_or_quarantined")
        reasons.append("candidate_mapping_blocked_by_ambiguity")
    if _any_ambiguity(_strings((candidate.get("generation_gate") or {}).get("block_reasons"))):
        blockers.append("ambiguous_or_quarantined")
        reasons.append("candidate_generation_gate_ambiguous_or_quarantined")
    if _any_ambiguity(risk_flags):
        blockers.append("ambiguous_or_quarantined")
        reasons.append("review_decision_has_blocking_risk_flags")
    if any(_any_ambiguity(_strings(mark_events.get(event_id, {}).get("review_flags"))) for event_id in event_ids):
        blockers.append("ambiguous_or_quarantined")
        reasons.append("canonical_mark_event_review_flags_block_runtime")
    if "generated_content" in candidate or "student_facing_text" in candidate:
        blockers.append("unsafe_advisory_text")
        reasons.append("candidate_contains_generated_or_student_facing_text")
    if _strings((candidate.get("mark_event_review_gate") or {}).get("advisory_event_ids")):
        blockers.append("unsafe_advisory_text")
        reasons.append("candidate_has_advisory_mark_event_ids")
    if region == "Unresolved P3 Skill Region":
        blockers.append("invalid_region_route")
        reasons.append("unresolved_p3_skill_region")
    if roles.get("canonical_practice") != "allow":
        blockers.append("asterion_runtime_schema")
        reasons.append("canonical_practice_role_not_allow")
    if roles.get("generated_warmup_pattern_source") == "allow" and roles.get("canonical_practice") != "allow":
        blockers.append("generation_seed_only")
        reasons.append("candidate_role_is_generation_seed_without_student_runtime_role")
    if (
        roles.get("field_guide_source") == "allow" or roles.get("guardian_candidate") == "allow"
    ) and roles.get("canonical_practice") != "allow":
        blockers.append("teacher_preview_only")
        reasons.append("candidate_role_is_preview_without_student_runtime_role")
    if quality_gate.get("canonical_assets_ok") is not True:
        blockers.append("unresolved_artifact_path")
        reasons.append("asterion_quality_gate_canonical_assets_not_ok")
    if candidate_id in regeneration_ids:
        blockers.append("candidate_regeneration_required")
        reasons.append("loop005_regeneration_backlog_candidate")

    blockers = _dedupe(blockers)
    reasons = _dedupe(reasons)
    contract_safe = not blockers
    non_runtime_blockers = [blocker for blocker in blockers if blocker != "asterion_runtime_schema"]
    promotion_eligible = content_lab_reviewed and not non_runtime_blockers
    promotion_valid = candidate_id in valid_promotions
    runtime_role_projection_applied = promotion_valid and promotion_eligible and "asterion_runtime_schema" in blockers
    final_runtime_safe = (contract_safe and (candidate.get("student_runtime_safe") is True or promotion_valid)) or runtime_role_projection_applied
    classification = _classification(
        contract_safe=contract_safe,
        final_runtime_safe=final_runtime_safe,
        reviewed=content_lab_reviewed,
        blockers=blockers,
        roles=roles,
        generation_gate=candidate.get("generation_gate") if isinstance(candidate.get("generation_gate"), dict) else {},
        promotion_eligible=promotion_eligible,
    )
    return RuntimeSafetyResult(
        candidate_id=candidate_id,
        question_id=question_id,
        subpart_id=subpart_id,
        region=region,
        current_student_runtime_safe=candidate.get("student_runtime_safe") is True,
        contract_safe=contract_safe,
        final_student_runtime_safe=final_runtime_safe,
        promotion_decision_valid=promotion_valid,
        promotion_eligible=promotion_eligible,
        runtime_role_projection_applied=runtime_role_projection_applied,
        classification=classification,
        blocker_classes=tuple(blockers),
        blocker_reasons=tuple(reasons),
        reviewed_decision_refs=tuple(_dedupe(reviewed_refs)),
        exact_skill_evidence_refs=tuple(_dedupe(exact_refs)),
        mark_event_evidence_refs=tuple(_dedupe(mark_refs)),
        canonical_question_image_path=question_image,
        canonical_mark_scheme_image_path=mark_scheme_image,
        risk_flags=tuple(_dedupe(risk_flags)),
        in_deterministic_sample=candidate_id in sample_ids,
        content_lab_reviewed_approval=content_lab_reviewed,
        current_runtime_role=_text(roles.get("canonical_practice") or "missing"),
        current_candidate_category=_text(candidate.get("review_status") or "|".join(_strings(candidate.get("possible_content_lab_roles"))) or "missing"),
    )


def _classification(
    *,
    contract_safe: bool,
    final_runtime_safe: bool,
    reviewed: bool,
    blockers: list[str],
    roles: dict[str, Any],
    generation_gate: dict[str, Any],
    promotion_eligible: bool,
) -> str:
    if final_runtime_safe:
        return "runtime_safe_now"
    if contract_safe:
        return "evidence_safe_schema_projection_blocked"
    if "candidate_regeneration_required" in blockers or "mapping_validation_contradiction" in blockers or "ambiguous_or_quarantined" in blockers:
        return "needs_candidate_regeneration"
    if "generation_seed_only" in blockers:
        return "generation_seed_only"
    if "teacher_preview_only" in blockers:
        return "teacher_preview_only"
    if "canonical_question_image" in blockers or "canonical_mark_scheme_image" in blockers or "unresolved_artifact_path" in blockers:
        return "artifact_or_schema_repairable" if not reviewed else "reviewed_but_not_runtime_safe"
    if "reviewed_exact_skill_mapping" in blockers or "reviewed_mark_event_evidence" in blockers:
        return "needs_reviewed_evidence"
    if promotion_eligible and "asterion_runtime_schema" in blockers:
        return "evidence_safe_runtime_role_blocked"
    if "asterion_runtime_schema" in blockers:
        return "artifact_or_schema_repairable" if not reviewed else "reviewed_but_not_runtime_safe"
    if reviewed:
        return "reviewed_but_not_runtime_safe"
    return "needs_reviewed_evidence"


def _review_evidence(
    *,
    reviewed_payload: dict[str, Any],
    source_review_payload: dict[str, Any],
    mark_review_payload: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    by_candidate: dict[str, dict[str, Any]] = {}
    if reviewed_payload.get("schema") == "exam_bank.content_lab.auto_reviewed_decisions":
        for record in reviewed_payload.get("records", []):
            if not isinstance(record, dict):
                continue
            if _review_record_satisfies_runtime(record, artifact_root=artifact_root):
                by_candidate[_text(record.get("candidate_id"))] = record
    source_records_by_subpart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_review_payload.get("records", []):
        if isinstance(record, dict) and _source_review_satisfies_runtime(record):
            source_records_by_subpart[_text(record.get("subpart_id"))].append(record)
    for subpart_id, records in automated_source_records(reviewed_payload).items():
        for record in records:
            if _source_review_satisfies_runtime(record):
                source_records_by_subpart[subpart_id].append(record)
    approved_mark_event_ids = {
        _text(decision.get("event_id"))
        for decision in mark_review_payload.get("decisions", [])
        if isinstance(decision, dict)
        and _text(decision.get("status")).lower() in {"approved", "reviewed"}
        and decision.get("satisfies_generation_gate") is not False
    }
    approved_mark_event_ids.update(automated_mark_event_ids(reviewed_payload))
    return {
        "by_candidate": by_candidate,
        "source_records_by_subpart": source_records_by_subpart,
        "approved_mark_event_ids": approved_mark_event_ids,
    }


def _review_record_satisfies_runtime(record: dict[str, Any], *, artifact_root: Path) -> bool:
    if record.get("review_source") != AUTO_REVIEW_SOURCE:
        return False
    if _text((record.get("adjudication") or {}).get("status")) != "approved":
        return False
    if float(record.get("confidence") or 0) < DEFAULT_CONFIDENCE_THRESHOLD:
        return False
    if _any_ambiguity(_strings(record.get("risk_flags"))):
        return False
    if not (_strings(record.get("approved_source_skill_ids")) or _strings(record.get("approved_exact_skill_ids"))):
        return False
    if not record.get("approved_mark_event_refs"):
        return False
    for path in (record.get("canonical_question_image_path"), record.get("canonical_mark_scheme_image_path")):
        if not _canonical_path(path, artifact_root=artifact_root):
            return False
    return True


def _source_review_satisfies_runtime(record: dict[str, Any]) -> bool:
    reviewer = record.get("reviewer") if isinstance(record.get("reviewer"), dict) else {}
    return (
        _text(record.get("route_status")).lower() == "clean"
        and _text(reviewer.get("review_status")).lower() in {"approved", "reviewed"}
        and bool(_strings(record.get("reviewed_source_skill_ids")))
        and not record.get("blockers")
    )


def _matching_source_review(candidate: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_skills = set(_strings(candidate.get("reviewed_source_skill_ids")) or _strings(candidate.get("source_skill_ids")))
    candidate_events = set(_strings(candidate.get("source_mark_event_ids")))
    for record in records:
        reviewed_skills = set(_strings(record.get("reviewed_source_skill_ids")))
        reviewed_events = {
            _text(ref.get("event_id"))
            for ref in record.get("mark_event_refs", [])
            if isinstance(ref, dict) and _text(ref.get("event_id"))
        }
        if candidate_skills and reviewed_skills and not candidate_skills.intersection(reviewed_skills):
            continue
        if candidate_events and reviewed_events and not candidate_events.issubset(reviewed_events):
            continue
        return record
    return None


def _promotion_decision(row: RuntimeSafetyResult, *, evidence: dict[str, Any]) -> dict[str, Any]:
    source_review = evidence["by_candidate"].get(row.candidate_id, {})
    decision_id = f"asterion_student_runtime_safe:{row.candidate_id}"
    role_reason = (
        "validated_runtime_role_projection_from_reviewed_evidence"
        if row.current_runtime_role != "allow"
        else "current_asterion_canonical_practice_role_allow"
    )
    return {
        "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
        "decision_id": decision_id,
        "candidate_id": row.candidate_id,
        "question_id": row.question_id,
        "subpart_id": row.subpart_id,
        "region_id": row.region,
        "promotion_decision": "student_runtime_safe",
        "review_source": "student_runtime_safe_audit",
        "reviewed_decision_refs": list(row.reviewed_decision_refs),
        "exact_source_skill_evidence_refs": list(row.exact_skill_evidence_refs),
        "mark_event_evidence_refs": list(row.mark_event_evidence_refs),
        "canonical_question_image_path": row.canonical_question_image_path,
        "canonical_mark_scheme_image_path": row.canonical_mark_scheme_image_path,
        "runtime_safety_reasons": [
            "satisfies_strict_runtime_safety_contract",
            role_reason,
        ],
        "risk_flags": list(row.risk_flags),
        "source_content_lab_review_decision_id": source_review.get("decision_id"),
        "provenance": {
            "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
            "created_by": "scripts/audit_asterion_student_runtime_safe.py",
            "source_review_decision_id": source_review.get("decision_id"),
        },
        "created_at": _utc_now_iso(),
    }


def _validate_promotion_decision(
    index: int,
    decision: dict[str, Any],
    *,
    candidates_by_id: dict[str, dict[str, Any]],
    artifact_root: Path,
    seen_decision_ids: set[str],
    seen_by_candidate: dict[str, dict[str, Any]],
    evidence: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    decision_id = _text(decision.get("decision_id"))
    candidate_id = _text(decision.get("candidate_id"))
    prefix = f"promotion:{index}:{decision_id or 'missing'}:{candidate_id or 'missing'}"
    required = {
        "schema_version",
        "decision_id",
        "candidate_id",
        "question_id",
        "subpart_id",
        "region_id",
        "promotion_decision",
        "review_source",
        "reviewed_decision_refs",
        "exact_source_skill_evidence_refs",
        "mark_event_evidence_refs",
        "canonical_question_image_path",
        "canonical_mark_scheme_image_path",
        "runtime_safety_reasons",
        "risk_flags",
        "provenance",
        "created_at",
    }
    for field in sorted(required):
        if field not in decision:
            errors.append(f"{prefix}:missing_required_field:{field}")
    if int(decision.get("schema_version") or 0) != PROMOTION_DECISIONS_SCHEMA_VERSION:
        errors.append(f"{prefix}:schema_version_mismatch")
    if decision_id in seen_decision_ids:
        errors.append(f"{prefix}:duplicate_decision_id")
    seen_decision_ids.add(decision_id)
    previous = seen_by_candidate.get(candidate_id)
    if previous and json.dumps(previous, sort_keys=True) != json.dumps(decision, sort_keys=True):
        errors.append(f"{prefix}:duplicate_conflicting_candidate_decision")
    seen_by_candidate[candidate_id] = decision
    candidate = candidates_by_id.get(candidate_id)
    if not candidate:
        errors.append(f"{prefix}:candidate_not_found")
    else:
        if _text(candidate.get("question_id")) != _text(decision.get("question_id")):
            errors.append(f"{prefix}:question_id_mismatch")
        if _text(candidate.get("subpart_id")) != _text(decision.get("subpart_id")):
            errors.append(f"{prefix}:subpart_id_mismatch")
        event_ids = set(_strings(candidate.get("source_mark_event_ids")))
        mark_refs = set(_strings(decision.get("mark_event_evidence_refs")))
        if event_ids and not event_ids.issubset(mark_refs):
            errors.append(f"{prefix}:mark_event_evidence_refs_do_not_cover_candidate")
    if decision.get("promotion_decision") != "student_runtime_safe":
        errors.append(f"{prefix}:promotion_decision_not_student_runtime_safe")
    if not _strings(decision.get("reviewed_decision_refs")):
        errors.append(f"{prefix}:reviewed_decision_refs_missing")
    if not _strings(decision.get("exact_source_skill_evidence_refs")):
        errors.append(f"{prefix}:exact_source_skill_evidence_refs_missing")
    if not _strings(decision.get("mark_event_evidence_refs")):
        errors.append(f"{prefix}:mark_event_evidence_refs_missing")
    if _any_ambiguity(_strings(decision.get("risk_flags"))):
        errors.append(f"{prefix}:blocking_risk_flags_present")
    if not _canonical_path(decision.get("canonical_question_image_path"), artifact_root=artifact_root):
        errors.append(f"{prefix}:canonical_question_image_missing")
    if not _canonical_path(decision.get("canonical_mark_scheme_image_path"), artifact_root=artifact_root):
        errors.append(f"{prefix}:canonical_mark_scheme_image_missing")
    provenance = decision.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{prefix}:provenance_missing")
    else:
        if provenance.get("contract_version") != RUNTIME_SAFE_CONTRACT_VERSION:
            errors.append(f"{prefix}:provenance_contract_version_mismatch")
        if not _text(provenance.get("created_by")):
            errors.append(f"{prefix}:provenance_created_by_missing")
    if not _text(decision.get("created_at")):
        errors.append(f"{prefix}:created_at_missing")
    if evidence is not None:
        errors.extend(_validate_promotion_review_evidence(prefix, decision, candidate=candidate, artifact_root=artifact_root, evidence=evidence))
    return errors


def _validate_promotion_review_evidence(
    prefix: str,
    decision: dict[str, Any],
    *,
    candidate: dict[str, Any] | None,
    artifact_root: Path,
    evidence: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    candidate_id = _text(decision.get("candidate_id"))
    review = evidence.get("by_candidate", {}).get(candidate_id)
    if not isinstance(review, dict):
        return _validate_promotion_source_review_evidence(
            prefix,
            decision,
            candidate=candidate,
            artifact_root=artifact_root,
            evidence=evidence,
        )

    decision_review_refs = set(_strings(decision.get("reviewed_decision_refs")))
    review_decision_id = _text(review.get("decision_id"))
    if review_decision_id and review_decision_id not in decision_review_refs:
        errors.append(f"{prefix}:reviewed_decision_refs_do_not_include_valid_review")

    approved_skills = set(_strings(review.get("approved_source_skill_ids")) or _strings(review.get("approved_exact_skill_ids")))
    exact_refs = set(_strings(decision.get("exact_source_skill_evidence_refs")))
    if not exact_refs or not exact_refs.issubset(approved_skills):
        errors.append(f"{prefix}:exact_source_skill_refs_not_validated_by_review")

    approved_mark_refs = {
        _text(ref.get("event_id"))
        for ref in review.get("approved_mark_event_refs", [])
        if isinstance(ref, dict) and _text(ref.get("event_id"))
    }
    mark_refs = set(_strings(decision.get("mark_event_evidence_refs")))
    if not mark_refs or not mark_refs.issubset(approved_mark_refs):
        errors.append(f"{prefix}:mark_event_refs_not_validated_by_review")
    if candidate:
        candidate_events = set(_strings(candidate.get("source_mark_event_ids")))
        if candidate_events and not candidate_events.issubset(approved_mark_refs):
            errors.append(f"{prefix}:candidate_mark_events_not_validated_by_review")

    for decision_field, review_field in (
        ("canonical_question_image_path", "canonical_question_image_path"),
        ("canonical_mark_scheme_image_path", "canonical_mark_scheme_image_path"),
    ):
        decision_path = _canonical_path(decision.get(decision_field), artifact_root=artifact_root)
        review_path = _canonical_path(review.get(review_field), artifact_root=artifact_root)
        if not decision_path or not review_path or Path(decision_path).resolve() != Path(review_path).resolve():
            errors.append(f"{prefix}:{decision_field}_not_validated_by_review")

    if _any_ambiguity(_strings(review.get("risk_flags"))):
        errors.append(f"{prefix}:reviewed_runtime_evidence_has_blocking_risk_flags")
    return errors


def _validate_promotion_source_review_evidence(
    prefix: str,
    decision: dict[str, Any],
    *,
    candidate: dict[str, Any] | None,
    artifact_root: Path,
    evidence: dict[str, Any],
) -> list[str]:
    if not candidate:
        return [f"{prefix}:reviewed_runtime_evidence_missing"]
    subpart_id = _text(candidate.get("subpart_id"))
    exact_record = _matching_source_review(candidate, evidence.get("source_records_by_subpart", {}).get(subpart_id, []))
    if not exact_record:
        return [f"{prefix}:reviewed_runtime_evidence_missing"]

    errors: list[str] = []
    evidence_id = _text(exact_record.get("evidence_id"))
    decision_review_refs = set(_strings(decision.get("reviewed_decision_refs")))
    exact_refs = set(_strings(decision.get("exact_source_skill_evidence_refs")))
    if evidence_id and evidence_id not in decision_review_refs:
        errors.append(f"{prefix}:reviewed_decision_refs_do_not_include_valid_review")
    if evidence_id and evidence_id not in exact_refs:
        errors.append(f"{prefix}:exact_source_skill_refs_not_validated_by_review")

    approved_mark_refs = set(evidence.get("approved_mark_event_ids", set()))
    mark_refs = set(_strings(decision.get("mark_event_evidence_refs")))
    candidate_events = set(_strings(candidate.get("source_mark_event_ids")))
    if not mark_refs or not mark_refs.issubset(approved_mark_refs):
        errors.append(f"{prefix}:mark_event_refs_not_validated_by_review")
    if candidate_events and not candidate_events.issubset(approved_mark_refs):
        errors.append(f"{prefix}:candidate_mark_events_not_validated_by_review")
    for field in ("canonical_question_image_path", "canonical_mark_scheme_image_path"):
        if not _canonical_path(decision.get(field), artifact_root=artifact_root):
            errors.append(f"{prefix}:{field}_not_validated_by_review")
    return errors


def _summary(
    rows: list[RuntimeSafetyResult],
    *,
    candidates_path: Path,
    question_bank_path: Path | None,
    topic_routing_path: Path | None,
    asterion_bank_path: Path,
    mark_events_path: Path | None,
    reviewed_decisions_path: Path | None,
    promotion_decisions_path: Path | None,
    target_pass_rate: float,
    promotion_validation_errors: list[str],
) -> dict[str, Any]:
    total = len(rows)
    current_true = sum(row.current_student_runtime_safe for row in rows)
    final_true = sum(row.final_student_runtime_safe for row in rows)
    target_count = int(target_pass_rate * total + 0.999999)
    classification_counts = Counter(row.classification for row in rows)
    blocker_counts = Counter(blocker for row in rows for blocker in row.blocker_classes)
    reviewed_count = sum(row.content_lab_reviewed_approval for row in rows)
    sample_rows = [row for row in rows if row.in_deterministic_sample]
    return {
        "schema": RUNTIME_SAFE_SCHEMA,
        "schema_version": RUNTIME_SAFE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
        "contract": CONTRACT,
        "source_files": {
            "candidates": str(candidates_path),
            "question_bank": str(question_bank_path) if question_bank_path else None,
            "topic_routing": str(topic_routing_path) if topic_routing_path else None,
            "asterion_bank": str(asterion_bank_path),
            "mark_events": str(mark_events_path) if mark_events_path else None,
            "reviewed_decisions": str(reviewed_decisions_path) if reviewed_decisions_path else None,
            "promotion_decisions": str(promotion_decisions_path) if promotion_decisions_path else None,
        },
        "total_p3_candidate_count": total,
        "current_student_runtime_safe_true_count": current_true,
        "current_student_runtime_safe_false_count": total - current_true,
        "current_student_runtime_safe_percentage": round(current_true / total, 4) if total else 0.0,
        "final_student_runtime_safe_true_count": final_true,
        "final_student_runtime_safe_false_count": total - final_true,
        "final_student_runtime_safe_percentage": round(final_true / total, 4) if total else 0.0,
        "target_pass_rate": target_pass_rate,
        "target_true_count_for_50_percent": target_count,
        "additional_candidates_needed_for_50_percent_baseline": max(0, target_count - current_true),
        "additional_candidates_needed_for_50_percent_final": max(0, target_count - final_true),
        "target_met": final_true >= target_count,
        "contract_satisfied_count": sum(row.contract_safe for row in rows),
        "safe_promotion_decision_count": sum(row.promotion_decision_valid for row in rows),
        "candidates_that_appear_safely_promotable_without_code_changes": sum(row.promotion_eligible for row in rows),
        "candidates_needing_export_schema_changes": sum(row.promotion_eligible and not row.promotion_decision_valid for row in rows),
        "candidates_evidence_safe_runtime_role_blocked": sum(
            row.promotion_eligible and "asterion_runtime_schema" in row.blocker_classes for row in rows
        ),
        "candidates_runtime_role_projected_by_valid_decision": sum(row.runtime_role_projection_applied for row in rows),
        "candidates_needing_reviewed_evidence": classification_counts.get("needs_reviewed_evidence", 0),
        "candidates_needing_regenerated_or_remapped_candidates": classification_counts.get("needs_candidate_regeneration", 0),
        "candidates_needing_artifact_or_schema_repair": classification_counts.get("artifact_or_schema_repairable", 0),
        "content_lab_reviewed_approval_overlap_count": reviewed_count,
        "content_lab_reviewed_approval_runtime_safe_final_count": sum(
            row.content_lab_reviewed_approval and row.final_student_runtime_safe for row in rows
        ),
        "deterministic_sample_overlap": {
            "sample_candidate_count": len(sample_rows),
            "current_student_runtime_safe_true_count": sum(row.current_student_runtime_safe for row in sample_rows),
            "contract_satisfied_count": sum(row.contract_safe for row in sample_rows),
            "final_student_runtime_safe_true_count": sum(row.final_student_runtime_safe for row in sample_rows),
            "reviewed_approval_count": sum(row.content_lab_reviewed_approval for row in sample_rows),
        },
        "classification_counts": dict(classification_counts),
        "blocker_class_counts": dict(blocker_counts),
        "region_summary": _region_rows(rows),
        "promotion_validation_error_count": len(promotion_validation_errors),
        "promotion_validation_errors": promotion_validation_errors,
        "student_runtime_app_changed": False,
        "trust_gates_weakened": False,
    }


def _classification_payload(rows: list[RuntimeSafetyResult], *, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": f"{RUNTIME_SAFE_SCHEMA}.classification",
        "schema_version": RUNTIME_SAFE_SCHEMA_VERSION,
        "created_at": summary["created_at"],
        "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
        "classification_counts": summary["classification_counts"],
        "rows": [_row_dict(row) for row in rows],
    }


def _write_promotion_payload(path: Path, decisions: list[dict[str, Any]], *, reviewed_decisions_path: Path | None) -> None:
    payload = {
        "schema": PROMOTION_DECISIONS_SCHEMA,
        "schema_version": PROMOTION_DECISIONS_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "decision_count": len(decisions),
        "reviewed_decisions_path": str(reviewed_decisions_path) if reviewed_decisions_path else None,
        "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
        "decisions": decisions,
    }
    write_atomic_json(payload, path, sort_keys=True)


def _write_student_runtime_safe_candidates(path: Path, rows: list[RuntimeSafetyResult]) -> None:
    safe_rows = [row for row in rows if row.final_student_runtime_safe]
    payload = {
        "schema": RUNTIME_SAFE_CANDIDATES_SCHEMA,
        "schema_version": RUNTIME_SAFE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
        "record_count": len(safe_rows),
        "candidates": [
            {
                "candidate_id": row.candidate_id,
                "question_id": row.question_id,
                "subpart_id": row.subpart_id,
                "region": row.region,
                "student_runtime_safe": True,
                "promotion_decision_valid": row.promotion_decision_valid,
                "runtime_role_projection_applied": row.runtime_role_projection_applied,
                "reviewed_decision_refs": list(row.reviewed_decision_refs),
                "exact_source_skill_evidence_refs": list(row.exact_skill_evidence_refs),
                "mark_event_evidence_refs": list(row.mark_event_evidence_refs),
                "canonical_question_image_path": row.canonical_question_image_path,
                "canonical_mark_scheme_image_path": row.canonical_mark_scheme_image_path,
                "provenance": {
                    "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
                    "promotion_decision_id": f"asterion_student_runtime_safe:{row.candidate_id}",
                    "source": "scripts/audit_asterion_student_runtime_safe.py",
                },
            }
            for row in safe_rows
        ],
    }
    write_atomic_json(payload, path, sort_keys=True)


DIAGNOSIS_SUBTYPE_KEYS = [
    "missing_question_image",
    "missing_mark_scheme_image",
    "path_resolution_bug",
    "runtime_role_denied",
    "canonical_practice_projection_missing",
    "schema_field_missing",
    "candidate_category_mismatch",
    "audit_contract_too_strict",
    "genuine_not_runtime_safe",
]


def _write_artifact_schema_diagnosis(out_dir: Path, rows: list[RuntimeSafetyResult]) -> None:
    diagnosis_rows = [_diagnosis_row(row) for row in rows if _legacy_artifact_schema_repairable(row)]
    subtype_counts = {key: sum(key in row["blocker_subtypes"].split("|") for row in diagnosis_rows) for key in DIAGNOSIS_SUBTYPE_KEYS}
    kind_counts = Counter(row["blocker_kind"] for row in diagnosis_rows)
    payload = {
        "schema": f"{RUNTIME_SAFE_SCHEMA}.artifact_schema_blocker_diagnosis",
        "schema_version": RUNTIME_SAFE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "contract_version": RUNTIME_SAFE_CONTRACT_VERSION,
        "candidate_count": len(diagnosis_rows),
        "blocker_subtype_counts": subtype_counts,
        "blocker_kind_counts": dict(kind_counts),
        "safely_promotable_after_repair_count": sum(row["safely_promotable_after_repair"] for row in diagnosis_rows),
        "rows": diagnosis_rows,
    }
    _write_json(out_dir / "artifact_schema_blocker_diagnosis.json", payload)
    _write_dict_csv(out_dir / "artifact_schema_blocker_candidates.csv", diagnosis_rows)
    _write_artifact_schema_diagnosis_markdown(out_dir / "artifact_schema_blocker_diagnosis.md", payload)


def _legacy_artifact_schema_repairable(row: RuntimeSafetyResult) -> bool:
    blockers = set(row.blocker_classes)
    if row.content_lab_reviewed_approval:
        return False
    if blockers.intersection({"candidate_regeneration_required", "mapping_validation_contradiction", "ambiguous_or_quarantined"}):
        return False
    if "generation_seed_only" in blockers or "teacher_preview_only" in blockers:
        return False
    return bool(blockers.intersection({"canonical_question_image", "canonical_mark_scheme_image", "unresolved_artifact_path", "asterion_runtime_schema"}))


def _diagnosis_row(row: RuntimeSafetyResult) -> dict[str, Any]:
    subtypes = _diagnosis_subtypes(row)
    question_exists = bool(row.canonical_question_image_path) and Path(row.canonical_question_image_path).exists()
    mark_exists = bool(row.canonical_mark_scheme_image_path) and Path(row.canonical_mark_scheme_image_path).exists()
    return {
        "candidate_id": row.candidate_id,
        "question_id": row.question_id,
        "region_id": row.region,
        "current_runtime_role": row.current_runtime_role,
        "current_candidate_category": row.current_candidate_category,
        "current_student_runtime_safe": row.current_student_runtime_safe,
        "canonical_question_image_path": row.canonical_question_image_path,
        "question_image_exists": question_exists,
        "canonical_mark_scheme_image_path": row.canonical_mark_scheme_image_path,
        "mark_scheme_image_exists": mark_exists,
        "reviewed_evidence_present": row.content_lab_reviewed_approval,
        "reviewed_decision_refs": "|".join(row.reviewed_decision_refs),
        "topic_region_route_valid": row.region != "Unresolved P3 Skill Region",
        "exact_blocker_reason": "|".join(row.blocker_reasons),
        "blocker_classes": "|".join(row.blocker_classes),
        "blocker_subtypes": "|".join(subtypes),
        "blocker_kind": _diagnosis_blocker_kind(row, subtypes),
        "safely_promotable_after_repair": row.promotion_eligible,
    }


def _diagnosis_subtypes(row: RuntimeSafetyResult) -> list[str]:
    blockers = set(row.blocker_classes)
    subtypes: list[str] = []
    if "canonical_question_image" in blockers:
        subtypes.append("missing_question_image")
    if "canonical_mark_scheme_image" in blockers:
        subtypes.append("missing_mark_scheme_image")
    if "unresolved_artifact_path" in blockers:
        subtypes.append("path_resolution_bug")
    if "asterion_runtime_schema" in blockers:
        subtypes.append("runtime_role_denied")
    if row.promotion_eligible and not row.promotion_decision_valid:
        subtypes.append("canonical_practice_projection_missing")
    if row.contract_safe and not row.current_student_runtime_safe:
        subtypes.append("schema_field_missing")
    if blockers.intersection({"generation_seed_only", "teacher_preview_only"}):
        subtypes.append("candidate_category_mismatch")
    if row.promotion_eligible and "asterion_runtime_schema" in blockers:
        subtypes.append("audit_contract_too_strict")
    if not row.promotion_eligible:
        subtypes.append("genuine_not_runtime_safe")
    return _dedupe(subtypes)


def _diagnosis_blocker_kind(row: RuntimeSafetyResult, subtypes: list[str]) -> str:
    if "missing_question_image" in subtypes or "missing_mark_scheme_image" in subtypes or "path_resolution_bug" in subtypes:
        return "artifact_path"
    if row.promotion_eligible and "runtime_role_denied" in subtypes:
        return "runtime_role"
    if "canonical_practice_projection_missing" in subtypes or "schema_field_missing" in subtypes:
        return "schema_projection"
    return "genuine_safety_issue"


def _write_artifact_schema_diagnosis_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Artifact/Schema Blocker Diagnosis",
        "",
        f"- Diagnosed candidates: `{payload['candidate_count']}`",
        f"- Safely promotable after repair: `{payload['safely_promotable_after_repair_count']}`",
        "",
        "## Blocker Subtypes",
        "",
    ]
    for key in DIAGNOSIS_SUBTYPE_KEYS:
        lines.append(f"- `{key}`: `{payload['blocker_subtype_counts'].get(key, 0)}`")
    lines.extend(["", "## Blocker Kinds", ""])
    for key, count in Counter(payload["blocker_kind_counts"]).most_common():
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Finding", ""])
    lines.append(
        "All diagnosed rows are retained in `artifact_schema_blocker_candidates.csv` with image existence, "
        "reviewed-evidence status, route validity, exact blocker reasons, and safe-promotion eligibility."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_result_csv(path: Path, rows: list[RuntimeSafetyResult]) -> None:
    fieldnames = list(_row_dict(rows[0]).keys()) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_dict(row))


def _write_baseline_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Asterion Student Runtime Safe Baseline",
        "",
        f"- Total P3 candidates: `{summary['total_p3_candidate_count']}`",
        f"- Current student_runtime_safe true: `{summary['current_student_runtime_safe_true_count']}` "
        f"(`{summary['current_student_runtime_safe_percentage']:.2%}`)",
        f"- Final projected student_runtime_safe true: `{summary['final_student_runtime_safe_true_count']}` "
        f"(`{summary['final_student_runtime_safe_percentage']:.2%}`)",
        f"- Target for 50%: `{summary['target_true_count_for_50_percent']}`",
        f"- Target met: `{summary['target_met']}`",
        f"- Additional needed after projection: `{summary['additional_candidates_needed_for_50_percent_final']}`",
        f"- Trust gates weakened: `{summary['trust_gates_weakened']}`",
        f"- Student runtime app changed: `{summary['student_runtime_app_changed']}`",
        "",
        "## Contract",
        "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(summary["contract"], start=1))
    lines.extend(["", "## Blockers", ""])
    for blocker, count in Counter(summary["blocker_class_counts"]).most_common(20):
        lines.append(f"- `{blocker}`: `{count}`")
    lines.extend(["", "## Regions", ""])
    for row in summary["region_summary"]:
        lines.append(
            f"- {row['region']}: `{row['final_student_runtime_safe_true_count']}` / `{row['total_candidates']}` "
            f"(`{row['final_student_runtime_safe_percentage']:.2%}`)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_classification_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Asterion Student Runtime Safe Classification", ""]
    for key, count in Counter(payload["classification_counts"]).most_common():
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## First Blocked Rows", ""])
    for row in payload["rows"][:30]:
        if row["classification"] != "runtime_safe_now":
            lines.append(f"- `{row['candidate_id']}` `{row['classification']}` blockers=`{row['blocker_classes']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_dict(row: RuntimeSafetyResult) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "question_id": row.question_id,
        "subpart_id": row.subpart_id,
        "region": row.region,
        "current_student_runtime_safe": row.current_student_runtime_safe,
        "contract_safe": row.contract_safe,
        "final_student_runtime_safe": row.final_student_runtime_safe,
        "promotion_decision_valid": row.promotion_decision_valid,
        "promotion_eligible": row.promotion_eligible,
        "runtime_role_projection_applied": row.runtime_role_projection_applied,
        "classification": row.classification,
        "blocker_classes": "|".join(row.blocker_classes),
        "blocker_reasons": "|".join(row.blocker_reasons),
        "reviewed_decision_refs": "|".join(row.reviewed_decision_refs),
        "exact_skill_evidence_refs": "|".join(row.exact_skill_evidence_refs),
        "mark_event_evidence_refs": "|".join(row.mark_event_evidence_refs),
        "canonical_question_image_path": row.canonical_question_image_path,
        "canonical_mark_scheme_image_path": row.canonical_mark_scheme_image_path,
        "risk_flags": "|".join(row.risk_flags),
        "in_deterministic_sample": row.in_deterministic_sample,
        "content_lab_reviewed_approval": row.content_lab_reviewed_approval,
        "current_runtime_role": row.current_runtime_role,
        "current_candidate_category": row.current_candidate_category,
    }


def _region_rows(rows: list[RuntimeSafetyResult]) -> list[dict[str, Any]]:
    result = []
    for region in sorted({row.region for row in rows}, key=_region_rank):
        region_rows = [row for row in rows if row.region == region]
        current = sum(row.current_student_runtime_safe for row in region_rows)
        final = sum(row.final_student_runtime_safe for row in region_rows)
        result.append(
            {
                "region": region,
                "total_candidates": len(region_rows),
                "current_student_runtime_safe_true_count": current,
                "current_student_runtime_safe_percentage": round(current / len(region_rows), 4) if region_rows else 0.0,
                "contract_satisfied_count": sum(row.contract_safe for row in region_rows),
                "final_student_runtime_safe_true_count": final,
                "final_student_runtime_safe_percentage": round(final / len(region_rows), 4) if region_rows else 0.0,
            }
        )
    return result


def _blocker_rows(rows: list[RuntimeSafetyResult]) -> list[dict[str, Any]]:
    counter = Counter((row.region, blocker) for row in rows for blocker in row.blocker_classes)
    return [
        {"region": region, "blocker_class": blocker, "count": count}
        for (region, blocker), count in sorted(counter.items(), key=lambda item: (_region_rank(item[0][0]), item[0][1]))
    ]


def _questions_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_text(record.get("question_id")): record for record in payload.get("questions", []) if isinstance(record, dict)}


def _topic_routing_by_question(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if isinstance(records, dict):
        return {_text(key): value for key, value in records.items() if isinstance(value, dict)}
    if isinstance(records, list):
        return {_text(record.get("question_id")): record for record in records if isinstance(record, dict)}
    return {}


def _mark_event_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        for event in record.get("mark_events", []):
            if isinstance(event, dict) and _text(event.get("event_id")):
                enriched = dict(event)
                enriched["source_question_id"] = record.get("question_id")
                result[_text(event.get("event_id"))] = enriched
    return result


def _skill_regions(payload: dict[str, Any]) -> dict[str, str]:
    result = {}
    for skill in payload.get("skills", []):
        if not isinstance(skill, dict):
            continue
        skill_id = _text(skill.get("skill_id"))
        section = _text(skill.get("section"))
        region = _region_from_section(section)
        if skill_id and region:
            result[skill_id] = region
    return result


def _question_skill_mapping_regions(payload: dict[str, Any], *, skill_regions: dict[str, str]) -> dict[str, tuple[str, str]]:
    result = {}
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
        for key in _strings([mapping.get("subpart_id"), mapping.get("question_id")]):
            result.setdefault(key, (region, "question_skill_mapping_for_runtime_route"))
    return result


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
    subpart_id = _text(candidate.get("subpart_id"))
    question_id = _text(candidate.get("question_id"))
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


def _subpart_by_id(question: dict[str, Any], subpart_id: str) -> dict[str, Any]:
    for subpart in question.get("subparts", []):
        if isinstance(subpart, dict) and _text(subpart.get("subpart_id")) == subpart_id:
            return subpart
    return {}


def _status(question: dict[str, Any], field: str) -> str:
    notes = question.get("notes") if isinstance(question.get("notes"), dict) else {}
    return _text(question.get(field) or notes.get(field)).lower()


def _canonical_path(path_value: Any, *, artifact_root: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path) if path.exists() else ""
    candidates = [path, artifact_root / path, Path.cwd() / path]
    if str(path).startswith("output/"):
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _candidate_ids_from_csv(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {_text(row.get("candidate_id")) for row in csv.DictReader(handle) if _text(row.get("candidate_id"))}


def _regeneration_backlog_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    payload = _read_json(path)
    rows = payload.get("rows") or payload.get("candidates") or payload.get("items") or []
    if isinstance(rows, dict):
        rows = rows.values()
    return {
        _text(row.get("candidate_id") or row.get("original_candidate_id"))
        for row in rows
        if isinstance(row, dict) and _text(row.get("candidate_id") or row.get("original_candidate_id"))
    }


def _region_from_section(section: str) -> str | None:
    for prefix, region in SECTION_REGION_NAMES.items():
        if section.startswith(prefix) or f" {prefix} " in section:
            return region
    return None


def _region_rank(region: str) -> tuple[int, str]:
    try:
        return (P3_REGION_ORDER.index(region), region)
    except ValueError:
        return (len(P3_REGION_ORDER), region)


def _any_ambiguity(values: Iterable[str]) -> bool:
    normalized = {_text(value).lower() for value in values}
    return any(any(term in value for term in AMBIGUITY_TERMS) for value in normalized)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
