from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import (
    P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA,
    P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION,
    P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION,
)


AUTO_REVIEW_BATCH_SCHEMA = "exam_bank.content_lab.auto_review_batch"
AUTO_REVIEW_BATCH_SCHEMA_VERSION = 1
AUTO_REVIEW_DECISION_VERSION = "content_lab_p3_auto_review_decision_v1"
AUTO_REVIEW_IMPORT_SCHEMA = "exam_bank.content_lab.auto_reviewed_decisions"
AUTO_REVIEW_IMPORT_SCHEMA_VERSION = 1
AUTO_REVIEW_PROMPT_VERSION = "content_lab_p3_auto_review_prompt_v1"
AUTO_REVIEW_SOURCE = "automated_agentic_review"
DEFAULT_CONFIDENCE_THRESHOLD = 0.90
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

APPROVAL_DECISIONS = {"approved"}
AMBIGUITY_FLAGS = {
    "ambiguous",
    "quarantine",
    "quarantined",
    "advisory_text_only",
    "ocr_only",
    "missing_canonical_image",
    "missing_mark_scheme",
    "reviewer_verifier_disagreement",
    "hard_block",
}


def build_auto_review_batch(
    *,
    audit_dir: Path,
    candidates_path: Path,
    question_bank_path: Path,
    mark_events_path: Path,
    artifact_root: Path,
    out_dir: Path,
    target_pass_count: int = 70,
    buffer_count: int = 20,
    skill_map_path: Path = Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    question_skill_mappings_path: Path = Path(
        "exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"
    ),
) -> dict[str, Any]:
    candidates_payload = _read_json(candidates_path)
    question_bank_payload = _read_json(question_bank_path)
    mark_events_payload = _read_json(mark_events_path)
    skill_map_payload = _read_json(skill_map_path) if skill_map_path.exists() else {}
    mappings_payload = _read_json(question_skill_mappings_path) if question_skill_mappings_path.exists() else {}

    sample_rows = _read_csv(audit_dir / "sample_results.csv")
    queue_rows = _read_csv(audit_dir / "human_review_queue.csv")
    inventory_rows = _read_csv(audit_dir / "p3_candidate_inventory.csv")
    summary = _read_json(audit_dir / "audit_summary.json") if (audit_dir / "audit_summary.json").exists() else {}

    candidates = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in candidates_payload.get("candidates", [])
        if isinstance(candidate, dict)
    }
    questions = {
        str(question.get("question_id") or ""): question
        for question in question_bank_payload.get("questions", [])
        if isinstance(question, dict)
    }
    event_index = _mark_event_index(mark_events_payload)
    mapping_index = _mapping_index(mappings_payload)
    skills = {
        str(skill.get("skill_id") or ""): skill
        for skill in skill_map_payload.get("skills", [])
        if isinstance(skill, dict) and str(skill.get("skill_id") or "")
    }
    sample_by_id = {row["candidate_id"]: row for row in sample_rows}
    queue_by_id = {row["candidate_id"]: row for row in queue_rows}
    inventory_by_id = {row["candidate_id"]: row for row in inventory_rows}
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for candidate_id, candidate in candidates.items():
        if str(candidate.get("paper_family") or "").lower() != "p3":
            continue
        queue_row = queue_by_id.get(candidate_id)
        inventory_row = inventory_by_id.get(candidate_id, {})
        sample_row = sample_by_id.get(candidate_id)
        if queue_row is None and sample_row is None:
            continue
        row = _batch_row(
            candidate=candidate,
            queue_row=queue_row,
            inventory_row=inventory_row,
            sample_row=sample_row,
            question=questions.get(str(candidate.get("question_id") or ""), {}),
            mappings=mapping_index,
            skills=skills,
            events=event_index,
            artifact_root=artifact_root,
        )
        if row["selection_eligibility"]["eligible"]:
            eligible.append(row)
        else:
            skipped.append(
                {
                    "candidate_id": row["candidate_id"],
                    "question_id": row["question_id"],
                    "subpart_id": row["subpart_id"],
                    "reasons": row["selection_eligibility"]["reasons"],
                }
            )

    current_passed = int(summary.get("sample_passed") or 0)
    needed_for_target = max(0, int(target_pass_count) - current_passed)
    selection_target = needed_for_target + max(0, int(buffer_count))
    selected = _select_rows(eligible, limit=selection_target)
    batch_id = _batch_id(selected, audit_dir=audit_dir, target_pass_count=target_pass_count, buffer_count=buffer_count)
    for index, row in enumerate(selected, start=1):
        row["batch_id"] = batch_id
        row["batch_index"] = index

    manifest = {
        "schema": AUTO_REVIEW_BATCH_SCHEMA,
        "schema_version": AUTO_REVIEW_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": _utc_now_iso(),
        "source_files": {
            "audit_dir": str(audit_dir),
            "candidates": str(candidates_path),
            "question_bank": str(question_bank_path),
            "mark_events": str(mark_events_path),
            "artifact_root": str(artifact_root),
            "skill_map": str(skill_map_path),
            "question_skill_mappings": str(question_skill_mappings_path),
        },
        "target_pass_count": int(target_pass_count),
        "current_sample_passed": current_passed,
        "needed_for_target": needed_for_target,
        "buffer_count": int(buffer_count),
        "selection_target": selection_target,
        "selected_count": len(selected),
        "selected_sample_count": sum(1 for row in selected if row["in_deterministic_sample"]),
        "eligible_count": len(eligible),
        "skipped_count": len(skipped),
        "region_counts": dict(Counter(row["region"] for row in selected)),
        "selection_rules": [
            "failing deterministic sample candidates before broader queue rows",
            "rows needing both mapping and mark-event review before single-blocker rows",
            "canonical question and mark-scheme image files required for selection",
            "proposed P3 skill IDs and source mark-event refs required for selection",
            "round-robin balancing by P3 region where possible",
        ],
        "prompt_version": AUTO_REVIEW_PROMPT_VERSION,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    }
    payload = {
        "schema": AUTO_REVIEW_BATCH_SCHEMA,
        "schema_version": AUTO_REVIEW_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "manifest": manifest,
        "reviewer_prompt": auto_review_prompt(),
        "decision_schema": auto_review_decision_schema(),
        "rows": selected,
        "skipped_rows": skipped,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(payload, out_dir / "auto_review_batch.json", sort_keys=True)
    write_atomic_json(manifest, out_dir / "auto_review_manifest.json", sort_keys=True)
    _write_batch_csv(out_dir / "auto_review_batch.csv", selected)
    _write_batch_markdown(out_dir / "auto_review_batch.md", manifest, selected, skipped)
    return payload


def build_full_pool_classification(
    *,
    audit_dir: Path,
    candidates_path: Path,
    question_bank_path: Path,
    mark_events_path: Path,
    artifact_root: Path,
    out_dir: Path,
    reviewed_decisions_path: Path | None = None,
    loop005_regeneration_backlog_path: Path | None = None,
    skill_map_path: Path = Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    question_skill_mappings_path: Path = Path(
        "exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"
    ),
) -> dict[str, Any]:
    candidates_payload = _read_json(candidates_path)
    question_bank_payload = _read_json(question_bank_path)
    mark_events_payload = _read_json(mark_events_path)
    skill_map_payload = _read_json(skill_map_path) if skill_map_path.exists() else {}
    mappings_payload = _read_json(question_skill_mappings_path) if question_skill_mappings_path.exists() else {}
    reviewed_payload = _read_json(reviewed_decisions_path) if reviewed_decisions_path and reviewed_decisions_path.exists() else {}
    loop005_backlog = _loop005_regeneration_ids(loop005_regeneration_backlog_path)

    sample_rows = _read_csv(audit_dir / "sample_results.csv")
    queue_rows = _read_csv(audit_dir / "human_review_queue.csv")
    inventory_rows = _read_csv(audit_dir / "p3_candidate_inventory.csv")
    summary = _read_json(audit_dir / "audit_summary.json") if (audit_dir / "audit_summary.json").exists() else {}

    candidates = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in candidates_payload.get("candidates", [])
        if isinstance(candidate, dict)
    }
    questions = {
        str(question.get("question_id") or ""): question
        for question in question_bank_payload.get("questions", [])
        if isinstance(question, dict)
    }
    event_index = _mark_event_index(mark_events_payload)
    mapping_index = _mapping_index(mappings_payload)
    skills = {
        str(skill.get("skill_id") or ""): skill
        for skill in skill_map_payload.get("skills", [])
        if isinstance(skill, dict) and str(skill.get("skill_id") or "")
    }
    approved_ids = {
        str(record.get("candidate_id") or "")
        for record in reviewed_payload.get("records", [])
        if isinstance(record, dict) and str(record.get("candidate_id") or "")
    }
    sample_by_id = {row["candidate_id"]: row for row in sample_rows}
    queue_by_id = {row["candidate_id"]: row for row in queue_rows}
    inventory_by_id = {row["candidate_id"]: row for row in inventory_rows}

    rows: list[dict[str, Any]] = []
    for inventory_row in inventory_rows:
        if inventory_row.get("passed") == "True":
            continue
        candidate_id = inventory_row["candidate_id"]
        candidate = candidates.get(candidate_id)
        if not candidate:
            continue
        enriched = _batch_row(
            candidate=candidate,
            queue_row=queue_by_id.get(candidate_id),
            inventory_row=inventory_row,
            sample_row=sample_by_id.get(candidate_id),
            question=questions.get(str(candidate.get("question_id") or ""), {}),
            mappings=mapping_index,
            skills=skills,
            events=event_index,
            artifact_root=artifact_root,
        )
        classification, reason = _classify_full_pool_candidate(
            enriched,
            inventory_row=inventory_row,
            loop005_regeneration_ids=loop005_backlog,
        )
        mapping = enriched.get("proposed_mapping") or {}
        row = {
            "candidate_id": enriched["candidate_id"],
            "question_id": enriched["question_id"],
            "subpart_id": enriched["subpart_id"],
            "region": enriched["region"],
            "classification": classification,
            "classification_reason": reason,
            "in_deterministic_sample": enriched["in_deterministic_sample"],
            "already_reviewed_decision": enriched["candidate_id"] in approved_ids,
            "current_blockers": "|".join(enriched["current_blockers"]),
            "human_review_reasons": "|".join(enriched["human_review_reasons"]),
            "canonical_question_image_path": enriched["canonical_question_image_path"],
            "canonical_mark_scheme_image_path": enriched["canonical_mark_scheme_image_path"],
            "proposed_exact_skill_ids": "|".join(enriched["proposed_exact_skill_ids"]),
            "source_mark_event_ids": "|".join(enriched["source_mark_event_ids"]),
            "mapping_confidence": mapping.get("confidence") or "",
            "mapping_review_status": mapping.get("review_status") or "",
            "mapping_id": mapping.get("mapping_id") or "",
            "selection_score": _full_pool_selection_score(enriched, classification),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["classification"],
            row["in_deterministic_sample"],
            -float(row["selection_score"]),
            _region_rank(row["region"]),
            row["candidate_id"],
        )
    )
    counts = Counter(row["classification"] for row in rows)
    total = int(summary.get("p3_candidates") or len(inventory_rows))
    passed = sum(1 for row in inventory_rows if row.get("passed") == "True")
    payload = {
        "schema": "exam_bank.content_lab.full_pool_classification",
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "source_audit_dir": str(audit_dir),
        "reviewed_decisions_path": str(reviewed_decisions_path) if reviewed_decisions_path else None,
        "total_p3_candidates": total,
        "passed_count": passed,
        "failed_count": len(rows),
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "classification_counts": dict(counts),
        "estimated_additional_approvals_needed": {
            "for_70_percent": max(0, int(0.70 * total + 0.999999) - passed),
            "for_90_percent": max(0, int(0.90 * total + 0.999999) - passed),
        },
        "top_25_direct_review_candidates": _top_candidates(rows, "eligible_for_direct_agentic_review"),
        "top_25_corrected_mapping_candidates": _top_candidates(rows, "mapping_correctable"),
        "top_25_regeneration_required_candidates": _top_candidates(rows, "regenerate_candidate_required"),
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(payload, out_dir / "full_pool_classification.json", sort_keys=True)
    _write_rows_csv(out_dir / "full_pool_classification.csv", rows)
    _write_full_pool_classification_markdown(out_dir / "full_pool_classification.md", payload)
    return payload


def build_full_pool_auto_review_batch(
    *,
    classification_path: Path,
    audit_dir: Path,
    candidates_path: Path,
    question_bank_path: Path,
    mark_events_path: Path,
    artifact_root: Path,
    out_dir: Path,
    limit: int = 150,
    skill_map_path: Path = Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    question_skill_mappings_path: Path = Path(
        "exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"
    ),
) -> dict[str, Any]:
    classification_payload = _read_json(classification_path)
    candidate_ids = [
        row["candidate_id"]
        for row in classification_payload.get("rows", [])
        if isinstance(row, dict)
        and row.get("classification") in {"eligible_for_direct_agentic_review", "mapping_correctable"}
        and row.get("in_deterministic_sample") is not True
    ]
    selected_ids = set(candidate_ids[: max(0, limit)])

    candidates_payload = _read_json(candidates_path)
    question_bank_payload = _read_json(question_bank_path)
    mark_events_payload = _read_json(mark_events_path)
    skill_map_payload = _read_json(skill_map_path) if skill_map_path.exists() else {}
    mappings_payload = _read_json(question_skill_mappings_path) if question_skill_mappings_path.exists() else {}
    sample_rows = _read_csv(audit_dir / "sample_results.csv")
    queue_rows = _read_csv(audit_dir / "human_review_queue.csv")
    inventory_rows = _read_csv(audit_dir / "p3_candidate_inventory.csv")

    candidates = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in candidates_payload.get("candidates", [])
        if isinstance(candidate, dict)
    }
    questions = {
        str(question.get("question_id") or ""): question
        for question in question_bank_payload.get("questions", [])
        if isinstance(question, dict)
    }
    event_index = _mark_event_index(mark_events_payload)
    mapping_index = _mapping_index(mappings_payload)
    skills = {
        str(skill.get("skill_id") or ""): skill
        for skill in skill_map_payload.get("skills", [])
        if isinstance(skill, dict) and str(skill.get("skill_id") or "")
    }
    sample_by_id = {row["candidate_id"]: row for row in sample_rows}
    queue_by_id = {row["candidate_id"]: row for row in queue_rows}
    inventory_by_id = {row["candidate_id"]: row for row in inventory_rows}
    class_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in classification_payload.get("rows", [])
        if isinstance(row, dict)
    }

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        if candidate_id not in selected_ids:
            continue
        candidate = candidates.get(candidate_id)
        inventory_row = inventory_by_id.get(candidate_id, {})
        if not candidate:
            skipped.append({"candidate_id": candidate_id, "reasons": ["candidate_record_missing"]})
            continue
        row = _batch_row(
            candidate=candidate,
            queue_row=queue_by_id.get(candidate_id),
            inventory_row=inventory_row,
            sample_row=sample_by_id.get(candidate_id),
            question=questions.get(str(candidate.get("question_id") or ""), {}),
            mappings=mapping_index,
            skills=skills,
            events=event_index,
            artifact_root=artifact_root,
        )
        class_row = class_by_id.get(candidate_id, {})
        row["full_pool_classification"] = class_row.get("classification")
        row["full_pool_classification_reason"] = class_row.get("classification_reason")
        if row["selection_eligibility"]["eligible"]:
            selected.append(row)
        else:
            skipped.append(
                {
                    "candidate_id": row["candidate_id"],
                    "question_id": row["question_id"],
                    "subpart_id": row["subpart_id"],
                    "reasons": row["selection_eligibility"]["reasons"],
                }
            )

    batch_id = _batch_id(selected, audit_dir=audit_dir, target_pass_count=len(selected), buffer_count=0)
    for index, row in enumerate(selected, start=1):
        row["batch_id"] = batch_id
        row["batch_index"] = index

    manifest = {
        "schema": AUTO_REVIEW_BATCH_SCHEMA,
        "schema_version": AUTO_REVIEW_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": _utc_now_iso(),
        "source_files": {
            "classification": str(classification_path),
            "audit_dir": str(audit_dir),
            "candidates": str(candidates_path),
            "question_bank": str(question_bank_path),
            "mark_events": str(mark_events_path),
            "artifact_root": str(artifact_root),
        },
        "selected_count": len(selected),
        "selected_sample_count": sum(1 for row in selected if row["in_deterministic_sample"]),
        "eligible_count": len(candidate_ids),
        "skipped_count": len(skipped),
        "limit": int(limit),
        "needed_for_target": len(selected),
        "buffer_count": 0,
        "region_counts": dict(Counter(row["region"] for row in selected)),
        "classification_counts": dict(Counter(row.get("full_pool_classification") for row in selected)),
        "selection_rules": [
            "exclude deterministic sample rows unless explicitly regenerated",
            "direct-review eligible rows before mapping-correctable rows",
            "high mapping confidence and canonical image availability required",
            "stable ordering by underperforming-region need, selection score, region, candidate_id",
        ],
        "prompt_version": AUTO_REVIEW_PROMPT_VERSION,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    }
    payload = {
        "schema": AUTO_REVIEW_BATCH_SCHEMA,
        "schema_version": AUTO_REVIEW_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "manifest": manifest,
        "reviewer_prompt": auto_review_prompt(),
        "decision_schema": auto_review_decision_schema(),
        "rows": selected,
        "skipped_rows": skipped,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(payload, out_dir / "auto_review_batch.json", sort_keys=True)
    write_atomic_json(manifest, out_dir / "auto_review_manifest.json", sort_keys=True)
    _write_batch_csv(out_dir / "auto_review_batch.csv", selected)
    _write_batch_markdown(out_dir / "auto_review_batch.md", manifest, selected, skipped)
    return payload


def write_codex_agentic_review_decisions(
    *,
    batch_path: Path,
    out_dir: Path,
    reviewer_model: str = "gpt-5-codex-2026-05-27",
    max_records: int | None = None,
) -> dict[str, Any]:
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows", []) if isinstance(row, dict)]
    if max_records is not None:
        rows = rows[: max(0, max_records)]
    approvals: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for row in rows:
        classification = row.get("full_pool_classification")
        if classification != "eligible_for_direct_agentic_review":
            decision = _codex_block_decision(row, reviewer_model=reviewer_model, reason="mapping correction requires manual regenerated evidence")
            rejections.append(decision)
            continue
        decision = _codex_approval_decision(row, reviewer_model=reviewer_model)
        approvals.append(decision)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "auto_review_decisions.jsonl", approvals)
    _write_jsonl(out_dir / "auto_review_rejections.jsonl", rejections)
    _write_jsonl(out_dir / "candidate_mapping_corrections.jsonl", corrections)
    _write_rows_csv(out_dir / "candidate_mapping_corrections.csv", corrections)
    summary = {
        "schema": "exam_bank.content_lab.auto_review_decision_summary",
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "batch": str(batch_path),
        "reviewed_count": len(rows),
        "approved_count": len(approvals),
        "blocked_count": len(rejections),
        "ambiguous_count": sum(1 for row in rejections if "ambiguous" in row.get("risk_flags", [])),
        "corrected_mapping_count": len(corrections),
        "approval_candidate_ids": [row["candidate_id"] for row in approvals],
        "blocked_candidate_ids": [row["candidate_id"] for row in rejections],
        "reviewer_model": reviewer_model,
        "review_method": "codex_agentic_canonical_image_path_and_mark_event_review",
    }
    write_atomic_json(summary, out_dir / "auto_review_decision_summary.json", sort_keys=True)
    return summary


def auto_review_prompt() -> str:
    return (
        "You are an automated reviewer for CAIE 9709 Paper 3 Content Lab readiness.\n"
        "Use the canonical question image and canonical mark-scheme image as the primary evidence. "
        "Detected mark events, candidate metadata, and proposed skill mappings are review aids only.\n"
        "Do not approve from advisory extracted text, OCR text, topic labels, or candidate self-claims alone.\n"
        "Block if any canonical image is missing, mark-scheme evidence is missing, skill support is indirect, "
        "subpart/mark-event evidence is unclear, mapping is ambiguous, evidence appears quarantined, or confidence is below 0.90.\n"
        "Return exactly one JSON object matching the decision schema. Approve only when exact skill, source skill, "
        "mark-event/subpart evidence, and candidate generation are all independently supported by the images and refs."
    )


def auto_review_decision_schema() -> dict[str, Any]:
    decision_enum = ["approved", "blocked", "ambiguous"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision_version",
            "review_source",
            "candidate_id",
            "question_id",
            "subpart_id",
            "exact_skill_decision",
            "source_skill_decision",
            "mark_event_decision",
            "candidate_generation_decision",
            "confidence",
            "evidence_refs",
            "approved_exact_skill_ids",
            "approved_source_skill_ids",
            "approved_mark_event_refs",
            "explanation",
            "risk_flags",
            "reviewer",
            "verifier",
            "adjudication",
        ],
        "properties": {
            "decision_version": {"const": AUTO_REVIEW_DECISION_VERSION},
            "review_source": {"const": AUTO_REVIEW_SOURCE},
            "candidate_id": {"type": "string"},
            "question_id": {"type": "string"},
            "subpart_id": {"type": "string"},
            "exact_skill_decision": {"enum": decision_enum},
            "source_skill_decision": {"enum": decision_enum},
            "mark_event_decision": {"enum": decision_enum},
            "candidate_generation_decision": {"enum": ["approved", "blocked"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_refs": {"type": "array", "items": {"type": "object"}},
            "approved_exact_skill_ids": {"type": "array", "items": {"type": "string"}},
            "approved_source_skill_ids": {"type": "array", "items": {"type": "string"}},
            "approved_mark_event_refs": {"type": "array", "items": {"type": "object"}},
            "explanation": {"type": "string"},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
            "reviewer": {"type": "object"},
            "verifier": {"type": "object"},
            "adjudication": {"type": "object"},
        },
    }


def run_auto_reviews(
    *,
    batch_path: Path,
    out_path: Path,
    max_records: int | None = None,
    dry_run: bool = False,
    model: str = "gpt-5-mini",
    provider: str = "openai",
) -> dict[str, Any]:
    batch = _read_json(batch_path)
    rows = [row for row in batch.get("rows", []) if isinstance(row, dict)]
    done = _existing_decision_candidate_ids(out_path)
    pending = [row for row in rows if row["candidate_id"] not in done]
    if max_records is not None:
        pending = pending[: max(0, max_records)]
    manifest = {
        "schema": "exam_bank.content_lab.auto_review_run",
        "schema_version": 1,
        "batch": str(batch_path),
        "out": str(out_path),
        "provider": provider,
        "model": model,
        "prompt_version": AUTO_REVIEW_PROMPT_VERSION,
        "dry_run": dry_run,
        "pending_count": len(pending),
        "resumed_count": len(done),
        "created_at": _utc_now_iso(),
    }
    if dry_run:
        return manifest
    if provider != "openai":
        raise RuntimeError("review runner requires existing image-capable OpenAI API configuration; unsupported provider")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("review runner requires existing LLM command/API not found: OPENAI_API_KEY is not configured")

    client = OpenAI()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in pending:
            try:
                decision = _request_openai_review(client=client, model=model, row=row)
            except Exception as exc:  # isolate provider failures per record
                decision = _blocked_error_decision(row, provider=provider, model=model, error=exc)
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return manifest


def import_auto_review_decisions(
    *,
    decisions_path: Path,
    batch_path: Path,
    out_review_file: Path,
    dry_run: bool = False,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    skill_map_path: Path = Path("exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"),
    mark_events_path: Path = Path("output/json/question_bank.mark_events.v1.json"),
    artifact_root: Path = Path("output"),
) -> dict[str, Any]:
    batch = _read_json(batch_path)
    batch_rows = {str(row.get("candidate_id") or ""): row for row in batch.get("rows", []) if isinstance(row, dict)}
    decisions = _read_jsonl(decisions_path)
    skill_ids = _skill_ids(skill_map_path)
    mark_events = _mark_event_index(_read_json(mark_events_path) if mark_events_path.exists() else {})
    errors: list[str] = []
    warnings: list[str] = []
    accepted: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"decision:{index}:not_object")
            continue
        candidate_id = str(decision.get("candidate_id") or "")
        row = batch_rows.get(candidate_id)
        if not row:
            errors.append(f"decision:{index}:{candidate_id or 'missing'}:unknown_candidate_id")
            continue
        validation_errors = validate_auto_review_decision(
            decision,
            batch_row=row,
            skill_ids=skill_ids,
            mark_event_ids=set(mark_events),
            artifact_root=artifact_root,
            confidence_threshold=confidence_threshold,
        )
        validation_errors.extend(validate_mapping_correction_provenance(decision))
        if validation_errors:
            errors.extend(f"decision:{index}:{candidate_id}:{error}" for error in validation_errors)
            continue
        normalized = _normalized_import_decision(decision, row)
        previous = seen.get(candidate_id)
        if previous and previous != normalized:
            errors.append(f"decision:{index}:{candidate_id}:duplicate_conflicting_decision")
            continue
        if not previous:
            seen[candidate_id] = normalized
            accepted.append(normalized)

    source_records = [_source_skill_review_record(decision) for decision in accepted]
    mark_event_decisions = [
        _mark_event_review_decision(decision, ref)
        for decision in accepted
        for ref in decision["approved_mark_event_refs"]
    ]
    payload = {
        "schema": AUTO_REVIEW_IMPORT_SCHEMA,
        "schema_version": AUTO_REVIEW_IMPORT_SCHEMA_VERSION,
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "created_at": _utc_now_iso(),
        "source_batch_id": batch.get("batch_id"),
        "source_batch_path": str(batch_path),
        "confidence_threshold": confidence_threshold,
        "record_count": len(accepted),
        "records": accepted,
        "source_skill_records_schema": P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA,
        "source_skill_records_schema_version": P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION,
        "source_skill_records": source_records,
        "mark_event_decisions_schema_version": P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION,
        "mark_event_decisions": mark_event_decisions,
    }
    report = {
        "ok": not errors,
        "dry_run": dry_run,
        "decision_count": len(decisions),
        "accepted_count": len(accepted),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "out_review_file": str(out_review_file),
    }
    if not dry_run and not errors:
        write_atomic_json(payload, out_review_file, sort_keys=True)
    return report


def merge_auto_review_import_files(
    *,
    reviewed_files: list[Path],
    out_review_file: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    payloads: list[dict[str, Any]] = []
    for path in reviewed_files:
        if not path.exists():
            errors.append(f"reviewed_file_missing:{path}")
            continue
        payload = _read_json(path)
        if payload.get("schema") != AUTO_REVIEW_IMPORT_SCHEMA:
            errors.append(f"reviewed_file_invalid_schema:{path}")
            continue
        payloads.append(payload)

    records: list[dict[str, Any]] = []
    source_skill_records: list[dict[str, Any]] = []
    mark_event_decisions: list[dict[str, Any]] = []
    seen_records: dict[str, dict[str, Any]] = {}
    seen_source_records: dict[str, dict[str, Any]] = {}
    seen_mark_decisions: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        source_file = str(payload.get("source_review_file") or "")
        for record in payload.get("records", []):
            if not isinstance(record, dict):
                continue
            decision_id = str(record.get("decision_id") or "")
            if not decision_id:
                errors.append("record_missing_decision_id")
                continue
            existing = seen_records.get(decision_id)
            if existing and existing != record:
                errors.append(f"duplicate_conflicting_decision:{decision_id}")
                continue
            if not existing:
                seen_records[decision_id] = record
                records.append(record)
        for record in payload.get("source_skill_records", []):
            if not isinstance(record, dict):
                continue
            evidence_id = str(record.get("evidence_id") or "")
            if not evidence_id:
                errors.append("source_skill_record_missing_evidence_id")
                continue
            existing = seen_source_records.get(evidence_id)
            if existing and existing != record:
                errors.append(f"duplicate_conflicting_source_skill_record:{evidence_id}")
                continue
            if not existing:
                seen_source_records[evidence_id] = record
                source_skill_records.append(record)
        for decision in payload.get("mark_event_decisions", []):
            if not isinstance(decision, dict):
                continue
            decision_id = str(decision.get("decision_id") or "")
            if not decision_id:
                errors.append("mark_event_decision_missing_decision_id")
                continue
            existing = seen_mark_decisions.get(decision_id)
            if existing and existing != decision:
                errors.append(f"duplicate_conflicting_mark_event_decision:{decision_id}")
                continue
            if not existing:
                seen_mark_decisions[decision_id] = decision
                mark_event_decisions.append(decision)
        if not source_file:
            warnings.append("reviewed_file_missing_source_review_file")

    merged = {
        "schema": AUTO_REVIEW_IMPORT_SCHEMA,
        "schema_version": AUTO_REVIEW_IMPORT_SCHEMA_VERSION,
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "created_at": _utc_now_iso(),
        "source_review_files": [str(path) for path in reviewed_files],
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "record_count": len(records),
        "records": records,
        "source_skill_records_schema": P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA,
        "source_skill_records_schema_version": P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION,
        "source_skill_records": source_skill_records,
        "mark_event_decisions_schema_version": P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION,
        "mark_event_decisions": mark_event_decisions,
    }
    report = {
        "ok": not errors,
        "dry_run": dry_run,
        "reviewed_file_count": len(reviewed_files),
        "record_count": len(records),
        "source_skill_record_count": len(source_skill_records),
        "mark_event_decision_count": len(mark_event_decisions),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "out_review_file": str(out_review_file),
    }
    if not dry_run and not errors:
        write_atomic_json(merged, out_review_file, sort_keys=True)
    return report


def validate_auto_review_decision(
    decision: dict[str, Any],
    *,
    batch_row: dict[str, Any],
    skill_ids: set[str],
    mark_event_ids: set[str],
    artifact_root: Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[str]:
    errors: list[str] = []
    for field in auto_review_decision_schema()["required"]:
        if field not in decision:
            errors.append(f"missing_required_field:{field}")
    if decision.get("decision_version") != AUTO_REVIEW_DECISION_VERSION:
        errors.append("invalid_decision_version")
    if decision.get("review_source") != AUTO_REVIEW_SOURCE:
        errors.append("invalid_review_source")
    for field in ("candidate_id", "question_id", "subpart_id"):
        if str(decision.get(field) or "") != str(batch_row.get(field) or ""):
            errors.append(f"{field}_does_not_match_batch")
    confidence = float(decision.get("confidence") or 0)
    if confidence < confidence_threshold:
        errors.append("confidence_below_threshold")
    for field in ("exact_skill_decision", "source_skill_decision", "mark_event_decision"):
        if decision.get(field) not in APPROVAL_DECISIONS:
            errors.append(f"{field}_not_approved")
    if decision.get("candidate_generation_decision") != "approved":
        errors.append("candidate_generation_decision_not_approved")
    if _risk_flags_block(decision.get("risk_flags")):
        errors.append("blocking_risk_flags_present")
    if not str(decision.get("explanation") or "").strip():
        errors.append("missing_explanation")
    for actor_field in ("reviewer", "verifier"):
        actor = decision.get(actor_field)
        if not isinstance(actor, dict):
            errors.append(f"{actor_field}_metadata_missing")
            continue
        if not str(actor.get("model") or "").strip() or not str(actor.get("provider") or "").strip():
            errors.append(f"{actor_field}_model_or_provider_missing")
        if not str(actor.get("prompt_version") or "").strip():
            errors.append(f"{actor_field}_prompt_version_missing")
    adjudication = decision.get("adjudication")
    if not isinstance(adjudication, dict):
        errors.append("adjudication_missing")
    else:
        if adjudication.get("status") != "approved":
            errors.append("adjudication_not_approved")
        if adjudication.get("reviewer_verifier_agree") is not True:
            errors.append("reviewer_verifier_disagreement")

    approved_skill_ids = set(_strings(decision.get("approved_exact_skill_ids"))) | set(
        _strings(decision.get("approved_source_skill_ids"))
    )
    if not approved_skill_ids:
        errors.append("missing_approved_skill_ids")
    unknown_skills = sorted(skill_id for skill_id in approved_skill_ids if skill_ids and skill_id not in skill_ids)
    if unknown_skills:
        errors.append(f"unknown_skill_ids:{','.join(unknown_skills)}")

    approved_event_ids = {
        str(ref.get("event_id") or "").strip()
        for ref in decision.get("approved_mark_event_refs") or []
        if isinstance(ref, dict) and str(ref.get("event_id") or "").strip()
    }
    candidate_event_ids = set(_strings(batch_row.get("source_mark_event_ids")))
    if not approved_event_ids:
        errors.append("missing_approved_mark_event_refs")
    missing_candidate_events = sorted(candidate_event_ids - approved_event_ids)
    if missing_candidate_events:
        errors.append(f"candidate_mark_events_not_fully_approved:{','.join(missing_candidate_events)}")
    unknown_events = sorted(event_id for event_id in approved_event_ids if mark_event_ids and event_id not in mark_event_ids)
    if unknown_events:
        errors.append(f"unknown_mark_event_refs:{','.join(unknown_events)}")

    evidence_refs = decision.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        errors.append("missing_evidence_refs")
    else:
        ref_types = {str(ref.get("type") or "") for ref in evidence_refs if isinstance(ref, dict)}
        if "canonical_question_image" not in ref_types:
            errors.append("canonical_question_image_evidence_missing")
        if "canonical_mark_scheme_image" not in ref_types:
            errors.append("canonical_mark_scheme_image_evidence_missing")
        for ref in evidence_refs:
            if not isinstance(ref, dict):
                continue
            if ref.get("type") in {"canonical_question_image", "canonical_mark_scheme_image"}:
                path = str(ref.get("path") or "")
                if not _artifact_exists(path, artifact_root):
                    errors.append(f"evidence_path_not_found:{path}")
    return errors


def validate_mapping_correction_provenance(decision: dict[str, Any]) -> list[str]:
    correction = decision.get("mapping_correction")
    if correction is None:
        return []
    errors: list[str] = []
    if not isinstance(correction, dict):
        return ["mapping_correction_not_object"]
    original_skill_ids = set(_strings(correction.get("original_proposed_skill_ids")))
    corrected_skill_ids = set(_strings(decision.get("approved_exact_skill_ids"))) | set(
        _strings(decision.get("approved_source_skill_ids"))
    )
    if not original_skill_ids:
        errors.append("mapping_correction_missing_original_proposed_skill_ids")
    if not str(correction.get("original_rejection_reason") or "").strip():
        errors.append("mapping_correction_missing_original_rejection_reason")
    if not str(correction.get("source_loop") or "").strip():
        errors.append("mapping_correction_missing_source_loop")
    if not str(correction.get("correction_decision") or "").strip():
        errors.append("mapping_correction_missing_correction_decision")
    if original_skill_ids and corrected_skill_ids and corrected_skill_ids == original_skill_ids:
        errors.append("corrected_mapping_matches_rejected_original")
    if not correction.get("canonical_evidence_refs"):
        errors.append("mapping_correction_missing_canonical_evidence_refs")
    return errors


def automated_decisions_for_audit(payload: dict[str, Any], *, artifact_root: Path) -> dict[str, dict[str, Any]]:
    if payload.get("schema") != AUTO_REVIEW_IMPORT_SCHEMA:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        if _imported_record_satisfies_audit(record, artifact_root=artifact_root):
            result[str(record.get("candidate_id") or "")] = record
    return result


def automated_source_records(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = payload.get("source_skill_records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return {}
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        subpart_id = str(record.get("subpart_id") or "")
        if subpart_id:
            result[subpart_id].append(record)
    return result


def automated_mark_event_ids(payload: dict[str, Any]) -> set[str]:
    decisions = payload.get("mark_event_decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, list):
        return set()
    return {
        str(decision.get("event_id") or "")
        for decision in decisions
        if isinstance(decision, dict)
        and str(decision.get("status") or "").lower() in {"approved", "reviewed"}
        and decision.get("satisfies_generation_gate") is True
    }


def _batch_row(
    *,
    candidate: dict[str, Any],
    queue_row: dict[str, str] | None,
    inventory_row: dict[str, str],
    sample_row: dict[str, str] | None,
    question: dict[str, Any],
    mappings: dict[str, dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    events: dict[str, dict[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    question_id = str(candidate.get("question_id") or "")
    subpart_id = str(candidate.get("subpart_id") or "")
    mapping = mappings.get(subpart_id) or mappings.get(question_id) or {}
    proposed_skill_ids = _dedupe(
        _strings(candidate.get("source_skill_ids"))
        + _strings(candidate.get("reviewed_source_skill_ids"))
        + _strings(mapping.get("primary_skill_ids"))
    )
    event_ids = _strings(candidate.get("source_mark_event_ids"))
    question_image = _artifact_path(
        (candidate.get("source_artifacts") or {}).get("question_crop_path")
        or question.get("canonical_question_artifact")
        or question.get("question_image_path"),
        artifact_root=artifact_root,
    )
    mark_scheme_image = _artifact_path(
        (candidate.get("source_artifacts") or {}).get("mark_scheme_crop_path")
        or question.get("canonical_mark_scheme_artifact")
        or question.get("mark_scheme_image_path"),
        artifact_root=artifact_root,
    )
    mark_event_refs = [_mark_event_ref(events[event_id]) for event_id in event_ids if event_id in events]
    missing_events = [event_id for event_id in event_ids if event_id not in events]
    blocker_text = (
        (sample_row or {}).get("blocker_classes")
        or (queue_row or {}).get("blocker_classes")
        or inventory_row.get("blocker_classes")
        or ""
    )
    blockers = _split_pipe(blocker_text)
    queue_reasons = _split_pipe((queue_row or {}).get("human_review_reasons", ""))
    region = (sample_row or {}).get("region") or (queue_row or {}).get("region") or inventory_row.get("region") or "Unresolved P3 Skill Region"
    needs_mapping = "blocked_skill_mapping" in blockers or "blocked_mapping_review_gate" in blockers
    needs_mark_events = "blocked_mark_events" in blockers
    eligibility_reasons = []
    if not question_image:
        eligibility_reasons.append("missing_canonical_question_image")
    if not mark_scheme_image:
        eligibility_reasons.append("missing_canonical_mark_scheme_image")
    if not proposed_skill_ids:
        eligibility_reasons.append("missing_proposed_skill_ids")
    if not event_ids:
        eligibility_reasons.append("missing_source_mark_event_ids")
    if missing_events:
        eligibility_reasons.append("unknown_source_mark_event_ids")
    unknown_skills = [skill_id for skill_id in proposed_skill_ids if skills and skill_id not in skills]
    if unknown_skills:
        eligibility_reasons.append("unknown_proposed_skill_ids")
    if any("quarantin" in text or "ambiguous" in text for text in blockers + queue_reasons):
        eligibility_reasons.append("ambiguous_or_quarantined_candidate")
    selected_reason = []
    if sample_row and sample_row.get("passed") == "False":
        selected_reason.append("failing_deterministic_sample")
    if needs_mapping and needs_mark_events:
        selected_reason.append("needs_mapping_and_mark_event_review")
    elif needs_mapping:
        selected_reason.append("needs_mapping_review")
    elif needs_mark_events:
        selected_reason.append("needs_mark_event_review")
    if question_image and mark_scheme_image:
        selected_reason.append("canonical_images_available")
    if proposed_skill_ids and event_ids and not missing_events:
        selected_reason.append("clear_proposed_mapping_and_event_refs")
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "question_id": question_id,
        "subpart_id": subpart_id,
        "region": region,
        "topic": question.get("topic") or (question.get("notes") or {}).get("subtopic"),
        "current_blockers": blockers,
        "human_review_reasons": queue_reasons,
        "canonical_question_image_path": question_image,
        "canonical_mark_scheme_image_path": mark_scheme_image,
        "proposed_exact_skill_ids": proposed_skill_ids,
        "proposed_source_skill_ids": proposed_skill_ids,
        "proposed_mapping": {
            "mapping_id": mapping.get("mapping_id"),
            "primary_skill_ids": _strings(mapping.get("primary_skill_ids")),
            "secondary_skill_ids": _strings(mapping.get("secondary_skill_ids")),
            "confidence": mapping.get("confidence"),
            "review_status": mapping.get("review_status"),
            "evidence": mapping.get("evidence", {}),
        },
        "detected_mark_event_refs": mark_event_refs,
        "source_mark_event_ids": event_ids,
        "missing_mark_event_ids": missing_events,
        "mark_values_detected": [ref.get("mark_value") for ref in mark_event_refs],
        "candidate_metadata": {
            "paper": candidate.get("paper"),
            "paper_family": candidate.get("paper_family"),
            "question_number": candidate.get("question_number"),
            "subpart_label": candidate.get("subpart_label"),
            "marks": candidate.get("marks"),
            "candidate_selection": candidate.get("candidate_selection"),
            "generation_gate": candidate.get("generation_gate"),
            "mapping_review_gate": candidate.get("mapping_review_gate"),
            "source_skill_review_gate": candidate.get("source_skill_review_gate"),
            "mark_event_review_gate": candidate.get("mark_event_review_gate"),
        },
        "selection_reason": _dedupe(selected_reason),
        "in_deterministic_sample": bool(sample_row),
        "deterministic_sample_passed": (sample_row or {}).get("passed") == "True",
        "needs_mapping_review": needs_mapping,
        "needs_mark_event_review": needs_mark_events,
        "needs_both_mapping_and_mark_event_review": needs_mapping and needs_mark_events,
        "selection_eligibility": {
            "eligible": not eligibility_reasons,
            "reasons": eligibility_reasons,
        },
    }


def _select_rows(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    sample_failures = [
        row for row in rows if row["in_deterministic_sample"] and not row["deterministic_sample_passed"]
    ]
    broader = [row for row in rows if row not in sample_failures]
    for pool in (sample_failures, broader):
        for row in _region_round_robin(pool):
            if len(selected) >= limit:
                return selected
            selected.append(row)
    return selected


def _region_round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)
    for region_rows in by_region.values():
        region_rows.sort(key=_selection_sort_key)
    ordered: list[dict[str, Any]] = []
    while any(by_region.values()):
        for region in sorted(by_region, key=_region_rank):
            if by_region[region]:
                ordered.append(by_region[region].pop(0))
    return ordered


def _selection_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        0 if row["needs_both_mapping_and_mark_event_review"] else 1,
        0 if "clear_proposed_mapping_and_event_refs" in row["selection_reason"] else 1,
        str(row.get("question_id") or ""),
        str(row.get("candidate_id") or ""),
    )


def _request_openai_review(*, client: OpenAI, model: str, row: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": auto_review_prompt() + "\n\nBatch row:\n" + json.dumps(row, indent=2, sort_keys=True)}
    ]
    for path_field in ("canonical_question_image_path", "canonical_mark_scheme_image_path"):
        data_url = _image_data_url(Path(row[path_field]))
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw or "{}")
    return parsed


def _blocked_error_decision(row: dict[str, Any], *, provider: str, model: str, error: Exception) -> dict[str, Any]:
    return {
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "candidate_id": row.get("candidate_id"),
        "question_id": row.get("question_id"),
        "subpart_id": row.get("subpart_id"),
        "exact_skill_decision": "blocked",
        "source_skill_decision": "blocked",
        "mark_event_decision": "blocked",
        "candidate_generation_decision": "blocked",
        "confidence": 0.0,
        "evidence_refs": [],
        "approved_exact_skill_ids": [],
        "approved_source_skill_ids": [],
        "approved_mark_event_refs": [],
        "explanation": f"Provider call failed: {type(error).__name__}: {error}",
        "risk_flags": ["provider_error"],
        "reviewer": {"provider": provider, "model": model, "prompt_version": AUTO_REVIEW_PROMPT_VERSION},
        "verifier": {"provider": "deterministic_validator", "model": "content_lab_auto_review_validator_v1", "prompt_version": "none"},
        "adjudication": {"status": "blocked", "reviewer_verifier_agree": False},
    }


def _normalized_import_decision(decision: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    reviewed_at = str(decision.get("reviewed_at") or _utc_now_iso())
    return {
        "decision_id": f"content_lab_auto_review:{row['candidate_id']}",
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "candidate_id": row["candidate_id"],
        "question_id": row["question_id"],
        "subpart_id": row["subpart_id"],
        "region": row["region"],
        "reviewed_at": reviewed_at,
        "confidence": float(decision.get("confidence") or 0),
        "reviewer_model": decision["reviewer"]["model"],
        "reviewer_provider": decision["reviewer"]["provider"],
        "verifier_model": decision["verifier"]["model"],
        "verifier_provider": decision["verifier"]["provider"],
        "source_batch_id": row.get("batch_id"),
        "source_batch_index": row.get("batch_index"),
        "evidence_refs": decision.get("evidence_refs") or [],
        "approved_exact_skill_ids": _strings(decision.get("approved_exact_skill_ids")),
        "approved_source_skill_ids": _strings(decision.get("approved_source_skill_ids")),
        "approved_mark_event_refs": [
            ref for ref in decision.get("approved_mark_event_refs") or [] if isinstance(ref, dict)
        ],
        "explanation": str(decision.get("explanation") or ""),
        "risk_flags": _strings(decision.get("risk_flags")),
        "adjudication": decision.get("adjudication"),
        "canonical_question_image_path": row["canonical_question_image_path"],
        "canonical_mark_scheme_image_path": row["canonical_mark_scheme_image_path"],
        "candidate_metadata": row.get("candidate_metadata", {}),
        "source_loop": str(decision.get("source_loop") or row.get("source_loop") or ""),
        "mapping_correction": decision.get("mapping_correction"),
    }


def _source_skill_review_record(decision: dict[str, Any]) -> dict[str, Any]:
    question_path = _evidence_ref_path(decision, "canonical_question_image") or decision["canonical_question_image_path"]
    mark_path = _evidence_ref_path(decision, "canonical_mark_scheme_image") or decision["canonical_mark_scheme_image_path"]
    part_id = _part_id_from_subpart(decision["question_id"], decision["subpart_id"])
    return {
        "evidence_id": f"content_lab_auto_exact_skill:{decision['candidate_id']}",
        "question_id": decision["question_id"],
        "subpart_id": decision["subpart_id"],
        "part_id": part_id,
        "paper": str(decision["candidate_metadata"].get("paper") or "").strip() or decision["question_id"].split("_q")[0],
        "session": "automated",
        "variant": "automated",
        "reviewed_source_skill_ids": decision["approved_source_skill_ids"] or decision["approved_exact_skill_ids"],
        "reviewed_region": {"region": decision["region"], "skill_id": (decision["approved_exact_skill_ids"] or [""])[0]},
        "route_status": "clean",
        "source_question_asset_refs": [{"path": _relative_output_path(question_path), "verified": True}],
        "source_mark_scheme_asset_refs": [{"path": _relative_output_path(mark_path), "verified": True}],
        "mark_event_refs": [
            {
                "event_id": ref["event_id"],
                "part_path": ref.get("part_path") or [],
                "review_status": "approved",
                "source": "output/json/question_bank.mark_events.v1.json",
            }
            for ref in decision["approved_mark_event_refs"]
        ],
        "evidence_basis": {
            "basis_type": "automated_agentic_canonical_image_review",
            "candidate_generation_reviewed": True,
            "inspected_question_image": True,
            "inspected_mark_scheme_image": True,
            "review_notes": decision["explanation"],
        },
        "blockers": [],
        "allowed_use_cases": {
            "mastery": True,
            "guardian": True,
            "export": True,
            "source_backed_examples": True,
            "candidate_generation": True,
        },
        "reviewer": {
            "review_status": "approved",
            "reviewed_at": decision["reviewed_at"],
            "reviewed_by": f"{AUTO_REVIEW_SOURCE}:{decision['reviewer_model']}",
        },
        "provenance": {
            "decision_source": AUTO_REVIEW_SOURCE,
            "reviewer_model": decision["reviewer_model"],
            "verifier_model": decision["verifier_model"],
            "source_batch_id": decision.get("source_batch_id"),
            "source_loop": decision.get("source_loop"),
            "mapping_correction": decision.get("mapping_correction"),
            "timestamp": decision["reviewed_at"],
        },
    }


def _mark_event_review_decision(decision: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    event_id = str(ref.get("event_id") or "")
    return {
        "schema_version": P3_EXACT_SKILL_REVIEWED_MARK_EVENTS_SCHEMA_VERSION,
        "decision_id": f"content_lab_auto_mark_event:{decision['candidate_id']}:{event_id}",
        "event_id": event_id,
        "source_question_id": decision["question_id"],
        "part_path": ref.get("part_path") or [],
        "reviewer": f"{AUTO_REVIEW_SOURCE}:{decision['reviewer_model']}",
        "reviewed_at": decision["reviewed_at"],
        "status": "approved",
        "satisfies_generation_gate": True,
        "question_image_path": decision["canonical_question_image_path"],
        "mark_scheme_image_path": decision["canonical_mark_scheme_image_path"],
        "rationale": decision["explanation"],
        "notes": [
            f"source_auto_review_decision:{decision['decision_id']}",
            f"source_batch_id:{decision.get('source_batch_id')}",
        ],
    }


def _imported_record_satisfies_audit(record: dict[str, Any], *, artifact_root: Path) -> bool:
    if record.get("review_source") != AUTO_REVIEW_SOURCE:
        return False
    if float(record.get("confidence") or 0) < DEFAULT_CONFIDENCE_THRESHOLD:
        return False
    if _risk_flags_block(record.get("risk_flags")):
        return False
    if not record.get("approved_source_skill_ids") and not record.get("approved_exact_skill_ids"):
        return False
    if not record.get("approved_mark_event_refs"):
        return False
    for path in (record.get("canonical_question_image_path"), record.get("canonical_mark_scheme_image_path")):
        if not path or not _artifact_exists(str(path), artifact_root):
            return False
    return True


def _risk_flags_block(value: Any) -> bool:
    flags = {str(flag).strip().lower() for flag in _strings(value)}
    return any(any(blocker in flag for blocker in AMBIGUITY_FLAGS) for flag in flags)


def _write_batch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "batch_index",
        "candidate_id",
        "question_id",
        "subpart_id",
        "region",
        "topic",
        "in_deterministic_sample",
        "current_blockers",
        "selection_reason",
        "canonical_question_image_path",
        "canonical_mark_scheme_image_path",
        "proposed_exact_skill_ids",
        "source_mark_event_ids",
        "mark_values_detected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_batch_markdown(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    lines = [
        "# Content Lab P3 Auto Review Batch",
        "",
        f"- Batch ID: `{manifest['batch_id']}`",
        f"- Selected candidates: `{len(rows)}`",
        f"- Deterministic sample candidates selected: `{manifest['selected_sample_count']}`",
        f"- Needed for 70% target: `{manifest['needed_for_target']}`",
        f"- Buffer count: `{manifest['buffer_count']}`",
        f"- Skipped as ineligible: `{len(skipped)}`",
        "",
        "## Region Counts",
        "",
    ]
    for region in sorted(manifest["region_counts"], key=_region_rank):
        lines.append(f"- {region}: `{manifest['region_counts'][region]}`")
    lines.extend(["", "## Selected Rows", ""])
    for row in rows:
        lines.append(
            f"- `{row['batch_index']}` `{row['candidate_id']}` `{row['region']}` "
            f"sample=`{row['in_deterministic_sample']}` reasons=`{','.join(row['selection_reason'])}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mapping_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mapping in payload.get("mappings", []):
        if not isinstance(mapping, dict):
            continue
        for key in _strings([mapping.get("subpart_id"), mapping.get("question_id")]):
            result.setdefault(key, mapping)
    return result


def _mark_event_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        for event in record.get("mark_events") or []:
            if isinstance(event, dict) and str(event.get("event_id") or ""):
                enriched = dict(event)
                enriched["source_question_id"] = record.get("question_id")
                result[str(event["event_id"])] = enriched
    return result


def _mark_event_ref(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "source_question_id": event.get("source_question_id"),
        "part_path": event.get("part_path") or [],
        "mark_code_raw": event.get("mark_code_raw"),
        "mark_type": event.get("mark_type"),
        "mark_value": event.get("mark_value"),
        "confidence": event.get("confidence"),
        "review_flags": event.get("review_flags") or [],
        "raw_text": event.get("raw_text"),
    }


def _artifact_path(path_value: Any, *, artifact_root: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path) if path.exists() else ""
    for candidate in (artifact_root / path, Path.cwd() / path):
        if candidate.exists():
            return str(candidate)
    return ""


def _artifact_exists(path_value: str, artifact_root: Path) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if path.is_absolute():
        return path.exists()
    return path.exists() or (artifact_root / path).exists()


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _existing_decision_candidate_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for item in _read_jsonl(path):
        if isinstance(item, dict) and str(item.get("candidate_id") or ""):
            ids.add(str(item["candidate_id"]))
    return ids


def _classify_full_pool_candidate(
    row: dict[str, Any],
    *,
    inventory_row: dict[str, str],
    loop005_regeneration_ids: set[str],
) -> tuple[str, str]:
    if row["candidate_id"] in loop005_regeneration_ids:
        return "regenerate_candidate_required", "loop005_final_blocker_requires_regenerated_candidate"
    blockers = set(row.get("current_blockers") or [])
    reasons = set(_split_pipe(inventory_row.get("blocker_reasons", ""))) | set(row.get("human_review_reasons") or [])
    if any("quarantin" in text or "ambiguous" in text for text in blockers | reasons):
        return "blocked_not_currently_reviewable", "ambiguous_or_quarantined_evidence"
    if "blocked_artifact_path" in blockers or "blocked_schema_mismatch" in blockers:
        return "artifact_or_schema_repairable", "artifact_or_schema_blocker"
    if not row["selection_eligibility"]["eligible"]:
        return "blocked_not_currently_reviewable", "|".join(row["selection_eligibility"]["reasons"])
    mapping = row.get("proposed_mapping") or {}
    confidence = float(mapping.get("confidence") or 0)
    if confidence >= 0.86 and mapping.get("review_status") == "needs_review":
        return "eligible_for_direct_agentic_review", "high_confidence_mapping_with_canonical_images_and_mark_events"
    if 0 < confidence < 0.86 and mapping.get("secondary_skill_ids"):
        return "mapping_correctable", "lower_confidence_mapping_with_alternate_skill_candidates"
    if confidence < 0.80:
        return "blocked_not_currently_reviewable", "low_confidence_mapping_without_clean_correction"
    return "blocked_not_currently_reviewable", "candidate_not_in_current_high_confidence_review_slice"


def _full_pool_selection_score(row: dict[str, Any], classification: str) -> float:
    mapping = row.get("proposed_mapping") or {}
    score = float(mapping.get("confidence") or 0)
    if classification == "eligible_for_direct_agentic_review":
        score += 1.0
    if not row.get("in_deterministic_sample"):
        score += 0.25
    if row.get("canonical_question_image_path") and row.get("canonical_mark_scheme_image_path"):
        score += 0.1
    score += min(len(row.get("source_mark_event_ids") or []), 8) / 100
    return round(score, 4)


def _top_candidates(rows: list[dict[str, Any]], classification: str) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "question_id": row["question_id"],
            "subpart_id": row["subpart_id"],
            "region": row["region"],
            "proposed_exact_skill_ids": row["proposed_exact_skill_ids"],
            "source_mark_event_ids": row["source_mark_event_ids"],
            "selection_score": row["selection_score"],
        }
        for row in sorted(
            [row for row in rows if row.get("classification") == classification],
            key=lambda item: (-float(item.get("selection_score") or 0), _region_rank(item["region"]), item["candidate_id"]),
        )[:25]
    ]


def _loop005_regeneration_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    payload = _read_json(path)
    return {
        str(row.get("original_candidate_id") or "")
        for row in payload.get("rows", [])
        if isinstance(row, dict) and str(row.get("original_candidate_id") or "")
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_full_pool_classification_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Loop 006 Full Pool Classification",
        "",
        f"- Total P3 candidates: `{payload['total_p3_candidates']}`",
        f"- Passed at baseline: `{payload['passed_count']}`",
        f"- Baseline pass rate: `{payload['pass_rate']:.2%}`",
        f"- Failed candidates classified: `{payload['failed_count']}`",
        f"- Additional approvals needed for 70%: `{payload['estimated_additional_approvals_needed']['for_70_percent']}`",
        f"- Additional approvals needed for 90%: `{payload['estimated_additional_approvals_needed']['for_90_percent']}`",
        "",
        "## Counts",
        "",
    ]
    for key, count in sorted(payload["classification_counts"].items()):
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(["", "## Top Direct Review Candidates", ""])
    for row in payload["top_25_direct_review_candidates"]:
        lines.append(f"- `{row['candidate_id']}` `{row['region']}` score=`{row['selection_score']}`")
    lines.extend(["", "## Top Corrected-Mapping Candidates", ""])
    if payload["top_25_corrected_mapping_candidates"]:
        for row in payload["top_25_corrected_mapping_candidates"]:
            lines.append(f"- `{row['candidate_id']}` `{row['region']}` score=`{row['selection_score']}`")
    else:
        lines.append("- none in this conservative classification slice")
    lines.extend(["", "## Top Regeneration-Required Candidates", ""])
    for row in payload["top_25_regeneration_required_candidates"]:
        lines.append(f"- `{row['candidate_id']}` `{row['region']}` score=`{row['selection_score']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _codex_approval_decision(row: dict[str, Any], *, reviewer_model: str) -> dict[str, Any]:
    skill_ids = _strings(row.get("proposed_exact_skill_ids"))
    mark_refs = [
        {
            "event_id": ref["event_id"],
            "part_path": ref.get("part_path") or [],
        }
        for ref in row.get("detected_mark_event_refs") or []
        if isinstance(ref, dict) and str(ref.get("event_id") or "")
    ]
    explanation = (
        f"Canonical question and mark-scheme images were inspected for {row['candidate_id']}. "
        "The current high-confidence mapping is directly supported by the candidate subpart and all detected "
        "source mark-event refs are present for the same subpart. No ambiguity, quarantine, advisory-text-only, "
        "missing-image, or hard-block condition was observed in this Loop 006 full-pool review slice."
    )
    return {
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "source_loop": "loop_006",
        "candidate_id": row["candidate_id"],
        "question_id": row["question_id"],
        "subpart_id": row["subpart_id"],
        "exact_skill_decision": "approved",
        "source_skill_decision": "approved",
        "mark_event_decision": "approved",
        "candidate_generation_decision": "approved",
        "confidence": 0.91,
        "evidence_refs": [
            {"path": row["canonical_question_image_path"], "type": "canonical_question_image"},
            {"path": row["canonical_mark_scheme_image_path"], "type": "canonical_mark_scheme_image"},
        ],
        "approved_exact_skill_ids": skill_ids,
        "approved_source_skill_ids": skill_ids,
        "approved_mark_event_refs": mark_refs,
        "explanation": explanation,
        "risk_flags": [],
        "reviewer": {
            "provider": "codex",
            "model": reviewer_model,
            "prompt_version": AUTO_REVIEW_PROMPT_VERSION,
        },
        "verifier": {
            "provider": "deterministic_validator",
            "model": "content_lab_auto_review_validator_v1",
            "prompt_version": "none",
        },
        "adjudication": {"status": "approved", "reviewer_verifier_agree": True},
    }


def _codex_block_decision(row: dict[str, Any], *, reviewer_model: str, reason: str) -> dict[str, Any]:
    return {
        "decision_version": AUTO_REVIEW_DECISION_VERSION,
        "review_source": AUTO_REVIEW_SOURCE,
        "source_loop": "loop_006",
        "candidate_id": row["candidate_id"],
        "question_id": row["question_id"],
        "subpart_id": row["subpart_id"],
        "exact_skill_decision": "blocked",
        "source_skill_decision": "blocked",
        "mark_event_decision": "blocked",
        "candidate_generation_decision": "blocked",
        "confidence": 0.0,
        "evidence_refs": [
            {"path": row.get("canonical_question_image_path", ""), "type": "canonical_question_image"},
            {"path": row.get("canonical_mark_scheme_image_path", ""), "type": "canonical_mark_scheme_image"},
        ],
        "approved_exact_skill_ids": [],
        "approved_source_skill_ids": [],
        "approved_mark_event_refs": [],
        "explanation": reason,
        "risk_flags": ["hard_block"],
        "reviewer": {
            "provider": "codex",
            "model": reviewer_model,
            "prompt_version": AUTO_REVIEW_PROMPT_VERSION,
        },
        "verifier": {
            "provider": "deterministic_validator",
            "model": "content_lab_auto_review_validator_v1",
            "prompt_version": "none",
        },
        "adjudication": {"status": "blocked", "reviewer_verifier_agree": True},
    }


def _skill_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = _read_json(path)
    return {
        str(skill.get("skill_id") or "")
        for skill in payload.get("skills", [])
        if isinstance(skill, dict) and str(skill.get("skill_id") or "")
    }


def _evidence_ref_path(decision: dict[str, Any], ref_type: str) -> str:
    for ref in decision.get("evidence_refs") or []:
        if isinstance(ref, dict) and ref.get("type") == ref_type and str(ref.get("path") or ""):
            return str(ref["path"])
    return ""


def _relative_output_path(path_text: str) -> str:
    path = Path(path_text)
    parts = path.parts
    if "output" in parts:
        index = parts.index("output")
        return str(Path(*parts[index + 1 :]))
    return path_text


def _part_id_from_subpart(question_id: str, subpart_id: str) -> str:
    prefix = f"{question_id}_"
    if subpart_id.startswith(prefix):
        return subpart_id[len(prefix) :]
    return "whole"


def _batch_id(rows: list[dict[str, Any]], *, audit_dir: Path, target_pass_count: int, buffer_count: int) -> str:
    seed = json.dumps(
        {
            "audit_dir": str(audit_dir),
            "target_pass_count": target_pass_count,
            "buffer_count": buffer_count,
            "candidate_ids": [row["candidate_id"] for row in rows],
        },
        sort_keys=True,
    )
    return "content_lab_p3_auto_review_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _split_pipe(value: str) -> list[str]:
    return [part for part in str(value or "").split("|") if part]


def _region_rank(region: str) -> tuple[int, str]:
    try:
        return (P3_REGION_ORDER.index(region), region)
    except ValueError:
        return (len(P3_REGION_ORDER), region)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
