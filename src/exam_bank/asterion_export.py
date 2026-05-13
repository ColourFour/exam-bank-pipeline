from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .output_layout import default_asterion_export_path


ASTERION_SCHEMA_NAME = "asterion.question_bank"
ASTERION_SCHEMA_VERSION = 1
ASTERION_EXPORT_FILENAME = "asterion_question_bank_v1.json"
CONTENT_LAB_SCHEMA_NAME = "asterion.content_lab_candidates"
CONTENT_LAB_SCHEMA_VERSION = 1
CONTENT_LAB_EXPORT_FILENAME = "asterion_content_lab_candidates_v1.json"
MARK_EVENT_REVIEW_STATUS = "machine_candidate"
MARK_EVENT_QUARANTINED_STATUS = "quarantined"
MARK_EVENT_TOTAL_DISAGREEMENT = "question_mark_scheme_total_disagreement"
CONTENT_LAB_BLOCKED_UNTIL_REVIEWED = "blocked_until_reviewed"

_MARK_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<dependent>D\s*)?(?P<kind>[MABCE])\s*(?P<value>[1-9]\d?)(?P<suffix>\s*(?:ft|dep))?\b",
    re.IGNORECASE,
)


def export_asterion_question_bank(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    artifact_root: str | Path | None = None,
    base_dir: str | Path | None = None,
    skill_map_path: str | Path | None = None,
) -> Path:
    input_path = Path(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    root = Path(artifact_root) if artifact_root is not None else infer_artifact_root(input_path)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    output = Path(output_path) if output_path is not None else default_asterion_export_path(input_path, ASTERION_EXPORT_FILENAME)
    skill_mappings = load_skill_mappings(skill_map_path) if skill_map_path else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_asterion_export(payload, artifact_root=root, base_dir=base, skill_mappings=skill_mappings),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output


def export_asterion_content_lab_candidates(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    artifact_root: str | Path | None = None,
    base_dir: str | Path | None = None,
    skill_map_path: str | Path | None = None,
) -> Path:
    input_path = Path(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    root = Path(artifact_root) if artifact_root is not None else infer_artifact_root(input_path)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    output = Path(output_path) if output_path is not None else default_asterion_export_path(input_path, CONTENT_LAB_EXPORT_FILENAME)
    skill_mappings = load_skill_mappings(skill_map_path) if skill_map_path else None
    asterion_payload = _ensure_asterion_payload(payload, artifact_root=root, base_dir=base, skill_mappings=skill_mappings)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_content_lab_candidates(asterion_payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output


def build_asterion_export(
    question_bank: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
    base_dir: str | Path | None = None,
    skill_mappings: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if question_bank.get("schema_name") != "exam_bank.question_bank":
        raise ValueError("Asterion export requires exam_bank.question_bank input")
    if int(question_bank.get("schema_version") or 0) != 2:
        raise ValueError("Asterion export requires exam_bank.question_bank schema_version 2")

    root = Path(artifact_root) if artifact_root is not None else None
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    skills = skill_mappings or {}
    records = [
        build_asterion_record(record, artifact_root=root, base_dir=base, skill_mappings=skills)
        for record in question_bank.get("questions", [])
    ]
    return {
        "schema_name": ASTERION_SCHEMA_NAME,
        "schema_version": ASTERION_SCHEMA_VERSION,
        "source_schema": {
            "schema_name": question_bank.get("schema_name"),
            "schema_version": question_bank.get("schema_version"),
            "record_count": question_bank.get("record_count"),
        },
        "record_count": len(records),
        "questions": records,
    }


def build_content_lab_candidates(asterion_question_bank: dict[str, Any]) -> dict[str, Any]:
    if asterion_question_bank.get("schema_name") != ASTERION_SCHEMA_NAME:
        raise ValueError("Content Lab candidates require asterion.question_bank input")
    if int(asterion_question_bank.get("schema_version") or 0) != ASTERION_SCHEMA_VERSION:
        raise ValueError("Content Lab candidates require asterion.question_bank schema_version 1")

    candidates: list[dict[str, Any]] = []
    for record in asterion_question_bank.get("questions", []):
        if not isinstance(record, dict):
            continue
        for subpart in record.get("subparts", []):
            if not isinstance(subpart, dict):
                continue
            if not _content_lab_candidate_subpart(record, subpart):
                continue
            candidates.append(_content_lab_candidate(record, subpart))

    return {
        "schema_name": CONTENT_LAB_SCHEMA_NAME,
        "schema_version": CONTENT_LAB_SCHEMA_VERSION,
        "source_schema": {
            "schema_name": asterion_question_bank.get("schema_name"),
            "schema_version": asterion_question_bank.get("schema_version"),
            "record_count": asterion_question_bank.get("record_count"),
        },
        "policy": {
            "no_student_facing_generated_content": True,
            "emits_candidates_and_metadata_only": True,
            "content_lab_generation_requires_reviewed_or_approved_mapping": True,
            "content_lab_generation_requires_reviewed_or_approved_mark_events": True,
        },
        "record_count": len(candidates),
        "candidates": candidates,
    }


def build_asterion_record(
    record: dict[str, Any],
    *,
    artifact_root: Path | None,
    base_dir: Path,
    skill_mappings: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    question_images = _image_integrity(_question_image_paths(record), artifact_root=artifact_root, base_dir=base_dir)
    mark_scheme_images = _image_integrity(_mark_scheme_image_paths(record), artifact_root=artifact_root, base_dir=base_dir)
    canonical_question = _first_path(question_images)
    canonical_mark_scheme = _first_path(mark_scheme_images)
    source_pdf = _source_pdf_metadata(record, base_dir=base_dir)
    quality_gate = _quality_gate(record, question_images, mark_scheme_images)
    total_marks = _total_marks(record)
    question_id = str(_get(record, "question_id") or "")
    subparts = _subpart_records(
        record,
        question_id=question_id,
        total_marks=total_marks,
        question_crop_path=canonical_question,
        mark_scheme_crop_path=canonical_mark_scheme,
        text_only_display_allowed=bool(quality_gate["text_only_display_allowed"]),
        mark_scheme_crop_ok=bool(quality_gate["mark_scheme_crop_ok"]),
        skill_mappings=skill_mappings or {},
    )

    return {
        "question_id": question_id,
        "paper": _get(record, "paper"),
        "paper_family": _get(record, "paper_family"),
        "question_number": _get(record, "question_number"),
        "total_marks": total_marks,
        "source_pdf": source_pdf,
        "canonical_question_artifact": canonical_question,
        "canonical_mark_scheme_artifact": canonical_mark_scheme,
        "artifact_integrity": {
            "question_images": question_images,
            "mark_scheme_images": mark_scheme_images,
            "source_pdf": source_pdf,
        },
        "quality_gate": quality_gate,
        "subparts": subparts,
        "usage_roles": _usage_roles(record, quality_gate),
    }


def infer_artifact_root(input_path: str | Path) -> Path | None:
    path = Path(input_path)
    if path.parent.name == "json":
        return path.parent.parent
    return path.parent


def load_skill_mappings(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        return {}

    by_subpart: dict[str, list[str]] = {}
    for item in mappings:
        if not isinstance(item, dict):
            continue
        subpart_id = str(item.get("subpart_id") or "")
        if not subpart_id:
            continue
        skill_ids = _dedupe(
            _list_string_values(item.get("skill_ids"))
            + _list_string_values(item.get("primary_skill_ids"))
            + _list_string_values(item.get("secondary_skill_ids"))
            + _list_string_values(item.get("prerequisite_skill_ids"))
        )
        if skill_ids:
            by_subpart[subpart_id] = skill_ids
    return by_subpart


def _ensure_asterion_payload(
    payload: dict[str, Any],
    *,
    artifact_root: Path | None,
    base_dir: Path,
    skill_mappings: dict[str, list[str]] | None,
) -> dict[str, Any]:
    schema_name = payload.get("schema_name")
    if schema_name == ASTERION_SCHEMA_NAME:
        return payload
    if schema_name == "exam_bank.question_bank":
        return build_asterion_export(payload, artifact_root=artifact_root, base_dir=base_dir, skill_mappings=skill_mappings)
    raise ValueError("Content Lab candidates require exam_bank.question_bank or asterion.question_bank input")


def _quality_gate(
    record: dict[str, Any],
    question_images: list[dict[str, Any]],
    mark_scheme_images: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_assets_ok = bool(question_images and mark_scheme_images) and all(
        bool(item.get("sha256")) for item in [question_images[0], mark_scheme_images[0]]
    )
    question_crop_ok = str(_get(record, "question_crop_confidence") or "").lower() == "high" and str(
        _get(record, "scope_quality_status") or ""
    ).lower() != "fail"
    mark_scheme_crop_ok = str(_get(record, "mark_scheme_crop_confidence") or "").lower() == "high"
    marks_consistent = _marks_consistent(record)
    paper_total_consistent = str(_get(record, "paper_total_status") or "").lower() in {"matched", "recovered_after_rescan"}
    text_only_display_allowed = _text_only_display_allowed(record)
    visual_required = _as_bool(_get(record, "visual_required"))
    content_lab_generation_allowed = _content_lab_generation_allowed(
        record,
        canonical_assets_ok=canonical_assets_ok,
        marks_consistent=marks_consistent,
    )
    reason_codes = _reason_codes(
        record,
        canonical_assets_ok=canonical_assets_ok,
        question_images=question_images,
        mark_scheme_images=mark_scheme_images,
        question_crop_ok=question_crop_ok,
        mark_scheme_crop_ok=mark_scheme_crop_ok,
        marks_consistent=marks_consistent,
        paper_total_consistent=paper_total_consistent,
        text_only_display_allowed=text_only_display_allowed,
        content_lab_generation_allowed=content_lab_generation_allowed,
    )
    return {
        "canonical_assets_ok": canonical_assets_ok,
        "question_crop_ok": question_crop_ok,
        "mark_scheme_crop_ok": mark_scheme_crop_ok,
        "marks_consistent": marks_consistent,
        "paper_total_consistent": paper_total_consistent,
        "text_only_display_allowed": text_only_display_allowed,
        "visual_required": visual_required,
        "content_lab_generation_allowed": content_lab_generation_allowed,
        "reason_codes": reason_codes,
    }


def _usage_roles(record: dict[str, Any], gate: dict[str, Any]) -> dict[str, str]:
    hard_blocked = _hard_blocked(record, gate)
    review_needed = _review_needed(record, gate)
    content_review_needed = review_needed or not gate["content_lab_generation_allowed"]
    return {
        "canonical_practice": "allow" if not hard_blocked and not review_needed else "block",
        "field_guide_source": _tri_state_role(hard_blocked=hard_blocked, review_needed=review_needed),
        "quick_check_source": _tri_state_role(hard_blocked=hard_blocked, review_needed=review_needed or not gate["text_only_display_allowed"]),
        "warmup_generator_source": _tri_state_role(hard_blocked=hard_blocked, review_needed=content_review_needed),
        "guardian_candidate": "allow" if not hard_blocked and not review_needed else "block",
        "p3_readiness_metric": "include" if str(_get(record, "paper_family") or "").lower() == "p3" else "exclude",
    }


def _hard_blocked(record: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        not gate["canonical_assets_ok"]
        or not gate["marks_consistent"]
        or str(_get(record, "mapping_status") or "").lower() == "fail"
        or str(_get(record, "validation_status") or "").lower() == "fail"
        or str(_get(record, "scope_quality_status") or "").lower() == "fail"
    )


def _review_needed(record: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        not gate["question_crop_ok"]
        or not gate["mark_scheme_crop_ok"]
        or not gate["paper_total_consistent"]
        or str(_get(record, "visual_curation_status") or "").lower() != "ready"
        or str(_get(record, "mapping_status") or "").lower() != "pass"
        or str(_get(record, "validation_status") or "").lower() != "pass"
    )


def _tri_state_role(*, hard_blocked: bool, review_needed: bool) -> str:
    if hard_blocked:
        return "block"
    if review_needed:
        return "block_until_reviewed"
    return "allow"


def _text_only_display_allowed(record: dict[str, Any]) -> bool:
    return (
        str(_get(record, "text_only_status") or "").lower() == "ready"
        and str(_get(record, "question_text_role") or "").lower() != "untrusted_math_text"
        and str(_get(record, "question_text_trust") or "").lower() == "high"
        and not _as_bool(_get(record, "visual_required"))
    )


def _content_lab_generation_allowed(
    record: dict[str, Any],
    *,
    canonical_assets_ok: bool,
    marks_consistent: bool,
) -> bool:
    topic_confidence = str(_get(record, "topic_confidence") or "").lower()
    return (
        canonical_assets_ok
        and marks_consistent
        and topic_confidence not in {"", "low"}
        and not _as_bool(_get(record, "topic_uncertain"))
        and str(_get(record, "mapping_status") or "").lower() == "pass"
        and str(_get(record, "validation_status") or "").lower() == "pass"
    )


def _reason_codes(
    record: dict[str, Any],
    *,
    canonical_assets_ok: bool,
    question_images: list[dict[str, Any]],
    mark_scheme_images: list[dict[str, Any]],
    question_crop_ok: bool,
    mark_scheme_crop_ok: bool,
    marks_consistent: bool,
    paper_total_consistent: bool,
    text_only_display_allowed: bool,
    content_lab_generation_allowed: bool,
) -> list[str]:
    reasons: set[str] = set()
    if not canonical_assets_ok:
        reasons.add("canonical_assets_missing_or_unhashed")
    if not question_images:
        reasons.add("missing_question_image_path")
    elif not question_images[0].get("sha256"):
        reasons.add("missing_question_image_file")
    if not mark_scheme_images:
        reasons.add("missing_mark_scheme_image_path")
    elif not mark_scheme_images[0].get("sha256"):
        reasons.add("missing_mark_scheme_image_file")
    if not question_crop_ok:
        reasons.add("question_crop_not_high_confidence")
    if not mark_scheme_crop_ok:
        reasons.add("mark_scheme_crop_not_high_confidence")
    if not marks_consistent:
        reasons.add("marks_inconsistent")
    if not paper_total_consistent:
        reasons.add("paper_total_not_consistent")

    mapping_status = str(_get(record, "mapping_status") or "").lower()
    validation_status = str(_get(record, "validation_status") or "").lower()
    scope_status = str(_get(record, "scope_quality_status") or "").lower()
    if mapping_status and mapping_status != "pass":
        reasons.add(f"mapping_status_{mapping_status}")
    if validation_status and validation_status != "pass":
        reasons.add(f"validation_status_{validation_status}")
    if scope_status == "fail":
        reasons.add("scope_quality_status_fail")

    if not text_only_display_allowed:
        text_status = str(_get(record, "text_only_status") or "").lower()
        text_role = str(_get(record, "question_text_role") or "").lower()
        text_trust = str(_get(record, "question_text_trust") or "").lower()
        if text_role == "untrusted_math_text":
            reasons.add("text_only_blocked_untrusted_math_text")
        if text_status:
            reasons.add(f"text_only_blocked_status_{text_status}")
        if text_trust and text_trust != "high":
            reasons.add(f"text_only_blocked_trust_{text_trust}")
        if _as_bool(_get(record, "visual_required")):
            reasons.add("text_only_blocked_visual_required")

    if not content_lab_generation_allowed:
        topic_confidence = str(_get(record, "topic_confidence") or "").lower()
        if topic_confidence in {"", "low"}:
            reasons.add("content_lab_blocked_topic_confidence_low")
        if _as_bool(_get(record, "topic_uncertain")):
            reasons.add("content_lab_blocked_topic_uncertain")

    if _subpart_marks_missing(record):
        reasons.add("subpart_marks_missing")
    return sorted(reasons)


def _subpart_records(
    record: dict[str, Any],
    *,
    question_id: str,
    total_marks: int | None,
    question_crop_path: str,
    mark_scheme_crop_path: str,
    text_only_display_allowed: bool,
    mark_scheme_crop_ok: bool,
    skill_mappings: dict[str, list[str]],
) -> list[dict[str, Any]]:
    labels = _subpart_labels(record)
    if not labels:
        labels = ["whole"]
    marks_by_label = _subpart_marks(record, labels=labels, total_marks=total_marks)
    question_texts = _split_labeled_text(str(_get(record, "question_text") or ""), labels)
    mark_scheme_texts = _split_labeled_text(str(_get(record, "mark_scheme_text") or ""), labels)
    detected_values = _detected_mark_values(record)
    review_status = _subpart_review_status(record, marks_by_label=marks_by_label, mark_scheme_crop_ok=mark_scheme_crop_ok)
    quarantine_reason = _mark_event_quarantine_reason(record)

    subparts = []
    for index, label in enumerate(labels):
        marks = marks_by_label.get(label)
        subpart_id = f"{question_id}_{_subpart_id_suffix(label)}"
        skill_ids = _skill_ids_for_subpart(record, subpart_id=subpart_id, skill_mappings=skill_mappings)
        mark_scheme_text = mark_scheme_texts.get(label)
        subparts.append(
            {
                "subpart_id": subpart_id,
                "label": label,
                "marks": marks,
                "question_crop_path": question_crop_path or None,
                "mark_scheme_crop_path": mark_scheme_crop_path or None,
                "question_text": {
                    "text": question_texts.get(label),
                    "trust_level": _get(record, "question_text_trust"),
                    "role": _get(record, "question_text_role"),
                    "text_only_display_allowed": text_only_display_allowed,
                },
                "mark_scheme_text": {
                    "text": mark_scheme_text,
                    "trust_level": _mark_scheme_text_trust(record, mark_scheme_crop_ok=mark_scheme_crop_ok),
                },
                "mark_events": _mark_events_from_text(
                    mark_scheme_text,
                    subpart_id=subpart_id,
                    skill_ids=skill_ids,
                    quarantine_reason=quarantine_reason,
                ),
                "detected_mark_values": _detected_values_for_subpart(index, label, labels, marks, detected_values),
                "review_status": review_status,
            }
        )
    return subparts


def _subpart_review_status(
    record: dict[str, Any],
    *,
    marks_by_label: dict[str, int | None],
    mark_scheme_crop_ok: bool,
) -> str:
    if str(_get(record, "mapping_status") or "").lower() == "fail" or str(_get(record, "validation_status") or "").lower() == "fail":
        return "blocked"
    if any(value is None for value in marks_by_label.values()):
        return "review"
    if not mark_scheme_crop_ok or str(_get(record, "visual_curation_status") or "").lower() != "ready":
        return "review"
    if str(_get(record, "question_crop_confidence") or "").lower() != "high":
        return "review"
    return "ready"


def _content_lab_candidate_subpart(record: dict[str, Any], subpart: dict[str, Any]) -> bool:
    if _subpart_reviewed_or_approved(subpart):
        return True
    events = _mark_events(subpart)
    if not events:
        return False
    if _content_lab_hard_blocked(record):
        return False
    if any(_mark_event_quarantined(event) for event in events):
        return False
    return _minimum_mark_event_confidence(events) >= 0.84


def _content_lab_candidate(record: dict[str, Any], subpart: dict[str, Any]) -> dict[str, Any]:
    subpart_id = str(subpart.get("subpart_id") or "")
    events = _mark_events(subpart)
    skill_ids = _dedupe(
        [
            skill_id
            for event in events
            for skill_id in _list_string_values(event.get("skill_ids"))
        ]
    )
    if not skill_ids:
        skill_ids = _list_string_values(subpart.get("skill_ids"))

    role_statuses = _content_lab_role_statuses(record, subpart, events, skill_ids)
    possible_roles = [role for role, status in role_statuses.items() if status != "block"]
    generation_gate = _content_lab_generation_gate(record, subpart, events, skill_ids)
    warmup_pattern = (
        _generated_warmup_pattern_source(subpart, events, skill_ids, generation_gate=generation_gate)
        if role_statuses["generated_warmup_pattern_source"] != "block"
        else None
    )

    return {
        "candidate_id": f"content_lab_{subpart_id}",
        "question_id": record.get("question_id"),
        "paper": record.get("paper"),
        "paper_family": record.get("paper_family"),
        "question_number": record.get("question_number"),
        "subpart_id": subpart_id,
        "subpart_label": subpart.get("label"),
        "marks": subpart.get("marks"),
        "candidate_selection": {
            "reviewed_or_approved_subpart": _subpart_reviewed_or_approved(subpart),
            "high_confidence_subpart": _high_confidence_content_lab_subpart(record, subpart),
            "minimum_mark_event_confidence": _minimum_mark_event_confidence(events),
        },
        "source_artifacts": {
            "question_crop_path": subpart.get("question_crop_path"),
            "mark_scheme_crop_path": subpart.get("mark_scheme_crop_path"),
        },
        "possible_content_lab_roles": possible_roles,
        "role_statuses": role_statuses,
        "source_skill_ids": skill_ids,
        "source_mark_event_count": len(events),
        "generated_warmup_pattern_source": warmup_pattern,
        "generation_gate": generation_gate,
        "review_status": _candidate_review_status(role_statuses, generation_gate),
    }


def _content_lab_role_statuses(
    record: dict[str, Any],
    subpart: dict[str, Any],
    events: list[dict[str, Any]],
    skill_ids: list[str],
) -> dict[str, str]:
    usage_roles = record.get("usage_roles") if isinstance(record.get("usage_roles"), dict) else {}
    generation_gate = _content_lab_generation_gate(record, subpart, events, skill_ids)
    review_needed = not _subpart_reviewed_or_approved(subpart)
    has_events = bool(events)
    has_skills = bool(skill_ids)

    roles = {
        "field_guide_source": _candidate_role_status(str(usage_roles.get("field_guide_source") or ""), review_needed=review_needed),
        "quick_check_source": _candidate_role_status(str(usage_roles.get("quick_check_source") or ""), review_needed=review_needed),
        "generated_warmup_pattern_source": generation_gate["status"] if has_events and has_skills else "block",
        "guardian_candidate": _candidate_role_status(str(usage_roles.get("guardian_candidate") or ""), review_needed=review_needed),
        "prerequisite_repair_source": _prerequisite_repair_role_status(record, subpart, skill_ids, review_needed=review_needed),
        "mixed_review_source": CONTENT_LAB_BLOCKED_UNTIL_REVIEWED if _mixed_review_candidate(record, subpart, events) else "block",
    }
    return roles


def _candidate_role_status(source_status: str, *, review_needed: bool) -> str:
    if source_status == "allow":
        return CONTENT_LAB_BLOCKED_UNTIL_REVIEWED if review_needed else "allow"
    if source_status == CONTENT_LAB_BLOCKED_UNTIL_REVIEWED:
        return CONTENT_LAB_BLOCKED_UNTIL_REVIEWED
    return "block"


def _prerequisite_repair_role_status(
    record: dict[str, Any],
    subpart: dict[str, Any],
    skill_ids: list[str],
    *,
    review_needed: bool,
) -> str:
    if not skill_ids:
        return "block"
    paper_family = str(record.get("paper_family") or "").lower()
    prerequisite_skill = any("_p1_" in skill_id or "_p2_" in skill_id or "prerequisite" in skill_id for skill_id in skill_ids)
    if paper_family in {"p1", "p2"} or prerequisite_skill:
        return CONTENT_LAB_BLOCKED_UNTIL_REVIEWED if review_needed or not _subpart_reviewed_or_approved(subpart) else "allow"
    return "block"


def _mixed_review_candidate(record: dict[str, Any], subpart: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    if _subpart_reviewed_or_approved(subpart):
        return False
    if _high_confidence_content_lab_subpart(record, subpart):
        return True
    return any(str(event.get("review_status") or "") == MARK_EVENT_REVIEW_STATUS for event in events)


def _content_lab_generation_gate(
    record: dict[str, Any],
    subpart: dict[str, Any],
    events: list[dict[str, Any]],
    skill_ids: list[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    quality_gate = record.get("quality_gate") if isinstance(record.get("quality_gate"), dict) else {}
    if not quality_gate.get("content_lab_generation_allowed"):
        reasons.append("question_quality_gate_blocks_content_lab_generation")
    if not _subpart_reviewed_or_approved(subpart):
        reasons.append("mapping_or_subpart_not_reviewed_or_approved")
    if not events:
        reasons.append("missing_source_mark_events")
    if any(not _mark_event_reviewed_or_approved(event) for event in events):
        reasons.append("mark_events_not_reviewed_or_approved")
    if any(_mark_event_quarantined(event) for event in events):
        reasons.append("mark_events_quarantined")
    if not skill_ids:
        reasons.append("missing_source_skill_ids")

    return {
        "status": "allow" if not reasons else CONTENT_LAB_BLOCKED_UNTIL_REVIEWED,
        "blocked": bool(reasons),
        "block_reasons": _dedupe(reasons),
    }


def _generated_warmup_pattern_source(
    subpart: dict[str, Any],
    events: list[dict[str, Any]],
    skill_ids: list[str],
    *,
    generation_gate: dict[str, Any],
) -> dict[str, Any]:
    mark_type_chain = _mark_type_chain(events)
    return {
        "method_pattern_id": _method_pattern_id(subpart, skill_ids, mark_type_chain),
        "source_skill_ids": skill_ids,
        "source_mark_events": [_content_lab_source_mark_event(event) for event in events],
        "suggested_generator_family": _suggested_generator_family(events),
        "required_parameter_constraints": _required_parameter_constraints(subpart, events, skill_ids, generation_gate=generation_gate),
        "common_errors_to_target": _common_errors_to_target(events),
        "review_status": "ready" if generation_gate["status"] == "allow" else CONTENT_LAB_BLOCKED_UNTIL_REVIEWED,
    }


def _content_lab_source_mark_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "mark_code": event.get("mark_code"),
        "mark_type": event.get("mark_type"),
        "student_action": event.get("student_action"),
        "answer_target": event.get("answer_target"),
        "dependency": event.get("dependency"),
        "evidence_text": event.get("evidence_text"),
        "confidence": event.get("confidence"),
        "review_status": event.get("review_status"),
    }


def _method_pattern_id(subpart: dict[str, Any], skill_ids: list[str], mark_type_chain: list[str]) -> str:
    skill_part = _slug(skill_ids[0] if skill_ids else str(subpart.get("subpart_id") or "unknown_skill"))
    chain_part = _slug("_".join(mark_type_chain) if mark_type_chain else "unknown_marks")
    return f"{skill_part}__{chain_part}"


def _suggested_generator_family(events: list[dict[str, Any]]) -> str:
    mark_types = {str(event.get("mark_type") or "") for event in events}
    if "follow_through" in mark_types:
        return "follow_through_method_variation"
    if "method" in mark_types and "accuracy" in mark_types:
        return "worked_method_variation"
    if mark_types == {"independent_fact"}:
        return "fact_or_result_check"
    if "communication" in mark_types:
        return "explanation_or_justification_pattern"
    return "mark_event_pattern"


def _required_parameter_constraints(
    subpart: dict[str, Any],
    events: list[dict[str, Any]],
    skill_ids: list[str],
    *,
    generation_gate: dict[str, Any],
) -> list[str]:
    constraints = [
        "emit_metadata_only_no_student_facing_generated_content",
        "preserve_source_mark_event_sequence",
    ]
    if subpart.get("marks") is not None:
        constraints.append(f"preserve_total_marks:{subpart.get('marks')}")
    if skill_ids:
        constraints.append("stay_within_source_skill_ids")
    mark_types = [str(event.get("mark_type") or "") for event in events]
    if "method" in mark_types and "accuracy" in mark_types:
        constraints.append("require_method_step_before_accuracy_award")
    if "follow_through" in mark_types:
        constraints.append("carry_forward_prior_result_for_follow_through_marks")
    if any(event.get("answer_target") for event in events):
        constraints.append("preserve_equivalent_answer_target_structure")
    if generation_gate.get("blocked"):
        constraints.append("generation_blocked_until_mapping_and_mark_events_reviewed")
    return _dedupe(constraints)


def _common_errors_to_target(events: list[dict[str, Any]]) -> list[str]:
    return _dedupe(
        [
            error
            for event in events
            for error in _list_string_values(event.get("common_errors"))
        ]
    )


def _candidate_review_status(role_statuses: dict[str, str], generation_gate: dict[str, Any]) -> str:
    if generation_gate["status"] == "allow" and all(status in {"allow", "block"} for status in role_statuses.values()):
        return "ready"
    if any(status == CONTENT_LAB_BLOCKED_UNTIL_REVIEWED for status in role_statuses.values()):
        return CONTENT_LAB_BLOCKED_UNTIL_REVIEWED
    return MARK_EVENT_REVIEW_STATUS


def _high_confidence_content_lab_subpart(record: dict[str, Any], subpart: dict[str, Any]) -> bool:
    events = _mark_events(subpart)
    if not events:
        return False
    if _content_lab_hard_blocked(record):
        return False
    if any(_mark_event_quarantined(event) for event in events):
        return False
    return _minimum_mark_event_confidence(events) >= 0.84


def _minimum_mark_event_confidence(events: list[dict[str, Any]]) -> float | None:
    values = []
    for event in events:
        try:
            values.append(float(event.get("confidence")))
        except (TypeError, ValueError):
            continue
    return round(min(values), 2) if values else None


def _mark_type_chain(events: list[dict[str, Any]]) -> list[str]:
    return [str(event.get("mark_type") or "unknown") for event in events]


def _mark_events(subpart: dict[str, Any]) -> list[dict[str, Any]]:
    events = subpart.get("mark_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _content_lab_hard_blocked(record: dict[str, Any]) -> bool:
    gate = record.get("quality_gate")
    if not isinstance(gate, dict):
        return True
    return _hard_blocked(record, gate)


def _subpart_reviewed_or_approved(subpart: dict[str, Any]) -> bool:
    return _reviewed_or_approved_status(subpart.get("review_status")) or _as_bool(subpart.get("content_lab_approved"))


def _mark_event_reviewed_or_approved(event: dict[str, Any]) -> bool:
    return _reviewed_or_approved_status(event.get("review_status")) or _as_bool(event.get("approved"))


def _mark_event_quarantined(event: dict[str, Any]) -> bool:
    return str(event.get("review_status") or "") == MARK_EVENT_QUARANTINED_STATUS or bool(event.get("quarantine_reason"))


def _reviewed_or_approved_status(value: Any) -> bool:
    return str(value or "").strip().lower() in {"ready", "reviewed", "approved", "human_reviewed", "explicitly_approved"}


def _mark_events_from_text(
    text: str | None,
    *,
    subpart_id: str,
    skill_ids: list[str],
    quarantine_reason: str | None,
) -> list[dict[str, Any]]:
    if _mark_scheme_text_too_degraded(text):
        return []

    review_status = MARK_EVENT_QUARANTINED_STATUS if quarantine_reason else MARK_EVENT_REVIEW_STATUS
    events: list[dict[str, Any]] = []
    for evidence in _mark_event_evidence_units(str(text or "")):
        matches = list(_MARK_CODE_RE.finditer(evidence))
        if not matches:
            continue
        action = _student_action_from_evidence(evidence)
        if _mark_event_action_too_degraded(action):
            continue
        common_errors = _common_errors_from_evidence(evidence)
        for match in matches:
            mark_code = _normalized_mark_code(match)
            mark_type = _mark_type(mark_code, evidence)
            answer_target = _answer_target_from_action(action, mark_code=mark_code)
            event = {
                "subpart_id": subpart_id,
                "mark_code": mark_code,
                "mark_type": mark_type,
                "student_action": action,
                "answer_target": answer_target,
                "dependency": _dependency_from_mark(match, evidence),
                "skill_ids": list(skill_ids),
                "common_errors": common_errors,
                "evidence_text": _compact_text(evidence),
                "confidence": _mark_event_confidence(
                    mark_code=mark_code,
                    action=action,
                    answer_target=answer_target,
                    dependency=_dependency_from_mark(match, evidence),
                    common_errors=common_errors,
                    quarantined=bool(quarantine_reason),
                ),
                "review_status": review_status,
            }
            if quarantine_reason:
                event["quarantine_reason"] = quarantine_reason
            events.append(event)
    return events


def _mark_scheme_text_too_degraded(text: str | None) -> bool:
    cleaned = _compact_text(str(text or ""))
    if not cleaned:
        return True
    if not _MARK_CODE_RE.search(cleaned):
        return True
    alnum_count = sum(1 for char in cleaned if char.isalnum())
    if alnum_count < 2:
        return True
    replacement_count = cleaned.count("\ufffd")
    if replacement_count >= 2 or replacement_count / max(len(cleaned), 1) > 0.02:
        return True
    if re.search(r"(?:[?#|]\s*){6,}", cleaned):
        return True
    return False


def _mark_event_evidence_units(text: str) -> list[str]:
    lines = [_compact_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if lines:
        return lines
    compact = _compact_text(text)
    return [compact] if compact else []


def _student_action_from_evidence(evidence: str) -> str:
    action = _MARK_CODE_RE.sub(" ", evidence)
    action = re.sub(
        r"^\s*\d{1,2}(?:\([a-h]\)|\((?:viii|vii|vi|iv|ix|iii|ii|i|v|x)\))?\s*",
        " ",
        action,
        flags=re.IGNORECASE,
    )
    action = re.sub(r"^\s*(?:\([a-h]\)|\((?:viii|vii|vi|iv|ix|iii|ii|i|v|x)\)|[a-h]\))\s*", " ", action, flags=re.IGNORECASE)
    action = _drop_trailing_marks_column_numbers(action, evidence=evidence)
    return _compact_text(action).strip(" ;,")


def _drop_trailing_marks_column_numbers(text: str, *, evidence: str) -> str:
    tokens = _compact_text(text).split()
    if len(tokens) < 3:
        return " ".join(tokens)
    dropped = 0
    while len(tokens) > 1 and re.fullmatch(r"\d{1,2}", tokens[-1]) and dropped < 3:
        previous = tokens[-2] if len(tokens) >= 2 else ""
        if dropped == 0 and not re.fullmatch(r"\d{1,2}", previous):
            if not _mark_code_followed_by_terminal_number(evidence):
                break
            tokens.pop()
            break
        tokens.pop()
        dropped += 1
    return " ".join(tokens)


def _mark_code_followed_by_terminal_number(evidence: str) -> bool:
    matches = list(_MARK_CODE_RE.finditer(evidence))
    if not matches:
        return False
    tail = evidence[matches[-1].end() :]
    return bool(re.fullmatch(r"\s+\d{1,2}\s*", tail))


def _mark_event_action_too_degraded(action: str) -> bool:
    if not action:
        return True
    if re.fullmatch(r"[\W_]+", action):
        return True
    if re.search(r"(?:[?#|]\s*){4,}", action):
        return True
    return False


def _normalized_mark_code(match: re.Match[str]) -> str:
    dependent = "D" if match.group("dependent") else ""
    kind = match.group("kind").upper()
    value = match.group("value")
    suffix = _compact_text(match.group("suffix") or "").lower()
    if suffix == "ft":
        return f"{dependent}{kind}{value}FT"
    return f"{dependent}{kind}{value}"


def _mark_type(mark_code: str, evidence: str) -> str:
    lowered = evidence.lower()
    if mark_code.endswith("FT") or (
        _single_mark_code_in_evidence(evidence) and re.search(r"\b(?:ft|follow(?:ed)?\s+through)\b", lowered)
    ):
        return "follow_through"
    core = mark_code[1:] if mark_code.startswith("D") else mark_code
    if core.startswith("M"):
        return "method"
    if core.startswith("A"):
        return "accuracy"
    if core.startswith("B"):
        return "independent_fact"
    if core.startswith(("C", "E")):
        return "communication"
    return "unknown"


def _dependency_from_mark(match: re.Match[str], evidence: str) -> str | None:
    lowered = evidence.lower()
    suffix = _compact_text(match.group("suffix") or "").lower()
    single_code = _single_mark_code_in_evidence(evidence)
    if suffix == "ft" or (single_code and re.search(r"\b(?:ft|follow(?:ed)?\s+through)\b", lowered)):
        return "follow_through_from_previous_work"
    if match.group("dependent") or suffix == "dep" or (single_code and re.search(r"\bdep(?:endent)?\b", lowered)):
        return "dependent_on_previous_mark"
    return None


def _single_mark_code_in_evidence(evidence: str) -> bool:
    return len(list(_MARK_CODE_RE.finditer(evidence))) == 1


def _answer_target_from_action(action: str, *, mark_code: str) -> str | None:
    cleaned = action.strip(" .;:,")
    if not cleaned:
        return None
    target_match = re.search(r"\b(?:answer|obtain|obtains|gives?|leading to|hence)\b\s*(?P<target>.+)$", cleaned, re.IGNORECASE)
    if target_match:
        target = target_match.group("target").strip(" .;:,")
        return target or None
    if _mark_type(mark_code, "") in {"accuracy", "independent_fact", "follow_through"} and len(cleaned) <= 160:
        if _has_math_or_answer_signal(cleaned):
            return cleaned
    return None


def _has_math_or_answer_signal(text: str) -> bool:
    if re.search(r"[=<>^+\-*/]", text):
        return True
    if re.search(r"\b(?:x|y|z|u|v|theta|sin|cos|tan|ln|log|e\^)\b", text, re.IGNORECASE):
        return True
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.strip()))


def _common_errors_from_evidence(evidence: str) -> list[str]:
    compact = _compact_text(evidence)
    patterns = [
        r"\bdo\s+not\s+allow\b[^.;]*",
        r"\bdo\s+not\s+accept\b[^.;]*",
        r"\bnot\s+accepted\b[^.;]*",
        r"\bincorrect\b[^.;]*",
        r"\bwrong\b[^.;]*",
        r"\b(?:M0|A0|B0)\b[^.;]*",
        r"\bscores?\s+0\b[^.;]*",
    ]
    errors: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact, re.IGNORECASE):
            errors.append(match.group(0).strip(" .;:,"))
    return _dedupe(errors)


def _mark_event_confidence(
    *,
    mark_code: str,
    action: str,
    answer_target: str | None,
    dependency: str | None,
    common_errors: list[str],
    quarantined: bool,
) -> float:
    confidence = 0.72
    if mark_code:
        confidence += 0.06
    if action:
        confidence += 0.06
    if answer_target:
        confidence += 0.04
    if dependency:
        confidence += 0.02
    if common_errors:
        confidence -= 0.03
    if quarantined:
        confidence = min(confidence, 0.55)
    return round(max(0.0, min(confidence, 0.9)), 2)


def _mark_event_quarantine_reason(record: dict[str, Any]) -> str | None:
    mark_scheme_total = _numeric(_get(record, "mark_scheme_total_detected"))
    question_total = _numeric(_get(record, "question_total_detected"))
    if question_total is None:
        question_total = _numeric(_get(record, "question_solution_marks"))
    if question_total is not None and mark_scheme_total is not None and question_total != mark_scheme_total:
        return MARK_EVENT_TOTAL_DISAGREEMENT
    return None


def _skill_ids_for_subpart(
    record: dict[str, Any],
    *,
    subpart_id: str,
    skill_mappings: dict[str, list[str]],
) -> list[str]:
    if subpart_id in skill_mappings:
        return list(skill_mappings[subpart_id])

    subpart_skill_ids = _get(record, "subpart_skill_ids")
    if isinstance(subpart_skill_ids, dict):
        return _list_string_values(subpart_skill_ids.get(subpart_id))

    skills_by_subpart = _get(record, "skills_by_subpart")
    if isinstance(skills_by_subpart, dict):
        return _list_string_values(skills_by_subpart.get(subpart_id))

    return []


def _mark_scheme_text_trust(record: dict[str, Any], *, mark_scheme_crop_ok: bool) -> str:
    if str(_get(record, "mapping_status") or "").lower() == "fail":
        return "low"
    if mark_scheme_crop_ok:
        return "high"
    return "medium"


def _subpart_marks(record: dict[str, Any], *, labels: list[str], total_marks: int | None) -> dict[str, int | None]:
    existing = _get(record, "subparts_solution_marks")
    if isinstance(existing, dict):
        values = {label: _numeric(existing.get(label)) for label in labels}
        if any(value is not None for value in values.values()):
            return values

    detected = _detected_mark_values(record)
    if labels == ["whole"]:
        if len(detected) == 1:
            return {"whole": detected[0]}
        return {"whole": total_marks}
    if len(detected) == len(labels):
        return {label: detected[index] for index, label in enumerate(labels)}
    return {label: None for label in labels}


def _subpart_marks_missing(record: dict[str, Any]) -> bool:
    labels = _subpart_labels(record)
    if not labels:
        return False
    values = _subpart_marks(record, labels=labels, total_marks=_total_marks(record))
    return all(value is None for value in values.values())


def _subpart_labels(record: dict[str, Any]) -> list[str]:
    values = _get(record, "subparts")
    if isinstance(values, list):
        return [str(value) for value in values if str(value)]
    return []


def _detected_mark_values(record: dict[str, Any]) -> list[int]:
    structure = _get(record, "question_structure_detected")
    values = structure.get("mark_values_detected") if isinstance(structure, dict) else []
    if not isinstance(values, list):
        return []
    detected: list[int] = []
    for value in values:
        numeric_value = _numeric(value)
        if numeric_value is not None:
            detected.append(numeric_value)
    return detected


def _detected_values_for_subpart(
    index: int,
    label: str,
    labels: list[str],
    marks: int | None,
    detected_values: list[int],
) -> list[int]:
    if labels == ["whole"]:
        return detected_values
    if len(detected_values) == len(labels):
        return [detected_values[index]]
    return [marks] if marks is not None else []


def _split_labeled_text(text: str, labels: list[str]) -> dict[str, str | None]:
    stripped = " ".join(text.split())
    if not stripped:
        return {label: None for label in labels}
    if labels == ["whole"]:
        return {"whole": stripped}

    spans: list[tuple[str, int, int]] = []
    for label in labels:
        pattern = re.compile(rf"(?:\({re.escape(label)}\)|(?<![A-Za-z0-9]){re.escape(label)}\))")
        match = pattern.search(stripped)
        if match:
            spans.append((label, match.start(), match.end()))
    if not spans:
        return {label: None for label in labels}

    spans.sort(key=lambda item: item[1])
    result: dict[str, str | None] = {label: None for label in labels}
    for index, (label, start, end) in enumerate(spans):
        next_start = spans[index + 1][1] if index + 1 < len(spans) else len(stripped)
        result[label] = stripped[start:next_start].strip() or stripped[end:next_start].strip() or None
    return result


def _question_image_paths(record: dict[str, Any]) -> list[str]:
    paths = _list_paths(_get(record, "question_image_paths"))
    for field in ["canonical_question_artifact", "question_image_path"]:
        value = _get(record, field)
        if value:
            paths.append(str(value))
    return _dedupe(paths)


def _mark_scheme_image_paths(record: dict[str, Any]) -> list[str]:
    paths = _list_paths(_get(record, "mark_scheme_image_paths"))
    value = _get(record, "mark_scheme_image_path")
    if value:
        paths.append(str(value))
    return _dedupe(paths)


def _image_integrity(paths: list[str], *, artifact_root: Path | None, base_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        digest = _sha256_for_path(path, artifact_root=artifact_root, base_dir=base_dir)
        items.append(
            {
                "path": path,
                "sha256": digest,
                "exists": digest is not None,
            }
        )
    return items


def _source_pdf_metadata(record: dict[str, Any], *, base_dir: Path) -> dict[str, dict[str, str | bool | None]]:
    question_pdf = _get(record, "source_pdf")
    mark_scheme_pdf = _get(record, "mark_scheme_source_pdf")
    return {
        "question_paper": _source_file_metadata(question_pdf, base_dir=base_dir),
        "mark_scheme": _source_file_metadata(mark_scheme_pdf, base_dir=base_dir),
    }


def _source_file_metadata(value: Any, *, base_dir: Path) -> dict[str, str | bool | None]:
    path = str(value or "")
    digest = _sha256_for_path(path, artifact_root=None, base_dir=base_dir) if path else None
    return {
        "path": path or None,
        "sha256": digest,
        "exists": digest is not None,
    }


def _sha256_for_path(path_value: str, *, artifact_root: Path | None, base_dir: Path) -> str | None:
    resolved = _resolve_existing_path(path_value, artifact_root=artifact_root, base_dir=base_dir)
    if resolved is None:
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_existing_path(path_value: str, *, artifact_root: Path | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if artifact_root is not None:
            candidates.append(artifact_root / path)
        candidates.append(base_dir / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _first_path(items: list[dict[str, Any]]) -> str:
    return str(items[0]["path"]) if items else ""


def _list_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _list_string_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _total_marks(record: dict[str, Any]) -> int | None:
    for field in ["question_solution_marks", "question_total_detected", "mark_scheme_total_detected"]:
        value = _numeric(_get(record, field))
        if value is not None:
            return value
    return None


def _marks_consistent(record: dict[str, Any]) -> bool:
    values = [
        value
        for value in [
            _numeric(_get(record, "question_solution_marks")),
            _numeric(_get(record, "question_total_detected")),
            _numeric(_get(record, "mark_scheme_total_detected")),
        ]
        if value is not None
    ]
    return bool(values) and all(value > 0 for value in values) and len(set(values)) == 1


def _numeric(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(record: dict[str, Any], field: str) -> Any:
    if field in record and record[field] not in (None, ""):
        return record[field]
    notes = record.get("notes")
    if isinstance(notes, dict):
        return notes.get(field)
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _subpart_id_suffix(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "whole"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "unknown"


def _compact_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())
