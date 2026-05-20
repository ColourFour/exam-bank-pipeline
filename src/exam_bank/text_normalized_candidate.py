from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_NAME = "exam_bank.advisory_normalized_text_candidate"
SCHEMA_VERSION = 1
RULE_VERSION = "math_normalization_rules_v1"
DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00Z"

SOURCE_TEXT_KINDS = {
    "selected_text",
    "native_pdf_text_raw",
    "ocr_text_raw",
    "profile_ocr_text",
    "native_pdf_text_candidate",
    "ocr_text_candidate",
}

REQUIRED_FIELDS = {
    "candidate_id",
    "record_id",
    "source_text_kind",
    "source_text_hash",
    "question_text_normalized",
    "normalization_flags",
    "normalization_confidence",
    "normalization_warnings",
    "normalization_is_advisory",
    "created_by_version",
    "created_at",
    "provenance",
    "display_allowed",
    "export_allowed",
}


class NormalizedTextCandidateContractError(ValueError):
    pass


def source_text_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def build_advisory_normalized_text_candidate(
    *,
    record_id: str,
    source_text_kind: str,
    source_text: str,
    question_text_normalized: str,
    normalization_flags: list[str],
    normalization_confidence: float,
    normalization_warnings: list[str],
    provenance: dict[str, Any],
    created_by_version: str = RULE_VERSION,
    created_at: str = DETERMINISTIC_CREATED_AT,
    display_allowed: bool = False,
    export_allowed: bool = False,
) -> dict[str, Any]:
    candidate = {
        "candidate_id": "",
        "record_id": str(record_id),
        "source_text_kind": source_text_kind,
        "source_text_hash": source_text_hash(source_text),
        "question_text_normalized": question_text_normalized,
        "normalization_flags": sorted(str(flag) for flag in normalization_flags),
        "normalization_confidence": float(normalization_confidence),
        "normalization_warnings": sorted(str(warning) for warning in normalization_warnings),
        "normalization_is_advisory": True,
        "created_by_version": created_by_version,
        "created_at": created_at,
        "provenance": provenance,
        "display_allowed": bool(display_allowed),
        "export_allowed": bool(export_allowed),
    }
    candidate["candidate_id"] = deterministic_candidate_id(candidate)
    validate_advisory_normalized_text_candidate(candidate)
    return candidate


def validate_advisory_normalized_text_candidate(candidate: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(candidate))
    if missing:
        raise NormalizedTextCandidateContractError(f"Missing required candidate fields: {', '.join(missing)}")

    if candidate["normalization_is_advisory"] is not True:
        raise NormalizedTextCandidateContractError("normalization_is_advisory must be true")
    if candidate["source_text_kind"] not in SOURCE_TEXT_KINDS:
        raise NormalizedTextCandidateContractError(f"Unsupported source_text_kind: {candidate['source_text_kind']!r}")
    if not isinstance(candidate["source_text_hash"], str) or len(candidate["source_text_hash"]) != 64:
        raise NormalizedTextCandidateContractError("source_text_hash must be a SHA-256 hex digest")
    if not isinstance(candidate["normalization_flags"], list):
        raise NormalizedTextCandidateContractError("normalization_flags must be a list")
    if not isinstance(candidate["normalization_warnings"], list):
        raise NormalizedTextCandidateContractError("normalization_warnings must be a list, even when empty")
    confidence = candidate["normalization_confidence"]
    if not isinstance(confidence, int | float) or confidence < 0 or confidence > 1:
        raise NormalizedTextCandidateContractError("normalization_confidence must be between 0 and 1")
    if not isinstance(candidate["provenance"], dict) or not candidate["provenance"]:
        raise NormalizedTextCandidateContractError("provenance must be a non-empty object")
    for field in ("candidate_id", "record_id", "question_text_normalized", "created_by_version", "created_at"):
        if not isinstance(candidate[field], str) or not candidate[field].strip():
            raise NormalizedTextCandidateContractError(f"{field} must be a non-empty string")
    for field in ("display_allowed", "export_allowed"):
        if not isinstance(candidate[field], bool):
            raise NormalizedTextCandidateContractError(f"{field} must be a boolean")


def deterministic_candidate_id(candidate: dict[str, Any]) -> str:
    stable_payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_id": candidate.get("record_id"),
        "source_text_kind": candidate.get("source_text_kind"),
        "source_text_hash": candidate.get("source_text_hash"),
        "question_text_normalized": candidate.get("question_text_normalized"),
        "created_by_version": candidate.get("created_by_version"),
    }
    digest = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"normcand_{digest[:24]}"
