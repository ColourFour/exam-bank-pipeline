from __future__ import annotations

import re
from typing import Any


class Confidence:
    """Ordered coarse confidence labels used by local and LLM classifiers."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VALUES = frozenset({LOW, MEDIUM, HIGH})


class ValidationStatus:
    """Question extraction validation outcome."""

    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    VALUES = frozenset({PASS, REVIEW, FAIL})


class ScopeQualityStatus:
    """Whether the selected question span appears clean, risky, or contaminated."""

    CLEAN = "clean"
    REVIEW = "review"
    FAIL = "fail"
    VALUES = frozenset({CLEAN, REVIEW, FAIL})


class TextSourceProfile:
    """Origin of the question text used for labels and downstream review."""

    NATIVE_PDF = "native_pdf"
    HYBRID = "hybrid"
    OCR = "ocr"
    VALUES = frozenset({NATIVE_PDF, HYBRID, OCR})


class TextFidelityStatus:
    """How trustworthy the extracted question text is for labeling."""

    CLEAN = "clean"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"
    VALUES = frozenset({CLEAN, DEGRADED, UNUSABLE})


class TopicTrustStatus:
    """Downstream interpretation of topic labels after extraction trust checks."""

    NORMAL = "normal"
    DEGRADED_TEXT = "degraded_text"
    REVIEW_REQUIRED = "review_required"
    VALUES = frozenset({NORMAL, DEGRADED_TEXT, REVIEW_REQUIRED})


class MappingStatus:
    """Question-to-mark-scheme mapping outcome."""

    PASS = "pass"
    FAIL = "fail"
    VALUES = frozenset({PASS, FAIL})


class CropConfidence:
    """Confidence that rendered question or mark-scheme crops contain the intended scope."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VALUES = frozenset({LOW, MEDIUM, HIGH})


class PaperTotalStatus:
    """Paper mark-total reconciliation result."""

    UNKNOWN_EXPECTED_TOTAL = "unknown_expected_total"
    MATCHED = "matched"
    MISMATCH = "mismatch"
    RECOVERED_AFTER_RESCAN = "recovered_after_rescan"
    MISMATCH_AFTER_RESCAN = "mismatch_after_rescan"
    VALUES = frozenset({UNKNOWN_EXPECTED_TOTAL, MATCHED, MISMATCH, RECOVERED_AFTER_RESCAN, MISMATCH_AFTER_RESCAN})


class RescanResult:
    """Result of the optional broader paper-total rescan pass."""

    NOT_TRIGGERED = "not_triggered"
    RECOVERED_EXACT_TOTAL = "recovered_exact_total"
    IMPROVED_BUT_STILL_MISMATCH = "improved_but_still_mismatch"
    NO_IMPROVEMENT = "no_improvement"
    VALUES = frozenset({NOT_TRIGGERED, RECOVERED_EXACT_TOTAL, IMPROVED_BUT_STILL_MISMATCH, NO_IMPROVEMENT})


class DeepSeekErrorType:
    """DeepSeek sidecar error categories."""

    PROVIDER_ERROR = "provider_error"
    VALUES = frozenset({PROVIDER_ERROR})


class ReconciliationStatus:
    """Local-vs-provider label reconciliation result."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNMAPPED_LABEL = "unmapped_label"
    NO_DEEPSEEK_LABEL = "no_deepseek_label"
    NO_LOCAL_LABEL = "no_local_label"
    VALUES = frozenset({MATCH, MISMATCH, UNMAPPED_LABEL, NO_DEEPSEEK_LABEL, NO_LOCAL_LABEL})


STRUCTURAL_MAPPING_FAILURES = frozenset(
    {
        "question_subparts_incomplete",
        "question_scope_contaminated",
        "missing_terminal_mark_total",
        "question_mark_total_mismatch",
        "question_mark_total_missing",
        "likely_truncated_question_crop",
    }
)


def derive_scope_quality_status(
    *,
    validation_flags: list[str],
    review_flags: list[str],
    question_structure_detected: dict[str, Any],
) -> str:
    if (
        question_structure_detected.get("contamination_detected")
        or any(
            flag in validation_flags or flag in review_flags
            for flag in {
                "question_scope_contaminated",
                "question_start_mismatch",
                "possible_next_question_contamination",
                "likely_truncated_question_crop",
            }
        )
    ):
        return ScopeQualityStatus.FAIL
    if any(
        flag in validation_flags or flag in review_flags
        for flag in {
            "weak_question_anchor",
            "question_start_uncertain",
            "question_sequence_gap",
            "paper_total_focus_candidate",
        }
    ):
        return ScopeQualityStatus.REVIEW
    return ScopeQualityStatus.CLEAN


def text_source_profile(review_flags: list[str]) -> str:
    if any(flag.startswith("ocr_merged") for flag in review_flags):
        return TextSourceProfile.HYBRID
    if "ocr_question_text" in review_flags:
        return TextSourceProfile.OCR
    return TextSourceProfile.NATIVE_PDF


def assess_text_fidelity(
    *,
    question_text: str,
    extraction_quality_flags: list[str],
    review_flags: list[str],
    validation_flags: list[str],
    question_structure_detected: dict[str, Any],
    mapping_failure_reason: str,
    text_source_profile: str,
) -> tuple[str, list[str]]:
    flags: set[str] = set()
    strong_corruption_flags = {
        "likely_needs_visual_review",
        "math_corruption_suspected",
        "broken_fraction_structure",
        "broken_superscript_or_power",
        "suspicious_symbol_run",
        "flattened_display_math",
        "diagram_text_mixed_with_body",
    }
    if any(flag in extraction_quality_flags for flag in strong_corruption_flags):
        flags.add("math_text_corruption_detected")
    if _text_has_suspicious_ocr_noise(question_text):
        flags.add("ocr_noise_fragment_present")
    if _text_has_corrupted_math_notation(question_text):
        flags.add("ocr_math_notation_degraded")
    if (
        mapping_failure_reason == "question_subparts_incomplete"
        or "question_subparts_incomplete" in validation_flags
        or bool(question_structure_detected.get("missing_internal_subparts"))
        or bool(question_structure_detected.get("impossible_subpart_sequence_detected"))
    ):
        flags.add("missing_visible_structure_in_text")
    if (
        text_source_profile != TextSourceProfile.NATIVE_PDF
        and "heavy_math_density" in extraction_quality_flags
        and ("ocr_noise_fragment_present" in flags or "ocr_math_notation_degraded" in flags)
    ):
        flags.add("hybrid_math_text_requires_review")
    if "weak_question_text" in review_flags and flags:
        flags.add("weak_extracted_text")

    status = TextFidelityStatus.CLEAN
    if "missing_visible_structure_in_text" in flags:
        status = TextFidelityStatus.UNUSABLE
    elif flags:
        status = TextFidelityStatus.DEGRADED
    return status, sorted(flags)


def derive_topic_trust_status(
    *,
    topic_confidence: str,
    topic_uncertain: bool,
    text_fidelity_status: str,
    validation_status: str,
    scope_quality_status: str,
) -> str:
    if validation_status == ValidationStatus.FAIL or scope_quality_status == ScopeQualityStatus.FAIL:
        return TopicTrustStatus.REVIEW_REQUIRED
    if text_fidelity_status != TextFidelityStatus.CLEAN:
        return TopicTrustStatus.DEGRADED_TEXT
    if topic_uncertain or topic_confidence == Confidence.LOW:
        return TopicTrustStatus.REVIEW_REQUIRED
    return TopicTrustStatus.NORMAL


def refine_validation_status(
    *,
    base_status: str,
    base_validation_flags: list[str],
    mapping_status: str,
    mapping_failure_reason: str,
    crop_uncertain: bool,
    extraction_quality_flags: list[str],
    review_flags: list[str],
    question_structure_detected: dict[str, Any],
) -> tuple[str, list[str]]:
    validation_flags = list(base_validation_flags)
    status = base_status

    if mapping_failure_reason in STRUCTURAL_MAPPING_FAILURES:
        status = ValidationStatus.FAIL
        if mapping_failure_reason not in validation_flags:
            validation_flags.append(mapping_failure_reason)

    support_groups = polluted_pass_signal_groups(
        crop_uncertain=crop_uncertain,
        base_validation_flags=base_validation_flags,
        extraction_quality_flags=extraction_quality_flags,
        review_flags=review_flags,
        question_structure_detected=question_structure_detected,
    )
    if mapping_status == MappingStatus.PASS and len(support_groups) >= 2:
        status = ValidationStatus.FAIL
        if "polluted_pass_requires_review" not in validation_flags:
            validation_flags.append("polluted_pass_requires_review")

    return status, sorted(set(validation_flags))


def polluted_pass_signal_groups(
    *,
    crop_uncertain: bool,
    base_validation_flags: list[str],
    extraction_quality_flags: list[str],
    review_flags: list[str],
    question_structure_detected: dict[str, Any],
) -> set[str]:
    if not crop_uncertain:
        return set()

    groups: set[str] = set()
    if any(
        flag in extraction_quality_flags
        for flag in {"likely_needs_visual_review", "math_corruption_suspected", "broken_fraction_structure", "suspicious_symbol_run"}
    ):
        groups.add("low_quality_text")
    if any(
        flag in review_flags
        for flag in {"low_confidence_question_crop", "crop_reaches_page_margin", "weak_question_text"}
    ):
        groups.add("crop_risk")
    if any(
        flag in base_validation_flags
        for flag in {"weak_question_anchor", "likely_truncated_question_crop", "missing_terminal_mark_total"}
    ):
        groups.add("question_validation_risk")
    contamination_indicators = question_structure_detected.get("contamination_indicators") or {}
    contamination_signal_score = int(contamination_indicators.get("signal_score", 0))
    if question_structure_detected.get("contamination_detected") or contamination_signal_score >= 2:
        groups.add("pollution_signals")
    return groups


def final_review_reasons(
    *,
    model_review_required: bool,
    validation_status: str | None,
    scope_quality_status: str | None,
    text_fidelity_status: str | None,
    topic_trust_status: str | None,
    topic_reconciliation_status: str,
    difficulty_reconciliation_status: str,
    local_difficulty_present: bool,
) -> list[str]:
    reasons: list[str] = []
    if model_review_required:
        _append_reason(reasons, "llm_review_required")
    if validation_status and validation_status != ValidationStatus.PASS:
        _append_reason(reasons, f"validation_status:{validation_status}")
    if scope_quality_status and scope_quality_status != ScopeQualityStatus.CLEAN:
        _append_reason(reasons, f"scope_quality_status:{scope_quality_status}")
    if text_fidelity_status == TextFidelityStatus.DEGRADED:
        _append_reason(reasons, "text_fidelity_status:degraded")
    elif text_fidelity_status == TextFidelityStatus.UNUSABLE:
        _append_reason(reasons, "text_fidelity_status:unusable")
    if topic_trust_status and topic_trust_status != TopicTrustStatus.NORMAL:
        _append_reason(reasons, f"topic_trust_status:{topic_trust_status}")
    if topic_reconciliation_status in {
        ReconciliationStatus.MISMATCH,
        ReconciliationStatus.UNMAPPED_LABEL,
        ReconciliationStatus.NO_DEEPSEEK_LABEL,
    }:
        _append_reason(reasons, f"topic_reconciliation_status:{topic_reconciliation_status}")
    if local_difficulty_present and difficulty_reconciliation_status in {
        ReconciliationStatus.MISMATCH,
        ReconciliationStatus.UNMAPPED_LABEL,
        ReconciliationStatus.NO_DEEPSEEK_LABEL,
    }:
        _append_reason(reasons, f"difficulty_reconciliation_status:{difficulty_reconciliation_status}")
    return reasons


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _text_has_suspicious_ocr_noise(text: str) -> bool:
    normalized_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in normalized_lines:
        lowered = line.lower()
        if lowered in {"anne", "fy"} or any(token in lowered for token in {"| anne", "| fy", "¢"}):
            return True
        if re.fullmatch(r"[|¦Il1]\s*[A-Za-z]{2,6}", line):
            return True
        if re.fullmatch(r"[A-Za-z]{2,6}[!?]", line):
            return True
        if re.search(r"\b[A-Za-z]{3,}[!?]\b", line):
            return True
    return False


def _text_has_corrupted_math_notation(text: str) -> bool:
    normalized = str(text)
    patterns = [
        r"(?:sin|cos|tan|sec|cot){2,}",
        r"\b[a-z]*(?:sin|cos|tan|sec|cot){2,}[a-z]*\b",
        r"\^\{[^}]{1,4}\}_\{[^}]{1,4}\}r",
        r"[A-Za-z]{1,3},,",
    ]
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)
