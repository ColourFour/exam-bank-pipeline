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


class QuestionTextRole:
    """How consumers should use extracted question text."""

    SEARCH_HINT = "search_hint"
    READABLE_TEXT = "readable_text"
    UNTRUSTED_MATH_TEXT = "untrusted_math_text"
    MISSING = "missing"
    VALUES = frozenset({SEARCH_HINT, READABLE_TEXT, UNTRUSTED_MATH_TEXT, MISSING})


class QuestionTextTrust:
    """Trust level for extracted question text as a representation of the visual question."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"
    VALUES = frozenset({HIGH, MEDIUM, LOW, UNUSABLE})


class CurationStatus:
    """Readiness for visual/manual curation or stricter text-only automation."""

    READY = "ready"
    REVIEW = "review"
    FAIL = "fail"
    VALUES = frozenset({READY, REVIEW, FAIL})


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
    visual_flags = visual_reason_flags(
        question_text=question_text,
        extraction_quality_flags=extraction_quality_flags,
        review_flags=review_flags,
        question_structure_detected=question_structure_detected,
        text_source_profile=text_source_profile,
    )
    if "contains_math_text_corruption" in visual_flags:
        flags.add("math_text_corruption_detected")
    if "contains_pdf_control_garbage" in visual_flags:
        flags.add("pdf_control_garbage_detected")
    if "native_text_missing_or_sparse" in visual_flags or "ocr_text_sparse_or_merged" in visual_flags:
        flags.add("sparse_or_merged_question_text")
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


def visual_reason_flags(
    *,
    question_text: str,
    extraction_quality_flags: list[str],
    review_flags: list[str],
    question_structure_detected: dict[str, Any],
    text_source_profile: str,
) -> list[str]:
    text = str(question_text or "")
    normalized = _normalize_for_visual_flags(text)
    lowered = normalized.lower()
    flags: set[str] = set()

    if not normalized or len(re.findall(r"[A-Za-z0-9]", normalized)) < 12:
        flags.add("native_text_missing_or_sparse")
    if text_source_profile != TextSourceProfile.NATIVE_PDF and (
        "weak_question_text" in review_flags or any(flag.startswith("ocr_merged") for flag in review_flags)
    ):
        flags.add("ocr_text_sparse_or_merged")

    if _text_has_pdf_control_garbage(text):
        flags.add("contains_pdf_control_garbage")
    if _text_has_page_furniture(text):
        flags.add("contains_page_furniture")
    if _text_has_math_corruption(text):
        flags.add("contains_math_text_corruption")
        flags.add("text_order_unreliable")

    if _contains_equation_layout(normalized):
        flags.add("contains_equation_layout")
    if _contains_fraction_or_integral_layout(normalized):
        flags.add("contains_fraction_or_integral_layout")
    if _contains_vector_notation(normalized):
        flags.add("contains_vector_notation")
    if _contains_complex_number_notation(normalized):
        flags.add("contains_complex_number_notation")
    if _contains_inequality_or_region_prompt(normalized):
        flags.add("contains_inequality_or_region_prompt")
    if _contains_graph_or_diagram_prompt(normalized):
        flags.add("contains_graph_or_diagram_prompt")
    if re.search(r"\b(?:sin|cos|tan|sec|cosec|cot)\b", lowered):
        flags.add("contains_trig_expression")
    if re.search(r"\b(?:ln|log|exp)\b|\be\s*(?:\^|[0-9]*x\b|[-+])", lowered):
        flags.add("contains_log_exponential_expression")

    structure_flags = set(extraction_quality_flags) | set(review_flags)
    if structure_flags & {
        "likely_needs_visual_review",
        "math_corruption_suspected",
        "broken_fraction_structure",
        "broken_superscript_or_power",
        "suspicious_symbol_run",
        "flattened_display_math",
        "diagram_text_mixed_with_body",
    }:
        flags.add("contains_math_text_corruption")
        flags.add("text_order_unreliable")
    if question_structure_detected.get("missing_internal_subparts") or question_structure_detected.get("impossible_subpart_sequence_detected"):
        flags.add("text_order_unreliable")

    return sorted(flags)


def derive_question_text_semantics(
    *,
    question_text: str,
    text_fidelity_status: str,
    visual_reason_flags: list[str],
) -> tuple[str, str, bool]:
    if not str(question_text or "").strip():
        return QuestionTextRole.MISSING, QuestionTextTrust.UNUSABLE, True

    flag_set = set(visual_reason_flags)
    visual_required = bool(
        flag_set
        & {
            "contains_equation_layout",
            "contains_fraction_or_integral_layout",
            "contains_vector_notation",
            "contains_complex_number_notation",
            "contains_inequality_or_region_prompt",
            "contains_graph_or_diagram_prompt",
            "contains_trig_expression",
            "contains_log_exponential_expression",
            "contains_math_text_corruption",
            "contains_pdf_control_garbage",
            "text_order_unreliable",
            "native_text_missing_or_sparse",
            "ocr_text_sparse_or_merged",
        }
    )

    corruption_flags = {
        "contains_math_text_corruption",
        "contains_pdf_control_garbage",
        "text_order_unreliable",
        "native_text_missing_or_sparse",
        "ocr_text_sparse_or_merged",
    }
    if text_fidelity_status == TextFidelityStatus.UNUSABLE or flag_set & corruption_flags:
        return QuestionTextRole.UNTRUSTED_MATH_TEXT, QuestionTextTrust.UNUSABLE if text_fidelity_status == TextFidelityStatus.UNUSABLE else QuestionTextTrust.LOW, True
    if text_fidelity_status == TextFidelityStatus.DEGRADED:
        return QuestionTextRole.SEARCH_HINT, QuestionTextTrust.LOW, True
    if visual_required:
        return QuestionTextRole.SEARCH_HINT, QuestionTextTrust.MEDIUM, True
    return QuestionTextRole.READABLE_TEXT, QuestionTextTrust.HIGH, False


def derive_visual_curation_status(
    *,
    validation_status: str,
    scope_quality_status: str,
    question_image_path: str,
    question_crop_confidence: str,
    mark_scheme_image_path: str,
    mark_scheme_crop_confidence: str,
) -> str:
    if not question_image_path:
        return CurationStatus.FAIL
    if validation_status == ValidationStatus.FAIL or scope_quality_status == ScopeQualityStatus.FAIL:
        return CurationStatus.FAIL
    if question_crop_confidence == CropConfidence.LOW or not mark_scheme_image_path:
        return CurationStatus.REVIEW
    if mark_scheme_crop_confidence and mark_scheme_crop_confidence != CropConfidence.HIGH:
        return CurationStatus.REVIEW
    if validation_status == ValidationStatus.REVIEW or scope_quality_status == ScopeQualityStatus.REVIEW:
        return CurationStatus.REVIEW
    return CurationStatus.READY


def derive_text_only_status(
    *,
    validation_status: str,
    scope_quality_status: str,
    question_text_role: str,
    question_text_trust: str,
) -> str:
    if validation_status == ValidationStatus.FAIL or scope_quality_status == ScopeQualityStatus.FAIL:
        return CurationStatus.FAIL
    if question_text_role in {QuestionTextRole.MISSING, QuestionTextRole.UNTRUSTED_MATH_TEXT}:
        return CurationStatus.FAIL
    if question_text_trust in {QuestionTextTrust.UNUSABLE, QuestionTextTrust.LOW}:
        return CurationStatus.FAIL
    if question_text_role == QuestionTextRole.SEARCH_HINT or question_text_trust == QuestionTextTrust.MEDIUM:
        return CurationStatus.REVIEW
    if validation_status == ValidationStatus.REVIEW or scope_quality_status == ScopeQualityStatus.REVIEW:
        return CurationStatus.REVIEW
    return CurationStatus.READY


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


def _normalize_for_visual_flags(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def _text_has_pdf_control_garbage(text: str) -> bool:
    raw = str(text or "")
    if any(char == "\ufffd" or ord(char) in {*range(0, 9), *range(14, 32)} for char in raw):
        return True
    if re.search(r"(?:\b\d{5,}\b\s*){2,}", raw):
        return True
    if re.search(r"[^\w\s.,;:()\[\]{}+\-*/=<>|^'\"%°]+", raw) and len(re.findall(r"[A-Za-z]", raw)) < 20:
        return True
    return False


def _text_has_page_furniture(text: str) -> bool:
    lowered = str(text or "").lower()
    furniture = [
        "do not write in this margin",
        "permission to reproduce items",
        "uc les",
        "cambridge university press",
        "this document has",
        "blank page",
        "additional page",
        "turn over",
        "© ucles",
    ]
    return any(item in lowered for item in furniture)


def _text_has_math_corruption(text: str) -> bool:
    normalized = _normalize_for_visual_flags(text)
    lowered = normalized.lower()
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[=+\-*/^<>≤≥|(){}\[\],]", normalized)
    if not tokens:
        return False

    if _text_has_corrupted_math_notation(normalized):
        return True
    if re.search(r"\b[xyzijkn]\s+[xyzijkn]\s+\d+\s+\d+\s*=", lowered):
        return True
    if re.search(r"(?:\b[xyzijk]\b\s*){3,}", lowered):
        return True
    if re.search(r"(?:[+\-*/=<>]\s*){3,}", normalized):
        return True
    if re.search(r"\b\d+\s+\d+\s+[A-Za-z]\s+[A-Za-z]\s*[-+<>=]", normalized):
        return True

    mathish = {token for token in tokens if re.fullmatch(r"[xyzijkedr]|sin|cos|tan|ln|log|exp|\d+(?:\.\d+)?|[=+\-*/^<>≤≥|]", token, re.I)}
    standalone_letter_count = sum(1 for token in tokens if re.fullmatch(r"[xyzijkedr]", token, re.I))
    operator_count = sum(1 for token in tokens if re.fullmatch(r"[=+\-*/^<>≤≥|]", token))
    if len(tokens) >= 8 and len(mathish) / len(tokens) >= 0.7 and standalone_letter_count >= 3 and operator_count >= 2:
        return True

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    short_math_lines = 0
    for line in lines:
        line_tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[=+\-*/^<>≤≥|]", line)
        if 4 <= len(line_tokens) <= 12:
            line_mathish = [token for token in line_tokens if re.fullmatch(r"[xyzijkedr]|sin|cos|tan|ln|log|exp|\d+(?:\.\d+)?|[=+\-*/^<>≤≥|]", token, re.I)]
            if len(line_mathish) / len(line_tokens) >= 0.75 and sum(1 for token in line_tokens if len(token) == 1) >= 4:
                short_math_lines += 1
    return short_math_lines >= 2


def _contains_equation_layout(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9)]\s*=\s*[-+A-Za-z0-9(|]", text) or re.search(r"\by\s*=", text, re.I))


def _contains_fraction_or_integral_layout(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\b(?:integral|differentiat|substitution)\b|∫", lowered)
        or re.search(r"\bd[xy]\b|d[xy]\s*/\s*d[xy]", lowered)
        or re.search(r"\b\d+\s*/\s*\d+\b|\\frac|over\s+\d", lowered)
        or "broken_fraction_structure" in lowered
    )


def _contains_vector_notation(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\bvector\b|\bposition vector\b|\bline\s+l\b", lowered)
        or re.search(r"\b[rijk]\b\s*[=+]\s*\d", lowered)
        or re.search(r"\([^)]+,\s*[^)]+,\s*[^)]+\)", text)
    )


def _contains_complex_number_notation(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\bargand\b|\bcomplex number\b|\barg\s*\(|\bmodulus\b", lowered) or re.search(r"\b[zw]\s*=\s*[^.\n]*\bi\b", lowered))


def _contains_inequality_or_region_prompt(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"[<>≤≥]", text) or re.search(r"\binequalit(?:y|ies)\b|\bregion\b|\bshade\b", lowered))


def _contains_graph_or_diagram_prompt(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(?:sketch|draw|graph|diagram|curve|asymptote|argand diagram)\b", lowered))
