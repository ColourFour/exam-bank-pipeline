from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import AppConfig
from .trust import QuestionTextRole, QuestionTextTrust, derive_question_text_semantics, visual_reason_flags


OCR_ENGINE = "tesseract"


@dataclass(frozen=True)
class OCRResult:
    ocr_ran: bool
    ocr_engine: str
    ocr_text: str
    ocr_text_trust: str
    ocr_failure_reason: str = ""
    ocr_text_role: str = QuestionTextRole.MISSING


@dataclass(frozen=True)
class TextCandidateScore:
    source: str
    score: int
    reasons: list[str]
    rejection_reasons: list[str]

    @property
    def has_hard_rejection(self) -> bool:
        return bool(self.rejection_reasons)


@dataclass(frozen=True)
class TextCandidateDecision:
    selected_text: str
    text_candidate_source: str
    native_text_score: int
    ocr_text_score: int
    selected_text_score: int
    text_candidate_decision: str
    text_candidate_decision_reasons: list[str]
    ocr_selected: bool
    ocr_rejected_reasons: list[str]


_PROMPT_WORD_RE = re.compile(
    r"\b(find|show that|solve|given|the diagram shows|calculate|determine|prove|sketch|evaluate|hence|use)\b",
    re.IGNORECASE,
)
_SUBPART_RE = re.compile(r"\(([a-h]|i{1,3}|iv|v|vi{0,3}|ix|x)\)", re.IGNORECASE)
_MARK_RE = re.compile(r"\[\d{1,2}\]")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_MATH_FUNCTION_RE = re.compile(r"\b(?:sin|cos|tan|ln|log)\s*[A-Za-z0-9(]")
_MATH_STRUCTURE_RE = re.compile(
    r"(?:\\frac|∫|√|≤|≥|[<>=]|\b(?:sin|cos|tan|sec|cosec|cot|ln|log|arg|vector|matrix|modulus|complex)\b)",
    re.IGNORECASE,
)
_LARGE_SAFE_OCR_MARGIN = 20
_SMALL_SAFE_OCR_MARGIN = 0
_PAGE_FURNITURE_RE = re.compile(
    r"\b(?:UCLES|Cambridge|BLANK PAGE|Additional Materials|READ THESE INSTRUCTIONS|INSTRUCTIONS|Question Paper|Mark Scheme)\b",
    re.IGNORECASE,
)
_BARCODE_OR_HEADER_RE = re.compile(r"(?:PUTT|RT TT|VARTA|ARTY|RACY|[A-Z]{4,}\s+[A-Z]{4,})")
_NEXT_QUESTION_RE = re.compile(r"\[\d{1,2}\]\s+(?:\d{1,2}\s+)?(?:Find|Show|Solve|Given|The diagram)", re.IGNORECASE)
_OCR_SYMBOL_GARBAGE_RE = re.compile(r"(?:[@?]{2,}|[~`\"|]{2,}|[A-Za-z]\?+[A-Za-z]|[0-9]\?+[0-9])")
_CONTROL_GARBAGE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
_MERGED_WORD_ARTIFACT_RE = re.compile(
    r"\b(?:"
    r"Thediagram|thediagram|thegraph|formgraph|straightof|andshows|shows[A-Z]|whichg|consistswhich|"
    r"thethe|twostraight|ofapplying|sequenceand|Thetwo|Foranother|Usean|Findthe|findthe|showthat"
    r")\b"
)


def disabled_ocr_result() -> OCRResult:
    return OCRResult(
        ocr_ran=False,
        ocr_engine="",
        ocr_text="",
        ocr_text_trust=QuestionTextTrust.UNUSABLE,
        ocr_failure_reason="disabled",
        ocr_text_role=QuestionTextRole.MISSING,
    )


def score_text_candidate(
    text: str,
    *,
    source: str,
    expected_question_number: str = "",
    expected_subparts: list[str] | None = None,
    expected_mark_count: int | None = None,
) -> TextCandidateScore:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split())
    expected_subparts = expected_subparts or []
    score = 0
    reasons: list[str] = []
    rejection_reasons: list[str] = []

    words = _WORD_RE.findall(normalized)
    word_count = len(words)
    mark_count = len(_MARK_RE.findall(normalized))
    prompt_count = len(_PROMPT_WORD_RE.findall(normalized))
    control_count = len(_CONTROL_GARBAGE_RE.findall(normalized))
    subparts = {label.lower() for label in _SUBPART_RE.findall(normalized)}

    if not normalized:
        return TextCandidateScore(source=source, score=-100, reasons=["empty_text"], rejection_reasons=["empty_text"])

    if expected_question_number and _contains_question_number(normalized, expected_question_number):
        score += 10
        reasons.append("expected_question_number_present")
    elif expected_question_number:
        score -= 8
        reasons.append("missing_expected_question_number")

    if expected_subparts:
        missing_subparts = [label for label in expected_subparts if label.lower() not in subparts]
        present_count = len(expected_subparts) - len(missing_subparts)
        score += present_count * 5
        if present_count:
            reasons.append("expected_subparts_present")
        if missing_subparts:
            score -= len(missing_subparts) * 5
            reasons.append("missing_expected_subparts")

    if expected_mark_count is not None:
        if mark_count >= expected_mark_count:
            score += min(mark_count, expected_mark_count) * 4
            reasons.append("expected_mark_brackets_present")
        else:
            score -= (expected_mark_count - mark_count) * 6
            reasons.append("missing_expected_mark_brackets")
    elif mark_count:
        score += min(mark_count, 6) * 3
        reasons.append("mark_brackets_present")

    if 8 <= word_count <= 240:
        score += min(word_count, 80)
        reasons.append("reasonable_text_length")
    elif word_count < 8:
        score -= 18
        reasons.append("suspiciously_short_text")
    else:
        score -= min(30, (word_count - 240) // 8)
        reasons.append("suspiciously_long_text")

    if prompt_count:
        score += min(prompt_count, 6) * 5
        reasons.append("prompt_words_present")

    if _looks_like_spaced_prose(normalized):
        score += 12
        reasons.append("readable_prose_spacing")
    else:
        score -= 8
        reasons.append("merged_or_sparse_prose")

    merged_artifact_count = len(_MERGED_WORD_ARTIFACT_RE.findall(normalized))
    if merged_artifact_count:
        score -= min(40, merged_artifact_count * 10)
        reasons.append("merged_word_artifacts")

    math_function_count = len(_MATH_FUNCTION_RE.findall(normalized))
    if math_function_count:
        score += min(math_function_count, 4) * 2
        reasons.append("math_function_tokens_present")

    if control_count:
        score -= control_count * 12
        reasons.append("pdf_control_or_replacement_garbage")

    if _PAGE_FURNITURE_RE.search(normalized) or _BARCODE_OR_HEADER_RE.search(normalized):
        score -= 35
        reasons.append("page_furniture_or_header_text")
        rejection_reasons.append("page_furniture_or_header_text")

    if _has_next_question_contamination(normalized, expected_question_number):
        score -= 35
        reasons.append("next_question_contamination")
        rejection_reasons.append("next_question_contamination")

    symbol_garbage_count = len(_OCR_SYMBOL_GARBAGE_RE.findall(normalized))
    if symbol_garbage_count:
        score -= symbol_garbage_count * 8
        reasons.append("ocr_symbol_garbage")

    isolated_symbol_count = len(re.findall(r"(?:^|\s)[^\w\s()\[\].,;:=+\-*/^]{1,2}(?=\s|$)", normalized))
    if isolated_symbol_count > max(6, word_count // 6):
        score -= 18
        reasons.append("excessive_isolated_symbols")

    if _diagram_labels_overwhelm_prompt(normalized):
        score -= 15
        reasons.append("diagram_labels_overwhelm_prompt")

    return TextCandidateScore(
        source=source,
        score=score,
        reasons=sorted(set(reasons)),
        rejection_reasons=sorted(set(rejection_reasons)),
    )


def select_text_candidate(
    *,
    native_text: str,
    ocr_text: str,
    expected_question_number: str = "",
    expected_subparts: list[str] | None = None,
    scope_quality_status: str = "",
    mapping_status: str = "",
    validation_status: str = "",
) -> TextCandidateDecision:
    native = " ".join(str(native_text or "").replace("\u00a0", " ").split())
    ocr = " ".join(str(ocr_text or "").replace("\u00a0", " ").split())
    scope_status = str(scope_quality_status or "").lower()
    mapping_status_normalized = str(mapping_status or "").lower()
    validation_status_normalized = str(validation_status or "").lower()
    expected_mark_count = max(len(_MARK_RE.findall(native)), 0) or None
    native_score = score_text_candidate(
        native,
        source="native",
        expected_question_number=expected_question_number,
        expected_subparts=expected_subparts,
        expected_mark_count=expected_mark_count,
    )
    ocr_score = score_text_candidate(
        ocr,
        source="ocr",
        expected_question_number=expected_question_number,
        expected_subparts=expected_subparts,
        expected_mark_count=expected_mark_count,
    )
    native_visual_flags = set(
        visual_reason_flags(
            question_text=native,
            extraction_quality_flags=[],
            review_flags=[],
            question_structure_detected={},
            text_source_profile="native_pdf",
        )
    )
    ocr_visual_flags = set(
        visual_reason_flags(
            question_text=ocr,
            extraction_quality_flags=[],
            review_flags=["ocr_question_text"],
            question_structure_detected={},
            text_source_profile="ocr",
        )
    )

    reasons = [
        f"native_score={native_score.score}",
        f"ocr_score={ocr_score.score}",
    ]
    rejected = list(ocr_score.rejection_reasons)
    if not ocr:
        rejected.append("empty_ocr_text")
    if scope_status == "fail" and ocr_score.rejection_reasons:
        rejected.append("scope_quality_failed")
    if scope_status != "fail":
        if mapping_status_normalized and mapping_status_normalized != "pass":
            rejected.append("ocr_mapping_status_not_pass")
        if validation_status_normalized and validation_status_normalized != "pass":
            rejected.append("ocr_validation_status_not_pass")
    if expected_question_number and _contains_question_number(native, expected_question_number) and not _contains_question_number(ocr, expected_question_number):
        if scope_status == "fail" and _ocr_missing_question_number_is_tolerable(ocr, expected_mark_count, expected_subparts):
            reasons.append("ocr_missing_question_number_tolerated")
        elif _can_recover_missing_ocr_question_number(
            ocr,
            expected_question_number=expected_question_number,
            expected_mark_count=expected_mark_count,
            expected_subparts=expected_subparts or [],
            scope_status=scope_status,
            mapping_status=mapping_status_normalized,
            validation_status=validation_status_normalized,
            native_visual_flags=native_visual_flags,
            ocr_visual_flags=ocr_visual_flags,
        ):
            ocr = f"{expected_question_number} {ocr}".strip()
            ocr_score = score_text_candidate(
                ocr,
                source="ocr",
                expected_question_number=expected_question_number,
                expected_subparts=expected_subparts,
                expected_mark_count=expected_mark_count,
            )
            ocr_visual_flags = set(
                visual_reason_flags(
                    question_text=ocr,
                    extraction_quality_flags=[],
                    review_flags=["ocr_question_text"],
                    question_structure_detected={},
                    text_source_profile="ocr",
                )
            )
            reasons[1] = f"ocr_score={ocr_score.score}"
            reasons.append("ocr_question_number_recovered_from_detector_anchor")
        else:
            rejected.append("ocr_missing_question_number")
    if expected_subparts:
        native_subparts = {label.lower() for label in _SUBPART_RE.findall(native)}
        ocr_subparts = {label.lower() for label in _SUBPART_RE.findall(ocr)}
        required = {label.lower() for label in expected_subparts if label.lower() in native_subparts}
        if required and not required.issubset(ocr_subparts):
            rejected.append("ocr_missing_subpart_labels")
    native_mark_count = len(_MARK_RE.findall(native))
    ocr_mark_count = len(_MARK_RE.findall(ocr))
    if native_mark_count >= 2 and ocr_mark_count < native_mark_count:
        rejected.append("ocr_lost_mark_brackets")
    if _ocr_lost_math_structure(native, ocr):
        rejected.append("ocr_lost_math_structure")
    if _ocr_lost_visual_dependency_prompt(native, ocr):
        rejected.append("ocr_lost_visual_dependency_prompt")
    if _ocr_lost_function_structure(native, ocr):
        rejected.append("ocr_lost_function_structure")
    if _ocr_lost_greek_symbol(native, ocr):
        rejected.append("ocr_lost_greek_symbol")
    if _ocr_lost_radical_structure(native, ocr):
        rejected.append("ocr_lost_radical_structure")
    if _ocr_lost_unit_structure(native, ocr):
        rejected.append("ocr_lost_unit_structure")

    if "contains_unit_corruption" in ocr_visual_flags:
        rejected.append("ocr_introduced_unit_corruption")
    if "contains_symbol_loss" in ocr_visual_flags:
        rejected.append("ocr_introduced_symbol_loss")
    if "contains_native_compacted_math_corruption" in ocr_visual_flags:
        rejected.append("ocr_introduced_compacted_math_corruption")
    if "contains_flattened_math_structure" in ocr_visual_flags and "contains_flattened_math_structure" not in native_visual_flags:
        rejected.append("ocr_introduced_flattened_math_structure")

    margin = ocr_score.score - native_score.score
    if rejected:
        if margin >= _LARGE_SAFE_OCR_MARGIN:
            rejected.append("ocr_large_margin_blocked_by_structural_rejection")
            reasons.append("ocr_large_margin_blocked_by_structural_rejection")
        reasons.append("ocr_rejected")
    elif margin >= _LARGE_SAFE_OCR_MARGIN:
        reasons.append("ocr_score_clear_margin")
        return TextCandidateDecision(
            selected_text=ocr,
            text_candidate_source="ocr",
            native_text_score=native_score.score,
            ocr_text_score=ocr_score.score,
            selected_text_score=ocr_score.score,
            text_candidate_decision="ocr_selected",
            text_candidate_decision_reasons=sorted(set(reasons + ocr_score.reasons)),
            ocr_selected=True,
            ocr_rejected_reasons=[],
        )
    elif margin >= _SMALL_SAFE_OCR_MARGIN and _native_text_has_known_corruption(native, native_score, native_visual_flags) and _ocr_text_is_structurally_clean(
        ocr, ocr_score, ocr_visual_flags
    ) and _small_margin_ocr_repair_is_safe(
        native_visual_flags,
        ocr_visual_flags,
    ):
        reasons.append("ocr_selected_to_repair_native_text")
        return TextCandidateDecision(
            selected_text=ocr,
            text_candidate_source="ocr",
            native_text_score=native_score.score,
            ocr_text_score=ocr_score.score,
            selected_text_score=ocr_score.score,
            text_candidate_decision="ocr_selected",
            text_candidate_decision_reasons=sorted(set(reasons + ocr_score.reasons)),
            ocr_selected=True,
            ocr_rejected_reasons=[],
        )
    else:
        rejected.append("ocr_not_clearly_better")
        reasons.append("native_retained")

    return TextCandidateDecision(
        selected_text=native,
        text_candidate_source="native",
        native_text_score=native_score.score,
        ocr_text_score=ocr_score.score,
        selected_text_score=native_score.score,
        text_candidate_decision="native_retained",
        text_candidate_decision_reasons=sorted(set(reasons + native_score.reasons)),
        ocr_selected=False,
        ocr_rejected_reasons=sorted(set(rejected)),
    )


def _ocr_missing_question_number_is_tolerable(
    ocr_text: str,
    expected_mark_count: int | None,
    expected_subparts: list[str],
) -> bool:
    normalized = " ".join(str(ocr_text or "").replace("\u00a0", " ").split())
    if len(_WORD_RE.findall(normalized)) < 8:
        return False
    if not _PROMPT_WORD_RE.search(normalized):
        return False
    if expected_mark_count is not None and len(_MARK_RE.findall(normalized)) < min(1, expected_mark_count):
        return False
    if expected_subparts:
        ocr_subparts = {label.lower() for label in _SUBPART_RE.findall(normalized)}
        required = {label.lower() for label in expected_subparts}
        if required and not required.issubset(ocr_subparts):
            return False
    return True


def _can_recover_missing_ocr_question_number(
    ocr_text: str,
    *,
    expected_question_number: str,
    expected_mark_count: int | None,
    expected_subparts: list[str],
    scope_status: str,
    mapping_status: str,
    validation_status: str,
    native_visual_flags: set[str],
    ocr_visual_flags: set[str],
) -> bool:
    if not expected_question_number or scope_status not in {"clean", "review"}:
        return False
    if mapping_status != "pass" or validation_status != "pass":
        return False
    if _PAGE_FURNITURE_RE.search(ocr_text) or _BARCODE_OR_HEADER_RE.search(ocr_text):
        return False
    if _has_next_question_contamination(ocr_text, expected_question_number):
        return False
    if not _ocr_missing_question_number_is_tolerable(ocr_text, expected_mark_count, expected_subparts):
        return False
    if not _ocr_subpart_order_is_clean(ocr_text, expected_subparts):
        return False
    if not _small_margin_ocr_repair_is_safe(native_visual_flags, ocr_visual_flags):
        return False
    return True


def _ocr_subpart_order_is_clean(ocr_text: str, expected_subparts: list[str]) -> bool:
    if not expected_subparts:
        return True
    normalized = " ".join(str(ocr_text or "").replace("\u00a0", " ").split())
    labels = [label.lower() for label in _SUBPART_RE.findall(normalized)]
    expected = [label.lower() for label in expected_subparts]
    if not labels:
        return False
    first_content = re.match(r"^(?:\s*\(([a-h]|i{1,3}|iv|v|vi{0,3}|ix|x)\)\s*)+", normalized, re.IGNORECASE)
    if first_content and labels[0] != expected[0]:
        return False
    expected_positions = [labels.index(label) for label in expected if label in labels]
    return expected_positions == sorted(expected_positions)


def _native_text_has_known_corruption(
    native_text: str,
    native_score: TextCandidateScore,
    native_visual_flags: set[str],
) -> bool:
    if native_visual_flags & {
        "contains_pdf_control_garbage",
        "contains_native_compacted_math_corruption",
        "contains_math_text_corruption",
        "contains_symbol_loss",
        "contains_unit_corruption",
    }:
        return True
    if set(native_score.reasons) & {"pdf_control_or_replacement_garbage", "merged_word_artifacts"}:
        return True
    return _has_long_joined_prose_artifact(native_text)


def _ocr_text_is_structurally_clean(
    ocr_text: str,
    ocr_score: TextCandidateScore,
    ocr_visual_flags: set[str],
) -> bool:
    if ocr_score.has_hard_rejection:
        return False
    if not ocr_text or ocr_score.score < 40:
        return False
    if ocr_visual_flags & {
        "contains_pdf_control_garbage",
        "contains_unit_corruption",
        "contains_symbol_loss",
        "contains_native_compacted_math_corruption",
        "contains_math_text_corruption",
        "text_order_unreliable",
        "contains_page_furniture",
    }:
        return False
    return True


def _small_margin_ocr_repair_is_safe(
    native_visual_flags: set[str],
    ocr_visual_flags: set[str],
) -> bool:
    native_risk_flags = {
        "contains_complex_number_notation",
        "contains_equation_layout",
        "contains_flattened_math_structure",
        "contains_fraction_or_integral_layout",
        "contains_graph_or_diagram_prompt",
        "contains_inequality_or_region_prompt",
        "contains_table_or_data_prompt",
        "contains_vector_notation",
    }
    ocr_risk_flags = native_risk_flags | {
        "contains_log_exponential_expression",
        "contains_trig_expression",
    }
    return not bool(native_visual_flags & native_risk_flags) and not bool(ocr_visual_flags & ocr_risk_flags)


def _has_long_joined_prose_artifact(text: str) -> bool:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split())
    joined_hints = {
        "arrangement",
        "calculate",
        "coefficient",
        "diagram",
        "different",
        "division",
        "equation",
        "expression",
        "friction",
        "members",
        "number",
        "probability",
        "random",
        "resistance",
        "transform",
    }
    for token in re.findall(r"[A-Za-z]{14,}", normalized):
        lowered = token.lower()
        if sum(1 for hint in joined_hints if hint in lowered) >= 1 and re.search(
            r"(?:the|that|and|of|in|to|for|with|which|find|given|are|is|has)",
            lowered,
        ):
            return True
    return False


def _ocr_lost_math_structure(native_text: str, ocr_text: str) -> bool:
    if _has_missing_math_operand(ocr_text):
        return True
    native_count = len(_MATH_STRUCTURE_RE.findall(native_text))
    if native_count < 3:
        return False
    ocr_count = len(_MATH_STRUCTURE_RE.findall(ocr_text))
    return ocr_count < max(1, native_count // 2)


def _has_missing_math_operand(text: str) -> bool:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split())
    return bool(
        re.search(r"=\s*[+\-*/]\s*(?:and|or|[.;,)\]]|$)", normalized, re.IGNORECASE)
        or re.search(r"(?:^|\s)[+\-*/]\s*(?:and|or|[.;,)\]]|$)", normalized, re.IGNORECASE)
    )


def _ocr_lost_visual_dependency_prompt(native_text: str, ocr_text: str) -> bool:
    native_flags = set(
        visual_reason_flags(
            question_text=native_text,
            extraction_quality_flags=[],
            review_flags=[],
            question_structure_detected={},
            text_source_profile="native_pdf",
        )
    )
    if not native_flags & {"contains_graph_or_diagram_prompt", "contains_table_or_data_prompt"}:
        return False
    ocr_flags = set(
        visual_reason_flags(
            question_text=ocr_text,
            extraction_quality_flags=[],
            review_flags=["ocr_question_text"],
            question_structure_detected={},
            text_source_profile="ocr",
        )
    )
    return not bool(ocr_flags & {"contains_graph_or_diagram_prompt", "contains_table_or_data_prompt"})


def _ocr_lost_function_structure(native_text: str, ocr_text: str) -> bool:
    return _function_structure_count(native_text) > 0 and _function_structure_count(ocr_text) == 0


def _function_structure_count(text: str) -> int:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split())
    return len(
        re.findall(
            r"\b(?:[fgh]\s*\(\s*x\s*\)|[fgh]\s*:\s*x|(?:sin|cos|tan|sec|cosec|cot|ln|log)\s*(?:\^\{?-?1\}?|\(|θ|[xyi1](?![A-Za-z])|[0-9]+(?:[A-Za-zθ])?))",
            normalized,
            re.IGNORECASE,
        )
    )


def _ocr_lost_greek_symbol(native_text: str, ocr_text: str) -> bool:
    return bool(_greek_symbols(native_text)) and not _greek_symbols(ocr_text)


def _greek_symbols(text: str) -> set[str]:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split()).lower()
    symbols: set[str] = set()
    for symbol, names in {
        "theta": ("θ", "theta"),
        "pi": ("π", " pi "),
        "sigma": ("σ", "sigma"),
        "alpha": ("α", "alpha"),
        "lambda": ("λ", "lambda"),
        "mu": ("μ", " mu "),
    }.items():
        if any(name in f" {normalized} " for name in names):
            symbols.add(symbol)
    return symbols


def _ocr_lost_radical_structure(native_text: str, ocr_text: str) -> bool:
    return _radical_signal_count(native_text) > 0 and _radical_signal_count(ocr_text) == 0


def _radical_signal_count(text: str) -> int:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split())
    return len(re.findall(r"√|\\sqrt|\bsqrt\b|\bsurd\b", normalized, re.IGNORECASE))


def _ocr_lost_unit_structure(native_text: str, ocr_text: str) -> bool:
    native_count = _valid_unit_signal_count(native_text)
    if native_count == 0:
        return False
    ocr_count = _valid_unit_signal_count(ocr_text)
    return ocr_count < max(1, native_count // 2)


def _valid_unit_signal_count(text: str) -> int:
    normalized = " ".join(str(text or "").replace("\u00a0", " ").split())
    unit_re = re.compile(
        r"(?<![A-Za-z0-9])(?:"
        r"kg|J|kW|"
        r"m\s*s\^\s*-?\d|m\s*s\^\{-?\d\}|m\s*s\^\{-?\}\s*\d|m\s*s\^-?\d|"
        r"ms\^-?\d|ms\^\{-?\d\}|ms\^\{-?\}\s*\d|m/s(?:\^?-?\d)?"
        r")(?=$|[^A-Za-z0-9])",
        re.IGNORECASE,
    )
    force_unit_re = re.compile(r"\b\d+(?:\.\d+)?\s*N\b")
    return len(unit_re.findall(normalized)) + len(force_unit_re.findall(normalized))


def _contains_question_number(text: str, question_number: str) -> bool:
    if not question_number:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(str(question_number))}(?:\s|\.|\(|$)", text))


def _looks_like_spaced_prose(text: str) -> bool:
    words = _WORD_RE.findall(text)
    if len(words) < 8:
        return False
    long_tokens = [word for word in words if len(word) >= 24]
    if len(long_tokens) >= 2:
        return False
    spaces = text.count(" ")
    letters = sum(1 for char in text if char.isalpha())
    return letters == 0 or spaces / max(letters, 1) >= 0.12


def _has_next_question_contamination(text: str, expected_question_number: str) -> bool:
    if not expected_question_number:
        return bool(_NEXT_QUESTION_RE.search(text))
    try:
        next_question = str(int(expected_question_number) + 1)
    except ValueError:
        return bool(_NEXT_QUESTION_RE.search(text))
    return bool(re.search(rf"\[\d{{1,2}}\]\s+{re.escape(next_question)}\s+(?:Find|Show|Solve|Given|The diagram)", text, re.IGNORECASE))


def _diagram_labels_overwhelm_prompt(text: str) -> bool:
    prefix = text[:80]
    label_count = len(re.findall(r"(?:^|\s)[A-Z](?=\s|$)", prefix))
    return label_count >= 5 and not _PROMPT_WORD_RE.search(prefix)


def run_question_crop_ocr(image_path: str | Path, config: AppConfig) -> OCRResult:
    if not config.ocr.enabled:
        return disabled_ocr_result()

    try:
        text = _tesseract_image_to_string(Path(image_path), config)
    except Exception as exc:
        return OCRResult(
            ocr_ran=True,
            ocr_engine=OCR_ENGINE,
            ocr_text="",
            ocr_text_trust=QuestionTextTrust.UNUSABLE,
            ocr_failure_reason=f"{exc.__class__.__name__}: {exc}",
            ocr_text_role=QuestionTextRole.MISSING,
        )

    normalized_text = " ".join(text.replace("\u00a0", " ").split())
    role, trust, _visual_required = derive_question_text_semantics(
        question_text=normalized_text,
        text_fidelity_status="clean",
        visual_reason_flags=visual_reason_flags(
            question_text=normalized_text,
            extraction_quality_flags=[],
            review_flags=["ocr_question_text"],
            question_structure_detected={},
            text_source_profile="ocr",
        ),
    )
    return OCRResult(
        ocr_ran=True,
        ocr_engine=OCR_ENGINE,
        ocr_text=normalized_text,
        ocr_text_trust=trust,
        ocr_failure_reason="" if normalized_text else "empty_ocr_text",
        ocr_text_role=role,
    )


def _tesseract_image_to_string(image_path: Path, config: AppConfig) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow are required for OCR. Install pytesseract and Tesseract.") from exc

    if config.ocr.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_cmd

    with Image.open(image_path) as image:
        kwargs = {
            "lang": config.ocr.language,
            "timeout": config.ocr.timeout_seconds,
        }
        return str(pytesseract.image_to_string(image, **kwargs))
