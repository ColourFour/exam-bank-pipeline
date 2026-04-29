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
_PAGE_FURNITURE_RE = re.compile(
    r"\b(?:UCLES|Cambridge|BLANK PAGE|Additional Materials|READ THESE INSTRUCTIONS|INSTRUCTIONS|Question Paper|Mark Scheme)\b",
    re.IGNORECASE,
)
_BARCODE_OR_HEADER_RE = re.compile(r"(?:PUTT|RT TT|VARTA|ARTY|RACY|[A-Z]{4,}\s+[A-Z]{4,})")
_NEXT_QUESTION_RE = re.compile(r"\[\d{1,2}\]\s+(?:\d{1,2}\s+)?(?:Find|Show|Solve|Given|The diagram)", re.IGNORECASE)
_OCR_SYMBOL_GARBAGE_RE = re.compile(r"(?:[@?]{2,}|[~`\"|]{2,}|[A-Za-z]\?+[A-Za-z]|[0-9]\?+[0-9])")
_CONTROL_GARBAGE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")


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
) -> TextCandidateDecision:
    native = " ".join(str(native_text or "").replace("\u00a0", " ").split())
    ocr = " ".join(str(ocr_text or "").replace("\u00a0", " ").split())
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

    reasons = [
        f"native_score={native_score.score}",
        f"ocr_score={ocr_score.score}",
    ]
    rejected = list(ocr_score.rejection_reasons)
    if not ocr:
        rejected.append("empty_ocr_text")
    if scope_quality_status == "fail":
        rejected.append("scope_quality_failed")
    if expected_question_number and _contains_question_number(native, expected_question_number) and not _contains_question_number(ocr, expected_question_number):
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

    margin = ocr_score.score - native_score.score
    if rejected:
        reasons.append("ocr_rejected")
    elif margin >= 30:
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
