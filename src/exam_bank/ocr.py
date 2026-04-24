from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def disabled_ocr_result() -> OCRResult:
    return OCRResult(
        ocr_ran=False,
        ocr_engine="",
        ocr_text="",
        ocr_text_trust=QuestionTextTrust.UNUSABLE,
        ocr_failure_reason="disabled",
        ocr_text_role=QuestionTextRole.MISSING,
    )


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
