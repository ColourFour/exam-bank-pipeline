from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


@dataclass(frozen=True)
class OCRProfile:
    name: str
    description: str
    preprocessing: str
    tesseract_config: str


@dataclass(frozen=True)
class OCRProfileRun:
    profile: str
    text: str
    runtime_seconds: float
    ok: bool
    error: str = ""


def available_ocr_profiles() -> list[OCRProfile]:
    return [
        OCRProfile(
            name="baseline_current",
            description="Current frozen fixture selected text; no OCR is re-run.",
            preprocessing="fixture_selected_text",
            tesseract_config="not_applicable",
        ),
        OCRProfile(
            name="grayscale_threshold",
            description="Grayscale, autocontrast, binary threshold for noisy crops.",
            preprocessing="grayscale_autocontrast_threshold",
            tesseract_config="--psm 6",
        ),
        OCRProfile(
            name="formula_heavy",
            description="High-contrast sharpened crop with math-heavy single-block OCR.",
            preprocessing="grayscale_autocontrast_sharpen",
            tesseract_config="--psm 6 -c preserve_interword_spaces=1",
        ),
        OCRProfile(
            name="table_preserving",
            description="Block-oriented OCR with preserved spaces for tables and probability layouts.",
            preprocessing="grayscale_autocontrast",
            tesseract_config="--psm 4 -c preserve_interword_spaces=1",
        ),
        OCRProfile(
            name="diagram_safe",
            description="Light preprocessing to avoid destroying diagram labels and geometry-adjacent text.",
            preprocessing="grayscale_autocontrast_light",
            tesseract_config="--psm 6",
        ),
        OCRProfile(
            name="padding_variant",
            description="Adds white padding before OCR to recover clipped anchors and mark brackets.",
            preprocessing="grayscale_autocontrast_pad",
            tesseract_config="--psm 6",
        ),
        OCRProfile(
            name="dense_algebra",
            description="Upscaled, high-contrast profile for compact algebra and calculus expressions.",
            preprocessing="grayscale_autocontrast_upscale_sharpen_threshold",
            tesseract_config="--psm 6 -c preserve_interword_spaces=1",
        ),
    ]


def run_profile_ocr(
    image_path: Path,
    profile: OCRProfile,
    *,
    language: str = "eng",
    timeout_seconds: int = 30,
    image_loader: Callable[[Path], Any] | None = None,
    tesseract_runner: Callable[..., str] | None = None,
) -> OCRProfileRun:
    if profile.preprocessing == "fixture_selected_text":
        raise ValueError("baseline_current uses fixture text and must be handled by the caller.")

    start = perf_counter()
    try:
        image, pytesseract = _load_dependencies(image_path, image_loader=image_loader)
        if tesseract_runner is None:
            tesseract_runner = pytesseract.image_to_string
        prepared = preprocess_image(image, profile.preprocessing)
        raw_text = tesseract_runner(
            prepared,
            lang=language,
            timeout=timeout_seconds,
            config=profile.tesseract_config,
        )
        return OCRProfileRun(
            profile=profile.name,
            text=normalize_ocr_output(str(raw_text)),
            runtime_seconds=perf_counter() - start,
            ok=True,
        )
    except Exception as exc:
        return OCRProfileRun(
            profile=profile.name,
            text="",
            runtime_seconds=perf_counter() - start,
            ok=False,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def preprocess_image(image: Any, preprocessing: str) -> Any:
    from PIL import Image, ImageFilter, ImageOps

    if preprocessing == "grayscale_autocontrast_threshold":
        prepared = ImageOps.autocontrast(image.convert("L"))
        return prepared.point(lambda pixel: 255 if pixel > 180 else 0)
    if preprocessing == "grayscale_autocontrast_sharpen":
        return ImageOps.autocontrast(image.convert("L")).filter(ImageFilter.SHARPEN)
    if preprocessing == "grayscale_autocontrast":
        return ImageOps.autocontrast(image.convert("L"))
    if preprocessing == "grayscale_autocontrast_light":
        return ImageOps.autocontrast(image.convert("L"), cutoff=1)
    if preprocessing == "grayscale_autocontrast_pad":
        prepared = ImageOps.autocontrast(image.convert("L"))
        return ImageOps.expand(prepared, border=24, fill=255)
    if preprocessing == "grayscale_autocontrast_upscale_sharpen_threshold":
        prepared = ImageOps.autocontrast(image.convert("L"))
        width, height = prepared.size
        prepared = prepared.resize((width * 2, height * 2), resample=Image.Resampling.LANCZOS)
        prepared = prepared.filter(ImageFilter.SHARPEN)
        return prepared.point(lambda pixel: 255 if pixel > 165 else 0)
    raise ValueError(f"Unknown OCR preprocessing profile: {preprocessing}")


def normalize_ocr_output(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def _load_dependencies(image_path: Path, *, image_loader: Callable[[Path], Any] | None) -> tuple[Any, Any]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pytesseract and Pillow are required for OCR profile experiments.") from exc

    if image_loader is not None:
        return image_loader(image_path), pytesseract
    with Image.open(image_path) as image:
        return image.copy(), pytesseract
