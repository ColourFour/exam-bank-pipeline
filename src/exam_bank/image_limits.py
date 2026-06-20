from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any


SAFE_RENDER_PIXELS = 40_000_000
SAFE_PROBE_PIXELS = 80_000_000
WHITESPACE_TRIM_NONWHITE_THRESHOLD = 245
WHITESPACE_TRIM_MAX_BLANK_MARGIN_RATIO = 0.75
WHITESPACE_TRIM_MIN_DIMENSION_PX = 120
WHITESPACE_TRIM_PADDING_PX = 24
EDGE_FURNITURE_MAX_BAND_HEIGHT_RATIO = 0.08
EDGE_FURNITURE_MAX_BAND_WIDTH_RATIO = 0.18
EDGE_FURNITURE_MIN_GAP_PX = 28
EDGE_FURNITURE_CENTER_TOLERANCE_RATIO = 0.12


def render_pdf_area(
    page: Any,
    fitz: Any,
    *,
    dpi: int,
    source_file: str | Path,
    page_number: int,
    context: str,
    clip: Any | None = None,
    max_pixels: int = SAFE_RENDER_PIXELS,
) -> tuple[Any, float]:
    """Render a PDF page or clip while capping oversized rasters."""

    rect = clip if clip is not None else page.rect
    width_pt = max(1.0, float(rect.width))
    height_pt = max(1.0, float(rect.height))
    requested_zoom = dpi / 72
    zoom = _safe_zoom(width_pt, height_pt, requested_zoom, max_pixels)
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    rendered_pixels = int(pix.width) * int(pix.height)
    if zoom < requested_zoom or rendered_pixels > max_pixels:
        logging.warning(
            "large_render_capped source=%s page=%s context=%s rendered=%sx%s requested_dpi=%s used_dpi=%.1f",
            source_file,
            page_number,
            context,
            pix.width,
            pix.height,
            dpi,
            zoom * 72,
        )
    return pixmap_to_image(pix), zoom


def pixmap_to_image(pix: Any) -> Any:
    """Convert a PyMuPDF pixmap without round-tripping through PNG decoding."""

    from PIL import Image

    size = (int(pix.width), int(pix.height))
    if pix.alpha:
        return Image.frombytes("RGBA", size, pix.samples).convert("RGB")
    if pix.n == 1:
        return Image.frombytes("L", size, pix.samples).convert("RGB")
    if pix.n == 3:
        return Image.frombytes("RGB", size, pix.samples)
    if pix.n == 4:
        return Image.frombytes("CMYK", size, pix.samples).convert("RGB")
    return Image.frombytes("RGB", size, pix.samples)


def cap_image_pixels(
    image: Any,
    *,
    source_file: str | Path,
    context: str,
    max_pixels: int = SAFE_RENDER_PIXELS,
) -> Any:
    pixels = int(image.width) * int(image.height)
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    logging.warning(
        "large_output_image_downscaled source=%s context=%s original=%sx%s output=%sx%s",
        source_file,
        context,
        image.width,
        image.height,
        new_size[0],
        new_size[1],
    )
    from PIL import Image

    return image.resize(new_size, Image.Resampling.LANCZOS)


def trim_excess_render_whitespace(
    image: Any,
    *,
    padding_px: int = WHITESPACE_TRIM_PADDING_PX,
    nonwhite_threshold: int = WHITESPACE_TRIM_NONWHITE_THRESHOLD,
    max_blank_margin_ratio: float = WHITESPACE_TRIM_MAX_BLANK_MARGIN_RATIO,
    min_dimension_px: int = WHITESPACE_TRIM_MIN_DIMENSION_PX,
) -> Any:
    """Trim extreme blank top/bottom margins from a rendered crop."""

    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    if width < min_dimension_px or height < min_dimension_px:
        return image

    grayscale = image.convert("L")
    mask = grayscale.point(lambda pixel: 255 if pixel < nonwhite_threshold else 0, mode="1")
    bbox = mask.getbbox()
    if bbox is None:
        return image

    _left, top, _right, bottom = bbox
    blank_top_ratio = top / height
    blank_bottom_ratio = (height - bottom) / height
    if blank_top_ratio < max_blank_margin_ratio and blank_bottom_ratio < max_blank_margin_ratio:
        return image

    trim_top = max(0, top - padding_px) if blank_top_ratio >= max_blank_margin_ratio else 0
    trim_bottom = min(height, bottom + padding_px) if blank_bottom_ratio >= max_blank_margin_ratio else height
    if trim_bottom <= trim_top:
        return image
    if trim_top == 0 and trim_bottom == height:
        return image
    return image.crop((0, trim_top, width, trim_bottom))


def clean_rendered_crop_image(image: Any) -> Any:
    return trim_excess_render_whitespace(trim_isolated_edge_furniture(image))


def trim_isolated_edge_furniture(
    image: Any,
    *,
    nonwhite_threshold: int = WHITESPACE_TRIM_NONWHITE_THRESHOLD,
    min_dimension_px: int = WHITESPACE_TRIM_MIN_DIMENSION_PX,
    max_band_height_ratio: float = EDGE_FURNITURE_MAX_BAND_HEIGHT_RATIO,
    max_band_width_ratio: float = EDGE_FURNITURE_MAX_BAND_WIDTH_RATIO,
    min_gap_px: int = EDGE_FURNITURE_MIN_GAP_PX,
    center_tolerance_ratio: float = EDGE_FURNITURE_CENTER_TOLERANCE_RATIO,
) -> Any:
    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    if width < min_dimension_px or height < min_dimension_px:
        return image

    mask = image.convert("L").point(lambda pixel: 255 if pixel < nonwhite_threshold else 0, mode="1")
    bbox = mask.getbbox()
    if bbox is None:
        return image

    bands = _nonwhite_row_bands(mask)
    if len(bands) < 2:
        return image

    crop_top = 0
    crop_bottom = height
    first = bands[0]
    second = bands[1]
    if _is_isolated_edge_furniture_band(
        mask,
        first,
        width=width,
        height=height,
        edge="top",
        adjacent_gap=second[0] - first[1],
        max_band_height_ratio=max_band_height_ratio,
        max_band_width_ratio=max_band_width_ratio,
        min_gap_px=min_gap_px,
        center_tolerance_ratio=center_tolerance_ratio,
    ):
        crop_top = first[1]

    last = bands[-1]
    previous = bands[-2]
    if _is_isolated_edge_furniture_band(
        mask,
        last,
        width=width,
        height=height,
        edge="bottom",
        adjacent_gap=last[0] - previous[1],
        max_band_height_ratio=max_band_height_ratio,
        max_band_width_ratio=max_band_width_ratio,
        min_gap_px=min_gap_px,
        center_tolerance_ratio=center_tolerance_ratio,
    ):
        crop_bottom = last[0]

    if crop_bottom <= crop_top:
        return image
    if crop_top == 0 and crop_bottom == height:
        return image
    return image.crop((0, crop_top, width, crop_bottom))


def _nonwhite_row_bands(mask: Any) -> list[tuple[int, int]]:
    width, height = mask.size
    rows = [bool(mask.crop((0, y, width, y + 1)).getbbox()) for y in range(height)]
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, has_content in enumerate(rows):
        if has_content and start is None:
            start = y
        elif not has_content and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, height))
    return bands


def _is_isolated_edge_furniture_band(
    mask: Any,
    band: tuple[int, int],
    *,
    width: int,
    height: int,
    edge: str,
    adjacent_gap: int,
    max_band_height_ratio: float,
    max_band_width_ratio: float,
    min_gap_px: int,
    center_tolerance_ratio: float,
) -> bool:
    top, bottom = band
    band_height = bottom - top
    if band_height <= 0 or band_height > height * max_band_height_ratio:
        return False
    if adjacent_gap < min_gap_px:
        return False
    if edge == "top" and top > height * 0.12:
        return False
    if edge == "bottom" and bottom < height * 0.88:
        return False

    bbox = mask.crop((0, top, width, bottom)).getbbox()
    if bbox is None:
        return False
    left, _band_top, right, _band_bottom = bbox
    band_width = right - left
    if band_width <= 0 or band_width > width * max_band_width_ratio:
        return False
    center = (left + right) / 2
    return abs(center - width / 2) <= width * center_tolerance_ratio


def _safe_zoom(width_pt: float, height_pt: float, requested_zoom: float, max_pixels: int) -> float:
    requested_pixels = width_pt * requested_zoom * height_pt * requested_zoom
    if requested_pixels <= max_pixels:
        return requested_zoom
    return max(0.1, math.sqrt(max_pixels / (width_pt * height_pt)))
