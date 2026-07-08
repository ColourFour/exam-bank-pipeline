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
BARCODE_BAND_MIN_WIDTH_RATIO = 0.18
BARCODE_BAND_MAX_WIDTH_RATIO = 0.72
BARCODE_BAND_MAX_HEIGHT_RATIO = 0.12
BARCODE_BAND_MIN_DARK_RATIO = 0.075
BARCODE_BAND_PADDING_PX = 3
EDGE_WATERMARK_MIN_WIDTH_RATIO = 0.12
EDGE_WATERMARK_MIN_HEIGHT_RATIO = 0.09
EDGE_WATERMARK_COMPONENT_DOWNSAMPLE = 4


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
    cleaned = trim_isolated_edge_furniture(image)
    cleaned = remove_dense_horizontal_furniture_bands(cleaned)
    cleaned = remove_isolated_edge_marks(cleaned)
    cleaned = remove_edge_watermark_fragments(cleaned)
    return trim_excess_render_whitespace(cleaned)


def remove_dense_horizontal_furniture_bands(
    image: Any,
    *,
    nonwhite_threshold: int = 80,
    min_width_ratio: float = BARCODE_BAND_MIN_WIDTH_RATIO,
    max_width_ratio: float = BARCODE_BAND_MAX_WIDTH_RATIO,
    max_height_ratio: float = BARCODE_BAND_MAX_HEIGHT_RATIO,
    min_dark_ratio: float = BARCODE_BAND_MIN_DARK_RATIO,
    padding_px: int = BARCODE_BAND_PADDING_PX,
) -> Any:
    """White out narrow barcode-like bands that appear inside rendered crops."""

    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    if width < WHITESPACE_TRIM_MIN_DIMENSION_PX or height < WHITESPACE_TRIM_MIN_DIMENSION_PX:
        return image

    grayscale = image.convert("L")
    dark_rows: list[tuple[int, int, int]] = []
    for y in range(height):
        row = grayscale.crop((0, y, width, y + 1))
        mask = row.point(lambda pixel: 255 if pixel < nonwhite_threshold else 0, mode="1")
        bbox = mask.getbbox()
        if bbox is None:
            continue
        x0, _top, x1, _bottom = bbox
        band_width = x1 - x0
        dark_count = sum(1 for pixel in row.tobytes() if pixel < nonwhite_threshold)
        dark_ratio = dark_count / width
        if min_width_ratio * width <= band_width <= max_width_ratio * width and dark_ratio >= min_dark_ratio:
            dark_rows.append((y, x0, x1))

    bands: list[tuple[int, int, int, int]] = []
    start: int | None = None
    end = 0
    xs: list[int] = []
    xe: list[int] = []
    previous_y: int | None = None
    for y, x0, x1 in dark_rows:
        if start is None or previous_y is None or y > previous_y + 1:
            if start is not None:
                bands.append((start, end, min(xs), max(xe)))
            start, end, xs, xe = y, y + 1, [x0], [x1]
        else:
            end = y + 1
            xs.append(x0)
            xe.append(x1)
        previous_y = y
    if start is not None:
        bands.append((start, end, min(xs), max(xe)))

    candidates = [
        (y0, y1, x0, x1)
        for y0, y1, x0, x1 in bands
        if y0 <= max(4, int(height * 0.12))
        and (y1 - y0) <= max(3, int(height * max_height_ratio))
        and _band_has_repeating_vertical_bars(
            grayscale,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            nonwhite_threshold=nonwhite_threshold,
        )
    ]
    if not candidates:
        return image

    from PIL import ImageDraw

    output = image.copy()
    draw = ImageDraw.Draw(output)
    for y0, y1, x0, x1 in candidates:
        draw.rectangle(
            (
                max(0, x0 - padding_px),
                max(0, y0 - padding_px),
                min(width, x1 + padding_px),
                min(height, y1 + padding_px),
            ),
            fill="white",
        )
    return output


def _band_has_repeating_vertical_bars(
    grayscale: Any,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    nonwhite_threshold: int,
) -> bool:
    band_height = max(1, y1 - y0)
    band_width = max(1, x1 - x0)
    if band_height < 2 or band_width < 40:
        return False
    vertical_bar_columns = 0
    runs = 0
    in_run = False
    pixels = grayscale.load()
    for x in range(x0, x1):
        dark_rows = 0
        for y in range(y0, y1):
            if pixels[x, y] < nonwhite_threshold:
                dark_rows += 1
        is_bar_column = dark_rows / band_height >= 0.64
        if is_bar_column:
            vertical_bar_columns += 1
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return vertical_bar_columns >= max(12, int(band_width * 0.18)) and runs >= 12


def remove_isolated_edge_marks(
    image: Any,
    *,
    nonwhite_threshold: int = 80,
    max_width_ratio: float = 0.07,
    max_height_ratio: float = 0.10,
    edge_ratio: float = 0.08,
) -> Any:
    """White out small, dense scan marks that sit alone near page edges."""

    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    if width < WHITESPACE_TRIM_MIN_DIMENSION_PX or height < WHITESPACE_TRIM_MIN_DIMENSION_PX:
        return image

    grayscale = image.convert("L")
    mask = grayscale.point(lambda pixel: 255 if pixel < nonwhite_threshold else 0, mode="1")
    pixels = mask.load()
    visited: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if (x, y) in visited or not pixels[x, y]:
                continue
            stack = [(x, y)]
            visited.add((x, y))
            x0 = x1 = x
            y0 = y1 = y
            count = 0
            while stack:
                cx, cy = stack.pop()
                count += 1
                x0 = min(x0, cx)
                x1 = max(x1, cx)
                y0 = min(y0, cy)
                y1 = max(y1, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if (nx, ny) in visited or not pixels[nx, ny]:
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))

            component_width = x1 - x0 + 1
            component_height = y1 - y0 + 1
            touches_side_zone = x0 <= width * edge_ratio or x1 >= width * (1 - edge_ratio)
            if not touches_side_zone:
                continue
            if component_width > width * max_width_ratio or component_height > height * max_height_ratio:
                continue
            if component_width < 8 or component_height < 8:
                continue
            density = count / max(1, component_width * component_height)
            if density < 0.75:
                continue
            candidates.append((x0, y0, x1 + 1, y1 + 1))

    if not candidates:
        return image

    from PIL import ImageDraw

    output = image.copy()
    draw = ImageDraw.Draw(output)
    for x0, y0, x1, y1 in candidates:
        draw.rectangle((max(0, x0 - 3), max(0, y0 - 3), min(width, x1 + 3), min(height, y1 + 3)), fill="white")
    return output


def remove_edge_watermark_fragments(
    image: Any,
    *,
    nonwhite_threshold: int = 235,
    component_downsample: int = EDGE_WATERMARK_COMPONENT_DOWNSAMPLE,
    min_width_ratio: float = EDGE_WATERMARK_MIN_WIDTH_RATIO,
    min_height_ratio: float = EDGE_WATERMARK_MIN_HEIGHT_RATIO,
) -> Any:
    """White out large connected diagonal watermark fragments at crop edges."""

    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    if width < WHITESPACE_TRIM_MIN_DIMENSION_PX or height < WHITESPACE_TRIM_MIN_DIMENSION_PX:
        return image

    scale = max(1, component_downsample)
    probe_width = max(1, width // scale)
    probe_height = max(1, height // scale)
    from PIL import Image, ImageDraw

    probe = image.convert("L").resize((probe_width, probe_height), Image.Resampling.NEAREST)
    mask = probe.point(lambda pixel: 255 if pixel < nonwhite_threshold else 0, mode="1")
    pixels = mask.load()
    visited: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, int, int]] = []
    x_floor = int(probe_width * 0.48)

    for y in range(probe_height):
        for x in range(x_floor, probe_width):
            if (x, y) in visited or not pixels[x, y]:
                continue
            stack = [(x, y)]
            visited.add((x, y))
            x0 = x1 = x
            y0 = y1 = y
            count = 0
            while stack:
                cx, cy = stack.pop()
                count += 1
                x0 = min(x0, cx)
                x1 = max(x1, cx)
                y0 = min(y0, cy)
                y1 = max(y1, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < x_floor or nx >= probe_width or ny < 0 or ny >= probe_height:
                        continue
                    if (nx, ny) in visited or not pixels[nx, ny]:
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))

            component_width = x1 - x0 + 1
            component_height = y1 - y0 + 1
            touches_edge = x1 >= probe_width - 2 or y0 <= 1
            if not touches_edge:
                continue
            if component_width < probe_width * min_width_ratio or component_height < probe_height * min_height_ratio:
                continue
            density = count / max(1, component_width * component_height)
            if density > 0.65:
                continue
            candidates.append((x0 * scale, y0 * scale, min(width, (x1 + 1) * scale), min(height, (y1 + 1) * scale)))

    if not candidates:
        return image

    output = image.copy()
    draw = ImageDraw.Draw(output)
    for x0, y0, x1, y1 in candidates:
        draw.rectangle((max(0, x0 - 12), max(0, y0 - 12), min(width, x1 + 12), min(height, y1 + 12)), fill="white")
    return output


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
