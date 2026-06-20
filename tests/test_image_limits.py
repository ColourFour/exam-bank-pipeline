from exam_bank.image_limits import (
    SAFE_RENDER_PIXELS,
    _safe_zoom,
    cap_image_pixels,
    trim_excess_render_whitespace,
    trim_isolated_edge_furniture,
)


def test_safe_zoom_keeps_normal_a4_render_dpi() -> None:
    requested_zoom = 220 / 72

    assert _safe_zoom(595, 842, requested_zoom, SAFE_RENDER_PIXELS) == requested_zoom


def test_safe_zoom_caps_oversized_page_render() -> None:
    requested_zoom = 220 / 72
    zoom = _safe_zoom(4000, 4000, requested_zoom, SAFE_RENDER_PIXELS)

    assert zoom < requested_zoom
    assert int(4000 * zoom) * int(4000 * zoom) <= SAFE_RENDER_PIXELS


def test_cap_image_pixels_downscales_large_output() -> None:
    from PIL import Image

    image = Image.new("RGB", (4000, 4000), "white")
    capped = cap_image_pixels(image, source_file="paper.pdf", context="test", max_pixels=4_000_000)

    assert capped.width * capped.height <= 4_000_000


def test_trim_excess_render_whitespace_removes_extreme_blank_bottom_margin() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 360, 130), fill="black")

    trimmed = trim_excess_render_whitespace(image, padding_px=20)

    assert trimmed.size == (400, 151)


def test_trim_excess_render_whitespace_preserves_normal_crop() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 80, 360, 360), fill="black")

    trimmed = trim_excess_render_whitespace(image)

    assert trimmed is image


def test_trim_excess_render_whitespace_preserves_blank_image() -> None:
    from PIL import Image

    image = Image.new("RGB", (400, 900), "white")

    trimmed = trim_excess_render_whitespace(image)

    assert trimmed is image


def test_trim_isolated_edge_furniture_removes_centered_top_page_number() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((190, 12, 210, 26), fill="black")
    draw.rectangle((45, 100, 360, 240), fill="black")

    trimmed = trim_isolated_edge_furniture(image)

    assert trimmed.size == (400, 473)


def test_trim_isolated_edge_furniture_preserves_left_aligned_question_label() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 12, 64, 26), fill="black")
    draw.rectangle((72, 100, 360, 240), fill="black")

    trimmed = trim_isolated_edge_furniture(image)

    assert trimmed is image
