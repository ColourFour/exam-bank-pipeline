from exam_bank.image_limits import (
    SAFE_RENDER_PIXELS,
    _safe_zoom,
    cap_image_pixels,
    clean_rendered_crop_image,
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


def test_clean_rendered_crop_image_removes_dense_barcode_band() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    for x in range(140, 480, 9):
        draw.rectangle((x, 12, x + 4, 25), fill="black")
    draw.rectangle((80, 150, 720, 230), fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.getbbox() is not None
    assert cleaned.crop((140, 12, 480, 26)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((80, 150, 720, 230)).convert("L").getextrema()[0] == 0


def test_clean_rendered_crop_image_removes_short_dense_barcode_band() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    for x in range(120, 420, 8):
        draw.rectangle((x, 12, x + 3, 14), fill="black")
    draw.text((80, 120), "Find the coordinates of the points of intersection.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((120, 12, 420, 15)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((80, 120, 430, 145)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_tall_antialiased_barcode_band() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 220), "white")
    draw = ImageDraw.Draw(image)
    for x in range(80, 630, 10):
        draw.rectangle((x, 0, x + 4, 14), fill="black")
    draw.text((70, 45), "Find the coordinates of the points of intersection.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((75, 0, 640, 18)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 45, 520, 75)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_preserves_normal_text_line() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((120, 80), "The point A lies on the curve and the line meets it at B.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((120, 80, 520, 110)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_isolated_edge_mark() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 250, 48, 282), fill="black")
    draw.text((80, 120), "A circle meets the y-axis at the points A and B.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((0, 245, 54, 288)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((80, 120, 430, 145)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_edge_watermark_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.line((620, -20, 910, 270), fill="black", width=18)
    draw.line((650, -20, 940, 270), fill="black", width=18)
    draw.text((690, 80), "Papacambridge", fill="black")
    draw.rectangle((80, 300, 700, 380), fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((600, 0, 900, 280)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((80, 300, 700, 380)).convert("L").getextrema()[0] == 0


def test_clean_rendered_crop_image_removes_midpage_right_edge_watermark_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.line((760, 255, 930, 425), fill="black", width=16)
    draw.line((790, 255, 960, 425), fill="black", width=16)
    draw.text((80, 250), "It is given that the coordinates of P are (3, 7).", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((735, 235, 900, 445)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((80, 250, 520, 285)).convert("L").getbbox() is not None
