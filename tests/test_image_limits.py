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


def test_clean_rendered_crop_image_removes_barcode_band_from_short_question_crop() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 82), "white")
    draw = ImageDraw.Draw(image)
    for x in range(60, 620, 9):
        draw.rectangle((x, 0, x + 4, 10), fill="black")
    draw.text((70, 34), "1 Solve the inequality |3x + 2| < 3|2x - 1|.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((55, 0, 630, 14)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 34, 560, 62)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_header_exposed_by_whitespace_trim() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 1800), "white")
    draw = ImageDraw.Draw(image)
    for x in range(220, 760, 9):
        draw.rectangle((x, 1400, x + 4, 1403), fill="black")
    draw.text((120, 1750), "Find the least value of arg z for points in this region.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((500, 0, 780, 40)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((100, cleaned.height - 80, 720, cleaned.height)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_preserves_normal_text_line() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((120, 80), "The point A lies on the curve and the line meets it at B.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((120, 80, 520, 110)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_preserves_real_top_question_line() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 16), "5", fill="black")
    draw.text((100, 16), "Two vectors, u and v, are such that", fill="black")
    draw.text((540, 95), "u = ( q 2 6 ) and v = ( 8 q - 1 q^2 - 7 ),", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((90, 10, 620, 50)).convert("L").getbbox() is not None
    assert cleaned.crop((30, 10, 70, 50)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_preserves_stacked_top_text_stroke_bands() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)
    for x in range(90, 610, 12):
        draw.rectangle((x, 18, x + 5, 21), fill="black")
        draw.rectangle((x + 1, 23, x + 4, 24), fill="black")
        draw.rectangle((x, 29, x + 5, 32), fill="black")
    draw.text((35, 16), "5", fill="black")
    draw.text((540, 95), "u = ( q 2 6 ) and v = ( 8 q - 1 q^2 - 7 ),", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((90, 18, 610, 33)).convert("L").getbbox() is not None
    assert cleaned.crop((30, 10, 70, 50)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_preserves_real_top_line_near_center() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 8), "8", fill="black")
    draw.text((100, 8), "Throughout this question the use of a calculator is not permitted.", fill="black")
    draw.text((100, 90), "The polynomial is denoted by p(z).", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((95, 5, 760, 42)).convert("L").getbbox() is not None
    assert cleaned.crop((95, 84, 520, 120)).convert("L").getbbox() is not None


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


def test_clean_rendered_crop_image_removes_short_right_edge_watermark_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1577, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.line((1320, -10, 1460, 120), fill="black", width=18)
    draw.line((1380, -10, 1570, 120), fill="black", width=18)
    draw.text((70, 30), "1 Solve the inequality |x - 3| > 2|x + 1|.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((1300, 0, 1577, 110)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 30, 650, 65)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_clipped_centered_top_page_number() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((495, -2), "3", fill="black")
    draw.text((35, 38), "8", fill="black")
    draw.line((330, 90, 700, 260), fill="black", width=3)

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((480, 0, 525, 20)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((30, 35, 65, 70)).convert("L").getbbox() is not None
    assert cleaned.crop((325, 85, 705, 265)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_edge_touching_scan_mark_but_preserves_mark_value() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((990, 310, 999, 352), fill="black")
    draw.text((890, 610), "[4]", fill="black")
    draw.text((70, 120), "Find the exact value of the x-coordinate.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((985, 305, 1000, 358)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((885, 605, 940, 640)).convert("L").getbbox() is not None
    assert cleaned.crop((70, 120, 430, 150)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_isolated_bottom_page_number_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "Solve the equation 2|x - 1| = 3|x|.", fill="black")
    draw.text((890, 45), "[3]", fill="black")
    draw.text((470, 252), "1", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.height < 248 or cleaned.crop((465, 248, 490, 260)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 40, 430, 75)).convert("L").getbbox() is not None
    assert cleaned.crop((885, 40, 940, 75)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_skinny_top_edge_rule_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((500, 0, 501, 25), fill="black")
    draw.text((70, 90), "The shaded region is shown in the Argand diagram.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((494, 0, 508, 32)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 90, 520, 120)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_faint_skinny_top_edge_rule_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((500, 0, 501, 25), fill=(210, 210, 210))
    draw.text((70, 90), "The shaded region is shown in the Argand diagram.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((494, 0, 508, 32)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 90, 520, 120)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_midcrop_dense_answer_rule_band() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1578, 251), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, 570, 14):
        draw.rectangle((x, 74, x + 7, 80), fill="black")
    draw.text((70, 20), "9 (a) Find the quotient and remainder. [3]", fill="black")
    draw.text((70, 130), "(b) Hence show the exact value. [5]", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((0, 68, 590, 86)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((70, 20, 520, 55)).convert("L").getbbox() is not None
    assert cleaned.crop((70, 130, 520, 165)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_preserves_centered_vector_notation_band() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1578, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (80, 35),
        "The points A, B and C have position vectors, relative to the origin O, given by",
        fill="black",
    )
    for x in range(420, 1060, 34):
        draw.rectangle((x, 100, x + 7, 148), fill="black")
    draw.text((80, 185), "(i) Find a vector equation for the line passing through A and B.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((410, 95, 1080, 155)).convert("L").getbbox() is not None
    assert cleaned.crop((80, 185, 720, 220)).convert("L").getbbox() is not None


def test_clean_rendered_crop_image_removes_short_top_right_watermark_fragment() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1500, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.line((1320, 0, 1490, 90), fill="black", width=18)
    draw.line((1360, 0, 1500, 130), fill=(150, 150, 150), width=24)
    draw.text((55, 85), "6", fill="black")
    draw.text((450, 210), "The diagram shows a semicircle.", fill="black")

    cleaned = clean_rendered_crop_image(image)

    assert cleaned.crop((1260, 0, 1500, 150)).convert("L").getextrema() == (255, 255)
    assert cleaned.crop((50, 80, 90, 120)).convert("L").getbbox() is not None
    assert cleaned.crop((445, 205, 780, 245)).convert("L").getbbox() is not None
