from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser(description="Render compact PNG contact sheets for visual audit review.")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--artifact-root", default="output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-sheet", type=int, default=9)
    args = parser.parse_args()

    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    rows = sample.get("questions", [])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(args.artifact_root)
    font = load_font(17)
    bold = load_font(19, bold=True)

    for start in range(0, len(rows), args.per_sheet):
        subset = rows[start : start + args.per_sheet]
        sheet = Image.new("RGB", (1800, 1110), "#e9edf3")
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, 1800, 45), fill="#172033")
        draw.text(
            (16, 12),
            f"Visual audit {start + 1:03d}–{start + len(subset):03d} / {len(rows)}",
            fill="white",
            font=bold,
        )
        for offset, row in enumerate(subset):
            render_card(
                sheet,
                draw,
                row,
                display_index=start + offset + 1,
                card_index=offset,
                artifact_root=artifact_root,
                font=font,
                bold=bold,
            )
        sheet_number = start // args.per_sheet + 1
        sheet.save(output_dir / f"contact_{sheet_number:03d}.jpg", quality=88, optimize=True)

    print(json.dumps({"questions": len(rows), "sheets": math.ceil(len(rows) / args.per_sheet)}, indent=2))
    return 0


def render_card(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: dict[str, Any],
    *,
    display_index: int,
    card_index: int,
    artifact_root: Path,
    font: ImageFont.ImageFont,
    bold: ImageFont.ImageFont,
) -> None:
    column = card_index % 3
    row_index = card_index // 3
    x0 = 10 + column * 596
    y0 = 55 + row_index * 350
    x1 = x0 + 586
    y1 = y0 + 340
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill="white", outline="#bfc8d6", width=2)
    draw.text((x0 + 10, y0 + 8), f"{display_index:03d}. {row.get('question_id', '')}", fill="#172033", font=bold)
    draw.text(
        (x0 + 10, y0 + 35),
        f"Q {row.get('question_ocr_similarity')}  ·  MS {row.get('mark_scheme_ocr_similarity')}",
        fill="#56637a",
        font=font,
    )
    draw.text((x0 + 100, y0 + 62), "question", fill="#172033", font=bold)
    draw.text((x0 + 398, y0 + 62), "markscheme", fill="#172033", font=bold)
    paste_thumbnail(sheet, artifact_root / str(row.get("question_image_path") or ""), (x0 + 8, y0 + 88, 285, 242))
    paste_thumbnail(sheet, artifact_root / str(row.get("mark_scheme_image_path") or ""), (x0 + 298, y0 + 88, 280, 242))


def paste_thumbnail(sheet: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    if not path.is_file():
        ImageDraw.Draw(sheet).text((x + 10, y + 10), "MISSING", fill="#a00")
        return
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        background = Image.new("RGB", (width, height), "#fafafa")
        left = (width - image.width) // 2
        background.paste(image, (left, 0))
        sheet.paste(background, (x, y))


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
