from __future__ import annotations

from typing import Any, Sequence


def stitch_images(images: Sequence[Any], gap_px: int) -> Any:
    """Stack Pillow images vertically on a white RGB canvas."""
    if not images:
        raise ValueError("At least one image is required for stitching.")

    from PIL import Image

    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap_px * max(0, len(images) - 1)
    stitched = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        stitched.paste(image, (0, y))
        y += image.height + gap_px
    return stitched
