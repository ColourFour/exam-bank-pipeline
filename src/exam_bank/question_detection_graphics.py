from __future__ import annotations

from .models import BoundingBox, PageLayout


def is_answer_rule_like(box: BoundingBox, layout: PageLayout) -> bool:
    width = max(0.0, box.x1 - box.x0)
    height = max(0.0, box.y1 - box.y0)
    return height <= 2.5 and width >= layout.width * 0.28
