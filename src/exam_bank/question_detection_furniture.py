from __future__ import annotations

import re

from .config import AppConfig
from .models import PageLayout, TextBlock


BOILERPLATE_PATTERNS = [
    (r"^Additional Page\b", "additional_page"),
    (r"If you use the following lined page", "lined_page_instruction"),
    (r"write the question number", "lined_page_instruction"),
    (r"^©\s*UCLES\b", "copyright_footer"),
    (r"^UCLES\b", "copyright_footer"),
    (r"^9709[/_ -]", "paper_code_footer"),
    (r"^\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}$", "paper_code_footer"),
    (r"^Cambridge International", "publisher_footer"),
    (r"DO NOT WRITE IN THIS MARGIN", "margin_furniture"),
    (r"^This document consists of", "page_furniture"),
    (r"^BLANK PAGE$", "blank_page"),
    (r"^Question Paper$", "page_furniture"),
    (r"^Mark Scheme$", "page_furniture"),
    (r"^Turn over$", "footer"),
]


def is_centered_page_number_block(block: TextBlock, page: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if not text.isdigit():
        return False
    if block.bbox.y0 > config.detection.min_question_start_y:
        return False
    center_x = (block.bbox.x0 + block.bbox.x1) / 2
    return page.width * 0.35 <= center_x <= page.width * 0.65


def is_footer_or_header_block(block: TextBlock, page: PageLayout, config: AppConfig) -> bool:
    return block.bbox.y1 < config.detection.crop_top_margin or block.bbox.y0 > page.height - config.detection.bottom_margin


def is_boilerplate_text(text: str) -> bool:
    return boilerplate_reason(text) is not None


def boilerplate_reason(text: str) -> str | None:
    text = _clean_text_line(text)
    for pattern, reason in BOILERPLATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return reason
    return None


def is_answer_space_text(text: str) -> bool:
    if re.fullmatch(r"[._\-–—\s]{6,}", text):
        return True
    if re.fullmatch(r"(?:\.\s*){6,}", text):
        return True
    return bool(re.search(r"\bAnswer\b\s*[._\-–—]{6,}", text, re.IGNORECASE))


def is_margin_furniture_text(block: TextBlock, page: PageLayout, config: AppConfig) -> bool:
    text = _clean_text_line(block.text)
    if re.search(r"DO NOT WRITE IN THIS MARGIN", text, re.IGNORECASE):
        return True
    narrow_edge = (block.bbox.x1 - block.bbox.x0) <= 70 and (
        block.bbox.x0 <= config.detection.crop_left_margin or block.bbox.x1 >= page.width - config.detection.crop_right_margin
    )
    tall = (block.bbox.y1 - block.bbox.y0) >= 80
    return narrow_edge and tall


def is_control_artifact_text(text: str) -> bool:
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r")
    if control_count == 0:
        return False
    cleaned = _strip_control_chars(text).strip()
    visible_count = sum(1 for char in cleaned if not char.isspace())
    if visible_count <= 3:
        return True
    return control_count >= max(4, visible_count)


def _clean_text_line(text: str) -> str:
    return " ".join(_strip_control_chars(text).replace("\u00a0", " ").split())


def _strip_control_chars(text: str) -> str:
    return "".join(char if ord(char) >= 32 or char in "\n\t\r" else " " for char in text)
