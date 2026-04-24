from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import BoundingBox
from .trust import CropConfidence, MappingStatus


@dataclass(frozen=True)
class MarkSchemeImageResult:
    question_number: str
    image_path: Path | None = None
    page_numbers: list[int] = field(default_factory=list)
    markscheme_question_number: str = ""
    crop_confidence: str = CropConfidence.LOW
    mapping_method: str = ""
    table_detected: bool = False
    table_header_detected: list[str] = field(default_factory=list)
    detected_anchor_pages: list[int] = field(default_factory=list)
    nearby_anchors: list[str] = field(default_factory=list)
    debug_paths: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    table_header_ok: bool = False
    continuation_rows_included: bool = False
    question_subparts: list[str] = field(default_factory=list)
    markscheme_subparts: list[str] = field(default_factory=list)
    question_marks_total: int | None = None
    markscheme_marks_total: int | None = None
    mapping_status: str = MappingStatus.FAIL
    failure_reason: str = ""


@dataclass(frozen=True)
class MarkSchemeTable:
    page_number: int
    bbox: BoundingBox
    question_col_right: float
    marks_col_left: float
    marks_col_right: float
    header_bottom: float
    confidence: str
    header_detected: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MarkSchemeAnchor:
    question_number: str
    page_number: int
    y0: float
    y1: float
    x0: float
    text: str
    table: MarkSchemeTable | None


@dataclass(frozen=True)
class MarkSchemeCropRegion:
    page_number: int
    bbox: BoundingBox
    table_detected: bool
    continuation_rows_included: bool = False


@dataclass(frozen=True)
class MarkSchemeWord:
    page_number: int
    text: str
    bbox: BoundingBox


@dataclass(frozen=True)
class MarkSchemeRow:
    page_number: int
    y0: float
    text: str
    marks_cell: str
    mark_values: tuple[int, ...]
    standalone_total: int | None
    question_label: str | None


@dataclass(frozen=True)
class HeaderGeometry:
    header_box: BoundingBox
    question_header: BoundingBox
    answer_header: BoundingBox
    marks_header: BoundingBox
    guidance_header: BoundingBox
    header_detected: list[str]
