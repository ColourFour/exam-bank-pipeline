from __future__ import annotations

from exam_bank.config import AppConfig
from exam_bank.models import BoundingBox, PageLayout
from exam_bank.question_regeneration import (
    _saved_crop_region_is_stale_furniture,
    _selected_record,
    _trim_saved_union_crop_stale_full_width_figure,
)


def test_question_regeneration_identity_uses_concrete_source_pdf_session_code() -> None:
    record = _selected_record(
        {
            "question_id": "12spring19_q05",
            "question_number": "5",
            "paper_family": "pm1",
            "canonical_year_folder": "2019",
            "canonical_session": "spring19",
            "canonical_question_artifact": "pm1/pm1_2019_m19_12_qp_q05_question.png",
            "notes": {
                "source_pdf": "input/pastpapers/9709/2019/question_papers/9709_m19_qp_12.pdf",
                "source_paper_code": "12",
            },
        }
    )

    assert record.identity.session_code == "m19"
    assert record.identity.component == "12"


def test_saved_context_figure_fallback_skips_stale_watermark_furniture() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=4, width=595, height=842, blocks=[], graphics=[BoundingBox(394, 0, 595, 200)])
    item = {
        "page_number": 4,
        "region_kind": "context_inferred_figure",
        "original_crop_bbox": {"x0": 394, "y0": 0, "x1": 595, "y1": 200},
        "final_crop_bbox": {"x0": 384, "y0": 45, "x1": 560, "y1": 210},
    }

    assert _saved_crop_region_is_stale_furniture(
        item,
        BoundingBox(384, 45, 560, 210),
        4,
        [layout],
        config,
    )


def test_saved_context_figure_fallback_keeps_non_furniture_diagram() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=1, width=595, height=842, blocks=[], graphics=[BoundingBox(120, 160, 410, 310)])
    item = {
        "page_number": 1,
        "region_kind": "context_inferred_figure",
        "original_crop_bbox": {"x0": 120, "y0": 160, "x1": 410, "y1": 310},
        "final_crop_bbox": {"x0": 110, "y0": 150, "x1": 420, "y1": 320},
    }

    assert not _saved_crop_region_is_stale_furniture(
        item,
        BoundingBox(110, 150, 420, 320),
        1,
        [layout],
        config,
    )


def test_saved_union_fallback_trims_stale_full_width_figure_to_text() -> None:
    config = AppConfig()
    layout = PageLayout(page_number=2, width=595, height=842, blocks=[])
    item = {
        "page_number": 2,
        "region_kind": "single_page_union",
        "final_crop_bbox": {"x0": 35, "y0": 297, "x1": 560, "y1": 365},
        "text_bbox": {"x0": 49, "y0": 299, "x1": 545, "y1": 339},
        "figure_bbox": {"x0": 0, "y0": 313, "x1": 595, "y1": 355},
    }

    trimmed = _trim_saved_union_crop_stale_full_width_figure(
        item,
        BoundingBox(35, 297, 560, 365),
        2,
        [layout],
        config,
    )

    assert trimmed.y0 > 296
    assert trimmed.y1 < 342
