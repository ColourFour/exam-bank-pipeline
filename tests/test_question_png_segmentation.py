from __future__ import annotations

from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.core.paper_identity import paper_identity_from_parts
from exam_bank.image_rendering import render_question_image
from exam_bank.pdf_extract import extract_pdf_layout
from exam_bank.question_detection import detect_question_spans


pytestmark = [pytest.mark.integration, pytest.mark.rendering]


REPO_S09_P1_QP = Path("input/pastpapers/9709/2009/question_papers/9709_s09_qp_1.pdf")
REPO_S08_P1_QP = Path("input/pastpapers/9709/2008/question_papers/9709_s08_qp_1.pdf")
REPO_W08_P1_QP = Path("input/pastpapers/9709/2008/question_papers/9709_w08_qp_1.pdf")


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    config.ocr.enabled = False
    return config


def _render_question(
    tmp_path: Path,
    *,
    pdf: Path,
    year: str,
    session: str,
    question_number: str,
):
    pytest.importorskip("fitz")
    if not pdf.exists():
        pytest.skip(f"Repo question paper PDF is not available: {pdf}")

    config = _config(tmp_path)
    layouts = extract_pdf_layout(pdf, config)
    span = next(item for item in detect_question_spans(layouts, pdf, config) if item.question_number == question_number)
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm1",
        year=year,
        session=session,
        component="01",
        question_number=question_number,
    )
    return render_question_image(pdf, span, layouts, config, identity=identity)


def test_legacy_2008_s08_q05_crop_trims_previous_question_diagram_hint(tmp_path: Path) -> None:
    result = _render_question(tmp_path, pdf=REPO_S08_P1_QP, year="2008", session="s08", question_number="5")

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "single_page_union_crop_used" in result.review_flags
    assert "page_diagram_union_skipped_neighbor_question" not in result.review_flags
    region = result.crop_diagnostics["regions"][0]
    assert region["region_kind"] == "single_page_union"
    assert region["final_crop_bbox"]["y0"] > 328
    assert region["final_crop_bbox"]["y1"] < 650


def test_legacy_2008_w08_q06_top_page_diagram_is_not_dropped_as_watermark(tmp_path: Path) -> None:
    result = _render_question(tmp_path, pdf=REPO_W08_P1_QP, year="2008", session="w08", question_number="6")

    assert result.screenshot_path and result.screenshot_path.exists()
    regions = result.crop_diagnostics["regions"]
    figure_region = regions[0]
    text_region = regions[1]
    assert figure_region["graphics_count"] == 1
    assert figure_region["final_crop_bbox"]["y0"] <= 45
    assert 250 <= figure_region["final_crop_bbox"]["y1"] <= 305
    assert figure_region["final_crop_bbox"]["x1"] < 470
    assert figure_region["figure_bbox"]["y0"] <= 45
    assert text_region["region_kind"] == "text"
    assert text_region["final_crop_bbox"]["y0"] >= 260
    assert text_region["final_crop_bbox"]["x1"] > 550


def test_legacy_2009_s09_q05_crop_trims_previous_graph_question(tmp_path: Path) -> None:
    result = _render_question(tmp_path, pdf=REPO_S09_P1_QP, year="2009", session="s09", question_number="5")

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "single_page_union_crop_used" in result.review_flags
    region = result.crop_diagnostics["regions"][0]
    assert region["region_kind"] == "single_page_union"
    assert region["final_crop_bbox"]["y0"] > 500
    assert region["figure_bbox"]["y0"] > 520


def test_legacy_2009_q02_question_crop_excludes_previous_question_watermark_region(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S09_P1_QP.exists():
        pytest.skip("Repo 2009 P1 question paper PDF is not available.")

    config = _config(tmp_path)
    layouts = extract_pdf_layout(REPO_S09_P1_QP, config)
    span = next(item for item in detect_question_spans(layouts, REPO_S09_P1_QP, config) if item.question_number == "2")
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm1",
        year="2009",
        session="s09",
        component="01",
        question_number="2",
    )

    result = render_question_image(REPO_S09_P1_QP, span, layouts, config, identity=identity)

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "watermark_excluded" in result.review_flags
    assert result.crop_diagnostics["regions"][0]["final_crop_bbox"]["y0"] >= 100
    assert result.crop_diagnostics["regions"][0]["final_crop_bbox"]["y1"] <= 155

    with Image.open(result.screenshot_path) as image:
        width, height = image.size
    assert width > 1500
    assert 110 <= height <= 180


def test_legacy_2009_q03_question_crop_excludes_q01_and_q02(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S09_P1_QP.exists():
        pytest.skip("Repo 2009 P1 question paper PDF is not available.")

    config = _config(tmp_path)
    layouts = extract_pdf_layout(REPO_S09_P1_QP, config)
    span = next(item for item in detect_question_spans(layouts, REPO_S09_P1_QP, config) if item.question_number == "3")
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm1",
        year="2009",
        session="s09",
        component="01",
        question_number="3",
    )

    result = render_question_image(REPO_S09_P1_QP, span, layouts, config, identity=identity)

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "watermark_excluded" in result.review_flags
    assert result.crop_diagnostics["regions"][0]["final_crop_bbox"]["y0"] >= 150
    assert result.crop_diagnostics["regions"][0]["final_crop_bbox"]["y1"] <= 235

    with Image.open(result.screenshot_path) as image:
        width, height = image.size
    assert width > 1500
    assert 200 <= height <= 270


def test_legacy_2009_q09_question_crop_excludes_previous_diagram(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S09_P1_QP.exists():
        pytest.skip("Repo 2009 P1 question paper PDF is not available.")

    config = _config(tmp_path)
    layouts = extract_pdf_layout(REPO_S09_P1_QP, config)
    span = next(item for item in detect_question_spans(layouts, REPO_S09_P1_QP, config) if item.question_number == "9")
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm1",
        year="2009",
        session="s09",
        component="01",
        question_number="9",
    )

    result = render_question_image(REPO_S09_P1_QP, span, layouts, config, identity=identity)

    assert result.screenshot_path and result.screenshot_path.exists()
    assert result.crop_diagnostics["regions"][0]["final_crop_bbox"]["y0"] >= 540
    assert result.crop_diagnostics["regions"][0]["final_crop_bbox"]["y1"] <= 795

    with Image.open(result.screenshot_path) as image:
        width, height = image.size
    assert width > 1500
    assert 650 <= height <= 800
