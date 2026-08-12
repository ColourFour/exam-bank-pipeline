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
REPO_M19_P12_QP = Path("input/pastpapers/9709/2019/question_papers/9709_m19_qp_12.pdf")
REPO_S19_P11_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_11.pdf")
REPO_W19_P11_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_11.pdf")
REPO_W19_P12_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_12.pdf")
REPO_S19_P13_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_13.pdf")
REPO_W19_P13_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_13.pdf")
REPO_S19_P31_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_31.pdf")
REPO_S19_P32_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_32.pdf")
REPO_M19_P32_QP = Path("input/pastpapers/9709/2019/question_papers/9709_m19_qp_32.pdf")
REPO_S19_P33_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_33.pdf")
REPO_W19_P31_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_31.pdf")
REPO_W19_P32_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_32.pdf")
REPO_W19_P33_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_33.pdf")
REPO_S19_P41_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_41.pdf")
REPO_W19_P41_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_41.pdf")
REPO_M19_P42_QP = Path("input/pastpapers/9709/2019/question_papers/9709_m19_qp_42.pdf")
REPO_W19_P42_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_42.pdf")
REPO_S19_P43_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_43.pdf")
REPO_W19_P43_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_43.pdf")
REPO_S19_P61_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_61.pdf")
REPO_W19_P62_QP = Path("input/pastpapers/9709/2019/question_papers/9709_w19_qp_62.pdf")
REPO_S19_P63_QP = Path("input/pastpapers/9709/2019/question_papers/9709_s19_qp_63.pdf")
REPO_S18_P41_QP = Path("input/pastpapers/9709/2018/question_papers/9709_s18_qp_41.pdf")
REPO_S18_P43_QP = Path("input/pastpapers/9709/2018/question_papers/9709_s18_qp_43.pdf")
REPO_S20_P43_QP = Path("input/pastpapers/9709/2020/question_papers/9709_s20_qp_43.pdf")
REPO_W20_P11_QP = Path("input/pastpapers/9709/2020/question_papers/9709_w20_qp_11.pdf")
REPO_S17_P43_QP = Path("input/pastpapers/9709/2017/question_papers/9709_s17_qp_43.pdf")
REPO_W10_P13_QP = Path("input/pastpapers/9709/2010/question_papers/9709_w10_qp_13.pdf")
REPO_W12_P13_QP = Path("input/pastpapers/9709/2012/question_papers/9709_w12_qp_13.pdf")
REPO_S14_P11_QP = Path("input/pastpapers/9709/2014/question_papers/9709_s14_qp_11.pdf")
REPO_M23_P13_QP = Path("input/pastpapers/9709/2023/question_papers/9709_m23_qp_13.pdf")
REPO_W23_P51_QP = Path("input/pastpapers/9709/2023/question_papers/9709_w23_qp_51.pdf")
REPO_W25_P13_QP = Path("input/pastpapers/9709/2025/question_papers/9709_w25_qp_13.pdf")
REPO_S21_P51_QP = Path("input/pastpapers/9709/2021/question_papers/9709_s21_qp_51.pdf")
REPO_S21_P52_QP = Path("input/pastpapers/9709/2021/question_papers/9709_s21_qp_52.pdf")
REPO_W21_P52_QP = Path("input/pastpapers/9709/2021/question_papers/9709_w21_qp_52.pdf")
REPO_W21_P53_QP = Path("input/pastpapers/9709/2021/question_papers/9709_w21_qp_53.pdf")
REPO_M24_P13_QP = Path("input/pastpapers/9709/2024/question_papers/9709_m24_qp_13.pdf")
REPO_M25_P31_QP = Path("input/pastpapers/9709/2025/question_papers/9709_m25_qp_31.pdf")
REPO_M25_P35_QP = Path("input/pastpapers/9709/2025/question_papers/9709_m25_qp_35.pdf")


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
    component: str = "01",
    subject_family: str = "pm1",
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
        subject_family=subject_family,
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )
    return render_question_image(pdf, span, layouts, config, identity=identity)


@pytest.mark.parametrize(
    ("pdf", "subject_family", "year", "session", "component", "question_number", "max_region_bottom"),
    [
        (REPO_M19_P12_QP, "pm1", "2019", "m19", "12", "5", 185),
        (REPO_S19_P31_QP, "pm3", "2019", "s19", "31", "8", 140),
        (REPO_M19_P32_QP, "pm3", "2019", "m19", "32", "8", 140),
        (REPO_S19_P33_QP, "pm3", "2019", "s19", "33", "9", 140),
        (REPO_W19_P31_QP, "pm3", "2019", "w19", "31", "2", 100),
        (REPO_W19_P31_QP, "pm3", "2019", "w19", "31", "8", 135),
    ],
)
def test_2019_prompt_crop_excludes_answer_blank_pages(
    tmp_path: Path,
    pdf: Path,
    subject_family: str,
    year: str,
    session: str,
    component: str,
    question_number: str,
    max_region_bottom: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family=subject_family,
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "full_region_mode" not in result.review_flags
    regions = result.crop_diagnostics["regions"]
    assert regions
    assert all(region["final_crop_bbox"]["y1"] <= max_region_bottom for region in regions)
    assert all(region["region_kind"] == "text" for region in regions)


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number", "max_region_bottom"),
    [
        (REPO_W19_P32_QP, "2019", "w19", "32", "10", 455),
        (REPO_M19_P32_QP, "2019", "m19", "32", "9", 130),
        (REPO_S19_P32_QP, "2019", "s19", "32", "9", 130),
        (REPO_S19_P33_QP, "2019", "s19", "33", "10", 130),
        (REPO_W19_P31_QP, "2019", "w19", "31", "7", 130),
        (REPO_W19_P33_QP, "2019", "w19", "33", "7", 415),
        (REPO_M25_P31_QP, "2025", "m25", "31", "8", 485),
        (REPO_M25_P35_QP, "2025", "m25", "35", "10", 455),
    ],
)
def test_vector_prompt_crops_trim_answer_blank_continuations(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
    max_region_bottom: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="pm3",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "crop_reaches_page_margin" not in result.review_flags
    regions = result.crop_diagnostics["regions"]
    assert regions
    assert all(region["final_crop_bbox"]["y1"] <= max_region_bottom for region in regions)


def test_2019_s19_p31_q09_diagram_crop_is_not_repeated(tmp_path: Path) -> None:
    result = _render_question(
        tmp_path,
        pdf=REPO_S19_P31_QP,
        subject_family="pm3",
        year="2019",
        session="s19",
        component="31",
        question_number="9",
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    regions = result.crop_diagnostics["regions"]
    page_14_regions = [region for region in regions if region["page_number"] == 14]

    assert len(page_14_regions) == 1
    assert page_14_regions[0]["region_kind"] == "page_diagram_union"
    assert page_14_regions[0]["graphics_count"] == 1
    assert page_14_regions[0]["final_crop_bbox"]["y1"] <= 370
    assert "duplicate_visual_regions_removed" in result.review_flags
    assert "page_diagram_union_used" in result.review_flags


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number"),
    [
        (REPO_S18_P41_QP, "2018", "s18", "41", "5"),
        (REPO_S20_P43_QP, "2020", "s20", "43", "4"),
        (REPO_S17_P43_QP, "2017", "s17", "43", "3"),
    ],
)
def test_text_only_graph_diagrams_are_preserved_as_single_crop(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="mechanics",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    diagram_regions = [region for region in result.crop_diagnostics["regions"] if region["region_kind"] == "text_diagram_union"]
    assert len(diagram_regions) == 1
    assert diagram_regions[0]["merged_blocks"] >= 2
    assert "text_only_diagram_union_used" in result.review_flags


def test_graph_tick_label_is_not_used_as_foreign_question_boundary(tmp_path: Path) -> None:
    result = _render_question(
        tmp_path,
        pdf=REPO_S18_P43_QP,
        subject_family="mechanics",
        year="2018",
        session="s18",
        component="43",
        question_number="1",
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    first_region = result.crop_diagnostics["regions"][0]
    assert first_region["region_kind"] == "page_diagram_union"
    assert first_region["figure_bbox"] is not None
    assert first_region["final_crop_bbox"]["y1"] > first_region["figure_bbox"]["y1"]
    assert "foreign_question_region_removed" not in result.review_flags


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number", "expected_kind"),
    [
        (REPO_W10_P13_QP, "2010", "w10", "13", "7", "single_page_union"),
        (REPO_W12_P13_QP, "2012", "w12", "13", "7", "single_page_union"),
        (REPO_S14_P11_QP, "2014", "s14", "11", "10", "single_page_union"),
        (REPO_S19_P13_QP, "2019", "s19", "13", "9", "page_diagram_union"),
        (REPO_W20_P11_QP, "2020", "w20", "11", "4", "single_page_union"),
        (REPO_M23_P13_QP, "2023", "m23", "13", "1", "single_page_union"),
        (REPO_W25_P13_QP, "2025", "w25", "13", "6", "page_diagram_union"),
    ],
)
def test_requested_graph_diagrams_are_preserved_as_unified_regions(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
    expected_kind: str,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="pm1",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    first_region = result.crop_diagnostics["regions"][0]
    assert first_region["region_kind"] == expected_kind
    assert first_region["figure_bbox"] is not None
    assert "missing_image_detection_failure" not in result.review_flags
    assert "missing_image_detection_failure" not in result.crop_diagnostics["flags"]


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number", "next_anchor_y"),
    [
        (REPO_W10_P13_QP, "2010", "w10", "13", "7", 490.1),
        (REPO_W12_P13_QP, "2012", "w12", "13", "7", 450.6),
        (REPO_S14_P11_QP, "2014", "s14", "11", "10", 509.7),
    ],
)
def test_unified_graph_crop_stops_before_following_question_anchor(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
    next_anchor_y: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="pm1",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    first_region = result.crop_diagnostics["regions"][0]
    assert first_region["region_kind"] == "single_page_union"
    assert first_region["final_crop_bbox"]["y1"] < next_anchor_y


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number", "max_region_bottom"),
    [
        (REPO_S19_P31_QP, "2019", "s19", "31", "1", 130),
        (REPO_S19_P31_QP, "2019", "s19", "31", "3", 105),
        (REPO_S19_P31_QP, "2019", "s19", "31", "5", 545),
        (REPO_S19_P31_QP, "2019", "s19", "31", "10", 445),
        (REPO_W19_P31_QP, "2019", "w19", "31", "10", 120),
        (REPO_S19_P33_QP, "2019", "s19", "33", "1", 105),
        (REPO_S19_P33_QP, "2019", "s19", "33", "8", 400),
        (REPO_W19_P33_QP, "2019", "w19", "33", "10", 315),
    ],
)
def test_2019_p3_topic_packet_crops_trim_stale_page_floor_blanks(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
    max_region_bottom: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="pm3",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "crop_reaches_page_margin" not in result.review_flags
    assert all(region["final_crop_bbox"]["y1"] <= max_region_bottom for region in result.crop_diagnostics["regions"])


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number", "max_region_bottom"),
    [
        (REPO_S19_P11_QP, "2019", "s19", "11", "6", 105),
        (REPO_W19_P11_QP, "2019", "w19", "11", "1", 105),
        (REPO_W19_P12_QP, "2019", "w19", "12", "2", 120),
        (REPO_S19_P13_QP, "2019", "s19", "13", "8", 115),
        (REPO_W19_P13_QP, "2019", "w19", "13", "7", 105),
        (REPO_W19_P13_QP, "2019", "w19", "13", "10", 480),
        (REPO_M24_P13_QP, "2024", "m24", "13", "5", 610),
    ],
)
def test_2019_p1_topic_packet_crops_trim_stale_page_floor_blanks(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
    max_region_bottom: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="pm1",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "crop_reaches_page_margin" not in result.review_flags
    assert all(region["final_crop_bbox"]["y1"] <= max_region_bottom for region in result.crop_diagnostics["regions"])


@pytest.mark.parametrize(
    ("pdf", "subject_family", "year", "session", "component", "question_number", "max_region_bottom"),
    [
        (REPO_S19_P61_QP, "stats", "2019", "s19", "61", "8", 440),
        (REPO_W19_P62_QP, "stats", "2019", "w19", "62", "7", 605),
        (REPO_S19_P63_QP, "stats", "2019", "s19", "63", "1", 458),
        (REPO_S19_P63_QP, "stats", "2019", "s19", "63", "5", 118),
        (REPO_S21_P51_QP, "stats", "2021", "s21", "51", "5", 645),
        (REPO_S21_P52_QP, "stats", "2021", "s21", "52", "7", 415),
        (REPO_W21_P52_QP, "stats", "2021", "w21", "52", "7", 690),
        (REPO_W21_P53_QP, "stats", "2021", "w21", "53", "3", 605),
    ],
)
def test_stats_topic_packet_crops_trim_answer_blanks(
    tmp_path: Path,
    pdf: Path,
    subject_family: str,
    year: str,
    session: str,
    component: str,
    question_number: str,
    max_region_bottom: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family=subject_family,
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "crop_reaches_page_margin" not in result.review_flags
    assert all(region["final_crop_bbox"]["y1"] <= max_region_bottom for region in result.crop_diagnostics["regions"])


def test_current_question_number_above_graph_is_not_removed_as_duplicate_label(tmp_path: Path) -> None:
    result = _render_question(
        tmp_path,
        pdf=REPO_W23_P51_QP,
        subject_family="stats",
        year="2023",
        session="w23",
        component="51",
        question_number="1",
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    first_region = result.crop_diagnostics["regions"][0]
    assert first_region["final_crop_bbox"]["y0"] <= 64
    assert first_region["text_bbox"]["y0"] <= 66


@pytest.mark.parametrize(
    ("pdf", "subject_family", "year", "session", "component", "question_number"),
    [
        (REPO_S19_P61_QP, "stats", "2019", "s19", "61", "3"),
        (REPO_W19_P62_QP, "stats", "2019", "w19", "62", "1"),
        (REPO_S19_P63_QP, "stats", "2019", "s19", "63", "7"),
        (REPO_W23_P51_QP, "stats", "2023", "w23", "51", "1"),
    ],
)
def test_prompt_text_before_answer_rules_keeps_safe_bottom_padding(
    tmp_path: Path,
    pdf: Path,
    subject_family: str,
    year: str,
    session: str,
    component: str,
    question_number: str,
) -> None:
    Image = pytest.importorskip("PIL.Image")
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family=subject_family,
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    with Image.open(result.screenshot_path) as image:
        grayscale = image.convert("L")
        assert grayscale.crop((0, max(0, image.height - 3), image.width, image.height)).getextrema() == (255, 255)


@pytest.mark.parametrize(
    ("pdf", "year", "session", "component", "question_number", "max_region_bottom"),
    [
        (REPO_S19_P41_QP, "2019", "s19", "41", "2", 525),
        (REPO_S19_P41_QP, "2019", "s19", "41", "3", 595),
        (REPO_S19_P41_QP, "2019", "s19", "41", "4", 455),
        (REPO_S19_P41_QP, "2019", "s19", "41", "6", 790),
        (REPO_W19_P41_QP, "2019", "w19", "41", "1", 105),
        (REPO_M19_P42_QP, "2019", "m19", "42", "4", 155),
        (REPO_W19_P42_QP, "2019", "w19", "42", "7", 365),
        (REPO_S19_P43_QP, "2019", "s19", "43", "6", 320),
        (REPO_W19_P43_QP, "2019", "w19", "43", "7", 595),
    ],
)
def test_mechanics_topic_packet_crops_trim_answer_blanks(
    tmp_path: Path,
    pdf: Path,
    year: str,
    session: str,
    component: str,
    question_number: str,
    max_region_bottom: float,
) -> None:
    result = _render_question(
        tmp_path,
        pdf=pdf,
        subject_family="mechanics",
        year=year,
        session=session,
        component=component,
        question_number=question_number,
    )

    assert result.screenshot_path and result.screenshot_path.exists()
    assert "crop_reaches_page_margin" not in result.review_flags
    assert all(region["final_crop_bbox"]["y1"] <= max_region_bottom for region in result.crop_diagnostics["regions"])


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
