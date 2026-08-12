from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank.config import AppConfig
from exam_bank.core.paper_identity import paper_identity_from_parts
from exam_bank.mark_schemes import render_mark_scheme_images


pytestmark = [pytest.mark.integration, pytest.mark.rendering]


REPO_S08_P1_MS = Path("input/pastpapers/9709/2008/mark_schemes/9709_s08_ms_1.pdf")
REPO_W08_P1_MS = Path("input/pastpapers/9709/2008/mark_schemes/9709_w08_ms_1.pdf")
REPO_S17_P33_MS = Path("input/pastpapers/9709/2017/mark_schemes/9709_s17_ms_33.pdf")
REPO_W17_P31_MS = Path("input/pastpapers/9709/2017/mark_schemes/9709_w17_ms_31.pdf")
REPO_W17_P33_MS = Path("input/pastpapers/9709/2017/mark_schemes/9709_w17_ms_33.pdf")
REPO_W18_P32_MS = Path("input/pastpapers/9709/2018/mark_schemes/9709_w18_ms_32.pdf")
REPO_S19_P33_MS = Path("input/pastpapers/9709/2019/mark_schemes/9709_s19_ms_33.pdf")
REPO_S20_P31_MS = Path("input/pastpapers/9709/2020/mark_schemes/9709_s20_ms_31.pdf")
REPO_S20_P33_MS = Path("input/pastpapers/9709/2020/mark_schemes/9709_s20_ms_33.pdf")
REPO_S21_P32_MS = Path("input/pastpapers/9709/2021/mark_schemes/9709_s21_ms_32.pdf")
REPO_S25_P32_MS = Path("input/pastpapers/9709/2025/mark_schemes/9709_s25_ms_32.pdf")
REPO_W25_P35_MS = Path("input/pastpapers/9709/2025/mark_schemes/9709_w25_ms_35.pdf")
REPO_W09_P32_MS = Path("input/pastpapers/9709/2009/mark_schemes/9709_w09_ms_32.pdf")


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    config.ocr.enabled = False
    return config


def _debug_records(config: AppConfig) -> list[dict]:
    path = config.output.debug_dir / "mark_scheme_crop_debug.jsonl"
    assert path.exists()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record(records: list[dict], question_id: str) -> dict:
    return next(item for item in records if item["question_id"] == question_id)


def test_legacy_formula_fragment_before_next_label_is_not_cross_question_contamination(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    if not REPO_W09_P32_MS.exists():
        pytest.skip("Repo 2009 P32 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm3",
        year="2009",
        session="w09",
        component="32",
        question_number="2",
    )
    result = render_mark_scheme_images(
        REPO_W09_P32_MS,
        config,
        ["2"],
        question_marks={"2": 5},
        question_subparts={"2": ["i", "ii"]},
        question_identities={"2": identity},
    )["2"]

    assert result.mapping_status == "pass"
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "32winter09_q02")
    assert debug["detected_primary_questions_in_left_column"] == ["2"]


def test_legacy_2008_q01_mark_scheme_uses_single_table_row_block(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S08_P1_MS.exists():
        pytest.skip("Repo 2008 P1 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "1": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm1",
            year="2008",
            session="s08",
            component="01",
            question_number="1",
        )
    }

    result = render_mark_scheme_images(
        REPO_S08_P1_MS,
        config,
        ["1"],
        question_marks={"1": 3},
        question_subparts={"1": []},
        question_identities=identities,
    )["1"]

    assert result.mapping_status == "pass"
    assert result.mapping_method == "table_grid"
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "01summer08_q01")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["1"]
    assert debug["crop_box"][0]["y0"] > 80
    assert debug["crop_box"][0]["y1"] <= 206

    with Image.open(result.image_path) as image:
        width, height = image.size
    assert width > 1500
    assert height > 330


def test_legacy_2008_q03_mark_scheme_excludes_previous_and_next_questions(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S08_P1_MS.exists():
        pytest.skip("Repo 2008 P1 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "3": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm1",
            year="2008",
            session="s08",
            component="01",
            question_number="3",
        )
    }

    result = render_mark_scheme_images(
        REPO_S08_P1_MS,
        config,
        ["3"],
        question_marks={"3": 6},
        question_subparts={"3": ["i", "ii"]},
        question_identities=identities,
    )["3"]

    assert result.mapping_status == "pass"
    assert result.mapping_method == "table_grid"
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "01summer08_q03")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["3"]
    assert debug["crop_box"][0]["y0"] >= 386
    assert debug["crop_box"][0]["y1"] <= 562

    with Image.open(result.image_path) as image:
        width, height = image.size
    assert width > 1500
    assert 480 < height < 620


def test_legacy_2008_winter_p1_row_bands_use_anchor_labels_at_boundaries(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_W08_P1_MS.exists():
        pytest.skip("Repo 2008 winter P1 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        number: paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm1",
            year="2008",
            session="w08",
            component="01",
            question_number=number,
        )
        for number in ["1", "2", "5", "6"]
    }

    results = render_mark_scheme_images(
        REPO_W08_P1_MS,
        config,
        ["1", "2", "5", "6"],
        question_marks={"1": 3, "2": 4, "5": 6, "6": 5},
        question_subparts={number: [] for number in ["1", "2", "5", "6"]},
        question_identities=identities,
    )

    assert {number: result.mapping_status for number, result in results.items()} == {
        "1": "pass",
        "2": "pass",
        "5": "pass",
        "6": "pass",
    }
    debug_records = _debug_records(config)
    assert _record(debug_records, "01winter08_q01")["detected_primary_questions_in_left_column"] == ["1"]
    assert _record(debug_records, "01winter08_q02")["detected_primary_questions_in_left_column"] == ["2"]
    assert _record(debug_records, "01winter08_q05")["detected_primary_questions_in_left_column"] == ["5"]
    assert _record(debug_records, "01winter08_q06")["detected_primary_questions_in_left_column"] == ["6"]

    for result in results.values():
        assert result.image_path and result.image_path.exists()
        with Image.open(result.image_path) as image:
            assert image.width > 1400
            assert image.height > 100


def test_modern_2025_q08_mark_scheme_keeps_all_subparts_without_neighbors(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_W25_P35_MS.exists():
        pytest.skip("Repo 2025 P35 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "8": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2025",
            session="w25",
            component="35",
            question_number="8",
        )
    }

    result = render_mark_scheme_images(
        REPO_W25_P35_MS,
        config,
        ["8"],
        question_marks={"8": 9},
        question_subparts={"8": ["a", "b"]},
        question_identities=identities,
    )["8"]

    assert result.mapping_status == "pass"
    assert result.markscheme_subparts == ["a", "b"]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "35winter25_q08")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["8(a)", "8(b)"]
    assert debug["page_numbers"] == [16]

    with Image.open(result.image_path) as image:
        width, height = image.size
    assert width > 2200
    assert 1300 < height < 1550


def test_modern_2025_p32_q08_mark_scheme_preserves_guidance_column(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S25_P32_MS.exists():
        pytest.skip("Repo 2025 P32 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "8": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2025",
            session="s25",
            component="32",
            question_number="8",
        )
    }

    result = render_mark_scheme_images(
        REPO_S25_P32_MS,
        config,
        ["8"],
        question_marks={"8": 7},
        question_subparts={"8": []},
        question_identities=identities,
    )["8"]

    assert result.mapping_status == "pass"
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "32summer25_q08")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["8"]
    assert debug["page_numbers"] == [18]

    with Image.open(result.image_path) as image:
        grayscale = image.convert("L")
        mask = grayscale.point(lambda pixel: 255 if pixel < 245 else 0, mode="1")
        content_box = mask.getbbox()
        assert content_box is not None
        assert content_box[2] > image.width * 0.9


def test_modern_2020_p31_q05_ignores_generic_misread_policy_row(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S20_P31_MS.exists():
        pytest.skip("Repo 2020 P31 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "5": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2020",
            session="s20",
            component="31",
            question_number="5",
        )
    }

    result = render_mark_scheme_images(
        REPO_S20_P31_MS,
        config,
        ["5"],
        question_marks={"5": 8},
        question_subparts={"5": ["a", "b"]},
        question_identities=identities,
    )["5"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [8]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "31summer20_q05")
    assert debug["validation_passed"] is True
    assert debug["crop_box"][0]["y1"] - debug["crop_box"][0]["y0"] > 250

    with Image.open(result.image_path) as image:
        assert image.height > 700


def test_modern_2020_p31_q01_ignores_mathematics_specific_policy_row(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S20_P31_MS.exists():
        pytest.skip("Repo 2020 P31 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "1": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2020",
            session="s20",
            component="31",
            question_number="1",
        )
    }

    result = render_mark_scheme_images(
        REPO_S20_P31_MS,
        config,
        ["1"],
        question_marks={"1": 4},
        question_subparts={"1": []},
        question_identities=identities,
    )["1"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [6]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "31summer20_q01")
    assert debug["validation_passed"] is True
    assert debug["crop_box"][0]["y1"] - debug["crop_box"][0]["y0"] > 100

    with Image.open(result.image_path) as image:
        assert image.height > 350


def test_modern_2020_p33_q01_includes_label_column_outside_answer_grid(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S20_P33_MS.exists():
        pytest.skip("Repo 2020 P33 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "1": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2020",
            session="s20",
            component="33",
            question_number="1",
        )
    }

    result = render_mark_scheme_images(
        REPO_S20_P33_MS,
        config,
        ["1"],
        question_marks={"1": 4},
        question_subparts={"1": []},
        question_identities=identities,
    )["1"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [6]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "33summer20_q01")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["1"]
    assert debug["crop_box"][0]["x0"] < 110

    with Image.open(result.image_path) as image:
        assert image.height > 650


def test_modern_2021_p32_q04_preserves_marks_and_guidance_columns(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S21_P32_MS.exists():
        pytest.skip("Repo 2021 P32 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "4": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2021",
            session="s21",
            component="32",
            question_number="4",
        )
    }

    result = render_mark_scheme_images(
        REPO_S21_P32_MS,
        config,
        ["4"],
        question_marks={"4": 5},
        question_subparts={"4": []},
        question_identities=identities,
    )["4"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [8, 9]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "32summer21_q04")
    assert debug["validation_passed"] is True
    assert debug["crop_box"][0]["x1"] - debug["crop_box"][0]["x0"] > 700

    with Image.open(result.image_path) as image:
        assert image.width > 2100
        assert image.height > 1400


def test_modern_2018_p32_q03_stops_before_following_table_header(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_W18_P32_MS.exists():
        pytest.skip("Repo 2018 P32 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "3": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2018",
            session="w18",
            component="32",
            question_number="3",
        )
    }

    result = render_mark_scheme_images(
        REPO_W18_P32_MS,
        config,
        ["3"],
        question_marks={"3": 5},
        question_subparts={"3": ["i", "ii"]},
        question_identities=identities,
    )["3"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [7, 8]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "32winter18_q03")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["3(i)", "3(ii)"]
    assert debug["crop_box"][1]["y1"] < 210

    with Image.open(result.image_path) as image:
        assert image.width > 2200
        assert 850 < image.height < 1000


def test_modern_2019_p33_q03_stops_before_following_table_header(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S19_P33_MS.exists():
        pytest.skip("Repo 2019 P33 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "3": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2019",
            session="s19",
            component="33",
            question_number="3",
        )
    }

    result = render_mark_scheme_images(
        REPO_S19_P33_MS,
        config,
        ["3"],
        question_marks={"3": 7},
        question_subparts={"3": ["i", "ii"]},
        question_identities=identities,
    )["3"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [7]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), "33summer19_q03")
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["3(i)", "3(ii)"]
    assert debug["crop_box"][0]["y1"] < 330

    with Image.open(result.image_path) as image:
        assert image.width > 2200
        assert 740 < image.height < 840


@pytest.mark.parametrize(
    ("source_pdf", "component", "question_id"),
    [
        (REPO_W17_P31_MS, "31", "31winter17_q08"),
        (REPO_W17_P33_MS, "33", "33winter17_q08"),
    ],
)
def test_modern_2017_winter_q08_stops_before_three_column_table_header(
    tmp_path: Path, source_pdf: Path, component: str, question_id: str
) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not source_pdf.exists():
        pytest.skip(f"Repo 2017 winter P{component} mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "8": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2017",
            session="w17",
            component=component,
            question_number="8",
        )
    }

    result = render_mark_scheme_images(
        source_pdf,
        config,
        ["8"],
        question_marks={"8": 9},
        question_subparts={"8": ["i", "ii"]},
        question_identities=identities,
    )["8"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [7]
    assert result.image_path and result.image_path.exists()
    debug = _record(_debug_records(config), question_id)
    assert debug["validation_passed"] is True
    assert debug["detected_primary_questions_in_left_column"] == ["8"]
    assert debug["crop_box"][0]["y1"] < 395

    with Image.open(result.image_path) as image:
        assert image.width > 1500
        assert 850 < image.height < 1050


def test_modern_2017_p33_q04_formulaic_fallback_stops_before_q05(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image")
    if not REPO_S17_P33_MS.exists():
        pytest.skip("Repo 2017 P33 mark scheme PDF is not available.")

    config = _config(tmp_path)
    identities = {
        "4": paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year="2017",
            session="s17",
            component="33",
            question_number="4",
        )
    }

    result = render_mark_scheme_images(
        REPO_S17_P33_MS,
        config,
        ["4"],
        question_marks={"4": 4},
        question_subparts={"4": []},
        question_identities=identities,
    )["4"]

    assert result.mapping_status == "pass"
    assert result.page_numbers == [5]
    assert result.image_path and result.image_path.exists()

    with Image.open(result.image_path) as image:
        assert 250 < image.height < 700
