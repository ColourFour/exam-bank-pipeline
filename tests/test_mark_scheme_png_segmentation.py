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
REPO_W25_P35_MS = Path("input/pastpapers/9709/2025/mark_schemes/9709_w25_ms_35.pdf")


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
