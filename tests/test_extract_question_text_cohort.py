from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_question_text_cohort import resolve_question_pdf, resolve_question_pdf_for_row


def test_resolve_question_pdf_supports_legacy_single_digit_variants(tmp_path: Path) -> None:
    expected = tmp_path / "2009" / "question_papers" / "9709_s09_qp_1.pdf"
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert resolve_question_pdf(tmp_path, "01summer09") == expected


def test_resolve_question_pdf_supports_two_digit_variants(tmp_path: Path) -> None:
    expected = tmp_path / "2025" / "question_papers" / "9709_w25_qp_33.pdf"
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert resolve_question_pdf(tmp_path, "33winter25") == expected


def test_resolve_question_pdf_rejects_unknown_paper_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported canonical paper identifier"):
        resolve_question_pdf(tmp_path, "not-a-paper")


def test_resolve_question_pdf_for_row_uses_rendered_source_session(tmp_path: Path) -> None:
    expected = tmp_path / "2021" / "question_papers" / "9709_m21_qp_11.pdf"
    expected.parent.mkdir(parents=True)
    expected.touch()

    row = {
        "paper": "11spring21",
        "question_image_path": "pm1/pm1_2021_m21_11_qp_q02_question.png",
    }

    assert resolve_question_pdf_for_row(tmp_path, row) == expected
