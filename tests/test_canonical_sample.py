from __future__ import annotations

import pytest

from exam_bank.canonical_sample import build_canonical_sample, validate_sample


def _row(year: int, paper: str, component: str, session_code: str, variant: int | None = None) -> dict[str, object]:
    suffix = f"{session_code}{str(year)[-2:]}-{component}"
    return {
        "board": "CAIE",
        "component": component,
        "id": f"caie-9709-{paper}-{suffix}",
        "mark_scheme_url": f"https://example.test/9709_{session_code}{str(year)[-2:]}_ms_{component}.pdf",
        "paper": paper,
        "paper_name": paper.replace("_", " ").title(),
        "qualification": "A Level",
        "question_paper_url": f"https://example.test/9709_{session_code}{str(year)[-2:]}_qp_{component}.pdf",
        "session": {"m": "March", "s": "May/June", "w": "October/November"}[session_code],
        "session_code": session_code,
        "source": "fixture",
        "source_page": f"https://example.test/{year}/{session_code}",
        "subject": "Mathematics",
        "syllabus": "9709",
        "variant": variant,
        "year": year,
    }


def test_build_canonical_sample_selects_exactly_one_per_available_type_per_year() -> None:
    rows = [
        _row(2008, "pure_math_1", "1", "s"),
        _row(2008, "pure_math_1", "1", "w"),
        _row(2008, "pure_math_3", "3", "s"),
        _row(2008, "mechanics_1", "4", "w"),
        _row(2009, "pure_math_1", "11", "s", 1),
        _row(2009, "pure_math_1", "12", "s", 2),
        _row(2009, "pure_math_3", "32", "m", 2),
        _row(2009, "statistics_1", "62", "w", 2),
    ]

    selected, report = build_canonical_sample(rows, year_start=2008, year_end=2009)

    selected_groups = [(row["year"], row["paper"]) for row in selected]
    assert len(selected_groups) == len(set(selected_groups))
    assert set(selected_groups) == {
        (2008, "pure_math_1"),
        (2008, "pure_math_3"),
        (2008, "mechanics_1"),
        (2009, "pure_math_1"),
        (2009, "pure_math_3"),
        (2009, "statistics_1"),
    }
    assert report["summary"]["total_selected"] == 6
    assert report["summary"]["missing_types_count"] == 2
    assert report["years"]["2008"]["missing_types"] == ["S1"]
    assert report["years"]["2009"]["missing_types"] == ["M1"]
    assert report["years"]["2009"]["selected"]["P1"]["variant"] == 2


def test_validate_sample_rejects_fabricated_or_duplicate_year_type_rows() -> None:
    rows = [
        _row(2008, "pure_math_1", "1", "s"),
        _row(2008, "pure_math_3", "3", "s"),
    ]

    with pytest.raises(ValueError, match="not present in source dataset"):
        validate_sample(rows, [_row(2008, "pure_math_1", "1", "w")], year_start=2008, year_end=2008)

    with pytest.raises(ValueError, match="Duplicate selected year/type"):
        validate_sample(rows, [rows[0], rows[0]], year_start=2008, year_end=2008)
