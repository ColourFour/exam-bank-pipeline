from collections import Counter

from scripts.validate_asterion_all_course_export import (
    build_parser,
    course_count_mismatch_error,
    source_course_counts,
)


def test_validation_report_default_is_stable_and_versioned() -> None:
    args = build_parser().parse_args([])

    assert args.output == "output/asterion/exports/latest/asterion_all_course_export_validation.v1.json"


def test_source_course_counts_split_shared_stats_family_by_component() -> None:
    question_bank = {
        "questions": [
            {"paper_family": "mechanics", "paper": "42winter25"},
            {"paper_family": "stats", "paper": "51summer19"},
            {"paper_family": "stats", "paper": "51summer25"},
            {"paper_family": "stats", "paper": "63winter19"},
            {"paper_family": "stats", "paper": "62winter25"},
        ]
    }

    assert source_course_counts(question_bank) == {"m1": 1, "s1": 2, "s2": 1}


def test_course_count_mismatch_is_blocking_and_reports_era_aware_delta() -> None:
    error = course_count_mismatch_error(
        "Catalog",
        expected=Counter({"s1": 434}),
        actual=Counter({"s2": 434}),
    )

    assert error is not None
    assert "era-aware source counts" in error
    assert "'s1': 434" in error
    assert "'s2': 434" in error
