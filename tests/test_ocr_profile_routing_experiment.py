from __future__ import annotations

from scripts.experiment_ocr_profiles import (
    infer_routing_slices,
    summarize_routing_slices,
)


def test_infer_routing_slices_covers_requested_paper_layout_and_failure_families() -> None:
    fixture = {
        "paper_family": "p3",
        "failure_tags": [
            "math_notation",
            "fraction_structure",
            "integral_bounds",
            "question_anchor_displaced",
            "mark_bracket_missing",
        ],
        "expected_normalized_text_or_structural_expectations": {
            "expectations": [
                "starts with question number 9",
                "preserves integral bounds",
                "contains mark bracket [6]",
            ]
        },
    }

    slices = {(row["slice_type"], row["slice"]) for row in infer_routing_slices(fixture)}

    assert ("paper_family", "P3") in slices
    assert ("layout_family", "dense_algebra") in slices
    assert ("layout_family", "calculus_integrals") in slices
    assert ("failure_type", "mark_bracket_recovery") in slices
    assert ("failure_type", "question_anchor_recovery") in slices
    assert ("failure_type", "symbol_heavy_cases") in slices


def test_summarize_routing_slices_separates_safe_improvements_from_regressions() -> None:
    records = [
        {
            "record_id": "p3_safe",
            "routing_slices": [{"slice_type": "layout_family", "slice": "dense_algebra"}],
            "profiles": [
                profile_row("baseline_current", 0),
                profile_row("formula_heavy", 20),
                profile_row("dense_algebra", 10),
            ],
        },
        {
            "record_id": "p3_regressed",
            "routing_slices": [{"slice_type": "layout_family", "slice": "dense_algebra"}],
            "profiles": [
                profile_row("baseline_current", 0),
                profile_row("formula_heavy", 0),
                profile_row("dense_algebra", -10),
            ],
        },
    ]

    summary = summarize_routing_slices(records)

    assert len(summary) == 1
    assert summary[0]["best_safe_profile"] == "formula_heavy"
    dense = next(row for row in summary[0]["profile_summary"] if row["profile"] == "dense_algebra")
    assert dense["safety_classification"] == "unsafe_regressions"
    assert summary[0]["regressions"] == [
        {
            "record_id": "p3_regressed",
            "profile": "dense_algebra",
            "delta": -10,
            "introduced_issue_keys": ["expected_structural_requirement_missing:theta_symbol"],
        }
    ]


def test_summarize_routing_slices_identifies_no_safe_profile_when_all_regress() -> None:
    records = [
        {
            "record_id": "symbol_heavy",
            "routing_slices": [{"slice_type": "failure_type", "slice": "symbol_heavy_cases"}],
            "profiles": [
                profile_row("baseline_current", 0),
                profile_row("formula_heavy", -20),
                profile_row("padding_variant", -5),
            ],
        }
    ]

    summary = summarize_routing_slices(records)

    assert summary[0]["best_safe_profile"] == ""
    assert summary[0]["no_safe_profile_reason"] == "all profiles regressed on at least one fixture in this slice"


def profile_row(profile: str, delta: int) -> dict[str, object]:
    return {
        "profile": profile,
        "fixture_score": 80 + delta,
        "score_delta_vs_baseline": delta,
        "runtime_seconds": 0.01,
        "introduced_issue_keys_vs_baseline": (
            ["expected_structural_requirement_missing:theta_symbol"] if delta < 0 else []
        ),
        "resolved_issue_keys_vs_baseline": (
            ["expected_structural_requirement_missing:question_number_start"] if delta > 0 else []
        ),
    }
