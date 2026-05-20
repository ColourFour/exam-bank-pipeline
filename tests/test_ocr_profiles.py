from __future__ import annotations

from pathlib import Path

from PIL import Image

from exam_bank.ocr_profiles import available_ocr_profiles, run_profile_ocr
from scripts.experiment_ocr_profiles import build_experiment_report, score_profile_text


def test_available_profiles_include_requested_experimental_names() -> None:
    names = [profile.name for profile in available_ocr_profiles()]

    assert names == [
        "baseline_current",
        "grayscale_threshold",
        "formula_heavy",
        "table_preserving",
        "diagram_safe",
        "padding_variant",
        "dense_algebra",
    ]


def test_run_profile_ocr_accepts_injected_runner_without_changing_production_ocr(tmp_path: Path) -> None:
    profile = next(profile for profile in available_ocr_profiles() if profile.name == "padding_variant")
    image_path = tmp_path / "q01.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    run = run_profile_ocr(
        image_path,
        profile,
        image_loader=lambda _path: Image.new("RGB", (20, 20), "white"),
        tesseract_runner=lambda *_args, **_kwargs: "  1   Find x.   [2]\n",
    )

    assert run.ok is True
    assert run.profile == "padding_variant"
    assert run.text == "1 Find x. [2]"
    assert run.runtime_seconds >= 0


def test_score_profile_text_reports_improvement_and_regression_against_fixture_expectations() -> None:
    fixture = {
        "record_id": "sample_q01",
        "question_number": "1",
        "native_pdf_text_raw": "",
        "failure_tags": ["math_notation"],
        "expected_normalized_text_or_structural_expectations": {
            "expectations": [
                "starts with question number 1",
                "contains mark bracket [2]",
                "preserves dy/dx derivative notation",
            ]
        },
    }

    good = score_profile_text(fixture, "1 Find dy/dx. [2]", "formula_heavy")
    bad = score_profile_text(fixture, "Find dydx", "grayscale_threshold")

    assert good["fixture_score"] > bad["fixture_score"]
    assert "expected_structural_requirement_missing:question_number_start" not in good["issue_keys"]
    assert "expected_structural_requirement_missing:derivative_notation" not in good["issue_keys"]
    assert "expected_structural_requirement_missing:question_number_start" in bad["issue_keys"]
    assert "expected_structural_requirement_missing:derivative_notation" in bad["issue_keys"]


def test_build_experiment_report_is_fixture_scoped_and_does_not_write_exports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fixture_image = tmp_path / "p1" / "sample" / "questions" / "q01.png"
    fixture_image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(fixture_image)
    manifest = {
        "schema_name": "text_fidelity_bad_text_fixture_manifest",
        "schema_version": 1,
        "record_count": 1,
        "records": [
            {
                "record_id": "sample_q01",
                "paper_id": "sample",
                "paper_family": "p1",
                "question_number": "1",
                "question_image_path": "p1/sample/questions/q01.png",
                "currently_selected_text": "Find x",
                "native_pdf_text_raw": "",
                "failure_tags": [],
                "expected_normalized_text_or_structural_expectations": {
                    "expectations": ["starts with question number 1", "contains mark bracket [2]"]
                },
            }
        ],
    }

    def fake_run_profile_ocr(_image_path, profile, **_kwargs):
        from exam_bank.ocr_profiles import OCRProfileRun

        text = "1 Find x. [2]" if profile.name == "padding_variant" else "Find x"
        return OCRProfileRun(profile=profile.name, text=text, runtime_seconds=0.01, ok=True)

    monkeypatch.setattr("scripts.experiment_ocr_profiles.run_profile_ocr", fake_run_profile_ocr)

    report = build_experiment_report(manifest, image_root=tmp_path)

    assert report["scope"] == "frozen_bad_text_fixtures_only"
    assert report["production_behavior_unchanged"] is True
    assert report["writes_question_bank"] is False
    assert report["records"][0]["best_profile"] == "padding_variant"
    padding = next(row for row in report["records"][0]["profiles"] if row["profile"] == "padding_variant")
    assert padding["comparison_vs_baseline"] == "improved"
