from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank.question_text_exact import (
    QuestionTextExactError,
    evaluate_question_text_exact,
    normalize_question_text_exact,
)
from scripts.evaluate_question_text_exact import main


def test_normalization_is_presentation_only() -> None:
    composed = "Caf\u00e9\r\nFind\u00a0 x  =\t2."
    decomposed = "Cafe\u0301\n Find x = 2."

    assert normalize_question_text_exact(composed) == "Caf\u00e9 Find x = 2."
    assert normalize_question_text_exact(composed) == normalize_question_text_exact(decomposed)

    assert normalize_question_text_exact("x \u2212 2") != normalize_question_text_exact("x - 2")
    assert normalize_question_text_exact("x = 2") != normalize_question_text_exact("x = 3")
    assert normalize_question_text_exact("Find x") != normalize_question_text_exact("find x")
    assert normalize_question_text_exact("(a) First (b) Second") != normalize_question_text_exact(
        "(b) Second (a) First"
    )


def test_exact_evaluation_reports_overall_family_and_mismatch_details() -> None:
    gold = {
        "records": [
            {"question_id": "q1", "question_text": "1 Find x = 2."},
            {"question_id": "q2", "question_text": "2 Solve y \u2265 3."},
            {"question_id": "q3", "question_text": "3 Caf\u00e9 question."},
        ]
    }
    candidate = {
        "questions": [
            {"question_id": "q1", "paper_family": "pm1", "question_text": " 1  Find x = 2.\r\n"},
            {"question_id": "q2", "paper_family": "pm1", "question_text": "2 Solve y > 3."},
            {"question_id": "q3", "paper_family": "stats", "question_text": "3 Cafe\u0301 question."},
            {"question_id": "outside", "paper_family": "pm3", "question_text": "ignored"},
        ]
    }

    report = evaluate_question_text_exact(gold, candidate, minimum_accuracy=2 / 3)

    assert report["schema_name"] == "exam_bank.question_text_exact_evaluation"
    assert report["schema_version"] == 1
    assert report["normalization"]["semantic_transformations"] is False
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["candidate_ids_outside_cohort_count"] == 1
    assert report["overall"] == {
        "exact_match_count": 2,
        "mismatch_count": 1,
        "total": 3,
        "accuracy": 2 / 3,
        "meets_minimum_accuracy": True,
    }
    assert report["by_family"]["pm1"]["exact_match_count"] == 1
    assert report["by_family"]["stats"]["exact_match_count"] == 1
    assert report["mismatches"] == [
        {
            "question_id": "q2",
            "paper_family": "pm1",
            "reason": "text_mismatch",
            "gold_text": "2 Solve y \u2265 3.",
            "candidate_text": "2 Solve y > 3.",
            "normalized_gold_text": "2 Solve y \u2265 3.",
            "normalized_candidate_text": "2 Solve y > 3.",
        }
    ]
    assert report["passed"] is True


def test_provided_cohort_requires_gold_candidate_and_valid_text_coverage() -> None:
    gold = {"records": [{"question_id": "q1", "question_text": "one"}]}
    candidate = {
        "questions": [
            {"question_id": "q1", "paper_family": "pm1", "question_text": "one"},
            {"question_id": "q2", "paper_family": "pm3"},
        ]
    }
    cohort = {
        "questions": [
            {"question_id": "q1", "paper_family": "pm1"},
            {"question_id": "q2", "paper_family": "pm3"},
            {"question_id": "q3", "paper_family": "stats"},
        ]
    }

    report = evaluate_question_text_exact(gold, candidate, cohort_sample=cohort, minimum_accuracy=0)

    assert report["coverage"]["complete"] is False
    assert report["coverage"]["missing_gold_question_ids"] == ["q2", "q3"]
    assert report["coverage"]["missing_candidate_question_ids"] == ["q3"]
    assert report["coverage"]["invalid_candidate_question_text_ids"] == ["q2"]
    assert report["overall"]["exact_match_count"] == 1
    assert report["overall"]["total"] == 3
    assert [row["reason"] for row in report["mismatches"]] == [
        "missing_gold_record",
        "missing_gold_record",
    ]
    assert report["passed"] is False


def test_gold_question_text_must_be_non_empty() -> None:
    with pytest.raises(QuestionTextExactError, match="non-empty question_text"):
        evaluate_question_text_exact(
            {"records": [{"question_id": "q1", "question_text": "  \n"}]},
            {"questions": [{"question_id": "q1", "question_text": "one"}]},
        )


@pytest.mark.parametrize("source", ["gold", "candidate", "cohort"])
def test_duplicate_question_ids_are_rejected(source: str) -> None:
    gold = {"records": [{"question_id": "q1", "question_text": "one"}]}
    candidate = {"questions": [{"question_id": "q1", "question_text": "one"}]}
    cohort = {"questions": [{"question_id": "q1"}]}
    if source == "gold":
        gold["records"].append({"question_id": "q1", "question_text": "one"})
    elif source == "candidate":
        candidate["questions"].append({"question_id": "q1", "question_text": "one"})
    else:
        cohort["questions"].append({"question_id": "q1"})

    with pytest.raises(QuestionTextExactError, match="Duplicate question_id"):
        evaluate_question_text_exact(gold, candidate, cohort_sample=cohort)


@pytest.mark.parametrize("minimum", [-0.01, 1.01, True, "0.85"])
def test_minimum_accuracy_must_be_numeric_and_bounded(minimum: object) -> None:
    with pytest.raises(QuestionTextExactError, match="minimum_accuracy"):
        evaluate_question_text_exact(
            {"records": [{"question_id": "q1", "question_text": "one"}]},
            {"questions": [{"question_id": "q1", "question_text": "one"}]},
            minimum_accuracy=minimum,  # type: ignore[arg-type]
        )


def test_cli_writes_report_and_returns_nonzero_below_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gold_path = tmp_path / "gold.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "report.json"
    gold_path.write_text(
        json.dumps({"records": [{"question_id": "q1", "question_text": "x = 2"}]}),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"questions": [{"question_id": "q1", "question_text": "x = 3"}]}),
        encoding="utf-8",
    )

    result = main(
        [
            "--gold",
            str(gold_path),
            "--candidate",
            str(candidate_path),
            "--minimum-accuracy",
            "0.85",
            "--output",
            str(output_path),
        ]
    )

    assert result == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["overall"]["accuracy"] == 0
    assert json.loads(capsys.readouterr().out) == report


def test_cli_returns_zero_when_complete_and_threshold_is_met(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    candidate_path = tmp_path / "candidate.json"
    payload = {"question_id": "q1", "question_text": "x = 2"}
    gold_path.write_text(json.dumps({"records": [payload]}), encoding="utf-8")
    candidate_path.write_text(json.dumps({"questions": [payload]}), encoding="utf-8")

    assert main(["--gold", str(gold_path), "--candidate", str(candidate_path)]) == 0


def test_cli_returns_nonzero_when_cohort_coverage_is_incomplete(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    candidate_path = tmp_path / "candidate.json"
    cohort_path = tmp_path / "cohort.json"
    gold_path.write_text(
        json.dumps({"records": [{"question_id": "q1", "question_text": "x = 2"}]}),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"questions": [{"question_id": "q1", "question_text": "x = 2"}]}),
        encoding="utf-8",
    )
    cohort_path.write_text(
        json.dumps({"questions": [{"question_id": "q1"}, {"question_id": "q2"}]}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--gold",
                str(gold_path),
                "--candidate",
                str(candidate_path),
                "--cohort",
                str(cohort_path),
                "--minimum-accuracy",
                "0",
            ]
        )
        == 1
    )
