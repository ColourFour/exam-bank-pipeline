from __future__ import annotations

import copy
import hashlib
from importlib import resources
import json
from pathlib import Path

import pytest

from exam_bank.command import main
from exam_bank.question_interchange import (
    QuestionInterchangeError,
    export_question_interchange,
    validate_question_interchange,
    validate_question_interchange_file,
)


def _question_record(*, question_id: str = "9709_s26_qp_12_q03a") -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "9709_s26_qp_12",
        "canonical_paper_id": "9709_s26_qp_12",
        "canonical_session": "s26",
        "canonical_year_folder": "2026",
        "question_number": "3(a)",
        "paper_family": "p1",
        "topic": "algebra",
        "question_text": "Find x.",
        "canonical_question_artifact": "p1/questions/q03a.png",
        "canonical_mark_scheme_artifact": "p1/mark_schemes/q03a.png",
        "mark_scheme_text": "Valid method and answer.",
        "question_solution_marks": 2,
        "rubric_status": "included",
        "rubric": [
            {
                "mark_id": "method-1",
                "mark_code": "M1",
                "mark_type": "M",
                "max_marks": 1,
                "criteria": "Uses a valid method.",
            },
            {
                "mark_id": "method-2",
                "mark_code": "M1",
                "mark_type": "M",
                "max_marks": 1,
                "criteria": "Completes the method.",
                "depends_on": ["method-1"],
            },
        ],
        "mark_scheme_confidence_score": 0.94,
        "notes": {
            "subtopic": "equations",
            "mapping_status": "pass",
            "validation_status": "pass",
            "question_text_trust": "high",
            "visual_curation_status": "ready",
            "text_only_status": "ready",
            "review_flags": [],
        },
    }


def _write_question_bank(path: Path, questions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 2,
                "record_count": len(questions),
                "questions": questions,
            }
        ),
        encoding="utf-8",
    )


def _write_assets(root: Path) -> None:
    for relative_path in ("p1/questions/q03a.png", "p1/mark_schemes/q03a.png"):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-image")


def test_export_builds_valid_self_identifying_question_handoff(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    question_bank_path = artifact_root / "json" / "question_bank.json"
    output_path = tmp_path / "handoff" / "questions.v1.json"
    _write_question_bank(question_bank_path, [_question_record()])
    _write_assets(artifact_root)

    payload = export_question_interchange(
        question_bank_path=question_bank_path,
        output_path=output_path,
        artifact_root=artifact_root,
        generated_at="2026-08-11T12:00:00Z",
        check_assets=True,
    )

    question = payload["questions"][0]
    assert payload["schema_name"] == "exam_bank.interchange.questions"
    assert payload["schema_version"] == 1
    assert payload["asset_root"] == "../output"
    assert payload["record_count"] == payload["source"]["record_count"] == 1
    assert question["schema_name"] == "exam_bank.interchange.question"
    assert question["schema_version"] == 1
    assert question["question_number"] == "3(a)"
    assert [item["mark_code"] for item in question["rubric"]] == ["M1", "M1"]
    assert [item["mark_id"] for item in question["rubric"]] == ["method-1", "method-2"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload

    report = validate_question_interchange_file(output_path, check_assets=True)
    assert report["ok"] is True
    assert report["error_count"] == 0


def test_validation_rejects_ambiguous_records_duplicate_ids_and_unsafe_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    question_bank_path = artifact_root / "json" / "question_bank.json"
    output_path = tmp_path / "handoff" / "questions.v1.json"
    _write_question_bank(question_bank_path, [_question_record()])
    _write_assets(artifact_root)
    payload = export_question_interchange(
        question_bank_path=question_bank_path,
        output_path=output_path,
        artifact_root=artifact_root,
        generated_at="2026-08-11T12:00:00Z",
        check_assets=True,
    )

    invalid = copy.deepcopy(payload)
    invalid["questions"][0].pop("schema_version")
    invalid["questions"][0]["question_image"] = "../outside.png"
    invalid["questions"].append(copy.deepcopy(invalid["questions"][0]))
    invalid["record_count"] = 2
    invalid["source"]["record_count"] = 2

    report = validate_question_interchange(invalid)

    assert report["ok"] is False
    assert "question[0]:schema_version_mismatch" in report["errors"]
    assert "question[0]:question_image_contains_parent_traversal" in report["errors"]
    assert any(error.startswith("question[1]:duplicate_question_id:") for error in report["errors"])


def test_export_fails_closed_for_duplicate_source_ids_and_missing_assets(tmp_path: Path) -> None:
    question_bank_path = tmp_path / "output" / "json" / "question_bank.json"
    duplicate = _question_record()
    _write_question_bank(question_bank_path, [_question_record(), duplicate])

    with pytest.raises(QuestionInterchangeError, match="must be unique"):
        export_question_interchange(question_bank_path=question_bank_path, output_path=tmp_path / "duplicate.json")

    _write_question_bank(question_bank_path, [_question_record()])
    output_path = tmp_path / "missing-assets.json"
    with pytest.raises(QuestionInterchangeError, match="file_missing"):
        export_question_interchange(
            question_bank_path=question_bank_path,
            output_path=output_path,
            artifact_root=tmp_path / "output",
            check_assets=True,
        )
    assert not output_path.exists()


def test_public_cli_exports_and_validates_questions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_root = tmp_path / "output"
    question_bank_path = artifact_root / "json" / "question_bank.json"
    output_path = tmp_path / "handoff" / "questions.v1.json"
    _write_question_bank(question_bank_path, [_question_record()])
    _write_assets(artifact_root)

    assert (
        main(
            [
                "data",
                "export-questions",
                "--input",
                str(question_bank_path),
                "--output",
                str(output_path),
                "--artifact-root",
                str(artifact_root),
                "--check-assets",
            ]
        )
        == 0
    )
    export_report = json.loads(capsys.readouterr().out)
    assert export_report["ok"] is True
    assert export_report["record_count"] == 1

    assert main(["data", "validate-questions", "--input", str(output_path), "--check-assets"]) == 0
    validation_report = json.loads(capsys.readouterr().out)
    assert validation_report["ok"] is True


def test_schema_contracts_self_identify_and_packaged_question_copy_is_exact() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    expected_names = {
        "question.v1.schema.json": "exam_bank.interchange.question",
        "submission.v1.schema.json": "homework_ingest.interchange.submission",
        "grade-result.v1.schema.json": "autograder.interchange.grade_result",
    }
    for filename, schema_name in expected_names.items():
        schema = json.loads((repository_root / "schemas" / filename).read_text(encoding="utf-8"))
        assert {"schema_name", "schema_version"} <= set(schema["required"])
        assert schema["properties"]["schema_name"] == {"const": schema_name}
        assert schema["properties"]["schema_version"] == {"const": 1}

    authoritative = (repository_root / "schemas" / "question.v1.schema.json").read_bytes()
    packaged = resources.files("exam_bank").joinpath("schemas", "question.v1.schema.json").read_bytes()
    assert packaged == authoritative
    assert hashlib.sha256(packaged).hexdigest() == hashlib.sha256(authoritative).hexdigest()
