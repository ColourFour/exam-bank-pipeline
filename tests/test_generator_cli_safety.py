from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(script_name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{script_name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "write_function"),
    [
        ("generate_topic_filter_maps", "write_json"),
        ("generate_skill_maps", "dump_json"),
    ],
)
def test_generator_help_does_not_read_or_write(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script_name: str,
    write_function: str,
) -> None:
    module = load_script_module(script_name)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--help must not read inputs or write outputs")

    monkeypatch.setattr(module, "load_json", fail)
    monkeypatch.setattr(module, write_function, fail)

    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])

    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_generate_skill_maps_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = load_script_module("generate_skill_maps")
    question_bank = tmp_path / "question_bank.json"
    asterion_question_bank = tmp_path / "asterion_question_bank_v1.json"
    content_lab_candidates = tmp_path / "asterion_content_lab_candidates_v1.json"

    def fake_load_json(path: Path) -> dict[str, object]:
        if path == question_bank:
            return {
                "questions": [
                    {
                        "question_id": "q1",
                        "paper": "11summer21",
                        "question_number": 1,
                        "topic": "quadratics",
                        "notes": {"topic_confidence": "high", "topic_uncertain": False},
                        "difficulty": "medium",
                    }
                ]
            }
        if path == asterion_question_bank:
            return {
                "questions": [
                    {
                        "question_id": "q1",
                        "paper": "11summer21",
                        "question_number": 1,
                        "paper_family": "p1",
                        "subparts": [
                            {
                                "subpart_id": "q1_a",
                                "label": "a",
                                "marks": 2,
                                "question_text": {
                                    "text": "complete the square completed square turning point",
                                    "trust_level": "high",
                                },
                                "mark_scheme_text": {
                                    "text": "completed square",
                                    "trust_level": "high",
                                },
                                "review_status": "needs_review",
                            }
                        ],
                    }
                ]
            }
        if path == content_lab_candidates:
            return {"candidates": []}
        raise AssertionError(f"Unexpected input read: {path}")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--dry-run must not write outputs")

    monkeypatch.setattr(module, "load_json", fake_load_json)
    monkeypatch.setattr(module, "dump_json", fail_write)
    monkeypatch.setattr(module, "load_git_baseline", lambda _path: {})

    exit_code = module.main(
        [
            "--dry-run",
            "--question-bank",
            str(question_bank),
            "--asterion-question-bank",
            str(asterion_question_bank),
            "--content-lab-candidates",
            str(content_lab_candidates),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["dry_run"] is True
    assert summary["components"]["p1"]["mapped_subparts"] == 1
    assert "exam_bank_taxonomy/canonical/indexes/skill_map_index_v1.json" in summary["would_write"]


def test_generate_topic_filter_maps_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    ) -> None:
    module = load_script_module("generate_topic_filter_maps")
    skill_map_index = tmp_path / "skill_map_index_v1.json"
    asterion_question_bank = tmp_path / "missing_asterion_question_bank_v1.json"
    legacy_question_bank = tmp_path / "missing_question_bank.json"

    def fake_load_json(path: Path) -> dict[str, object]:
        if path == skill_map_index:
            return {
                "source_syllabus_reference": "test syllabus",
                "source_syllabus_url": "https://example.test/syllabus.pdf",
                "components": [
                    {
                        "skill_map_file": "skill_map_9709_p1_v1.json",
                        "mapping_file": "question_skill_mappings_9709_p1_v1.json",
                        "coverage_report_file": "coverage_report_9709_p1_v1.json",
                    }
                ],
            }
        if path.name == "skill_map_9709_p1_v1.json":
            return {
                "syllabus_code": "9709",
                "subject_name": "Cambridge International AS & A Level Mathematics",
                "caie_class_or_component": "Paper 1",
                "component_label": "Pure Mathematics 1",
                "sections": [{"section": "1.1", "name": "Quadratics"}],
                "skills": [
                    {
                        "skill_id": "9709_p1_quadratics_completed_square",
                        "syllabus_code": "9709",
                        "subject_name": "Cambridge International AS & A Level Mathematics",
                        "caie_class_or_component": "Paper 1",
                        "component_label": "Pure Mathematics 1",
                        "section": "1.1 Quadratics",
                        "name": "Completing the square",
                        "description": "Rewrite quadratic expressions in completed-square form.",
                        "assessment_role": "direct_assessed",
                        "recognizer_signals": ["complete the square"],
                        "common_errors": [],
                        "content_lab_priority": "high",
                    }
                ],
            }
        if path.name == "question_skill_mappings_9709_p1_v1.json":
            return {
                "mappings": [
                    {
                        "question_id": "q1",
                        "subpart_id": "q1_a",
                        "paper_id": "11summer21",
                        "syllabus_code": "9709",
                        "subject_name": "Cambridge International AS & A Level Mathematics",
                        "caie_class_or_component": "Paper 1",
                        "component_label": "Pure Mathematics 1",
                        "year": 2021,
                        "session": "June",
                        "variant": "1",
                        "question_number": "1",
                        "subpart_label": "a",
                        "primary_skill_ids": ["9709_p1_quadratics_completed_square"],
                        "secondary_skill_ids": [],
                        "prerequisite_skill_ids": [],
                        "confidence": 0.86,
                        "evidence_granularity": "subpart",
                        "mapping_source": "question_text_inferred",
                        "review_status": "needs_review",
                        "evidence": {
                            "counts_as_direct_readiness_evidence": True,
                            "matched_signals": ["signal:complete the square"],
                            "question_text_snippet": "complete the square",
                            "mark_scheme_text_snippet": "completed square",
                            "question_text_trust": "high",
                            "mark_scheme_text_trust": "high",
                        },
                    }
                ]
            }
        raise AssertionError(f"Unexpected input read: {path}")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--dry-run must not write outputs")

    monkeypatch.setattr(module, "load_json", fake_load_json)
    monkeypatch.setattr(module, "write_json", fail_write)

    exit_code = module.main(
        [
            "--dry-run",
            "--skill-map-index",
            str(skill_map_index),
            "--asterion-question-bank",
            str(asterion_question_bank),
            "--legacy-question-bank",
            str(legacy_question_bank),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["dry_run"] is True
    assert summary["components"]["p1"]["assignments"] == 1
    assert "exam_bank_taxonomy/canonical/indexes/topic_filter_map_index_v1.json" in summary["would_write"]
