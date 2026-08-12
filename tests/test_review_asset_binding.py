from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exam_bank.review_asset_binding import bind_review_evidence_to_question_bank, main


def test_matching_review_assets_are_rebound_to_current_canonical_paths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, reviewed_bytes_match=True)

    bound = bind_review_evidence_to_question_bank(
        fixture["question_bank"],
        artifact_root=fixture["artifact_root"],
        base_dir=tmp_path,
        source_skill_payload=fixture["source"],
        mark_event_payload=fixture["marks"],
        content_lab_payload=fixture["content_lab"],
    )

    report = bound["report"]
    assert report["review_provenance_ok"] is True
    assert report["active_invalid_count"] == 0
    assert report["source_skill_records"]["rebound_path_count"] == 2
    assert report["mark_event_decisions"]["rebound_path_count"] == 4
    assert report["content_lab_records"]["rebound_path_count"] == 2
    assert bound["source_skill_payload"]["records"][0]["route_status"] == "clean"
    assert bound["mark_event_payload"]["decisions"][0]["status"] == "approved"
    assert bound["content_lab_payload"]["records"][0]["adjudication"]["status"] == "approved"
    assert (
        bound["content_lab_payload"]["records"][0]["canonical_question_image_path"]
        == "output/pm3/question.png"
    )


def test_changed_review_assets_are_demoted_without_mutating_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, reviewed_bytes_match=False)

    bound = bind_review_evidence_to_question_bank(
        fixture["question_bank"],
        artifact_root=fixture["artifact_root"],
        base_dir=tmp_path,
        source_skill_payload=fixture["source"],
        mark_event_payload=fixture["marks"],
        content_lab_payload=fixture["content_lab"],
    )

    assert bound["report"]["review_provenance_ok"] is False
    assert bound["report"]["fail_closed_applied"] is True
    assert bound["report"]["active_invalid_count"] == 3
    source = bound["source_skill_payload"]["records"][0]
    marks = bound["mark_event_payload"]["decisions"][0]
    content_lab = bound["content_lab_payload"]["records"][0]
    assert source["route_status"] == "blocked"
    assert source["allowed_use_cases"] == {"candidate_generation": False}
    assert "stale_review_asset_binding" in source["blockers"]
    assert marks["status"] == "advisory"
    assert marks["satisfies_generation_gate"] is False
    assert content_lab["adjudication"]["status"] == "blocked"
    assert "stale_review_asset_binding" in content_lab["risk_flags"]
    assert fixture["source"]["records"][0]["route_status"] == "clean"
    assert fixture["marks"]["decisions"][0]["status"] == "approved"


def test_cli_report_keeps_mark_event_results_separate_from_input_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, reviewed_bytes_match=True)
    inputs = {
        "question_bank": fixture["question_bank"],
        "source_skills": fixture["source"],
        "mark_events": fixture["marks"],
        "content_lab": fixture["content_lab"],
    }
    paths = {name: tmp_path / f"{name}.json" for name in inputs}
    for name, payload in inputs.items():
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "review_asset_binding_validation.json"

    exit_code = main(
        [
            "--question-bank",
            str(paths["question_bank"]),
            "--artifact-root",
            str(fixture["artifact_root"]),
            "--source-skills",
            str(paths["source_skills"]),
            "--mark-events",
            str(paths["mark_events"]),
            "--content-lab",
            str(paths["content_lab"]),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["mark_event_decisions"]["record_count"] == 1
    assert report["mark_event_decisions_path"] == str(paths["mark_events"])
    assert report["source_skill_decisions_path"] == str(paths["source_skills"])
    assert report["content_lab_decisions_path"] == str(paths["content_lab"])


def _fixture(tmp_path: Path, *, reviewed_bytes_match: bool) -> dict[str, object]:
    artifact_root = tmp_path / "output"
    current_question = artifact_root / "pm3" / "question.png"
    current_mark = artifact_root / "pm3" / "mark.png"
    current_question.parent.mkdir(parents=True)
    current_question.write_bytes(b"current-question")
    current_mark.write_bytes(b"current-mark")

    legacy_question = artifact_root / "p3" / "question.png"
    legacy_mark = artifact_root / "p3" / "mark.png"
    legacy_question.parent.mkdir(parents=True)
    legacy_question.write_bytes(
        b"current-question" if reviewed_bytes_match else b"reviewed-old-question"
    )
    legacy_mark.write_bytes(b"current-mark" if reviewed_bytes_match else b"reviewed-old-mark")

    question_hash = _sha256(legacy_question)
    mark_hash = _sha256(legacy_mark)
    question_bank = {
        "questions": [
            {
                "question_id": "31summer24_q01",
                "question_image_paths": ["pm3/question.png"],
                "mark_scheme_image_paths": ["pm3/mark.png"],
            }
        ]
    }
    source = {
        "records": [
            {
                "evidence_id": "evidence-1",
                "question_id": "31summer24_q01",
                "route_status": "clean",
                "blockers": [],
                "allowed_use_cases": {"candidate_generation": True},
                "source_question_asset_refs": [
                    {"path": "p3/question.png", "sha256": question_hash, "verified": True}
                ],
                "source_mark_scheme_asset_refs": [
                    {"path": "p3/mark.png", "sha256": mark_hash, "verified": True}
                ],
            }
        ]
    }
    marks = {
        "decisions": [
            {
                "decision_id": "decision-1",
                "source_question_id": "31summer24_q01",
                "status": "approved",
                "satisfies_generation_gate": True,
                "question_image_path": str(legacy_question),
                "question_image_ref": {
                    "path": "p3/question.png",
                    "sha256": question_hash,
                    "verified": True,
                },
                "mark_scheme_image_path": str(legacy_mark),
                "mark_scheme_image_ref": {
                    "path": "p3/mark.png",
                    "sha256": mark_hash,
                    "verified": True,
                },
            }
        ]
    }
    content_lab = {
        "records": [
            {
                "decision_id": "content-lab-1",
                "question_id": "31summer24_q01",
                "canonical_question_image_path": str(legacy_question),
                "canonical_mark_scheme_image_path": str(legacy_mark),
                "adjudication": {"status": "approved"},
                "risk_flags": [],
            }
        ]
    }
    return {
        "artifact_root": artifact_root,
        "current_question": current_question,
        "current_mark": current_mark,
        "question_bank": question_bank,
        "source": source,
        "marks": marks,
        "content_lab": content_lab,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
