from __future__ import annotations

import json
from pathlib import Path

from exam_bank.audit import audit_question_bank, write_audit


def test_visual_first_audit_reports_text_and_curation_distributions(tmp_path: Path) -> None:
    records = [
        {
            "question_id": "12spring24_q01",
            "question_number": "1",
            "question_text": "Sketch the graph of y x 3 6 = -.",
            "question_text_role": "untrusted_math_text",
            "question_text_trust": "low",
            "visual_required": True,
            "visual_reason_flags": ["contains_graph_or_diagram_prompt", "contains_math_text_corruption"],
            "visual_curation_status": "ready",
            "text_only_status": "fail",
            "question_image_path": "p1/12spring24/questions/q01.png",
            "notes": {"text_fidelity_status": "degraded"},
        },
        {
            "question_id": "12spring24_q02",
            "question_number": "2",
            "question_text": "A committee has 5 members chosen from 8 people.",
            "question_text_role": "readable_text",
            "question_text_trust": "high",
            "visual_required": False,
            "visual_reason_flags": [],
            "visual_curation_status": "ready",
            "text_only_status": "ready",
            "notes": {"text_fidelity_status": "clean"},
        },
        {
            "question_id": "12spring24_q03",
            "question_number": "3",
            "question_text": "Solve tan x = 1.",
            "question_text_role": "search_hint",
            "question_text_trust": "medium",
            "visual_required": True,
            "visual_reason_flags": ["contains_trig_expression", "contains_equation_layout"],
            "visual_curation_status": "review",
            "text_only_status": "review",
            "notes": {"text_fidelity_status": "clean"},
        },
    ]

    report = audit_question_bank(records)

    assert report["question_text_role_counts"] == {
        "readable_text": 1,
        "search_hint": 1,
        "untrusted_math_text": 1,
    }
    assert report["question_text_trust_counts"] == {"high": 1, "low": 1, "medium": 1}
    assert report["visual_required_counts"] == {"false": 1, "true": 2}
    assert report["visual_curation_status_counts"] == {"ready": 2, "review": 1}
    assert report["text_only_status_counts"] == {"fail": 1, "ready": 1, "review": 1}
    assert report["visual_reason_flag_counts"]["contains_equation_layout"] == 1
    assert report["examples_clean_text_fidelity_but_visual_required"][0]["question_id"] == "12spring24_q03"

    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "audit.json"
    input_path.write_text(json.dumps({"questions": records}), encoding="utf-8")
    written = write_audit(input_path, output_path)

    assert written["record_count"] == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["record_count"] == 3
