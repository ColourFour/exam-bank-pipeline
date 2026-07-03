from __future__ import annotations

from exam_bank.question_regeneration import _selected_record


def test_question_regeneration_identity_uses_concrete_source_pdf_session_code() -> None:
    record = _selected_record(
        {
            "question_id": "12summer19_q05",
            "question_number": "5",
            "paper_family": "pm1",
            "canonical_year_folder": "2019",
            "canonical_session": "summer19",
            "canonical_question_artifact": "pm1/pm1_2019_m19_12_qp_q05_question.png",
            "notes": {
                "source_pdf": "input/pastpapers/9709/2019/question_papers/9709_m19_qp_12.pdf",
                "source_paper_code": "12",
            },
        }
    )

    assert record.identity.session_code == "m19"
    assert record.identity.component == "12"
