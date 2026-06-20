from __future__ import annotations

from exam_bank.mark_scheme_regeneration import _selected_record


def test_mark_scheme_regeneration_identity_uses_canonical_artifact_session_code() -> None:
    record = _selected_record(
        {
            "question_id": "52summer20_q01",
            "question_number": "1",
            "paper_family": "mechanics",
            "canonical_year_folder": "2020",
            "canonical_session": "summer20",
            "canonical_mark_scheme_artifact": "mechanics/mechanics_2020_m20_52_ms_q01_markscheme.png",
            "notes": {
                "mark_scheme_source_pdf": "input/pastpapers/9709/2020/mark_schemes/9709_m20_ms_52.pdf",
                "source_paper_code": "52",
            },
        }
    )

    assert record.identity.session_code == "m20"
    assert record.identity.component == "52"
