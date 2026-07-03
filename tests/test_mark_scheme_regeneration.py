from __future__ import annotations

from exam_bank.mark_scheme_regeneration import _record_match_keys, _selected_record


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


def test_mark_scheme_regeneration_uses_nested_canonical_asset_identity() -> None:
    record = {
        "question_id": "32winter09_q05",
        "question_number": "5",
        "paper_family": "pm3",
        "canonical_year_folder": "2009",
        "canonical_session": "winter09",
        "canonical_mark_scheme_artifact": "",
        "mark_scheme_image_path": "",
        "notes": {
            "mark_scheme_source_pdf": "input/pastpapers/9709/2009/mark_schemes/9709_w09_ms_32.pdf",
            "source_paper_code": "32",
            "mark_scheme_structure_detected": {
                "asset_identity": {
                    "canonical_path": "pm3/pm3_2009_w09_32_ms_q05_markscheme.png",
                }
            },
        },
    }

    selected = _selected_record(record)

    assert selected.identity.session_code == "w09"
    assert selected.identity.component == "32"
    assert "pm3_2009_w09_32_ms_q05" in _record_match_keys(record)
