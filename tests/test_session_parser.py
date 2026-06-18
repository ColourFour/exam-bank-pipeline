from pathlib import Path

from exam_bank.document_metadata import parse_filename_metadata
from exam_bank.ingress.pastpapers_co import parse_pdf_resource
from exam_bank.output_layout import paper_instance_id
from exam_bank.shared.session_parser import parse_session


def test_canonical_session_parser_maps_compact_series_codes() -> None:
    assert parse_session("m21") == {
        "year": 2021,
        "session": "summer",
        "canonical_session": "summer21",
        "season": "m",
        "component_year_key": "m21",
        "canonical_year_folder": "2021",
    }
    assert parse_session("s08")["canonical_session"] == "summer08"
    assert parse_session("w22")["canonical_session"] == "winter22"


def test_session_outputs_are_deterministic_across_modules() -> None:
    metadata = parse_filename_metadata("9709_m21_qp_12.pdf")
    resource = parse_pdf_resource(
        "https://pastpapers.co/caie/a-level/mathematics-9709/2021-may-june/9709_m21_qp_12.pdf"
    )

    assert metadata.session == "summer21"
    assert metadata.year == "2021"
    assert resource is not None
    assert resource.canonical_session == "summer21"
    assert paper_instance_id(metadata.component, metadata.session, metadata.year) == "12summer21"


def test_no_legacy_session_fallback_tables_remain_in_parsing_path() -> None:
    root = Path(__file__).resolve().parents[1]
    document_metadata = (root / "src/exam_bank/document_metadata.py").read_text(encoding="utf-8")
    output_layout = (root / "src/exam_bank/output_layout.py").read_text(encoding="utf-8")
    ingress = (root / "src/exam_bank/ingress/pastpapers_co.py").read_text(encoding="utf-8")

    assert "SESSION_ALIASES" not in document_metadata
    assert "_SESSION_FOLDER_LABELS" not in output_layout
    assert "SERIES_NAMES" not in ingress
    assert "parse_two_digit_year" not in ingress
