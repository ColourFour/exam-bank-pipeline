import pytest

from exam_bank.config import AppConfig
from exam_bank.core.asset_paths import AssetPathResolver
from exam_bank.core.paper_identity import (
    IdentityError,
    build_paper_id,
    build_question_id,
    canonical_subject_family,
    paper_identity_from_parts,
    parse_session,
)
from exam_bank.document_metadata import parse_filename_metadata
from exam_bank.mark_scheme_pairing import find_mark_scheme
from exam_bank.output_layout import mark_scheme_image_output_path, question_image_output_path


def test_m21_maps_to_summer21_with_compact_session_code() -> None:
    parsed = parse_session("m21")

    assert parsed.year == 2021
    assert parsed.session_code == "m21"
    assert parsed.canonical_session == "summer21"
    assert parsed.canonical_year_folder == "2021"


def test_pm1_identity_is_deterministic_across_calls() -> None:
    first = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm1",
        year=2021,
        session="m21",
        component="12",
        question_number="1",
    )
    second = paper_identity_from_parts(
        syllabus=9709,
        subject_family="p1",
        year="2021",
        session="March",
        component=12,
        question_number=1,
    )

    assert first == second
    assert first.paper_id == "12summer21"
    assert first.question_id == "12summer21_q01"
    assert first.subject_family == "pm1"


def test_same_file_metadata_produces_same_identity_every_time() -> None:
    path = "input/question_papers/9709 Mathematics March 2021 Question paper  12.pdf"
    first_metadata = parse_filename_metadata(path)
    second_metadata = parse_filename_metadata(path)

    first = paper_identity_from_parts(
        syllabus=first_metadata.syllabus or "9709",
        subject_family=first_metadata.paper_family,
        year=first_metadata.year,
        session=first_metadata.session,
        component=first_metadata.component,
        question_number="3",
    )
    second = paper_identity_from_parts(
        syllabus=second_metadata.syllabus or "9709",
        subject_family=second_metadata.paper_family,
        year=second_metadata.year,
        session=second_metadata.session,
        component=second_metadata.component,
        question_number="3",
    )

    assert first == second
    assert first.paper_id == "12summer21"
    assert first.question_id == "12summer21_q03"


def test_legacy_and_modern_subject_families_use_same_canonical_values() -> None:
    assert canonical_subject_family("p1") == "pm1"
    assert canonical_subject_family("pm1") == "pm1"
    assert canonical_subject_family("p3") == "pm3"
    assert canonical_subject_family("pm3") == "pm3"
    assert canonical_subject_family("p4") == "stats"
    assert canonical_subject_family("p6") == "stats"
    assert canonical_subject_family("stats") == "stats"
    assert canonical_subject_family("p5") == "mechanics"
    assert canonical_subject_family("mechanics") == "mechanics"


def test_builders_are_canonical_and_reject_inconsistent_identity() -> None:
    paper_id = build_paper_id("9709", "w22", "31")

    assert paper_id == "31winter22"
    assert build_question_id(paper_id, "7(a)") == "31winter22_q07"

    with pytest.raises(IdentityError, match="paper identity mismatch"):
        paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm3",
            year=2022,
            session="w22",
            component="31",
            question_number="7",
            expected_paper_id="31summer22",
        )

    with pytest.raises(IdentityError, match="session"):
        paper_identity_from_parts(
            syllabus="9709",
            subject_family="pm1",
            year=2021,
            session="w22",
            component="12",
        )


def test_asset_path_resolver_binds_paths_to_identity(tmp_path) -> None:
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="p1",
        year=2008,
        session="s08",
        component="1",
        question_number="1",
    )
    resolver = AssetPathResolver(tmp_path / "output")

    question_asset = resolver.question_image(identity)
    mark_asset = resolver.mark_scheme_image(identity)

    assert question_asset.question_id == "01summer08_q01"
    assert question_asset.paper_id == "01summer08"
    assert question_asset.component == "01"
    assert question_asset.canonical_path == "pm1/pm1_2008_s08_qp_q01_question.png"
    assert mark_asset.canonical_path == "pm1/pm1_2008_s08_ms_q01_markscheme.png"
    assert "unknown" not in question_asset.canonical_path
    assert question_asset.absolute_path == tmp_path / "output" / question_asset.canonical_path


def test_output_layout_uses_identity_resolver_without_unknown_paths(tmp_path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")

    qp_path = question_image_output_path("input/pastpapers/9709/2008/question_papers/9709_s08_qp_1.pdf", "1", config)
    ms_path = mark_scheme_image_output_path("input/pastpapers/9709/2008/mark_schemes/9709_s08_ms_1.pdf", "1", config)

    assert qp_path == tmp_path / "output" / "pm1" / "pm1_2008_s08_qp_q01_question.png"
    assert ms_path == tmp_path / "output" / "pm1" / "pm1_2008_s08_ms_q01_markscheme.png"
    assert "unknown" not in str(qp_path)
    assert "unknown" not in str(ms_path)


def test_mark_scheme_pairing_uses_paper_identity_not_fuzzy_names(tmp_path) -> None:
    qp = tmp_path / "9709_s08_qp_1.pdf"
    qp.write_text("fake", encoding="utf-8")
    ms_dir = tmp_path / "mark_schemes"
    ms_dir.mkdir()
    exact = ms_dir / "9709_s08_ms_1.pdf"
    exact.write_text("fake", encoding="utf-8")
    near_miss = ms_dir / "9709_s08_ms_3.pdf"
    near_miss.write_text("fake", encoding="utf-8")

    assert find_mark_scheme(qp, ms_dir) == exact

    exact.unlink()
    assert find_mark_scheme(qp, ms_dir) is None
