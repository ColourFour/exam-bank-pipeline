from pathlib import Path

from exam_bank.config import AppConfig
from exam_bank import mark_schemes
from exam_bank import pipeline
from exam_bank.models import QuestionSpan


def test_analyze_mark_scheme_builds_one_reusable_paper_context(monkeypatch) -> None:
    source = Path("paper_ms.pdf")
    layouts = [object()]
    words = {1: [object()]}
    tables = {1: object()}
    anchors = [object()]
    calls: list[str] = []

    monkeypatch.setattr(
        mark_schemes,
        "extract_pdf_layout",
        lambda path, config: calls.append(f"layout:{path}") or layouts,
    )
    monkeypatch.setattr(
        mark_schemes,
        "_extract_mark_scheme_words",
        lambda path: calls.append(f"words:{path}") or words,
    )
    monkeypatch.setattr(
        mark_schemes,
        "_detect_mark_scheme_tables",
        lambda seen_layouts, config, seen_words: (
            calls.append("tables")
            or tables
            if seen_layouts is layouts and seen_words is words
            else None
        ),
    )
    monkeypatch.setattr(
        mark_schemes,
        "_detect_table_question_anchors",
        lambda seen_layouts, seen_tables, config, expected, seen_words: (
            calls.append("anchors")
            or anchors
            if seen_layouts is layouts and seen_tables is tables and seen_words is words
            else None
        ),
    )

    analysis = mark_schemes.analyze_mark_scheme(source, AppConfig())

    assert analysis.source_pdf == source
    assert analysis.layouts is layouts
    assert analysis.words_by_page is words
    assert analysis.tables is tables
    assert analysis.anchors is anchors
    assert calls == ["layout:paper_ms.pdf", "words:paper_ms.pdf", "tables", "anchors"]


def test_legacy_block_builder_reuses_analyzed_words(tmp_path, monkeypatch) -> None:
    source = tmp_path / "paper_ms.pdf"
    source.touch()
    words = {1: [object()]}

    monkeypatch.setattr(
        mark_schemes,
        "_extract_mark_scheme_words",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected word extraction: {path}")),
    )
    monkeypatch.setattr(mark_schemes, "_legacy_table_grid_row_bands", lambda *args: [])
    monkeypatch.setattr(mark_schemes, "_detect_legacy_mark_scheme_anchors", lambda *args: [])

    blocks = mark_schemes._build_legacy_mark_scheme_blocks(
        source,
        [],
        AppConfig(),
        [],
        question_marks={},
        question_subparts={},
        words_by_page=words,
    )

    assert blocks == {}


def test_pipeline_passes_one_mark_scheme_analysis_to_both_consumers(tmp_path, monkeypatch) -> None:
    question_pdf = tmp_path / "paper_qp.pdf"
    mark_scheme_pdf = tmp_path / "paper_ms.pdf"
    mark_scheme_pdf.touch()
    analysis = object()
    calls: list[tuple[str, object | None]] = []
    span = QuestionSpan(
        source_pdf=question_pdf,
        paper_name="paper",
        question_number="1",
        start_page=1,
        start_y=80,
        end_page=1,
        end_y=160,
        page_numbers=[1],
        blocks=[],
        full_question_label="1",
        question_total_detected=2,
    )

    monkeypatch.setattr(pipeline, "_paper_identity_for_metadata", lambda *args: (None, []))
    monkeypatch.setattr(pipeline, "_question_identities_for_spans", lambda *args: ({}, {}))
    monkeypatch.setattr(
        pipeline,
        "analyze_mark_scheme",
        lambda path, config: calls.append(("analyze", None)) or analysis,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_mark_scheme_answers",
        lambda path, config, expected, *, analysis=None: calls.append(("extract", analysis)) or {},
    )
    monkeypatch.setattr(
        pipeline,
        "render_mark_scheme_images",
        lambda path, config, expected, **kwargs: calls.append(("render", kwargs.get("analysis"))) or {},
    )

    records = pipeline._build_records_from_spans(
        question_pdf=question_pdf,
        layouts=[],
        spans=[span],
        config=AppConfig(),
        mark_scheme_pdf=mark_scheme_pdf,
        examiner_report_paths=None,
        document_metadata=object(),
        registry_warnings=[],
        source_paper_code="11",
    )

    assert records == []
    assert calls == [("analyze", None), ("extract", analysis), ("render", analysis)]
