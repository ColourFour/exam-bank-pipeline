import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from exam_bank import pipeline
from exam_bank.config import AppConfig
from exam_bank.document_metadata import DocumentMetadata, parse_filename_metadata, reconcile_document_metadata
from exam_bank.document_registry import build_document_registry


@dataclass
class _WorkerRecord:
    paper_name: str
    question_number: str
    full_question_label: str
    screenshot_path: str
    syllabus_code: str
    paper_family: str
    year: str
    session: str
    component: str
    source_paper_code: str


def touch_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n")
    return path


def test_filename_metadata_parser_handles_long_cambridge_names() -> None:
    metadata = parse_filename_metadata("9709 Mathematics November 2025 Question Paper 12.pdf")

    assert metadata.syllabus == "9709"
    assert metadata.subject == "Mathematics"
    assert metadata.session == "winter25"
    assert metadata.normalized_session_key == "winter25"
    assert metadata.year == "2025"
    assert metadata.document_type == "question_paper"
    assert metadata.component == "12"
    assert metadata.canonical_key == "9709_2025_winter25_12"


def test_examiner_report_without_component_is_session_level() -> None:
    metadata = parse_filename_metadata("9709 Mathematics November 2025 Examiner Report.pdf")

    assert metadata.document_type == "examiner_report"
    assert metadata.component == ""
    assert metadata.session_key == "9709_2025_winter25"
    assert metadata.canonical_key == ""


def test_filename_metadata_normalizes_compact_and_phrase_sessions() -> None:
    compact = parse_filename_metadata("9709_s21_qp_12.pdf")
    phrase = parse_filename_metadata("9709 Mathematics October November 2025 Question Paper 12.pdf")

    assert compact.normalized_session_key == "summer21"
    assert compact.canonical_key == "9709_2021_summer21_12"
    assert phrase.normalized_session_key == "winter25"
    assert phrase.canonical_key == "9709_2025_winter25_12"


def test_filename_metadata_handles_loose_exam_paper_p_family_names() -> None:
    question = parse_filename_metadata("March 2019 Exam Paper P1 (2).pdf")
    mark_scheme = parse_filename_metadata("March 2019 Mark Scheme P1 (2).pdf")

    assert question.syllabus == ""
    assert question.year == "2019"
    assert question.session == "summer19"
    assert question.document_type == "question_paper"
    assert question.component == "1"
    assert question.paper_family == "P1"
    assert question.canonical_key == "unknown_2019_summer19_1"
    assert mark_scheme.document_type == "mark_scheme"
    assert mark_scheme.component == "1"
    assert mark_scheme.canonical_key == question.canonical_key


def test_reconcile_metadata_treats_november_and_octnov_as_compatible() -> None:
    filename = DocumentMetadata(
        syllabus="9709",
        year="2025",
        session="November",
        original_session_label="November",
        normalized_session_key="November",
        document_type="question_paper",
        component="12",
        source="filename",
    )
    internal = DocumentMetadata(
        syllabus="9709",
        year="2025",
        session="OctNov",
        original_session_label="OctNov",
        normalized_session_key="OctNov",
        document_type="question_paper",
        component="12",
        source="internal",
    )

    reconciled = reconcile_document_metadata(filename, internal)

    assert reconciled.session == "OctNov"
    assert reconciled.warnings == ()


def test_reconcile_metadata_ignores_lone_internal_session_mismatch() -> None:
    filename = DocumentMetadata(
        syllabus="9231",
        year="2025",
        session="OctNov",
        original_session_label="OctNov",
        normalized_session_key="OctNov",
        document_type="question_paper",
        component="33",
        source="filename",
    )
    internal = DocumentMetadata(
        session="MayJune",
        original_session_label="MayJune",
        normalized_session_key="MayJune",
        source="internal",
    )

    reconciled = reconcile_document_metadata(filename, internal)

    assert reconciled.session == "OctNov"
    assert reconciled.normalized_session_key == "OctNov"
    assert reconciled.source == "filename"
    assert reconciled.warnings == ()


def test_folder_registry_classifies_and_pairs_companion_files(tmp_path: Path) -> None:
    qp12 = touch_pdf(tmp_path / "9709 Mathematics November 2025 Question Paper 12.pdf")
    ms12 = touch_pdf(tmp_path / "9709 Mathematics November 2025 Mark Scheme 12.pdf")
    ms13 = touch_pdf(tmp_path / "9709 Mathematics November 2025 Mark Scheme 13.pdf")
    er = touch_pdf(tmp_path / "9709 Mathematics November 2025 Examiner Report.pdf")

    registry = build_document_registry(tmp_path)

    entry = registry.entries["9709_2025_winter25_12"]
    assert entry.question_paper == qp12
    assert entry.mark_scheme == ms12
    assert entry.mark_scheme != ms13
    assert entry.examiner_reports == [er]
    assert registry.session_reports["9709_2025_winter25"] == [er]


def test_missing_companion_files_do_not_remove_question_paper_entry(tmp_path: Path) -> None:
    qp = touch_pdf(tmp_path / "9709 Mathematics March 2022 Question Paper 42.pdf")

    registry = build_document_registry(tmp_path)

    entry = registry.entries["9709_2022_summer22_42"]
    assert entry.question_paper == qp
    assert entry.mark_scheme is None
    assert entry.examiner_reports == []
    assert entry.missing_companions == ["mark_scheme"]


def test_process_registry_routes_only_question_papers_to_question_extraction(tmp_path: Path, monkeypatch) -> None:
    touch_pdf(tmp_path / "9709 Mathematics November 2025 Question Paper 12.pdf")
    touch_pdf(tmp_path / "9709 Mathematics November 2025 Mark Scheme 12.pdf")
    touch_pdf(tmp_path / "9709 Mathematics November 2025 Examiner Report.pdf")
    registry = build_document_registry(tmp_path)
    config = AppConfig()
    calls: list[dict[str, object]] = []

    def fake_build_records_for_pdf(question_pdf, config, mark_scheme_pdf=None, examiner_report_paths=None, **kwargs):
        calls.append(
            {
                "question_pdf": Path(question_pdf).name,
                "mark_scheme_pdf": Path(mark_scheme_pdf).name if mark_scheme_pdf else "",
                "examiner_report_paths": [Path(path).name for path in examiner_report_paths or []],
                "metadata": kwargs.get("filename_metadata"),
            }
        )
        return []

    monkeypatch.setattr(pipeline, "build_records_for_pdf", fake_build_records_for_pdf)
    monkeypatch.setattr(pipeline, "export_records", lambda records, config: tmp_path / "question_bank.json")
    monkeypatch.setattr(pipeline, "_write_batch_diagnostic", lambda records, config: tmp_path / "diagnostics.json")

    pipeline._process_registry_entries(registry, config)

    assert calls == [
        {
            "question_pdf": "9709 Mathematics November 2025 Question Paper 12.pdf",
            "mark_scheme_pdf": "9709 Mathematics November 2025 Mark Scheme 12.pdf",
            "examiner_report_paths": ["9709 Mathematics November 2025 Examiner Report.pdf"],
                "metadata": registry.entries["9709_2025_winter25_12"].metadata_by_path[
                str(tmp_path / "9709 Mathematics November 2025 Question Paper 12.pdf")
            ],
        }
    ]


def test_parallel_workers_match_single_worker_order_content_and_artifact_hashes(tmp_path: Path, monkeypatch) -> None:
    touch_pdf(tmp_path / "9709 Mathematics November 2025 Question Paper 13.pdf")
    touch_pdf(tmp_path / "9709 Mathematics November 2025 Question Paper 12.pdf")
    registry = build_document_registry(tmp_path)

    def fake_build_records(question_pdf, config, **_kwargs):
        metadata = parse_filename_metadata(question_pdf)
        artifact = config.output.root_dir() / "p1" / f"{metadata.component}.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact-{metadata.component}".encode())
        return [
            _WorkerRecord(
                paper_name=f"{metadata.component}{metadata.normalized_session_key}",
                question_number="1",
                full_question_label="1",
                screenshot_path=str(artifact),
                syllabus_code="9709",
                paper_family="p1",
                year=metadata.year,
                session=metadata.session,
                component=metadata.component,
                source_paper_code=metadata.component,
            )
        ]

    def fake_export(records, config):
        output = config.output.json_dir / config.naming.json_name
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "paper": record.paper_name,
                "question_number": record.question_number,
                "artifact": Path(record.screenshot_path).relative_to(config.output.root_dir()).as_posix(),
            }
            for record in records
        ]
        output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return output

    monkeypatch.setattr(pipeline, "build_records_for_pdf", fake_build_records)
    monkeypatch.setattr(pipeline, "export_records", fake_export)
    monkeypatch.setattr(pipeline, "_write_missing_image_repair_report", lambda *_args, **_kwargs: None)

    single_config = AppConfig()
    single_config.output.apply_root(tmp_path / "single")
    multi_config = AppConfig()
    multi_config.output.apply_root(tmp_path / "multi")
    single = pipeline._process_registry_entries(registry, single_config, workers=1)
    multi = pipeline._process_registry_entries(registry, multi_config, workers=2)

    assert single.json_path.read_text(encoding="utf-8") == multi.json_path.read_text(encoding="utf-8")
    assert [(row.paper_name, row.question_number) for row in single.records] == [
        (row.paper_name, row.question_number) for row in multi.records
    ]
    assert _artifact_hashes(single_config.output.root_dir()) == _artifact_hashes(multi_config.output.root_dir())
    assert not (multi_config.output.root_dir() / ".workers").exists()


def test_parallel_workers_reject_debug_mode(tmp_path: Path) -> None:
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    config.debug.enabled = True

    try:
        pipeline._process_registry_entries(build_document_registry(tmp_path), config, workers=2)
    except ValueError as exc:
        assert "debug mode" in str(exc)
    else:
        raise AssertionError("workers > 1 should reject shared debug mode")


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.bin"))
    }
