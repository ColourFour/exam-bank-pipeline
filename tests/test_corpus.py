from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from exam_bank.corpus import (
    CorpusManifestError,
    _documents_sha256,
    build_corpus_manifest,
    hydrate_corpus,
    load_corpus_manifest,
    quarantine_structural_failures,
    sha256_file,
    verify_corpus,
)


def test_build_manifest_uses_canonical_paths_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "input"
    question = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    mark_scheme = root / "pastpapers/9709/2024/mark_schemes/9709_s24_ms_11.pdf"
    _write(question, _valid_pdf_bytes("question"))
    _write(mark_scheme, _valid_pdf_bytes("mark scheme"))

    manifest = build_corpus_manifest(root, generated_at="2026-07-13T00:00:00Z")

    assert manifest["record_count"] == 2
    by_type = {row["document_type"]: row for row in manifest["documents"]}
    assert by_type["question_paper"]["local_path"] == "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    assert by_type["question_paper"]["component"] == "11"
    assert by_type["question_paper"]["session"] == "June"
    assert by_type["mark_scheme"]["sha256"] == sha256_file(mark_scheme)


def test_build_manifest_rejects_structurally_blank_pdf(tmp_path: Path) -> None:
    root = tmp_path / "input"
    question = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    document = fitz.open()
    document.new_page()
    _write(question, document.tobytes())
    document.close()

    with pytest.raises(CorpusManifestError, match="Structurally invalid corpus PDF"):
        build_corpus_manifest(root)


def test_verify_reports_missing_size_and_checksum_failures(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)

    missing = verify_corpus(manifest_path, root=root)
    assert missing["missing_count"] == 1

    target = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    _write(target, b"short")
    wrong_size = verify_corpus(manifest_path, root=root)
    assert wrong_size["size_mismatch_count"] == 1

    altered = bytearray(source.read_bytes())
    altered[-1] ^= 1
    _write(target, bytes(altered))
    wrong_checksum = verify_corpus(manifest_path, root=root)
    assert wrong_checksum["checksum_mismatch_count"] == 1


def test_verify_rejects_checksum_valid_pdf_without_renderable_content(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page()
    _write(source, document.tobytes())
    document.close()
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)
    target = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    _write(target, source.read_bytes())

    report = verify_corpus(manifest_path, root=root)

    assert report["ok"] is False
    assert report["structural_failure_count"] == 1
    assert report["structural_failures"] == [
        {
            "local_path": "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf",
            "reason": "pdf_has_no_renderable_content",
        }
    ]


def test_hydrate_downloads_missing_file_from_verified_source(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)

    report = hydrate_corpus(manifest_path, root=root)

    assert report["ok"] is True
    assert report["hydrated_count"] == 1
    assert verify_corpus(manifest_path, root=root)["ok"] is True


def test_hydrate_leaves_complete_file_untouched(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)
    target = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    _write(target, source.read_bytes())

    report = hydrate_corpus(manifest_path, root=root)

    assert report["ok"] is True
    assert report["hydrated_count"] == 0
    assert report["already_verified_count"] == 1


def test_hydrate_partial_corpus_downloads_only_missing_files(tmp_path: Path) -> None:
    question_source = tmp_path / "question.pdf"
    mark_scheme_source = tmp_path / "mark-scheme.pdf"
    _write(question_source, _valid_pdf_bytes("question"))
    _write(mark_scheme_source, _valid_pdf_bytes("mark scheme"))
    manifest_path = _manifest(tmp_path, question_source)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mark_scheme = dict(payload["documents"][0])
    mark_scheme.update(
        {
            "document_id": "9709_s24_ms_11",
            "document_type": "mark_scheme",
            "local_path": "pastpapers/9709/2024/mark_schemes/9709_s24_ms_11.pdf",
            "source_url": mark_scheme_source.resolve().as_uri(),
            "sha256": sha256_file(mark_scheme_source),
            "size_bytes": mark_scheme_source.stat().st_size,
        }
    )
    payload["documents"].append(mark_scheme)
    payload["record_count"] = 2
    payload["documents_sha256"] = _documents_sha256(payload["documents"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    root = tmp_path / "input"
    existing = root / payload["documents"][0]["local_path"]
    _write(existing, question_source.read_bytes())

    report = hydrate_corpus(manifest_path, root=root)

    assert report["ok"] is True
    assert report["already_verified_count"] == 1
    assert report["hydrated"] == [mark_scheme["local_path"]]


def test_hydrate_uses_verified_mirror_after_primary_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    manifest_path = _manifest(tmp_path, source)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["documents"][0]["source_url"] = (tmp_path / "missing.pdf").resolve().as_uri()
    payload["documents"][0]["mirror_urls"] = [source.resolve().as_uri()]
    payload["documents_sha256"] = _documents_sha256(payload["documents"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = hydrate_corpus(manifest_path, root=tmp_path / "input")

    assert report["ok"] is True
    assert report["hydrated_count"] == 1


def test_failed_download_removes_partial_file(tmp_path: Path) -> None:
    expected = tmp_path / "expected.pdf"
    invalid = tmp_path / "invalid.pdf"
    _write(expected, _valid_pdf_bytes("expected"))
    _write(invalid, b"not-the-expected-payload")
    manifest_path = _manifest(tmp_path, expected)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["documents"][0]["source_url"] = invalid.resolve().as_uri()
    payload["documents_sha256"] = _documents_sha256(payload["documents"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    root = tmp_path / "input"

    report = hydrate_corpus(manifest_path, root=root)

    assert report["ok"] is False
    assert report["failed"][0]["reason"] == "download_failed"
    target = root / payload["documents"][0]["local_path"]
    assert not target.with_name(f".{target.name}.partial").exists()


def test_hydrate_requires_repair_and_quarantines_corrupt_file(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)
    target = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    _write(target, b"corrupt-data")

    refused = hydrate_corpus(manifest_path, root=root)
    assert refused["ok"] is False
    assert refused["failed"][0]["repair_required"] is True
    assert target.read_bytes() == b"corrupt-data"

    repaired = hydrate_corpus(manifest_path, root=root, repair=True)
    assert repaired["ok"] is True
    assert repaired["quarantined_count"] == 1
    assert target.read_bytes() == source.read_bytes()
    assert Path(repaired["quarantined"][0]).read_bytes() == b"corrupt-data"


def test_offline_hydration_is_non_mutating(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)

    report = hydrate_corpus(manifest_path, root=root, offline=True)

    assert report["ok"] is False
    assert report["failed"][0]["reason"] == "offline_missing"
    assert not (root / "pastpapers").exists()


def test_manifest_rejects_paths_outside_corpus_root(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    manifest_path = _manifest(tmp_path, source)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["documents"][0]["local_path"] = "../escape.pdf"
    payload["documents_sha256"] = _documents_sha256(payload["documents"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="unsafe_local_path"):
        load_corpus_manifest(manifest_path)


def test_structural_quarantine_is_recoverable_and_builds_partial_active_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    valid_source = tmp_path / "valid.pdf"
    invalid_source = tmp_path / "invalid.pdf"
    _write(valid_source, _valid_pdf_bytes("question"))
    blank = fitz.open()
    blank.new_page()
    _write(invalid_source, blank.tobytes())
    blank.close()
    manifest_path = _manifest(tmp_path, valid_source)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid = dict(payload["documents"][0])
    invalid.update(
        {
            "document_id": "9709_s24_qp_12",
            "component": "12",
            "local_path": "pastpapers/9709/2024/question_papers/9709_s24_qp_12.pdf",
            "source_url": invalid_source.resolve().as_uri(),
            "sha256": sha256_file(invalid_source),
            "size_bytes": invalid_source.stat().st_size,
        }
    )
    payload["documents"].append(invalid)
    payload["record_count"] = 2
    payload["documents_sha256"] = _documents_sha256(payload["documents"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    for document in payload["documents"]:
        source = valid_source if document["component"] == "11" else invalid_source
        _write(root / document["local_path"], source.read_bytes())
    report_path = tmp_path / "quarantine-validation.json"
    active_path = tmp_path / "active-manifest.json"

    dry_run = quarantine_structural_failures(
        manifest_path,
        root=root,
        report_path=report_path,
        active_manifest_path=active_path,
        generated_at="2026-08-11T00:00:00Z",
    )

    assert dry_run["operation_ok"] is True
    assert dry_run["ok"] is False
    assert dry_run["planned_count"] == 1
    assert not active_path.exists()
    assert (root / invalid["local_path"]).is_file()

    applied = quarantine_structural_failures(
        manifest_path,
        root=root,
        report_path=report_path,
        active_manifest_path=active_path,
        generated_at="2026-08-11T00:00:00Z",
        apply=True,
    )

    assert applied["operation_ok"] is True
    assert applied["ok"] is False
    assert applied["quarantined_count"] == 1
    assert applied["active_record_count"] == 1
    assert not (root / invalid["local_path"]).exists()
    quarantine_path = Path(applied["entries"][0]["quarantine_absolute_path"])
    assert quarantine_path.read_bytes() == invalid_source.read_bytes()
    active = load_corpus_manifest(active_path)
    assert active["corpus_state"] == "partial_quarantined"
    assert active["record_count"] == 1
    assert verify_corpus(active_path, root=root)["ok"] is True
    assert load_corpus_manifest(manifest_path)["record_count"] == 2

    repeated = quarantine_structural_failures(
        manifest_path,
        root=root,
        report_path=report_path,
        active_manifest_path=active_path,
        generated_at="2026-08-11T00:00:00Z",
        apply=True,
    )
    assert repeated["already_quarantined_count"] == 1
    assert repeated["blocking_count"] == 0


def test_structural_quarantine_requires_outputs_outside_corpus_root(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, _valid_pdf_bytes("source"))
    manifest_path = _manifest(tmp_path, source)

    with pytest.raises(CorpusManifestError, match="must be outside the corpus root"):
        quarantine_structural_failures(
            manifest_path,
            root=tmp_path / "input",
            report_path=tmp_path / "input/report.json",
        )


def _manifest(tmp_path: Path, source: Path) -> Path:
    document = {
        "document_id": "9709_s24_qp_11",
        "document_type": "question_paper",
        "syllabus": "9709",
        "year": 2024,
        "session": "June",
        "session_code": "s",
        "component": "11",
        "local_path": "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf",
        "source_url": source.resolve().as_uri(),
        "mirror_urls": [],
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }
    payload = {
        "schema_name": "exam_bank.corpus_manifest",
        "schema_version": 1,
        "corpus_id": "fixture",
        "generated_at": "2026-07-13T00:00:00Z",
        "record_count": 1,
        "documents_sha256": _documents_sha256([document]),
        "documents": [document],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _valid_pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload
