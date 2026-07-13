from __future__ import annotations

import json
from pathlib import Path

import pytest

from exam_bank.corpus import (
    CorpusManifestError,
    _documents_sha256,
    build_corpus_manifest,
    hydrate_corpus,
    load_corpus_manifest,
    sha256_file,
    verify_corpus,
)


def test_build_manifest_uses_canonical_paths_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "input"
    question = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    mark_scheme = root / "pastpapers/9709/2024/mark_schemes/9709_s24_ms_11.pdf"
    _write(question, b"%PDF-question")
    _write(mark_scheme, b"%PDF-mark-scheme")

    manifest = build_corpus_manifest(root, generated_at="2026-07-13T00:00:00Z")

    assert manifest["record_count"] == 2
    by_type = {row["document_type"]: row for row in manifest["documents"]}
    assert by_type["question_paper"]["local_path"] == "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    assert by_type["question_paper"]["component"] == "11"
    assert by_type["question_paper"]["session"] == "June"
    assert by_type["mark_scheme"]["sha256"] == sha256_file(mark_scheme)


def test_verify_reports_missing_size_and_checksum_failures(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, b"%PDF-source")
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)

    missing = verify_corpus(manifest_path, root=root)
    assert missing["missing_count"] == 1

    target = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    _write(target, b"short")
    wrong_size = verify_corpus(manifest_path, root=root)
    assert wrong_size["size_mismatch_count"] == 1

    _write(target, b"%PDF-other!")
    wrong_checksum = verify_corpus(manifest_path, root=root)
    assert wrong_checksum["checksum_mismatch_count"] == 1


def test_hydrate_downloads_missing_file_from_verified_source(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, b"%PDF-source")
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)

    report = hydrate_corpus(manifest_path, root=root)

    assert report["ok"] is True
    assert report["hydrated_count"] == 1
    assert verify_corpus(manifest_path, root=root)["ok"] is True


def test_hydrate_leaves_complete_file_untouched(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, b"%PDF-source")
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
    _write(question_source, b"%PDF-question")
    _write(mark_scheme_source, b"%PDF-mark-scheme")
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
    _write(source, b"%PDF-source")
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
    _write(expected, b"%PDF-expected")
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
    _write(source, b"%PDF-source")
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
    _write(source, b"%PDF-source")
    root = tmp_path / "input"
    manifest_path = _manifest(tmp_path, source)

    report = hydrate_corpus(manifest_path, root=root, offline=True)

    assert report["ok"] is False
    assert report["failed"][0]["reason"] == "offline_missing"
    assert not (root / "pastpapers").exists()


def test_manifest_rejects_paths_outside_corpus_root(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write(source, b"%PDF-source")
    manifest_path = _manifest(tmp_path, source)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["documents"][0]["local_path"] = "../escape.pdf"
    payload["documents_sha256"] = _documents_sha256(payload["documents"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusManifestError, match="unsafe_local_path"):
        load_corpus_manifest(manifest_path)


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
