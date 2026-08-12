from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

from exam_bank.corpus_session_identity import (
    detect_pdf_session,
    normalize_corpus_session_identity,
)


def test_detect_pdf_session_uses_first_page_publisher_label(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _write_pdf(path, "Cambridge International AS & A Level May/June 2024")

    evidence = detect_pdf_session(path)

    assert evidence.status == "resolved"
    assert evidence.session_code == "s"
    assert evidence.matched_session_codes == ("s",)
    assert len(evidence.first_page_text_sha256 or "") == 64


def test_normalization_transactionally_swaps_all_session_bearing_document_types(
    tmp_path: Path,
) -> None:
    root = tmp_path / "input"
    report_path = tmp_path / "session-validation.json"
    paths: dict[str, Path] = {}
    for folder, code in (
        ("question_papers", "qp"),
        ("mark_schemes", "ms"),
        ("exam_reports", "er"),
        ("grade_thresholds", "gt"),
    ):
        component = "12" if code in {"qp", "ms"} else "0"
        march = root / f"pastpapers/9709/2021/{folder}/9709_s21_{code}_{component}.pdf"
        june = root / f"pastpapers/9709/2021/{folder}/9709_m21_{code}_{component}.pdf"
        _write_pdf(march, "Cambridge International March 2021")
        _write_pdf(june, "Cambridge International May/June 2021")
        paths[f"{code}_march"] = march
        paths[f"{code}_june"] = june
    before = {name: _sha256(path) for name, path in paths.items()}

    dry_run = normalize_corpus_session_identity(
        root=root,
        report_path=report_path,
        generated_at="2026-08-11T00:00:00Z",
    )

    assert dry_run["operation_ok"] is True
    assert dry_run["ok"] is False
    assert dry_run["mismatch_count"] == 8
    assert {path.name: _sha256(path) for path in paths.values()} == {
        path.name: before[name] for name, path in paths.items()
    }

    applied = normalize_corpus_session_identity(
        root=root,
        report_path=report_path,
        apply=True,
        generated_at="2026-08-11T00:00:00Z",
    )

    assert applied["ok"] is True
    assert applied["renamed_count"] == 8
    assert applied["post_apply_mismatch_count"] == 0
    for code in ("qp", "ms", "er", "gt"):
        assert _sha256(paths[f"{code}_march"].with_name(paths[f"{code}_march"].name.replace("_s21_", "_m21_"))) == before[
            f"{code}_march"
        ]
        assert _sha256(paths[f"{code}_june"].with_name(paths[f"{code}_june"].name.replace("_m21_", "_s21_"))) == before[
            f"{code}_june"
        ]
    assert json.loads(report_path.read_text(encoding="utf-8"))["operation_state"] == "completed"
    assert not (root / ".corpus_session_identity_stage").exists()


def test_unreadable_question_paper_can_use_matching_mark_scheme_evidence(tmp_path: Path) -> None:
    root = tmp_path / "input"
    qp = root / "pastpapers/9709/2022/question_papers/9709_s22_qp_12.pdf"
    ms = root / "pastpapers/9709/2022/mark_schemes/9709_s22_ms_12.pdf"
    qp.parent.mkdir(parents=True, exist_ok=True)
    qp.write_bytes(b"not a PDF")
    _write_pdf(ms, "Cambridge International February/March 2022")

    report = normalize_corpus_session_identity(
        root=root,
        report_path=tmp_path / "report.json",
        generated_at="2026-08-11T00:00:00Z",
    )

    assert report["operation_ok"] is True
    assert report["unresolved_count"] == 0
    assert report["mismatch_count"] == 2
    qp_entry = next(item for item in report["entries"] if item["document_type"] == "qp")
    assert qp_entry["evidence_source"] == "paired_document"
    assert qp_entry["internal_session_code"] == "m"


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
