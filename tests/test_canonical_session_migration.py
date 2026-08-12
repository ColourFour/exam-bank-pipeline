from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from exam_bank.canonical_session_migration import (
    CanonicalSessionMigrationError,
    build_session_identity_aliases,
    migrate_canonical_session_identity,
)


def test_migration_rewrites_only_provenance_matched_march_identity(tmp_path: Path) -> None:
    bank_path = tmp_path / "question_bank.json"
    routing_path = tmp_path / "routing.json"
    reviewed_path = tmp_path / "reviewed.json"
    alias_path = tmp_path / "aliases.json"
    march_source = "input/pastpapers/9709/2019/question_papers/9709_m19_qp_12.pdf"
    mislabeled_june_source = "input/pastpapers/9709/2021/question_papers/9709_m21_qp_12.pdf"
    _write_pdf(tmp_path / march_source, "Cambridge International February/March 2019")
    _write_pdf(tmp_path / mislabeled_june_source, "Cambridge International May/June 2021")
    _write_json(
        bank_path,
        {
            "schema_name": "exam_bank.question_bank",
            "schema_version": 2,
            "record_count": 2,
            "questions": [
                _record(
                    "12summer19_q01",
                    "12summer19",
                    march_source,
                    "m19",
                ),
                _record(
                    "12summer21_q01",
                    "12summer21",
                    mislabeled_june_source,
                    "m21",
                ),
            ],
        },
    )
    _write_json(
        routing_path,
        {
            "schema_name": "exam_bank.topic_routing_sidecar",
            "schema_version": 1,
            "record_count": 2,
            "records": {
                "12summer19_q01": {"paper": "12summer19", "question_number": "1"},
                "12summer21_q01": {"paper": "12summer21", "question_number": "1"},
            },
        },
    )
    _write_json(
        reviewed_path,
        {
            "decisions": [
                {
                    "question_id": "12summer19_q01",
                    "candidate_id": "content_lab_12summer19_q01_whole",
                    "paper": "12summer19",
                },
                {"question_id": "12summer21_q01", "paper": "12summer21"},
            ]
        },
    )

    report = migrate_canonical_session_identity(
        question_bank_path=bank_path,
        artifact_paths=[routing_path, reviewed_path],
        alias_manifest_path=alias_path,
        source_root=tmp_path,
        write=True,
    )

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    routes = json.loads(routing_path.read_text(encoding="utf-8"))["records"]
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))["decisions"]
    aliases = json.loads(alias_path.read_text(encoding="utf-8"))
    assert bank["questions"][0]["question_id"] == "12spring19_q01"
    assert bank["questions"][0]["paper"] == "12spring19"
    assert bank["questions"][0]["canonical_session"] == "spring19"
    assert bank["questions"][0]["notes"]["question_crop_diagnostics"]["question_id"] == "12spring19_q01"
    assert bank["questions"][0]["question_image_path"].endswith("_m19_12_qp_q01_question.png")
    assert bank["questions"][1]["question_id"] == "12summer21_q01"
    assert set(routes) == {"12spring19_q01", "12summer21_q01"}
    assert routes["12spring19_q01"]["source_session_code"] == "m19"
    assert reviewed[0]["candidate_id"] == "content_lab_12spring19_q01_whole"
    assert reviewed[1]["question_id"] == "12summer21_q01"
    assert aliases["alias_count"] == 1
    assert aliases["aliases"][0]["legacy_question_id"] == "12summer19_q01"
    assert aliases["aliases"][0]["raw_source_session_code"] == "m19"
    assert len(aliases["aliases"][0]["source_sha256"]) == 64
    assert report["march_alias_count"] == 1
    assert report["question_bank_records_rewritten"] == 1


def test_migration_dry_run_does_not_write(tmp_path: Path) -> None:
    bank_path = tmp_path / "question_bank.json"
    artifact_path = tmp_path / "reviewed.json"
    alias_path = tmp_path / "aliases.json"
    source = "input/pastpapers/9709/2019/question_papers/9709_m19_qp_12.pdf"
    _write_pdf(tmp_path / source, "Cambridge International February/March 2019")
    _write_json(
        bank_path,
        {
            "questions": [
                _record(
                    "12summer19_q01",
                    "12summer19",
                    source,
                    "m19",
                )
            ]
        },
    )
    _write_json(artifact_path, {"question_id": "12summer19_q01"})
    before = bank_path.read_bytes()

    report = migrate_canonical_session_identity(
        question_bank_path=bank_path,
        artifact_paths=[artifact_path],
        alias_manifest_path=alias_path,
        source_root=tmp_path,
        write=False,
    )

    assert bank_path.read_bytes() == before
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["question_id"] == "12summer19_q01"
    assert not alias_path.exists()
    assert report["write"] is False


def test_migration_rejects_alias_that_would_overwrite_new_june_identity(tmp_path: Path) -> None:
    march_source = "input/pastpapers/9709/2019/question_papers/9709_m19_qp_12.pdf"
    june_source = "input/pastpapers/9709/2019/question_papers/9709_s19_qp_12.pdf"
    _write_pdf(tmp_path / march_source, "Cambridge International February/March 2019")
    _write_pdf(tmp_path / june_source, "Cambridge International May/June 2019")
    records = [
        _record(
            "12spring19_q01",
            "12spring19",
            march_source,
            "m19",
        ),
        _record(
            "12summer19_q01",
            "12summer19",
            june_source,
            "s19",
        ),
    ]

    with pytest.raises(CanonicalSessionMigrationError, match="shared by multiple raw sessions"):
        build_session_identity_aliases(records, source_root=tmp_path)


def test_migration_uses_publisher_evidence_when_raw_m_filename_contains_june(tmp_path: Path) -> None:
    source = "input/pastpapers/9709/2021/question_papers/9709_m21_qp_12.pdf"
    _write_pdf(tmp_path / source, "Cambridge International May/June 2021")

    aliases = build_session_identity_aliases(
        [_record("12summer21_q01", "12summer21", source, "m21")],
        source_root=tmp_path,
    )

    assert aliases == []


def _record(question_id: str, paper: str, source_pdf: str, session_code: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": paper,
        "canonical_paper_id": paper,
        "canonical_session": f"summer{session_code[1:]}",
        "question_image_path": f"pm1/pm1_2021_{session_code}_12_qp_q01_question.png",
        "notes": {
            "source_pdf": source_pdf,
            "question_crop_diagnostics": {
                "question_id": question_id,
                "paper_id": paper,
            },
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()
