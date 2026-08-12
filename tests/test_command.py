from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import fitz

from exam_bank.command import COMMANDS, main, render_command_reference
from exam_bank.corpus import _documents_sha256, sha256_file


def test_public_help_is_namespaced_and_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "exam_bank.command", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "exam-bank <domain> <command>" in result.stdout
    assert "extract" in result.stdout
    assert "data" in result.stdout
    assert "exam_bank.cli" not in result.stdout

    extract_help = subprocess.run(
        [sys.executable, "-m", "exam_bank.command", "extract", "run", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--workers" in extract_help.stdout


def test_all_active_product_domains_have_commands() -> None:
    assert set(COMMANDS) == {
        "extract",
        "data",
        "topic",
        "asterion",
        "release",
        "ai",
        "triage",
        "review",
        "marks",
        "advisory",
    }
    assert {"run", "audit", "integrity"} <= set(COMMANDS["extract"])
    assert {"hydrate", "verify", "manifest"} <= set(COMMANDS["data"])


def test_data_verify_command_uses_manifest_contract(tmp_path: Path, capsys) -> None:
    root = tmp_path / "input"
    document_path = root / "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf"
    document_path.parent.mkdir(parents=True)
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Cambridge International May/June 2024")
    pdf.save(document_path)
    pdf.close()
    document = {
        "document_id": "9709_s24_qp_11",
        "document_type": "question_paper",
        "syllabus": "9709",
        "year": 2024,
        "session": "June",
        "session_code": "s",
        "component": "11",
        "local_path": "pastpapers/9709/2024/question_papers/9709_s24_qp_11.pdf",
        "source_url": "https://example.invalid/9709_s24_qp_11.pdf",
        "mirror_urls": [],
        "sha256": sha256_file(document_path),
        "size_bytes": document_path.stat().st_size,
    }
    manifest = {
        "schema_name": "exam_bank.corpus_manifest",
        "schema_version": 1,
        "corpus_id": "fixture",
        "generated_at": "2026-07-13T00:00:00Z",
        "record_count": 1,
        "documents_sha256": _documents_sha256([document]),
        "documents": [document],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = main(["data", "verify", "--manifest", str(manifest_path), "--root", str(root)])

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["verified_count"] == 1


def test_every_domain_help_smokes_without_loading_flat_cli(capsys) -> None:
    for domain in COMMANDS:
        assert main([domain, "--help"]) == 0
    output = capsys.readouterr().out
    assert "exam_bank.cli" not in output


def test_every_namespaced_command_help_smokes() -> None:
    for domain, actions in COMMANDS.items():
        for action in actions:
            result = subprocess.run(
                [sys.executable, "-m", "exam_bank.command", domain, action, "--help"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result.returncode == 0, f"{domain} {action}: {result.stdout}\n{result.stderr}"
            assert "usage:" in result.stdout.lower()


def test_old_flat_commands_are_absent_from_public_surface() -> None:
    for old_name in [
        "process",
        "output-integrity-audit",
        "topic-packets",
        "asterion-export",
        "classroom",
        "email-smoke-test",
        "grade-quiz-bma",
    ]:
        assert old_name not in COMMANDS


def test_generated_command_reference_is_current() -> None:
    assert Path("docs/COMMAND_REFERENCE.md").read_text(encoding="utf-8") == render_command_reference()
