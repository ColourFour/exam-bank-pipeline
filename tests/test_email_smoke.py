from __future__ import annotations

import json
from pathlib import Path

from exam_bank.cli import main
from exam_bank.emailing.providers import FakeEmailProvider
from exam_bank.emailing.smoke import DEFAULT_REPORTS_ROOT, run_email_smoke_test


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_fake_provider_email_check_succeeds(capsys) -> None:
    exit_code = main(["email-check", "--provider", "fake"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "fake"
    assert payload["smtp_ok"] is True
    assert payload["imap_ok"] is True


def test_fake_provider_email_send_test_sends_one_message(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "email-send-test",
            "--provider",
            "fake",
            "--to",
            "brooker@rdfzcygj.cn",
            "--subject",
            "Exam Bank Email Smoke Test",
            "--body",
            "This is a controlled test email from the exam-bank system.",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sent"] is True
    audit = _jsonl(DEFAULT_REPORTS_ROOT / "email_audit.jsonl")
    assert len(audit) == 1
    assert audit[0]["recipient_domain"] == "rdfzcygj.cn"
    assert "This is a controlled test email" not in json.dumps(audit)


def test_email_send_test_blocks_multiple_recipients() -> None:
    exit_code = main(
        [
            "email-send-test",
            "--provider",
            "fake",
            "--to",
            "brooker@rdfzcygj.cn,student@example.com",
            "--subject",
            "Exam Bank Email Smoke Test",
            "--body",
            "body",
        ]
    )

    assert exit_code == 2


def test_email_send_test_blocks_attachments_by_default(tmp_path: Path) -> None:
    attachment = tmp_path / "file.txt"
    attachment.write_text("not sent", encoding="utf-8")

    exit_code = main(
        [
            "email-send-test",
            "--provider",
            "fake",
            "--to",
            "brooker@rdfzcygj.cn",
            "--subject",
            "Exam Bank Email Smoke Test",
            "--body",
            "body",
            "--attachment",
            str(attachment),
        ]
    )

    assert exit_code == 2


def test_email_send_test_blocks_non_smoke_subject_unless_allowed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    blocked = main(
        [
            "email-send-test",
            "--provider",
            "fake",
            "--to",
            "brooker@rdfzcygj.cn",
            "--subject",
            "Operational message",
            "--body",
            "body",
        ]
    )
    allowed = main(
        [
            "email-send-test",
            "--provider",
            "fake",
            "--to",
            "brooker@rdfzcygj.cn",
            "--subject",
            "Operational message",
            "--body",
            "body",
            "--allow-non-smoke-subject",
        ]
    )

    assert blocked == 2
    assert allowed == 0


def test_email_smoke_test_creates_html_and_json_report(tmp_path: Path) -> None:
    provider = FakeEmailProvider(account_email="brooker@rdfzcygj.cn")

    report = run_email_smoke_test(provider=provider, to="brooker@rdfzcygj.cn", reports_root=tmp_path / "reports")

    assert report.send_ok is True
    assert report.receive_ok is True
    assert report.report_path.is_file()
    assert report.json_report_path.is_file()
    payload = json.loads(report.json_report_path.read_text(encoding="utf-8"))
    assert payload["student_email_block_enforced"] is True
    assert payload["scores_sent"] is False
    assert payload["attachments_sent"] is False


def test_report_contains_no_password_or_auth_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXAM_BANK_EMAIL_PASSWORD", "super-secret-password")
    monkeypatch.setenv("EXAM_BANK_AUTH_TOKEN", "super-secret-token")
    provider = FakeEmailProvider(account_email="brooker@rdfzcygj.cn")

    report = run_email_smoke_test(provider=provider, to="brooker@rdfzcygj.cn", reports_root=tmp_path / "reports")
    report_text = report.report_path.read_text(encoding="utf-8") + report.json_report_path.read_text(encoding="utf-8")

    assert "super-secret-password" not in report_text
    assert "super-secret-token" not in report_text


def test_audit_log_excludes_password_token_and_body(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXAM_BANK_EMAIL_PASSWORD", "super-secret-password")
    monkeypatch.setenv("EXAM_BANK_AUTH_TOKEN", "super-secret-token")

    exit_code = main(
        [
            "email-send-test",
            "--provider",
            "fake",
            "--to",
            "brooker@rdfzcygj.cn",
            "--subject",
            "Exam Bank Email Smoke Test",
            "--body",
            "raw body must not be logged",
        ]
    )

    assert exit_code == 0
    audit_text = (DEFAULT_REPORTS_ROOT / "email_audit.jsonl").read_text(encoding="utf-8")
    assert "super-secret-password" not in audit_text
    assert "super-secret-token" not in audit_text
    assert "raw body must not be logged" not in audit_text


def test_receive_test_returns_safe_message_summaries() -> None:
    provider = FakeEmailProvider(account_email="brooker@rdfzcygj.cn")
    provider.send_message(
        to="brooker@rdfzcygj.cn",
        subject="Exam Bank Email Smoke Test [safe]",
        body_text="safe preview",
    )

    messages = provider.search_messages(query="Smoke Test", limit=5)

    assert len(messages) == 1
    assert messages[0].subject == "Exam Bank Email Smoke Test [safe]"
    assert messages[0].has_attachments is False


def test_smoke_commands_reject_assignment_roster_student_bulk_options(tmp_path: Path) -> None:
    roster = tmp_path / "roster.csv"
    scores = tmp_path / "scores.csv"
    roster.write_text("student\n", encoding="utf-8")
    scores.write_text("score\n", encoding="utf-8")

    assert (
        main(
            [
                "email-send-test",
                "--provider",
                "fake",
                "--to",
                "brooker@rdfzcygj.cn",
                "--subject",
                "Exam Bank Email Smoke Test",
                "--body",
                "body",
                "--assignment-id",
                "quiz-1",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "email-smoke-test",
                "--provider",
                "fake",
                "--to",
                "brooker@rdfzcygj.cn",
                "--roster-file",
                str(roster),
                "--score-file",
                str(scores),
            ]
        )
        == 2
    )
