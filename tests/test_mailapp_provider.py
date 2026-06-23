from __future__ import annotations

from exam_bank.emailing.mailapp import (
    MAILAPP_PERMISSION_NOTE,
    MailAppCommandResult,
    MailAppEmailProvider,
    build_mailapp_send_script,
)


def test_mailapp_provider_builds_escaped_applescript_safely() -> None:
    script = build_mailapp_send_script(
        to='brooker"@rdfzcygj.cn',
        subject='Exam Bank Email Smoke Test" & do shell script "echo bad"',
        body_text='Line 1\nLine 2 "quoted"',
        from_address='brooker@rdfzcygj.cn',
    )

    assert '\\" & do shell script \\"echo bad\\"' in script
    assert 'Line 1\\nLine 2 \\"quoted\\"' in script
    assert 'brooker\\"@rdfzcygj.cn' in script


def test_mailapp_provider_reports_permission_errors_safely() -> None:
    def runner(script: str) -> MailAppCommandResult:
        return MailAppCommandResult(returncode=1, stdout="", stderr="Not authorized to send Apple events to Mail (-1743)")

    provider = MailAppEmailProvider(runner=runner)

    status = provider.check_connection()

    assert status.connected is False
    assert status.error_code == "mailapp_automation_permission_blocked"
    assert status.error_message_safe == MAILAPP_PERMISSION_NOTE


def test_mailapp_provider_refuses_default_account_unless_allowed() -> None:
    scripts: list[str] = []

    def runner(script: str) -> MailAppCommandResult:
        scripts.append(script)
        return MailAppCommandResult(returncode=1, stdout="", stderr="mailapp_sender_refused")

    provider = MailAppEmailProvider(requested_from_address="brooker@rdfzcygj.cn", runner=runner)

    result = provider.send_message(
        to="brooker@rdfzcygj.cn",
        subject="Exam Bank Email Smoke Test",
        body_text="body",
        from_address="brooker@rdfzcygj.cn",
    )

    assert result.sent is False
    assert result.error_code == "mailapp_sender_refused"
    assert "-- Continue with Mail.app default sender" not in scripts[0]


def test_mailapp_provider_allows_default_account_when_explicit() -> None:
    scripts: list[str] = []

    def runner(script: str) -> MailAppCommandResult:
        scripts.append(script)
        return MailAppCommandResult(returncode=0, stdout="mailapp-message-1\n", stderr="")

    provider = MailAppEmailProvider(
        requested_from_address="brooker@rdfzcygj.cn",
        allow_default_mail_account=True,
        runner=runner,
    )

    result = provider.send_message(
        to="brooker@rdfzcygj.cn",
        subject="Exam Bank Email Smoke Test",
        body_text="body",
        from_address="brooker@rdfzcygj.cn",
    )

    assert result.sent is True
    assert result.provider_message_id == "mailapp-message-1"
    assert "-- Continue with Mail.app default sender" in scripts[0]


def test_mailapp_receive_unsupported_is_reported_honestly() -> None:
    def runner(script: str) -> MailAppCommandResult:
        return MailAppCommandResult(returncode=1, stdout="", stderr="Can't get messages of mailbox")

    provider = MailAppEmailProvider(runner=runner)

    try:
        provider.search_messages(query="Exam Bank Email Smoke Test", limit=5)
    except RuntimeError as exc:
        assert str(exc) == "mailapp_search_unsupported"
    else:
        raise AssertionError("Expected receive/search unsupported error")


def test_mailapp_connection_verifies_requested_account() -> None:
    def runner(script: str) -> MailAppCommandResult:
        if "application file id" in script:
            return MailAppCommandResult(returncode=0, stdout="true\n", stderr="")
        return MailAppCommandResult(returncode=0, stdout="brooker@rdfzcygj.cn\nother@example.com\n", stderr="")

    provider = MailAppEmailProvider(requested_from_address="brooker@rdfzcygj.cn", runner=runner)

    status = provider.check_connection()

    assert status.mailapp_available is True
    assert status.account_verified is True
    assert status.error_code is None
