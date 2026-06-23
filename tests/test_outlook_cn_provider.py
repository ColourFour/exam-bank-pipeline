from __future__ import annotations

from exam_bank.emailing.outlook_cn import (
    DEFAULT_IMAP_HOST,
    DEFAULT_IMAP_PORT,
    DEFAULT_SMTP_HOST,
    DEFAULT_SMTP_PORT,
    OutlookCnConfig,
    OutlookCnEmailProvider,
)


def test_outlook_cn_provider_reads_host_port_address_from_env() -> None:
    provider = OutlookCnEmailProvider.from_env(
        {
            "EXAM_BANK_EMAIL_ADDRESS": "brooker@rdfzcygj.cn",
            "EXAM_BANK_EMAIL_PASSWORD": "not-used",
            "EXAM_BANK_IMAP_HOST": "imap.example.test",
            "EXAM_BANK_IMAP_PORT": "1993",
            "EXAM_BANK_SMTP_HOST": "smtp.example.test",
            "EXAM_BANK_SMTP_PORT": "1587",
        }
    )

    assert provider.config.account_email == "brooker@rdfzcygj.cn"
    assert provider.config.password == "not-used"
    assert provider.config.imap_host == "imap.example.test"
    assert provider.config.imap_port == 1993
    assert provider.config.smtp_host == "smtp.example.test"
    assert provider.config.smtp_port == 1587


def test_outlook_cn_provider_defaults_to_21vianet_hosts() -> None:
    provider = OutlookCnEmailProvider.from_env(
        {
            "EXAM_BANK_EMAIL_ADDRESS": "brooker@rdfzcygj.cn",
            "EXAM_BANK_EMAIL_PASSWORD": "not-used",
        }
    )

    assert provider.config.imap_host == DEFAULT_IMAP_HOST
    assert provider.config.imap_port == DEFAULT_IMAP_PORT
    assert provider.config.smtp_host == DEFAULT_SMTP_HOST
    assert provider.config.smtp_port == DEFAULT_SMTP_PORT


def test_outlook_cn_provider_missing_password_fails_without_network() -> None:
    provider = OutlookCnEmailProvider.from_env({"EXAM_BANK_EMAIL_ADDRESS": "brooker@rdfzcygj.cn"})

    status = provider.check_connection()

    assert status.connected is False
    assert status.smtp_ok is False
    assert status.imap_ok is False
    assert status.error_code == "bad_credentials_or_app_password_required"


def test_partial_smtp_imap_failure_is_reported_safely(monkeypatch) -> None:
    provider = OutlookCnEmailProvider(OutlookCnConfig(account_email="brooker@rdfzcygj.cn", password="not-used"))
    monkeypatch.setattr(provider, "_check_smtp", lambda: (False, "smtp_disabled_or_blocked"))
    monkeypatch.setattr(provider, "_check_imap", lambda: (True, None))

    status = provider.check_connection()

    assert status.connected is False
    assert status.smtp_ok is False
    assert status.imap_ok is True
    assert status.error_code == "smtp_disabled_or_blocked"
    assert "password" not in (status.error_message_safe or "").lower()
