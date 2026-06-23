from __future__ import annotations

from .models import (
    EmailConnectionStatus,
    EmailMessageSummary,
    EmailSendResult,
    EmailSmokeTestReport,
)
from .providers import EmailProvider, FakeEmailProvider, build_email_provider

__all__ = [
    "EmailConnectionStatus",
    "EmailMessageSummary",
    "EmailProvider",
    "EmailSendResult",
    "EmailSmokeTestReport",
    "FakeEmailProvider",
    "build_email_provider",
]
