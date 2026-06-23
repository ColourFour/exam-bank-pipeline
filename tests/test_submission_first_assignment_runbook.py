from __future__ import annotations

from pathlib import Path


def test_live_email_connector_contract_and_runbook_exist() -> None:
    contract = Path("docs/SUBMISSION_LIVE_EMAIL_CONNECTOR_CONTRACT.md")
    runbook = Path("docs/SUBMISSION_FIRST_ASSIGNMENT_RUNBOOK.md")

    assert contract.is_file()
    assert runbook.is_file()

    contract_text = contract.read_text(encoding="utf-8")
    runbook_text = runbook.read_text(encoding="utf-8")
    assert "read-only" in contract_text.lower()
    assert "transport adapter only" in contract_text
    assert "Dry-run is the default" in contract_text
    assert "fixture-backed intake remains the test source" in contract_text.lower()
    assert "--dry-run" in runbook_text
    assert "--apply" in runbook_text
    assert "build_outgoing_email_queue.py" in runbook_text
    assert "Do not commit real config" in runbook_text


def test_runbook_points_to_ignored_private_roots() -> None:
    text = Path("docs/SUBMISSION_FIRST_ASSIGNMENT_RUNBOOK.md").read_text(encoding="utf-8")

    assert "data/submissions/<assignment_id>/" in text
    assert "output/submissions/" in text
    assert "reports/submissions/" in text
