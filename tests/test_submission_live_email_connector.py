from __future__ import annotations

import json
import subprocess
from pathlib import Path

from exam_bank.submissions.email_connectors import FakeEmailConnector
from exam_bank.submissions.email_fixtures import load_email_fixtures
from exam_bank.submissions.live_email_import import import_live_email_submissions
from exam_bank.submissions.live_email_import_cli import build_parser


FIXTURES = Path("tests/fixtures/submissions")
ASSIGNMENT_ID = "p3_vectors_hw1"


def _connector_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "data" / "submissions" / ASSIGNMENT_ID
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "email_connector_config.json"
    path.write_text(
        json.dumps(
            {
                "provider": "fake",
                "account_label": "teacher_assignment_inbox",
                "assignment_id": ASSIGNMENT_ID,
                "mailbox_scope": "assignment-label",
                "search_query": ASSIGNMENT_ID,
                "since": "2026-06-01T00:00:00+00:00",
                "limit": 20,
                "dry_run": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_connector_config_template_exists_and_contains_no_secrets() -> None:
    template = Path("templates/submissions/email_connector_config.template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert payload["dry_run"] is True
    assert payload["assignment_id"] == "p3_quiz_2026_06_23"
    serialized = json.dumps(payload).lower()
    for forbidden in ["token", "secret", "password", "refresh"]:
        assert forbidden not in serialized


def test_live_import_defaults_to_dry_run_and_apply_requires_flag(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--assignment",
            str(FIXTURES / "assignment_p3_vectors_hw1.json"),
            "--roster",
            str(FIXTURES / "roster_class_12a.csv"),
            "--connector-config",
            str(_connector_config(tmp_path)),
        ]
    )
    assert args.apply is False

    result = import_live_email_submissions(
        assignment_path=FIXTURES / "assignment_p3_vectors_hw1.json",
        roster_path=FIXTURES / "roster_class_12a.csv",
        connector_config_path=args.connector_config,
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    assert result["dry_run"] is True
    assert result["apply"] is False
    assert result["phase4_result"] is None


def test_fake_connector_messages_convert_to_existing_email_models() -> None:
    fixtures = load_email_fixtures(FIXTURES / "email_inbox", ASSIGNMENT_ID)
    connector = FakeEmailConnector(fixtures)
    refs = connector.list_messages(mailbox_scope="assignment-label", search_query=ASSIGNMENT_ID, since=None, limit=10)
    message = connector.fetch_message(refs[0].message_id)
    attachments = connector.fetch_attachments(refs[0].message_id)

    assert message.assignment_id == ASSIGNMENT_ID
    assert message.source == "fake_connector"
    assert message.attachment_count == len(message.attachments)
    assert set(attachments) == {attachment.attachment_id for attachment in message.attachments}


def test_dry_run_does_not_stage_pdfs_or_call_phase1(tmp_path: Path) -> None:
    fixtures = load_email_fixtures(FIXTURES / "email_inbox", ASSIGNMENT_ID)
    result = import_live_email_submissions(
        assignment_path=FIXTURES / "assignment_p3_vectors_hw1.json",
        roster_path=FIXTURES / "roster_class_12a.csv",
        connector_config_path=_connector_config(tmp_path),
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        connector=FakeEmailConnector(fixtures),
    )

    output_root = tmp_path / "output" / "submissions" / ASSIGNMENT_ID
    summary = result["summary"]
    dry_run_report = _json(Path(result["dry_run_report"]))

    assert summary["dry_run"] is True
    assert summary["messages_found"] == 7
    assert summary["phase1_accepted_count"] == 0
    assert result["phase4_result"] is None
    assert not (output_root / "email_intake").exists()
    assert not (output_root / "live_email_import" / "materialized_fixtures").exists()
    assert len(dry_run_report["decisions"]) == 7
    assert Path(result["readiness_report"]).is_file()


def test_apply_with_fake_connector_reuses_phase4_intake(tmp_path: Path) -> None:
    fixtures = load_email_fixtures(FIXTURES / "email_inbox", ASSIGNMENT_ID)
    result = import_live_email_submissions(
        assignment_path=FIXTURES / "assignment_p3_vectors_hw1.json",
        roster_path=FIXTURES / "roster_class_12a.csv",
        connector_config_path=_connector_config(tmp_path),
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        apply=True,
        connector=FakeEmailConnector(fixtures),
    )

    phase4 = result["phase4_result"]
    assert phase4 is not None
    assert result["dry_run"] is False
    assert result["summary"]["phase1_accepted_count"] == 3
    assert Path(phase4["email_intake_dir"]).is_dir()
    assert Path(phase4["phase1_result"]["completion_report"]).is_file()
    provenance = _json(Path(phase4["email_intake_dir"]) / "provenance.json")
    assert {item["source"] for item in provenance} == {"live_connector"}


def test_no_tokens_credentials_or_sending_are_written_to_outputs(tmp_path: Path) -> None:
    result = import_live_email_submissions(
        assignment_path=FIXTURES / "assignment_p3_vectors_hw1.json",
        roster_path=FIXTURES / "roster_class_12a.csv",
        connector_config_path=_connector_config(tmp_path),
        submission_output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )
    output_text = Path(result["dry_run_report"]).read_text(encoding="utf-8") + Path(result["audit_log"]).read_text(encoding="utf-8")

    for forbidden in ["access_token", "refresh_token", "client_secret", "password", "sendmail"]:
        assert forbidden not in output_text.lower()


def test_private_connector_config_path_is_ignored() -> None:
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            f"data/submissions/{ASSIGNMENT_ID}/email_connector_config.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert ignored.stdout.strip() == f"data/submissions/{ASSIGNMENT_ID}/email_connector_config.json"


def test_no_network_sending_behavior_exists() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("src/exam_bank/submissions/email_connectors.py"),
            Path("src/exam_bank/submissions/live_email_import.py"),
            Path("src/exam_bank/submissions/live_email_import_cli.py"),
        ]
    ).lower()
    for forbidden in ["smtplib", "sendmail", "requests.post", "requests.put", "urllib.request"]:
        assert forbidden not in source
