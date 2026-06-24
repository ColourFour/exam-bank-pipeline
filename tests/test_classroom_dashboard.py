from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import fitz

from exam_bank.classroom_dashboard.server import ClassroomDashboardApp, ClassroomDashboardConfig
from exam_bank.emailing.models import EmailSendResult


def _app(tmp_path: Path) -> ClassroomDashboardApp:
    return ClassroomDashboardApp(
        ClassroomDashboardConfig(
            classes_root=tmp_path / "data" / "classes",
            output_root=tmp_path / "output" / "submissions",
            reports_root=tmp_path / "reports" / "submissions",
            audit_path=tmp_path / "reports" / "classroom_dashboard" / "classroom_dashboard_audit.jsonl",
        )
    )


def _json(response) -> dict[str, object]:
    return json.loads(response.body.decode("utf-8"))


def _pdf_bytes(text: str = "Question 1 complete\nQuestion 2 complete") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _multipart(fields: dict[str, str], files: list[tuple[str, str, bytes, str]] | None = None) -> tuple[dict[str, str], bytes]:
    boundary = "----exam-bank-test"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for field, filename, content, content_type in files or []:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return {"Content-Type": f"multipart/form-data; boundary={boundary}"}, b"".join(chunks)


def _post_json(app: ClassroomDashboardApp, path: str, payload: dict[str, object]):
    return app.handle("POST", path, headers={"Content-Type": "application/json"}, body=json.dumps(payload).encode("utf-8"))


def _create_class(app: ClassroomDashboardApp) -> None:
    response = _post_json(
        app,
        "/api/classes",
        {
            "class_id": "class_12a",
            "display_name": "Class 12A",
            "course_id": "p3",
            "teacher_email": "brooker@rdfzcygj.cn",
            "timezone": "Asia/Shanghai",
        },
    )
    assert response.status == 201


def _save_roster(app: ClassroomDashboardApp, email: str = "one@example.invalid", *, second_email: str = "two@example.invalid") -> None:
    response = _post_json(
        app,
        "/api/classes/class_12a/roster",
        {
            "rows": [
                {"student_id": "S0001", "display_name": "Student One", "email": email, "active": "true"},
                {"student_id": "S0002", "display_name": "Student Two", "email": second_email, "active": "true"},
            ]
        },
    )
    assert response.status == 200


def _create_assignment(
    app: ClassroomDashboardApp,
    *,
    assignment_id: str = "hw1",
    title: str = "test 1 - topic - content",
    send_at: str = "2026-06-24T09:00",
    due_at: str = "2026-06-30T17:00",
    pdf: bytes | None = None,
) -> object:
    headers, body = _multipart(
        {
            "assignment_id": assignment_id,
            "title": title,
            "send_at": send_at,
            "due_at": due_at,
            "timezone": "Asia/Shanghai",
        },
        [("pdf", "assignment.pdf", pdf if pdf is not None else _pdf_bytes("Assignment"), "application/pdf")],
    )
    return app.handle("POST", "/api/classes/class_12a/assignments", headers=headers, body=body)


class ConnectedFakeMailProvider:
    sent_messages: list[dict[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def check_connection(self):
        return type("Status", (), {"connected": True})()

    def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return EmailSendResult(
            provider="mailapp",
            sent=True,
            dry_run=False,
            to=str(kwargs["to"]),
            subject=str(kwargs["subject"]),
            provider_message_id=f"fake-{len(self.sent_messages)}",
            sent_at=datetime.now(timezone.utc),
            from_address=str(kwargs.get("from_address") or ""),
        )


def test_classroom_route_app_serves_home() -> None:
    response = ClassroomDashboardApp().handle("GET", "/")

    assert response.status == 200
    assert b"Exam Bank Classroom" in response.body
    assert b"sendLaterBtn" in response.body
    assert b"sendNowBtn" in response.body
    assert b"confirmModal" in response.body
    assert b"Exam Bank" in response.body
    assert b"+ Add class" in response.body
    assert b"classSidebar" in response.body
    assert b"editClassRosterBtn" in response.body
    assert b"deleteClassBtn" in response.body
    assert b"more-menu" in response.body
    assert b"submissions-card" in response.body
    assert b"add-class-modal" in response.body


def test_dashboard_assets_include_modern_state_and_modal_ui() -> None:
    root = Path(__file__).parents[1] / "src" / "exam_bank" / "classroom_dashboard" / "static"
    js = (root / "app.js").read_text(encoding="utf-8")
    css = (root / "style.css").read_text(encoding="utf-8")

    assert "class-nav-item" in js
    assert "draft_assignment_count" in js
    assert "archived_assignment_count" in js
    assert "openAddClassModal" in js
    assert "openRosterForSelectedClass" in js
    assert "openDeleteClassModal" in js
    assert "afterConfirm: \"class_deleted\"" in js
    assert "localStorage" in js
    assert "openConfirmModal" in js
    assert "--surface" in css
    assert "--sidebar-width: 300px" in css
    assert "grid-template-columns: var(--sidebar-width) minmax(0, 1fr)" in css
    assert "height: 100vh" in css
    assert "background: linear-gradient" in css
    assert ".add-class-modal" in css
    assert ".modal-backdrop" in css
    assert ".class-nav-item.selected" in css
    assert ".button.danger" in css


def test_create_class_writes_class_json_and_roster(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)

    assert (tmp_path / "data" / "classes" / "class_12a" / "class.json").is_file()
    assert (tmp_path / "data" / "classes" / "class_12a" / "roster.csv").is_file()


def test_delete_class_requires_confirmation_and_exact_class_id(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)

    missing = _post_json(app, "/api/classes/class_12a/delete", {})
    wrong_text = _post_json(app, "/api/classes/class_12a/delete", {"confirm": True, "confirm_text": "DELETE"})
    deleted = _post_json(app, "/api/classes/class_12a/delete", {"confirm": True, "confirm_text": "class_12a"})

    assert missing.status == 400
    assert _json(missing)["error"] == "delete_confirmation_required"
    assert wrong_text.status == 400
    assert _json(wrong_text)["error"] == "delete_confirmation_text_required"
    assert deleted.status == 200
    assert not (tmp_path / "data" / "classes" / "class_12a").exists()


def test_roster_save_writes_csv_and_backup(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    response = _post_json(app, "/api/classes/class_12a/roster", {"rows": [{"student_id": "S0001", "display_name": "Updated", "email": "one@example.invalid", "active": "true"}]})

    assert response.status == 200
    assert list((tmp_path / "data" / "classes" / "class_12a").glob("roster.backup.*.csv"))


def test_duplicate_student_ids_are_rejected(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    response = _post_json(
        app,
        "/api/classes/class_12a/roster",
        {"rows": [{"student_id": "S0001", "display_name": "One"}, {"student_id": "S0001", "display_name": "Dup"}]},
    )

    assert response.status == 400
    assert _json(response)["error"] == "duplicate_student_id"


def test_assignment_pdf_upload_saves_pdf_and_assignment_json(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    response = _create_assignment(app)

    assert response.status == 201
    assert (tmp_path / "data" / "classes" / "class_12a" / "assignments" / "hw1" / "assignment.pdf").is_file()
    payload = json.loads((tmp_path / "data" / "classes" / "class_12a" / "assignments" / "hw1" / "assignment.json").read_text(encoding="utf-8"))
    assert payload["send_at"].endswith("+08:00")
    assert payload["due_at"].endswith("+08:00")


def test_non_pdf_upload_is_rejected(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    response = _create_assignment(app, pdf=b"not a pdf")

    assert response.status == 400
    assert _json(response)["error"] == "invalid_assignment_pdf"


def test_preview_dispatch_does_not_send_email(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)

    response = app.handle("POST", "/api/classes/class_12a/assignments/hw1/preview-dispatch")
    payload = _json(response)

    assert response.status == 200
    assert payload["preview"]["recipient_count"] == 2
    assert payload["preview"]["subject"] == "assignment: test 1 - topic - content"
    assert payload["preview"]["scores_included"] is False


def test_assignment_list_uses_plain_text_assignment_label(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)

    response = app.handle("GET", "/api/classes/class_12a/assignments")
    payload = _json(response)

    assert response.status == 200
    assert payload["assignments"][0]["title"] == "assignment: test 1 - topic - content"


def test_class_list_reports_assignment_status_counts(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app, assignment_id="draft_hw", send_at="2999-06-24T09:00", due_at="2999-06-30T17:00")
    _create_assignment(app, assignment_id="sent_hw", send_at="2999-06-24T09:00", due_at="2999-06-30T17:00")
    _create_assignment(app, assignment_id="past_hw", send_at="2000-06-24T09:00", due_at="2000-06-30T17:00")
    sent_meta_path = tmp_path / "data" / "classes" / "class_12a" / "assignments" / "sent_hw" / "assignment.json"
    sent_meta = json.loads(sent_meta_path.read_text(encoding="utf-8"))
    sent_meta["email_status"] = "sent"
    sent_meta_path.write_text(json.dumps(sent_meta), encoding="utf-8")

    response = app.handle("GET", "/api/classes")
    payload = _json(response)

    assert response.status == 200
    class_payload = payload["classes"][0]
    assert class_payload["active_assignment_count"] == 1
    assert class_payload["draft_assignment_count"] == 1
    assert class_payload["archived_assignment_count"] == 1


def test_live_dispatch_requires_explicit_confirmation(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)

    response = _post_json(app, "/api/classes/class_12a/assignments/hw1/send-dispatch", {"now": "2026-06-24T10:00:00+08:00"})

    assert response.status == 400
    assert _json(response)["error"] == "send_confirmation_required"


def test_live_dispatch_blocks_missing_provider(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    class BlockedProvider:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_connection(self):
            return type("Status", (), {"connected": False})()

    monkeypatch.setattr(server, "MailAppEmailProvider", BlockedProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)

    response = _post_json(
        app,
        "/api/classes/class_12a/assignments/hw1/send-dispatch",
        {"now": "2026-06-24T10:00:00+08:00", "confirm": True, "confirm_text": "SEND"},
    )

    assert response.status == 400
    assert _json(response)["error"] == "email_provider_not_configured"


def test_live_dispatch_blocks_invalid_roster_emails(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app, email="bad-email")
    _create_assignment(app)

    response = _post_json(
        app,
        "/api/classes/class_12a/assignments/hw1/send-dispatch",
        {"now": "2026-06-24T10:00:00+08:00", "confirm": True, "confirm_text": "SEND"},
    )

    assert response.status == 400
    assert _json(response)["error"] == "invalid_roster_emails"


def test_send_later_requires_explicit_confirmation(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)

    response = _post_json(app, "/api/classes/class_12a/assignments/hw1/confirm-send-later", {})

    assert response.status == 400
    assert _json(response)["error"] == "send_confirmation_required"


def test_confirm_send_later_does_not_send_assignment_messages(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    ConnectedFakeMailProvider.sent_messages = []
    monkeypatch.setattr(server, "MailAppEmailProvider", ConnectedFakeMailProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app, email="brooker@rdfzcygj.cn", second_email="")
    _create_assignment(app, send_at="2026-07-01T09:00", due_at="2026-07-30T17:00")

    response = _post_json(
        app,
        "/api/classes/class_12a/assignments/hw1/confirm-send-later",
        {"confirm": True},
    )
    payload = _json(response)

    assert response.status == 200
    assert payload["sent"] == 0
    assert payload["scheduled"] is True
    assert ConnectedFakeMailProvider.sent_messages == []


def test_confirm_send_date_alias_does_not_send_assignment_messages(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    ConnectedFakeMailProvider.sent_messages = []
    monkeypatch.setattr(server, "MailAppEmailProvider", ConnectedFakeMailProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app, email="brooker@rdfzcygj.cn", second_email="")
    _create_assignment(app, send_at="2026-07-01T09:00", due_at="2026-07-30T17:00")

    response = _post_json(
        app,
        "/api/classes/class_12a/assignments/hw1/confirm-send-date",
        {"confirmed": True},
    )

    assert response.status == 200
    assert _json(response)["sent"] == 0
    assert ConnectedFakeMailProvider.sent_messages == []


def test_send_now_sends_assignment_even_when_send_date_is_future(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    ConnectedFakeMailProvider.sent_messages = []
    monkeypatch.setattr(server, "MailAppEmailProvider", ConnectedFakeMailProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app, email="brooker@rdfzcygj.cn", second_email="")
    _create_assignment(app, assignment_id="hw_future", send_at="2026-07-15T09:00", due_at="2026-07-30T17:00")

    response = _post_json(
        app,
        "/api/classes/class_12a/assignments/hw_future/send-now",
        {"confirm": True},
    )
    payload = _json(response)

    assert response.status == 200
    assert payload["result"]["sent"] == 1
    assert ConnectedFakeMailProvider.sent_messages[0]["to"] == "brooker@rdfzcygj.cn"


def test_send_now_allows_resending_sent_assignment(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    ConnectedFakeMailProvider.sent_messages = []
    monkeypatch.setattr(server, "MailAppEmailProvider", ConnectedFakeMailProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app, email="brooker@rdfzcygj.cn", second_email="")
    _create_assignment(app, send_at="2026-07-15T09:00", due_at="2026-07-30T17:00")

    first = _post_json(app, "/api/classes/class_12a/assignments/hw1/send-now", {"confirm": True})
    preview = app.handle("POST", "/api/classes/class_12a/assignments/hw1/preview-dispatch")
    second = _post_json(app, "/api/classes/class_12a/assignments/hw1/send-now", {"confirm": True})

    assert first.status == 200
    assert second.status == 200
    assert _json(second)["result"]["sent"] == 1
    assert len(ConnectedFakeMailProvider.sent_messages) == 2
    preview_payload = _json(preview)["preview"]
    assert preview_payload["already_sent"] is True
    assert preview_payload["sent_recipient_count"] == 1
    assert preview_payload["sent_recipients"][0]["email"] == "brooker@rdfzcygj.cn"


def test_send_now_to_multiple_recipients_requires_send_text(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    ConnectedFakeMailProvider.sent_messages = []
    monkeypatch.setattr(server, "MailAppEmailProvider", ConnectedFakeMailProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app, email="one@example.invalid", second_email="two@example.invalid")
    _create_assignment(app, send_at="2026-07-15T09:00", due_at="2026-07-30T17:00")

    response = _post_json(
        app,
        "/api/classes/class_12a/assignments/hw1/send-now",
        {"confirm": True},
    )

    assert response.status == 400
    assert _json(response)["error"] == "send_confirmation_text_required"
    assert ConnectedFakeMailProvider.sent_messages == []


def test_submissions_panel_has_distinct_styling() -> None:
    css = (Path(__file__).parents[1] / "src" / "exam_bank" / "classroom_dashboard" / "static" / "style.css").read_text(encoding="utf-8")

    assert ".submissions-card" in css
    assert "#f6fef9" in css


def test_upload_submissions_saves_pdfs_to_inbox(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)
    headers, body = _multipart({}, [("submissions", "S0001.pdf", _pdf_bytes(), "application/pdf")])

    response = app.handle("POST", "/api/classes/class_12a/assignments/hw1/upload-submissions", headers=headers, body=body)

    assert response.status == 200
    assert (tmp_path / "data" / "classes" / "class_12a" / "assignments" / "hw1" / "inbox" / "S0001.pdf").is_file()


def test_ingest_submissions_calls_existing_workflow(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)
    headers, body = _multipart({}, [("submissions", "S0001.pdf", _pdf_bytes(), "application/pdf")])
    app.handle("POST", "/api/classes/class_12a/assignments/hw1/upload-submissions", headers=headers, body=body)

    response = _post_json(app, "/api/classes/class_12a/assignments/hw1/ingest-submissions", {})

    assert response.status == 200
    assert (tmp_path / "reports" / "submissions" / "hw1_completion.csv").is_file()


def test_acknowledgement_preview_does_not_send_email(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)
    headers, body = _multipart({}, [("submissions", "S0001.pdf", _pdf_bytes(), "application/pdf")])
    app.handle("POST", "/api/classes/class_12a/assignments/hw1/upload-submissions", headers=headers, body=body)
    _post_json(app, "/api/classes/class_12a/assignments/hw1/ingest-submissions", {})

    response = app.handle("POST", "/api/classes/class_12a/assignments/hw1/preview-acknowledgements")

    assert response.status == 200
    assert _json(response)["count"] == 1
    assert _json(response)["scores_included"] is False
    assert _json(response)["attachments"] == []
    assert _json(response)["missing_students_receive_email"] is False


def test_acknowledgement_later_requires_confirmation_and_does_not_send(tmp_path: Path, monkeypatch) -> None:
    import exam_bank.classroom_dashboard.server as server

    ConnectedFakeMailProvider.sent_messages = []
    monkeypatch.setattr(server, "MailAppEmailProvider", ConnectedFakeMailProvider)
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)
    headers, body = _multipart({}, [("submissions", "S0001.pdf", _pdf_bytes(), "application/pdf")])
    app.handle("POST", "/api/classes/class_12a/assignments/hw1/upload-submissions", headers=headers, body=body)
    _post_json(app, "/api/classes/class_12a/assignments/hw1/ingest-submissions", {})

    missing = _post_json(app, "/api/classes/class_12a/assignments/hw1/confirm-acknowledgements-later", {})
    confirmed = _post_json(app, "/api/classes/class_12a/assignments/hw1/confirm-acknowledgements-later", {"confirm": True})

    assert missing.status == 400
    assert _json(missing)["error"] == "send_confirmation_required"
    assert confirmed.status == 200
    assert _json(confirmed)["sent"] == 0
    assert ConnectedFakeMailProvider.sent_messages == []


def test_acknowledgement_live_send_requires_confirmation(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)
    headers, body = _multipart({}, [("submissions", "S0001.pdf", _pdf_bytes(), "application/pdf")])
    app.handle("POST", "/api/classes/class_12a/assignments/hw1/upload-submissions", headers=headers, body=body)
    _post_json(app, "/api/classes/class_12a/assignments/hw1/ingest-submissions", {})

    response = _post_json(app, "/api/classes/class_12a/assignments/hw1/send-acknowledgements", {})

    assert response.status == 400
    assert _json(response)["error"] == "send_confirmation_required"


def test_dashboard_binds_to_localhost_by_default() -> None:
    config = ClassroomDashboardConfig()

    assert config.classes_root == Path("data/classes")


def test_audit_log_excludes_credentials_and_email_bodies(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _create_class(app)
    _save_roster(app)
    _create_assignment(app)
    app.handle("POST", "/api/classes/class_12a/assignments/hw1/preview-dispatch")
    _post_json(app, "/api/classes/class_12a/assignments/hw1/dry-run-dispatch", {"now": "2026-06-24T10:00:00+08:00"})

    audit_text = (tmp_path / "reports" / "classroom_dashboard" / "classroom_dashboard_audit.jsonl").read_text(encoding="utf-8")
    assert "password" not in audit_text.lower()
    assert "Please complete the assignment" not in audit_text
