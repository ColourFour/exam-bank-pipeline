from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import fitz

from exam_bank.classroom import (
    add_assignment,
    dispatch_due_messages,
    ingest_class_assignment,
    init_class_workspace,
)
from exam_bank.emailing.mailapp import MailAppEmailProvider


def _write_pdf(path: Path, text: str = "Question 1 complete\nQuestion 2 complete") -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _write_roster(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["student_id", "class_id", "display_name", "email", "active"])
        writer.writeheader()
        writer.writerow({"student_id": "S0001", "class_id": "class_12a", "display_name": "Student One", "email": "one@example.invalid", "active": "true"})
        writer.writerow({"student_id": "S0002", "class_id": "class_12a", "display_name": "Student Two", "email": "two@example.invalid", "active": "true"})


def test_class_workspace_creates_roster_and_assignments_folder(tmp_path: Path) -> None:
    source_roster = tmp_path / "source_roster.csv"
    _write_roster(source_roster)

    result = init_class_workspace(class_id="class_12a", roster_source=source_roster, classes_root=tmp_path / "data" / "classes")

    assert Path(result["roster"]).is_file()
    assert Path(result["assignments_dir"]).is_dir()
    rows = list(csv.DictReader(Path(result["roster"]).open("r", encoding="utf-8", newline="")))
    assert rows[0]["last_assignment_status"] == ""


def test_add_assignment_builds_distribution_and_reminder_schedule(tmp_path: Path) -> None:
    classes_root = tmp_path / "data" / "classes"
    source_roster = tmp_path / "source_roster.csv"
    assignment_pdf = tmp_path / "assignment.pdf"
    _write_roster(source_roster)
    _write_pdf(assignment_pdf)
    init_class_workspace(class_id="class_12a", roster_source=source_roster, classes_root=classes_root)

    result = add_assignment(
        class_id="class_12a",
        pdf_path=assignment_pdf,
        assignment_id="hw1",
        title="Homework 1",
        due_at=datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc),
        send_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        classes_root=classes_root,
    )

    schedule = json.loads(Path(result["schedule"]).read_text(encoding="utf-8"))
    assert Path(result["assignment_pdf"]).is_file()
    assert len(schedule) == 8
    assert {item["message_type"] for item in schedule} == {"assignment_distribution", "reminder_48h", "reminder_24h", "reminder_12h"}


def test_dispatch_due_messages_dry_run_preserves_schedule(tmp_path: Path) -> None:
    classes_root = tmp_path / "data" / "classes"
    source_roster = tmp_path / "source_roster.csv"
    assignment_pdf = tmp_path / "assignment.pdf"
    _write_roster(source_roster)
    _write_pdf(assignment_pdf)
    init_class_workspace(class_id="class_12a", roster_source=source_roster, classes_root=classes_root)
    result = add_assignment(
        class_id="class_12a",
        pdf_path=assignment_pdf,
        assignment_id="hw1",
        title="Homework 1",
        due_at=datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc),
        send_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        classes_root=classes_root,
    )

    dispatch = dispatch_due_messages(
        class_id="class_12a",
        assignment_id="hw1",
        now=datetime(2026, 6, 24, 10, 0, tzinfo=timezone.utc),
        send_live=False,
        classes_root=classes_root,
    )

    assert dispatch["dry_run"] == 2
    schedule = json.loads(Path(result["schedule"]).read_text(encoding="utf-8"))
    assert all(item["status"] == "scheduled" for item in schedule if item["message_type"] == "assignment_distribution")


def test_ingest_class_assignment_updates_roster_with_completion_and_questions(tmp_path: Path) -> None:
    classes_root = tmp_path / "data" / "classes"
    source_roster = tmp_path / "source_roster.csv"
    assignment_pdf = tmp_path / "assignment.pdf"
    submission_pdf = tmp_path / "S0001.pdf"
    _write_roster(source_roster)
    _write_pdf(assignment_pdf, "Question 1\nQuestion 2")
    _write_pdf(submission_pdf, "Question 1 done\nQuestion 2 done")
    init_class_workspace(class_id="class_12a", roster_source=source_roster, classes_root=classes_root)
    result = add_assignment(
        class_id="class_12a",
        pdf_path=assignment_pdf,
        assignment_id="hw1",
        title="Homework 1",
        due_at=datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc),
        send_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        classes_root=classes_root,
    )
    inbox = Path(result["inbox"])
    submission_pdf.rename(inbox / submission_pdf.name)

    ingest = ingest_class_assignment(
        class_id="class_12a",
        assignment_id="hw1",
        classes_root=classes_root,
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
    )

    assert ingest["accepted_count"] == 1
    assert Path(str(ingest["answer_check_results"])).is_file()
    assert Path(str(ingest["answer_check_csv"])).is_file()
    assert ingest["answer_check_summary"]["question_count"] == 2
    roster_path = classes_root / "class_12a" / "roster.csv"
    rows = {row["student_id"]: row for row in csv.DictReader(roster_path.open("r", encoding="utf-8", newline=""))}
    assert rows["S0001"]["last_assignment_status"] == "submitted"
    assert rows["S0001"]["last_questions_completed"] == "1;2"
    assert rows["S0002"]["last_assignment_status"] == "missing"


def test_mailapp_import_uses_assignment_label_and_sender_prefixed_files(tmp_path: Path) -> None:
    classes_root = tmp_path / "data" / "classes"
    source_roster = tmp_path / "source_roster.csv"
    assignment_pdf = tmp_path / "assignment.pdf"
    _write_roster(source_roster)
    _write_pdf(assignment_pdf, "Assignment")
    init_class_workspace(class_id="class_12a", roster_source=source_roster, classes_root=classes_root)
    add_assignment(
        class_id="class_12a",
        pdf_path=assignment_pdf,
        assignment_id="hw1",
        title="Homework 1",
        due_at=datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc),
        send_at=datetime(2026, 6, 24, 9, 0, tzinfo=timezone.utc),
        classes_root=classes_root,
    )

    class FakeMailAppProvider(MailAppEmailProvider):
        seen_query = ""
        seen_roster: dict[str, str] = {}

        def __init__(self) -> None:
            super().__init__(runner=lambda script: None)  # type: ignore[arg-type]

        def export_student_pdf_attachments(self, **kwargs):
            self.seen_query = str(kwargs["query"])
            self.seen_roster = dict(kwargs["roster_email_to_student_id"])
            target_dir = Path(kwargs["target_dir"])
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "S0001__assignment.pdf"
            _write_pdf(target, "Question 1 done")
            return [target]

    provider = FakeMailAppProvider()

    ingest = ingest_class_assignment(
        class_id="class_12a",
        assignment_id="hw1",
        classes_root=classes_root,
        output_root=tmp_path / "output" / "submissions",
        reports_root=tmp_path / "reports" / "submissions",
        import_from_mailapp=True,
        from_address="brooker@rdfzcygj.cn",
        provider=provider,
    )

    assert provider.seen_query == "assignment: Homework 1"
    assert provider.seen_roster == {"one@example.invalid": "S0001", "two@example.invalid": "S0002"}
    assert ingest["accepted_count"] == 1
