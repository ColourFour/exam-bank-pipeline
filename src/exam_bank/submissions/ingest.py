from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from exam_bank.submissions.audit_log import AuditLog
from exam_bank.submissions.feedback_drafts import (
    build_acknowledgement_draft,
    build_missing_reminder_draft,
    build_resend_draft,
    write_drafts_jsonl,
)
from exam_bank.submissions.models import Assignment, CompletionRow, FeedbackDraft, Student, Submission, dataclass_to_json_dict
from exam_bank.submissions.reports import write_completion_csv
from exam_bank.submissions.validation import sha256_file, validate_pdf


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_assignment(path: Path) -> Assignment:
    data = json.loads(path.read_text(encoding="utf-8"))
    due_at_raw = data.get("due_at")
    return Assignment(
        assignment_id=str(data["assignment_id"]),
        course_id=str(data["course_id"]),
        title=str(data["title"]),
        class_id=str(data["class_id"]),
        due_at=parse_datetime(str(due_at_raw)) if due_at_raw else None,
        timezone=str(data["timezone"]),
        accepted_file_types=[str(item).lower().lstrip(".") for item in data.get("accepted_file_types", ["pdf"])],
        max_files_per_student=int(data["max_files_per_student"]),
        max_file_size_mb=int(data["max_file_size_mb"]),
        allow_late=bool(data["allow_late"]),
        source_question_ids=[str(item) for item in data.get("source_question_ids", [])],
    )


def _parse_active(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "active"}


def load_roster(path: Path) -> list[Student]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"student_id", "class_id", "display_name", "email"}
    if not rows:
        return []
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Roster missing required columns: {sorted(missing)}")
    return [
        Student(
            student_id=row["student_id"].strip(),
            class_id=row["class_id"].strip(),
            display_name=row["display_name"].strip(),
            email=row.get("email", "").strip(),
            active=_parse_active(row.get("active", "true")),
            source_file=row.get("source_file", "").strip(),
        )
        for row in rows
    ]


def _normalize_student_match_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def match_student_id_from_filename(path: Path, student_ids: set[str], assignment_id: str = "") -> str | None:
    stem = path.stem
    candidates = [stem]
    if "_" in stem:
        candidates.extend(part for part in stem.split("_") if part)
    if assignment_id and stem.startswith(f"{assignment_id}_"):
        candidates.append(stem.removeprefix(f"{assignment_id}_"))

    normalized_student_ids = {_normalize_student_match_token(student_id): student_id for student_id in student_ids}
    for candidate in candidates:
        matched = normalized_student_ids.get(_normalize_student_match_token(candidate))
        if matched is not None:
            return matched
    normalized_stem = _normalize_student_match_token(stem)
    for student_id in student_ids:
        normalized_student_id = _normalize_student_match_token(student_id)
        if normalized_stem.startswith(f"{normalized_student_id}_") or normalized_stem.endswith(f"_{normalized_student_id}"):
            return student_id
    return None


def _received_at(path: Path, override: datetime | None) -> datetime:
    if override is not None:
        return override
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _require_private_roots(output_root: Path, reports_root: Path) -> None:
    output_parts = output_root.parts
    reports_parts = reports_root.parts
    if len(output_parts) < 2 or output_parts[-2:] != ("output", "submissions"):
        raise ValueError("Submission output_root must end with output/submissions")
    if len(reports_parts) < 2 or reports_parts[-2:] != ("reports", "submissions"):
        raise ValueError("Submission reports_root must end with reports/submissions")


def _copy_submission(source: Path, target_dir: Path) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return target.as_posix()


def _make_submission(
    *,
    assignment: Assignment,
    student_id: str,
    source: Path,
    stored_path: str,
    received_at: datetime,
    status: str,
    late: bool,
    reasons: list[str],
) -> Submission:
    digest = sha256_file(source) if source.exists() and source.is_file() else ""
    return Submission(
        submission_id=f"{assignment.assignment_id}:{source.name}:{digest[:12]}",
        assignment_id=assignment.assignment_id,
        student_id=student_id,
        source_filename=source.name,
        stored_pdf_path=stored_path,
        sha256=digest,
        received_at=received_at,
        submitted_via="local_folder",
        status=status,
        late=late,
        validation_reasons=reasons,
    )


def _choose_duplicate_winner(submissions: list[Submission]) -> Submission:
    return sorted(submissions, key=lambda item: (item.received_at, item.source_filename))[-1]


def _completion_rows(
    assignment: Assignment,
    roster: list[Student],
    accepted_by_student: dict[str, Submission],
    rejected_by_student: dict[str, list[Submission]],
    duplicate_student_ids: set[str],
) -> list[CompletionRow]:
    rows: list[CompletionRow] = []
    for student in sorted((item for item in roster if item.active), key=lambda item: item.student_id):
        accepted = accepted_by_student.get(student.student_id)
        rejected = rejected_by_student.get(student.student_id, [])
        if accepted is not None:
            notes = "duplicates_found" if student.student_id in duplicate_student_ids else ""
            rows.append(
                CompletionRow(
                    assignment_id=assignment.assignment_id,
                    assignment_title=assignment.title,
                    class_id=assignment.class_id,
                    student_id=student.student_id,
                    display_name=student.display_name,
                    email=student.email,
                    status="late" if accepted.late else "submitted",
                    submitted_at=accepted.received_at.isoformat(),
                    late=accepted.late,
                    source_filename=accepted.source_filename,
                    stored_pdf_path=accepted.stored_pdf_path,
                    rejection_reasons="",
                    notes=notes,
                )
            )
        elif rejected:
            reasons = sorted({reason for submission in rejected for reason in submission.validation_reasons})
            rows.append(
                CompletionRow(
                    assignment_id=assignment.assignment_id,
                    assignment_title=assignment.title,
                    class_id=assignment.class_id,
                    student_id=student.student_id,
                    display_name=student.display_name,
                    email=student.email,
                    status="rejected",
                    submitted_at=max((submission.received_at for submission in rejected)).isoformat(),
                    late=any(submission.late for submission in rejected),
                    source_filename=";".join(submission.source_filename for submission in rejected),
                    stored_pdf_path=";".join(submission.stored_pdf_path for submission in rejected),
                    rejection_reasons=";".join(reasons),
                    notes="",
                )
            )
        else:
            rows.append(
                CompletionRow(
                    assignment_id=assignment.assignment_id,
                    assignment_title=assignment.title,
                    class_id=assignment.class_id,
                    student_id=student.student_id,
                    display_name=student.display_name,
                    email=student.email,
                    status="missing",
                    submitted_at="",
                    late=False,
                    source_filename="",
                    stored_pdf_path="",
                    rejection_reasons="",
                    notes="",
                )
            )
    return rows


def _build_drafts(
    assignment: Assignment,
    roster_by_id: dict[str, Student],
    accepted: list[Submission],
    rejected: list[Submission],
    rows: list[CompletionRow],
) -> tuple[list[FeedbackDraft], list[FeedbackDraft], list[FeedbackDraft]]:
    acknowledgement = [
        build_acknowledgement_draft(assignment, roster_by_id[submission.student_id], submission)
        for submission in accepted
        if submission.student_id in roster_by_id
    ]
    resend = [
        build_resend_draft(assignment, roster_by_id[submission.student_id], submission)
        for submission in rejected
        if submission.student_id in roster_by_id
    ]
    reminder = [
        build_missing_reminder_draft(assignment, roster_by_id[row.student_id])
        for row in rows
        if row.status == "missing" and row.student_id in roster_by_id
    ]
    return acknowledgement, resend, reminder


def ingest_assignment_submissions(
    *,
    assignment_path: Path,
    roster_path: Path,
    submissions_dir: Path,
    output_root: Path = Path("output/submissions"),
    reports_root: Path = Path("reports/submissions"),
    received_at_override: datetime | None = None,
    reset_audit: bool = True,
) -> dict[str, object]:
    _require_private_roots(output_root, reports_root)
    assignment = load_assignment(assignment_path)
    roster = [student for student in load_roster(roster_path) if student.class_id == assignment.class_id and student.active]
    roster_by_id = {student.student_id: student for student in roster}
    student_ids = set(roster_by_id)
    student_id_by_source_file = {
        student.source_file: student.student_id
        for student in roster
        if student.source_file
    }

    assignment_output = output_root / assignment.assignment_id
    accepted_dir = assignment_output / "accepted_pdfs"
    rejected_dir = assignment_output / "rejected_pdfs"
    drafts_dir = assignment_output / "drafts"
    audit_path = assignment_output / "audit.jsonl"
    if reset_audit and audit_path.exists():
        audit_path.unlink()
    audit = AuditLog(audit_path, assignment.assignment_id)
    audit.write("assignment_loaded", status="loaded")
    audit.write("roster_loaded", status="loaded", roster_count=len(roster))

    valid_candidates: dict[str, list[Submission]] = {}
    rejected_submissions: list[Submission] = []

    for source in sorted(path for path in submissions_dir.iterdir() if path.is_file()):
        audit.write("file_seen", source_filename=source.name, status="seen")
        student_id = student_id_by_source_file.get(source.name) or match_student_id_from_filename(source, student_ids, assignment.assignment_id)
        reasons = validate_pdf(source, assignment)
        if student_id is None:
            reasons.append("unknown_student")
        received_at = _received_at(source, received_at_override)
        late = bool(assignment.due_at and received_at > assignment.due_at)
        if late and not assignment.allow_late:
            reasons.append("late_not_allowed")

        if reasons:
            stored = _copy_submission(source, rejected_dir)
            status = "late_rejected" if "late_not_allowed" in reasons else "rejected"
            submission = _make_submission(
                assignment=assignment,
                student_id=student_id or "",
                source=source,
                stored_path=stored,
                received_at=received_at,
                status=status,
                late=late,
                reasons=reasons,
            )
            rejected_submissions.append(submission)
            audit.write("file_rejected", student_id=student_id or "", source_filename=source.name, status=status, reasons=reasons)
            continue

        submission = _make_submission(
            assignment=assignment,
            student_id=student_id or "",
            source=source,
            stored_path="",
            received_at=received_at,
            status="late" if late else "accepted",
            late=late,
            reasons=[],
        )
        valid_candidates.setdefault(submission.student_id, []).append(submission)

    accepted_submissions: list[Submission] = []
    duplicate_student_ids: set[str] = set()
    for student_id, candidates in valid_candidates.items():
        if len(candidates) == 1:
            candidate = candidates[0]
            accepted = Submission(
                submission_id=candidate.submission_id,
                assignment_id=candidate.assignment_id,
                student_id=candidate.student_id,
                source_filename=candidate.source_filename,
                stored_pdf_path=_copy_submission(submissions_dir / candidate.source_filename, accepted_dir),
                sha256=candidate.sha256,
                received_at=candidate.received_at,
                submitted_via=candidate.submitted_via,
                status=candidate.status,
                late=candidate.late,
                validation_reasons=candidate.validation_reasons,
            )
            accepted_submissions.append(accepted)
            audit.write("file_accepted", student_id=student_id, source_filename=accepted.source_filename, status=accepted.status)
            continue

        duplicate_student_ids.add(student_id)
        candidate_winner = _choose_duplicate_winner(candidates)
        winner = Submission(
            submission_id=candidate_winner.submission_id,
            assignment_id=candidate_winner.assignment_id,
            student_id=candidate_winner.student_id,
            source_filename=candidate_winner.source_filename,
            stored_pdf_path=_copy_submission(submissions_dir / candidate_winner.source_filename, accepted_dir),
            sha256=candidate_winner.sha256,
            received_at=candidate_winner.received_at,
            submitted_via=candidate_winner.submitted_via,
            status=candidate_winner.status,
            late=candidate_winner.late,
            validation_reasons=candidate_winner.validation_reasons,
        )
        accepted_submissions.append(winner)
        audit.write("file_accepted", student_id=student_id, source_filename=winner.source_filename, status=winner.status, reasons=["duplicate_winner"])
        for duplicate in candidates:
            if duplicate == candidate_winner:
                continue
            rejected = Submission(
                submission_id=duplicate.submission_id,
                assignment_id=duplicate.assignment_id,
                student_id=duplicate.student_id,
                source_filename=duplicate.source_filename,
                stored_pdf_path=_copy_submission(Path(submissions_dir / duplicate.source_filename), rejected_dir),
                sha256=duplicate.sha256,
                received_at=duplicate.received_at,
                submitted_via=duplicate.submitted_via,
                status="rejected",
                late=duplicate.late,
                validation_reasons=["duplicate_submission"],
            )
            rejected_submissions.append(rejected)
            audit.write(
                "duplicate_detected",
                student_id=student_id,
                source_filename=duplicate.source_filename,
                status="rejected",
                reasons=["duplicate_submission"],
                accepted_source_filename=winner.source_filename,
            )

    accepted_by_student = {submission.student_id: submission for submission in accepted_submissions}
    rejected_by_student: dict[str, list[Submission]] = {}
    for submission in rejected_submissions:
        if submission.student_id:
            rejected_by_student.setdefault(submission.student_id, []).append(submission)

    rows = _completion_rows(assignment, roster, accepted_by_student, rejected_by_student, duplicate_student_ids)
    report_path = reports_root / f"{assignment.assignment_id}_completion.csv"
    write_completion_csv(rows, report_path)
    audit.write("completion_report_written", status="written", report_path=report_path.as_posix())

    acknowledgement, resend, reminder = _build_drafts(assignment, roster_by_id, accepted_submissions, rejected_submissions, rows)
    draft_paths = {
        "acknowledgement": drafts_dir / "acknowledgement_drafts.jsonl",
        "resend": drafts_dir / "resend_drafts.jsonl",
        "reminder": drafts_dir / "reminder_drafts.jsonl",
    }
    write_drafts_jsonl(acknowledgement, draft_paths["acknowledgement"])
    write_drafts_jsonl(resend, draft_paths["resend"])
    write_drafts_jsonl(reminder, draft_paths["reminder"])
    for draft in acknowledgement + resend + reminder:
        audit.write("feedback_draft_created", student_id=draft.student_id, status="draft", draft_type=draft.draft_type, send_allowed=draft.send_allowed)

    manifest = {
        "assignment": dataclass_to_json_dict(assignment),
        "counts": {
            "accepted": len(accepted_submissions),
            "rejected": len(rejected_submissions),
            "missing": sum(1 for row in rows if row.status == "missing"),
            "late": sum(1 for row in rows if row.late),
            "drafts": len(acknowledgement) + len(resend) + len(reminder),
        },
        "completion_report": report_path.as_posix(),
        "audit_log": audit_path.as_posix(),
        "drafts": {key: path.as_posix() for key, path in draft_paths.items()},
        "accepted_submissions": [dataclass_to_json_dict(item) for item in accepted_submissions],
        "rejected_submissions": [dataclass_to_json_dict(item) for item in rejected_submissions],
    }
    manifest_path = assignment_output / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.write("ingest_finished", status="finished", reasons=[])

    return {
        "assignment_id": assignment.assignment_id,
        "manifest": manifest_path,
        "audit_log": audit_path,
        "completion_report": report_path,
        "draft_paths": draft_paths,
        "accepted": accepted_submissions,
        "rejected": rejected_submissions,
        "completion_rows": rows,
    }
