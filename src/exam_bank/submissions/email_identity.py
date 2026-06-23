from __future__ import annotations

import re
from pathlib import Path

import exam_bank.submissions.email_reasons as reasons
from exam_bank.submissions.email_models import InboundEmailMessage, StudentMatchCandidate, StudentMatchResult
from exam_bank.submissions.ingest import match_student_id_from_filename
from exam_bank.submissions.models import Student


def match_student_for_email(message: InboundEmailMessage, roster: list[Student]) -> StudentMatchResult:
    """Match a synthetic inbound email to a roster entry without guessing names."""

    active_roster = [student for student in roster if student.active]
    student_ids = {student.student_id for student in active_roster}
    roster_email_to_student_id = {
        student.email.strip().lower(): student.student_id
        for student in active_roster
        if student.email.strip()
    }

    candidates: list[StudentMatchCandidate] = []
    match_reasons: list[str] = []

    for attachment in message.attachments:
        student_id = match_student_id_from_filename(Path(attachment.filename), student_ids, message.assignment_id)
        if student_id:
            candidates.append(
                StudentMatchCandidate(
                    student_id=student_id,
                    match_source="attachment_filename_student_id",
                    confidence="high",
                    evidence=[attachment.filename],
                )
            )
            match_reasons.append(reasons.MATCHED_BY_ATTACHMENT_FILENAME)

    subject_matches = _student_ids_in_text(message.subject, student_ids)
    for student_id in subject_matches:
        candidates.append(
            StudentMatchCandidate(
                student_id=student_id,
                match_source="subject_student_id",
                confidence="high",
                evidence=[message.subject],
            )
        )
        match_reasons.append(reasons.MATCHED_BY_SUBJECT)

    sender_student_id = roster_email_to_student_id.get(message.from_email.strip().lower())
    if sender_student_id:
        candidates.append(
            StudentMatchCandidate(
                student_id=sender_student_id,
                match_source="sender_email",
                confidence="high",
                evidence=[message.from_email],
            )
        )
        match_reasons.append(reasons.MATCHED_BY_SENDER_EMAIL)

    body_matches = _student_ids_in_text(message.body_preview, student_ids)
    for student_id in body_matches:
        candidates.append(
            StudentMatchCandidate(
                student_id=student_id,
                match_source="body_student_id",
                confidence="low",
                evidence=[message.body_preview],
            )
        )
        match_reasons.append(reasons.MATCHED_BY_BODY)

    distinct_student_ids = {candidate.student_id for candidate in candidates}
    if len(distinct_student_ids) == 1:
        return StudentMatchResult(
            status="matched",
            student_id=next(iter(distinct_student_ids)),
            candidates=_dedupe_candidates(candidates),
            reasons=_dedupe(match_reasons),
        )
    if len(distinct_student_ids) > 1:
        return StudentMatchResult(
            status="ambiguous",
            student_id="",
            candidates=_dedupe_candidates(candidates),
            reasons=_dedupe([*match_reasons, reasons.AMBIGUOUS_STUDENT_MATCH, reasons.MULTIPLE_STUDENT_IDS_FOUND]),
        )
    return StudentMatchResult(
        status="unknown",
        student_id="",
        candidates=[],
        reasons=[reasons.UNKNOWN_STUDENT, reasons.NO_ROSTER_EMAIL_MATCH],
    )


def _student_ids_in_text(text: str, student_ids: set[str]) -> list[str]:
    found: list[str] = []
    for student_id in sorted(student_ids):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(student_id)}(?![A-Za-z0-9])"
        if re.search(pattern, text or ""):
            found.append(student_id)
    return found


def _dedupe_candidates(candidates: list[StudentMatchCandidate]) -> list[StudentMatchCandidate]:
    seen: set[tuple[str, str, str]] = set()
    result: list[StudentMatchCandidate] = []
    for candidate in candidates:
        key = (candidate.student_id, candidate.match_source, candidate.confidence)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
