# Submission Outgoing Email Contract

This contract defines Phase 5A: the controlled outgoing email queue for assignment-submission workflows.

## Purpose

Phase 5A collects draft-only acknowledgement, resend, reminder, notice, and teacher-approved feedback messages created by earlier submission phases. It normalizes those drafts into a single outgoing draft format, creates a teacher approval template, builds a queue only from explicitly approved safe drafts, records audit events, and supports dry-run delivery reporting.

Phase 5A does not send real email by default. It does not require live credentials, does not connect to live mail services, and does not change canonical exam-bank extraction, OCR, grading, or Asterion export behavior.

## Allowed Message Types

These message types may be queued only after explicit teacher approval:

- `submission_acknowledgement`
- `resend_request`
- `missing_submission_reminder`
- `late_submission_notice`
- `teacher_review_notice`
- `generic_teacher_approved_feedback`

Every outgoing message starts as an `OutgoingEmailDraft` with `send_allowed=false` and `approval_status=draft` unless policy blocks it earlier.

## Blocked Message Types

These message types must not be queued in Phase 5A unless a future contract explicitly allows them:

- `final_grade`
- `ai_generated_score_feedback`
- `draft_auto_grading_feedback`

AI draft grades from Phase 3 cannot be sent as final feedback. Phase 3 draft-auto grading records remain teacher-facing only. A generic teacher-approved feedback message may be queued only when it is not AI score feedback, not a final grade, and has explicit approval.

## Approval Requirements

The approval CSV template is:

```text
draft_id,assignment_id,student_id,recipient_email,message_type,subject,approval_status,approved_by,teacher_note
```

Only rows with `approval_status=approved` may become queue items. Unknown `draft_id` values are rejected. The CSV recipient email must match the original draft recipient email. Phase 5A does not add recipient override behavior.

Approval changes a safe draft to `send_allowed=true`, records `approved_by` and `approved_at`, and creates an outgoing queue item. Unapproved, unknown, mismatched, unsupported, and blocked drafts remain blocked.

## Dry-Run Default

Dry-run is the default delivery mode. Building the queue writes artifacts and audit records only. The dry-run report reads the queue and writes:

```text
reports/submissions/<assignment_id>_outgoing_email_dry_run.csv
```

Dry-run must not send email, open network connections, require credentials, or mutate student-facing state.

## Sender Adapter Rules

Phase 5A includes a fake adapter only. It may run only with an explicit `--use-fake-adapter` flag and writes local JSONL records under:

```text
output/submissions/<assignment_id>/outgoing_email/fake_sent_messages.jsonl
```

The fake adapter must not use credentials or network requests. Live sending is deferred. Any future live sender adapter must be disabled by default, require an explicit command-line flag, use a non-test adapter, and preserve all approval and audit gates.

The Phase 5B live email connector is inbound transport only. It does not grant permission to send outgoing email, auto-send acknowledgements, or bypass this approval gate.

## Audit Log Requirements

Outgoing audit logs are JSONL under:

```text
output/submissions/<assignment_id>/outgoing_email/outgoing_email_audit.jsonl
```

Each event records:

- `timestamp`
- `event_type`
- `assignment_id`
- `draft_id`
- `queue_id`
- `student_id`
- `recipient_email`
- `status`
- `reasons`

Audit events must cover draft normalization, approval-template writing, approval rejection, draft blocking, queue creation, and summary writing. Audit logs must not include credentials, raw student submissions, or final grades.

## Privacy Rules

All outgoing email artifacts are private generated outputs and must remain under ignored submission roots:

- `output/submissions/`
- `reports/submissions/`

Do not commit real student emails, message bodies, feedback, grades, PDFs, rosters, or delivery reports. Tests and fixtures must use fake data and reserved domains such as `example.invalid`.

## Fixture And Test Rules

Tests must use fake data only. Tests must not require live credentials, live mailboxes, network access, or real student data. Tests may use the fake adapter because it writes local JSONL only.

No test may send real email. No test may depend on a live sender implementation.
