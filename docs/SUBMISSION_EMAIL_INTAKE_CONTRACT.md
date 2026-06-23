# Submission Email Intake Contract

This contract defines the Phase 4 readiness boundary for inbound email intake. It adds identity, provenance, quarantine, duplicate/resend, and dry-run rules before any live mailbox integration exists.

## Phase 4 Purpose

Phase 4 is inbound email intake for PDF assignment submissions only. Its job is to inspect already-available inbound message data, identify the assignment and rostered student, preserve message and attachment provenance, quarantine ambiguous or unsafe messages, and prepare data that can later feed the existing local Phase 1 submission flow.

Phase 4 does not replace Phase 1 PDF validation, duplicate handling, late handling, review queues, or draft grading. It should feed those workflows only after intake decisions are explicit and auditable.

## Fixture-Backed Implementation Status

Phase 4 fixture-backed intake is implemented for synthetic local message fixtures only. It does not connect to a live mailbox, does not require email credentials, and does not send outgoing email. Accepted PDF attachments are copied into an ignored private staging folder and then passed through the Phase 1 local submission tracker, which remains responsible for PDF validation, duplicate-file handling, late policy, completion CSVs, and accepted/rejected submission records.

Email intake writes private artifacts under `output/submissions/<assignment_id>/email_intake/`, including message decisions, quarantine records, draft-only acknowledgement/resend records, and provenance linking each staged Phase 1 filename back to the original synthetic message and attachment. Quarantined messages remain teacher-review territory. A live mailbox connector is intentionally deferred.

## Non-Goals

Phase 4 email intake must not:

- send outgoing email
- require live mailbox credentials in tests
- add Gmail, Outlook, SMTP, IMAP, or live mailbox behavior to the test suite
- run OCR
- grade submissions
- create final scores
- create student-facing feedback
- import manual scores
- alter Asterion exports
- alter canonical exam-bank extraction
- commit real messages, real email addresses, real rosters, real PDFs, real grades, or real feedback

## Required Message Provenance

Every inbound message record must preserve:

- `message_id`
- `thread_id`
- `assignment_id`
- `received_at`
- `from_email`
- `from_name`
- `subject`
- `body_preview`
- `attachment_count`
- `attachments`
- `source`
- `status`
- `reasons`
- `created_at`

Message status must be one of `accepted`, `quarantined`, or `rejected`.

## Required Attachment Provenance

Every attachment record must preserve:

- `attachment_id`
- `message_id`
- `filename`
- `content_type`
- `size_bytes`
- `sha256`
- `stored_path`
- `attachment_index`
- `status`
- `reasons`

Attachment status must be one of `accepted`, `quarantined`, or `rejected`.

## Student Matching Policy

Student matching must be deterministic. Preferred order:

1. Student ID in PDF attachment filename.
2. Student ID in subject.
3. Sender email exactly matches a roster email.
4. Student ID in body preview, only as low-confidence fallback.

Exact student ID and exact sender-email matches are high confidence. Body preview matches are low confidence. The intake layer must not guess from display name, fuzzy match names, or silently choose one student when multiple different students match.

If multiple different students match, the message is ambiguous and must be quarantined. If no student matches, the message is unknown and must be quarantined or rejected according to the caller's policy. Filename matching is allowed, but when email metadata is available it is a fallback identity signal and must be preserved alongside message provenance.

Machine-readable matching reasons include:

- `matched_by_attachment_filename`
- `matched_by_subject`
- `matched_by_sender_email`
- `ambiguous_student_match`
- `unknown_student`
- `multiple_student_ids_found`
- `no_roster_email_match`

## Assignment Matching Policy

Email intake must not infer an assignment from vague text. Assignment identity must come from explicit message fixture metadata, a teacher-selected intake run, or a deterministic assignment token that can be audited. If the message assignment does not match the active intake assignment, the message must be quarantined with `assignment_mismatch`.

## Duplicate And Resend Policy

Duplicate and resend decisions must preserve provenance and avoid destructive updates.

- Same `message_id` seen twice: quarantine the second message with `duplicate_message_id`.
- Same attachment hash seen twice: quarantine or mark duplicate with `duplicate_attachment_hash`; do not restage the attachment.
- Same student sends multiple different PDFs: preserve email provenance and defer final winner selection to Phase 1 deterministic duplicate handling.
- Same student resends after an accepted submission: mark `resubmission_detected`; received time must be preserved so Phase 1 policy can choose if explicitly allowed.
- Same student sends a non-PDF after an accepted submission: reject or quarantine the attachment; do not affect an already accepted PDF.
- Late resend: preserve `received_at`; Phase 1 late policy decides whether it is accepted or rejected.

## Quarantine Policy

Quarantine is required when a message or attachment is ambiguous, unsafe, duplicate, or missing required metadata. Supported reasons include:

- `unknown_student`
- `ambiguous_student_match`
- `missing_message_id`
- `duplicate_message_id`
- `missing_received_at`
- `no_pdf_attachment`
- `non_pdf_attachment`
- `too_many_pdf_attachments`
- `attachment_missing_file`
- `attachment_empty`
- `attachment_too_large`
- `unsafe_attachment_path`
- `pdf_validation_failed`
- `assignment_mismatch`
- `duplicate_attachment_hash`
- `resubmission_detected`
- `dry_run_only`

Quarantined records are teacher-facing intake artifacts. They must not be treated as submitted work until the teacher or a later explicit workflow resolves them.

## Dry-Run Policy

Dry-run mode must not:

- copy attachments
- write durable manifests
- call Phase 1 ingestion
- send email

Dry-run mode may:

- parse synthetic message fixture objects
- evaluate student matching
- evaluate attachment metadata
- report whether a PDF would be staged
- report whether a submission would be created
- report whether a draft acknowledgement would be created

Dry-run decisions must include `dry_run_only` and must set `would_stage_pdf=false` and `would_create_submission=false`.

## Draft-Only Message Policy

Email intake may prepare acknowledgement, resend, or quarantine-review draft decisions only. Any draft record or decision must remain non-sending. Phase 4 must not add outgoing mail APIs or live send paths. Controlled outgoing email remains Phase 5.

Outgoing email approval, queueing, dry-run reports, and fake-adapter behavior are handled separately by `docs/SUBMISSION_OUTGOING_EMAIL_CONTRACT.md`. Email intake remains inbound-only.

## Live Connector Boundary

The live email connector is a transport adapter only. It must convert scoped mailbox messages into the same inbound message and attachment models used by fixture-backed intake. Fixture-backed intake remains the test source, real mailbox import defaults to dry-run, and apply mode must reuse this intake path rather than bypassing quarantine, provenance, or Phase 1 validation.

## Email Fixture Privacy Rules

Email fixtures must be synthetic and local. Use reserved domains such as `example.invalid`, fake student IDs, fake names, fake subjects, and generated tiny PDFs or metadata-only attachment records. Fixtures must not include real mailbox exports, real message bodies, real email addresses, real student submissions, real grades, or real feedback.

Live credentials must not appear in tests, docs, fixtures, `.env` examples, or committed configuration.
