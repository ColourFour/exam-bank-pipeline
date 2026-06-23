# Submission Tracking Contract

This contract defines the privacy, storage, and behavior boundaries for the automated assignment submission system. It prepares the repository for a local-first submission tracker without changing canonical exam-bank extraction, Asterion exports, OCR, grading, or email behavior.

## Purpose

The submission tracking system exists to help a teacher reconcile expected assignment submissions against locally available PDF files. Its first useful version should answer:

- which rostered students submitted an acceptable PDF
- which students are missing a submission
- which files were rejected and why
- which submissions are late or duplicated
- what acknowledgement, resend, or reminder text could be drafted for teacher review

The system is not part of canonical CAIE 9709 exam-bank extraction. Student submission data must remain private, local-first, and separated from Asterion/static-site exports.

## Phase Ladder

- Phase 0 - contracts/privacy: define this contract, privacy boundaries, ignored private roots, and fake-fixture rules.
- Phase 1 - local submission tracker: validate teacher-provided assignment metadata, a teacher-provided roster, and a local folder of PDFs; produce completion reports, audit logs, and draft text only.
- Phase 2 - teacher review/grading preparation: add review queues, grading-prep records, and manual teacher notes without sending grade or feedback messages.
- Phase 3 - draft automated grading: experiment with draft marks only, with confidence flags and teacher approval gates.
- Phase 4 - email intake: add controlled inbound email/attachment intake after the local workflow is stable.
- Phase 5 - controlled outgoing email: add a teacher-approved outgoing queue and full outgoing audit log.

Phase 4 is currently implemented as fixture-backed intake only. It reads synthetic local message fixtures, stages accepted PDFs under ignored private submission roots, and reuses Phase 1 for validation, duplicate handling, late policy, completion CSVs, and accepted/rejected records. It has no live mailbox connector, no outgoing email, and no student-facing feedback. Quarantine remains a teacher-review workflow, and live mailbox integration is deferred.

Phase 5A is implemented as a controlled outgoing email queue and approval gate. It normalizes existing draft acknowledgement, resend, reminder, notice, and teacher-approved feedback records; writes an approval template; queues only explicitly approved safe drafts; writes outgoing audit logs; and defaults to dry-run delivery reporting. Phase 5A includes a fake local adapter only, behind an explicit flag. Live sending is deferred. AI draft grades and Phase 3 draft-auto grading feedback cannot be sent as final feedback.

Phase 5B adds a read-only live email connector scaffold. The connector is transport only: it converts scoped mailbox messages into the same internal Phase 4 intake models. Fixture-backed intake remains the test source, real mailbox import defaults to dry-run, and apply mode must reuse Phase 4 and Phase 1 rather than bypassing quarantine or validation. Phase 6 high-confidence automation requires evidence from successful real runs.

## Phase 1 Allowed Scope

Phase 1 may:

- load fake or teacher-provided assignment metadata from local files
- load fake or teacher-provided roster data from local CSV or JSON files
- scan a teacher-selected local folder for PDF files
- validate basic PDF acceptability without OCR or grading
- match submissions to rostered students using explicit metadata or deterministic filename conventions
- mark rostered students as submitted, missing, rejected, late, or duplicate
- export a local completion CSV under an ignored private output root
- write a local JSONL audit log under an ignored private reports root
- generate acknowledgement, resend, and reminder message drafts for teacher review
- use fake fixtures with reserved domains for tests

## Phase 1 Forbidden Scope

Phase 1 must not:

- ingest live email
- send any email or other student-facing message
- call Gmail, Outlook, SMTP, IMAP, or messaging APIs
- run OCR on student submissions
- grade, score, rank, or evaluate student work
- produce student-facing feedback or grades
- add cloud storage, database, queue, or external service integration
- add new dependencies for the submission tracker
- write private data into canonical exam-bank output paths
- write private data into Asterion/static exports
- touch Asterion export behavior
- touch canonical exam-bank extraction behavior
- commit real rosters, real student IDs, real emails, real PDFs, grades, feedback, or submissions

No student-facing email, score, feedback, or grade may be sent automatically in Phase 1. OCR and grading are explicitly out of scope for Phase 1.

## Phase 2 Teacher Review And Grading Preparation

Phase 2 may create teacher-facing review and grading-prep artifacts for accepted local submissions. These artifacts may include review queue records, manual teacher-note templates, review status, review reasons, and manual grading placeholders.

Phase 2 does not grade submissions, does not run OCR, does not email students, and does not produce student-facing feedback. Scores must remain null unless they are manually entered in a future controlled workflow. Teacher review is required before any student-facing feedback, score, or grade can be prepared for delivery.

Rejected, missing, and duplicate-rejected submissions must not receive grading-prep records by default. Accepted duplicate winners may receive a review reason that a duplicate was seen, but still require teacher review.

## Phase 3 Draft Automated Grading

Phase 3 may audit and salvage older grading work only when it is compatible with the submission-tracking privacy boundary and draft-only grading contract. It may create teacher-facing draft automated grading artifacts for Phase 2 accepted submissions, including native PDF extraction results, draft grading records, a summary CSV/JSON, and a teacher review packet.

Phase 3 may attempt safe native PDF text extraction and page counting. It must store only a short text preview, must not run heavy OCR by default, and must not use OCR or generated advisory mark events as scoring authority. It must fail closed with null draft scores when extraction is unavailable, native text is empty, reviewed-rubric mapping is missing, question mapping is missing, or deterministic student-answer mapping is unavailable.

Every Phase 3 draft grading record must include:

```json
{
  "grading_mode": "draft_auto",
  "student_facing": false,
  "teacher_review_required": true
}
```

Phase 3 may not send emails, may not create student-facing feedback, may not finalize grades, may not overwrite teacher-entered notes, and may not destructively overwrite Phase 2 records. Teacher review remains required for every draft result.

## Phase 4 Readiness

Email identity, message provenance, attachment provenance, quarantine, duplicate/resend, and dry-run contracts must be defined before live inbound email intake. Phase 4 must preserve all Phase 0-3 safety rules: private data stays under ignored submission roots, all outgoing messages remain drafts, and no OCR, grading, final scores, or student-facing feedback may be introduced by email intake.

Live mailbox integration is not required for Phase 4 tests. Tests should use synthetic local fixtures and reserved email domains, and filename matching must become a fallback rather than the primary identity source when email metadata is available.

## Phase 5A Controlled Outgoing Queue

Phase 5A may collect previously generated draft-only messages and convert teacher-approved safe drafts into outgoing queue items. Every outgoing email starts as a draft with `send_allowed=false`. Approval must be imported and recorded before a draft becomes send-eligible. Dry-run is the default, and the dry-run report must not send anything.

Allowed Phase 5A message types are acknowledgement, resend request, missing-submission reminder, late notice, teacher-review notice, and generic teacher-approved feedback. Final grades, AI-generated score feedback, and draft-auto grading feedback remain blocked. Phase 3 draft grades are not final grades and must not become student-facing feedback.

Phase 5A includes a fake adapter only. It writes local JSONL under ignored output roots and requires an explicit flag. Live sender adapters remain future work and must require a separate contract update, focused tests, explicit flags, and non-test credentials outside the test suite.

## Phase 5B Live Email Connector

Phase 5B may read a configured assignment label, folder, or search query and convert messages to existing inbound email models. It must not scan the whole inbox by default, store credentials in the repo, stage attachments in dry-run, call Phase 1 in dry-run, queue outgoing email, or send messages.

Apply mode requires an explicit `--apply` flag and must hand off through Phase 4 fixture-compatible intake, preserving quarantine and provenance. Outgoing email remains approval-gated by Phase 5A.

## Fail-Closed Rules

Submission tracking must fail closed when required information is missing, ambiguous, private-data unsafe, or outside the allowed phase.

- Unknown student identity: mark the file rejected or review-required; do not guess.
- Ambiguous duplicate: preserve the evidence, choose a deterministic status, and require teacher review.
- Invalid PDF: reject with a reason and do not treat as submitted.
- Missing assignment metadata: stop before writing completion reports.
- Missing roster: stop before matching submissions.
- Unsafe output path: refuse to write outside approved ignored roots.
- Any request to send, grade, OCR, or publish in Phase 1: refuse and record the blocked operation if an audit log exists.
- Any weak Phase 3 extraction, rubric, question, or answer mapping: create a teacher-review-required draft record with no score.
- Any private data detected in fixtures or committed paths: treat as a release blocker.

## Data Storage Rules

Private or generated submission data must live only under ignored local roots:

- `data/submissions/`
- `output/submissions/`
- `reports/submissions/`

Phase 1 should use these roots as follows:

- `data/submissions/`: local teacher inputs such as private rosters, assignment metadata, and copied or downloaded PDFs when the teacher chooses to place them in the repo workspace.
- `output/submissions/`: generated completion CSVs, validation summaries, matched-submission indexes, and draft-message files.
- `reports/submissions/`: audit JSONL, run summaries, and validation diagnostics.

Committed files under these roots are prohibited except safe placeholders such as `.gitkeep` if the repository uses them. Absolute local paths, student emails, student IDs, grades, feedback, and submission contents must not appear in committed docs, JSON, CSV, fixtures, or tests.

## Audit Log Requirements

Phase 1 audit logs must be local and append-friendly, preferably JSONL under `reports/submissions/`. Each event should include:

- timestamp
- run ID
- assignment ID
- event type
- input path or path hash when a raw path would expose private details
- roster/student reference using a local pseudonymous or teacher-provided ID
- status before and after where applicable
- validation reason for rejected files
- draft-message type when draft text is produced
- `sent=false` for all message-related events in Phase 1

Audit logs must not include raw PDF text, grades, feedback, or live email message bodies.

## Draft-Message Rules

Phase 1 may generate draft acknowledgement, resend, and reminder text only.

- Drafts must be labeled as drafts.
- Draft records must include `send_allowed=false`.
- Drafts must not be sent automatically.
- Drafts must not include scores, grades, or evaluative feedback.
- Drafts must be written only under ignored private roots.
- Draft text must require teacher review before any future controlled sending phase.

## Student-Data Privacy Rules

Real student PDFs, rosters, emails, grades, feedback, and submissions are private data. They must never be committed. They must not be copied into docs, tests, fixtures, Asterion exports, static-site assets, canonical question-bank JSON, topic packets, or public reports.

Local-first is the default. Any future cloud storage, database, email, or outgoing-message integration requires an explicit contract update, tests, and teacher-approval flow.

## Fixture Rules

Fixtures and tests must use only synthetic data:

- fake student names
- fake student IDs
- reserved email domains such as `example.invalid`
- tiny synthetic PDFs generated by tests or clearly fake committed fixtures
- fake assignments with no real class, school, or student data

Fixtures must not include real rosters, real email addresses, real submissions, real grades, teacher feedback, screenshots of student work, or private filenames copied from a real class.

## Asterion And Canonical Extraction Boundary

Submission tracking is a separate private workflow. It must not alter:

- `output/json/question_bank.json`
- canonical question or mark-scheme image generation
- Asterion catalog or runtime exports
- topic-routing sidecars
- OCR behavior
- auto-grading contracts

Asterion/static exports must not contain private student submission data.
