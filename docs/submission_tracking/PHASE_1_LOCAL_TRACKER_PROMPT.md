# 2026-06-22 - Submission Tracking Phase 1 Prompt

Goal:
Implement the local-first assignment submission tracker v1.

Context:
Phase 0 should already have added `docs/SUBMISSION_TRACKING_CONTRACT.md` and confirmed private storage protections. Follow that contract and the roadmap section `Automated Assignment Submission / Grading Assets` in `ROADMAP.md`.

Preferred v1:

- teacher creates or imports assignment metadata
- roster exists as CSV or JSON
- teacher places downloaded student PDFs in a local folder
- system validates PDFs
- system marks each student as submitted, missing, rejected, or late
- system exports a CSV completion report
- system writes a JSONL audit log
- system generates draft acknowledgement, resend, and reminder text
- no automatic email sending
- no OCR
- no automated grading

Hard boundaries:

- Do not add live email intake.
- Do not add Gmail, Outlook, SMTP, or IMAP code.
- Do not send email.
- Do not add OCR services or OCR-based submission grading.
- Do not add automated grading.
- Do not add dependencies.
- Do not touch Asterion exports or canonical question-bank generation.
- Do not create real student data.
- Do not commit real rosters, emails, submissions, grades, or feedback.

Required work:

1. Read before editing:
   - `docs/SUBMISSION_TRACKING_CONTRACT.md`
   - `ROADMAP.md`
   - `.gitignore`
   - existing package and test patterns under `src/exam_bank/`, `scripts/`, and `tests/`
2. Add a small `src/exam_bank/submissions/` package.
3. Add implementation-friendly models first:
   - `Assignment`
   - `Student`
   - `Submission`
   - `SubmissionValidationResult`
   - `CompletionReportRow`
   - `FeedbackDraft`
   - `AuditEvent`
4. Add fake fixtures only:
   - fake assignment JSON
   - fake roster CSV or JSON
   - tiny test PDFs generated inside tests or stored only if clearly synthetic
   - invalid non-PDF fixture
   - no real names or real email domains
5. Add tests before or alongside implementation for:
   - loading fake assignment metadata
   - loading fake roster data
   - validating an accepted PDF
   - rejecting invalid files with reasons
   - matching submissions to students
   - reporting missing students
   - marking late submissions
   - handling duplicate submissions deterministically
   - exporting completion CSV
   - writing audit JSONL
   - generating acknowledgement, resend, and reminder drafts without sending
6. Implement local-folder ingest only.
7. Add thin script or CLI entrypoint only if it follows existing repo patterns.
8. Keep generated private outputs under ignored roots.
9. Do not change existing extraction, OCR, Asterion, topic routing, or auto-grade behavior.

V1 acceptance criteria:

- A fake roster can be loaded.
- A fake assignment can be loaded.
- A local folder of test PDFs can be ingested.
- Invalid files are rejected with reasons.
- Missing students are reported.
- Late submissions are marked.
- Duplicate submissions are handled deterministically.
- A completion CSV is exported.
- An audit JSONL is written.
- Acknowledgement, resend, and reminder drafts are generated but not sent.
- No real emails are sent.
- No real student data is committed.
- Focused tests pass, or unrelated existing failures are documented.

Verification:

- Run focused submission tests.
- Run `git diff --check`.
- Do not run the full extraction/rendering suite unless touched code requires it.
- If the existing repo has unrelated failing tests, document them separately and do not mask them.

Expected final answer:

- files changed
- what v1 supports
- what remains explicitly out of scope
- fixture/privacy confirmation
- verification commands run and results
- suggested next step
- suggested commit message

Suggested commit message:
`feat: add local submission tracker v1`
