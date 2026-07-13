# Phase 0-3 Submission Readiness Audit Before Email Intake

Date: 2026-06-22

Scope: audit completed submission-tracking and draft-grading work from Phases 0-3 before Phase 4 email intake. Phase 4 was not implemented in this pass.

## Executive Summary

Final recommendation: GO WITH CAUTION.

The local submission workflow is clean enough to use as the base for the next phase, provided Phase 4 starts by adding email intake models and quarantine/dry-run behavior rather than live sending, grading, OCR, or final scoring. The audited implementation keeps private submission artifacts under ignored local roots, uses synthetic fixtures, produces non-sending draft messages, and fails closed for Phase 3 draft grading.

The main caution is that the current identity and provenance model is intentionally Phase 1-local: student identity comes from deterministic filename matching, and there is no inbound message, attachment, thread, or message-id model yet. Those are Phase 4 blockers, not regressions in Phases 0-3.

## Phase 0 Findings

Files checked:

- `docs/SUBMISSION_TRACKING_CONTRACT.md`
- `docs/SUBMISSION_PRIVACY_BOUNDARIES.md`
- `.gitignore`
- `tests/test_repo_hygiene.py`

Findings:

- Private roots are documented and ignored: `data/submissions/`, `output/submissions/`, and `reports/submissions/`.
- `.gitignore` ignores private submission roots while allowing only `.gitkeep` placeholders.
- Repo hygiene tests assert that private rosters, PDFs, completion reports, draft messages, and audit logs under submission roots are ignored.
- Contracts forbid committing real rosters, real PDFs, live emails, grades, feedback, and real student identifiers in tracked fixtures.
- Draft message rules require `send_allowed=false` and teacher review before any future sending phase.
- Submission tracking is explicitly separated from canonical exam-bank extraction and Asterion/static exports.
- Phase 4 is framed as inbound email intake only, with outgoing email deferred to Phase 5.

Assessment: PASS. Phase 4 should preserve these rules and add tests for message fixtures, provenance, quarantine, and dry-run behavior.

## Phase 1 Findings

Files checked:

- `src/exam_bank/submissions/ingest.py`
- `src/exam_bank/submissions/validation.py`
- `src/exam_bank/submissions/feedback_drafts.py`
- `src/exam_bank/submissions/cli.py`
- `scripts/ingest_assignment_submissions.py`
- `tests/test_submission_ingest.py`
- `tests/test_submission_pdf_validation.py`

Findings:

- Assignment loading and roster loading are local file based.
- Roster loading requires `student_id`, `class_id`, `display_name`, `email`, and `active`.
- Matching is deterministic but filename-based. It checks exact stem, underscore-separated tokens, assignment-prefixed filenames, and student-id prefix/suffix.
- Unknown students are rejected with `unknown_student`.
- PDF validation rejects non-PDF files, empty files, oversized files, encrypted PDFs, no-page PDFs, and files PyMuPDF cannot open.
- Duplicate handling is deterministic. The latest `received_at`, then filename order, wins; rejected duplicates receive `duplicate_submission`.
- Late submissions are marked, or rejected with `late_not_allowed` when the assignment disallows late work.
- Completion CSV, manifest JSON, copied accepted/rejected files, draft JSONL files, and audit JSONL are written under required private roots.
- Root checks reject output roots that do not end in `output/submissions` or `reports/submissions`.
- Draft acknowledgement, resend, and reminder records all set `send_allowed=false`.
- Phase 1 source and tests show no email API, OCR, or grading behavior.

Assessment: PASS for local tracker scope. Phase 4 must not rely only on filename matching once email metadata is available.

## Phase 2 Findings

Files checked:

- `src/exam_bank/submissions/review_queue.py`
- `src/exam_bank/submissions/grading_packets.py`
- `src/exam_bank/submissions/review_cli.py`
- `scripts/build_submission_review_queue.py`
- `tests/test_submission_review_queue.py`
- `tests/test_submission_grading_prep.py`

Findings:

- Review queue generation reads the Phase 1 manifest and creates records only for accepted submissions.
- Each accepted submission receives a teacher review record and a manual grading-prep placeholder.
- Rejected and missing submissions do not receive grading-prep records by default.
- Manual grading prep uses `grading_mode=manual_placeholder`, `status=not_started`, `score=null`, `max_score=null`, and `review_required=true`.
- Review records start in `needs_review`.
- Review CSV, review summary JSON, grading-prep JSON, and teacher notes template are written under ignored submission output/report roots.
- No student-facing output is created.
- Phase 2 appends audit events and does not destructively overwrite Phase 1 manifest data.

Assessment: PASS.

## Phase 3 Findings

Files checked:

- `src/exam_bank/submissions/extraction.py`
- `src/exam_bank/submissions/draft_grading.py`
- `src/exam_bank/submissions/draft_grading_cli.py`
- `src/exam_bank/submissions/grading_packets.py`
- `scripts/build_submission_draft_grades.py`
- `tests/test_submission_extraction.py`
- `tests/test_submission_draft_grading.py`

Findings:

- Native PDF extraction stores page count, extractability, warnings, and a short text preview only.
- Blank or weak native extraction fails closed with `status=partial` or `status=failed`.
- Draft grading outputs are validated to use `grading_mode=draft_auto`, `student_facing=false`, and `teacher_review_required=true`.
- `DraftGradingSummary` rejects nonzero `student_facing_count`.
- Missing reviewed rubrics, missing question mappings, missing native text, and missing deterministic student-answer mapping all keep `draft_score=null`.
- Advisory mark events are audited as context but do not authorize scoring.
- Draft maximum score may be populated from approved rubrics, but draft score remains null without deterministic student-answer mapping.
- Teacher grading review packets are teacher-facing and explicitly state no student-facing feedback or final grades are produced.
- Phase 3 appends audit events and does not modify Phase 2 review queue contents in the tested path.
- Phase 3 source and tests show no email sending, OCR, final score, or final grade behavior.

Assessment: PASS.

## End-to-End Fake Workflow Result

Fixture inputs:

- Assignment: `tests/fixtures/submissions/assignment_p3_vectors_hw1.json`
- Roster: `tests/fixtures/submissions/roster_class_12a.csv`
- Inbox: `tests/fixtures/submissions/inbox`
- Received-at override: `2026-06-29T10:00:00+08:00`
- Run root: `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/`

Commands run:

```bash
.venv/bin/python scripts/ingest_assignment_submissions.py --assignment tests/fixtures/submissions/assignment_p3_vectors_hw1.json --roster tests/fixtures/submissions/roster_class_12a.csv --submissions-dir tests/fixtures/submissions/inbox --output-root .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions --reports-root .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/reports/submissions --received-at 2026-06-29T10:00:00+08:00
.venv/bin/python scripts/build_submission_review_queue.py --assignment-id p3_vectors_hw1 --submission-output-root .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions --reports-root .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/reports/submissions
.venv/bin/python scripts/build_submission_draft_grades.py --assignment-id p3_vectors_hw1 --submission-output-root .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions --reports-root .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/reports/submissions --reviewed-rubrics-path .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/missing_reviewed_rubrics.json --mark-events-path .agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/missing_mark_events.json
```

Observed counts:

- Accepted submissions: 2
- Rejected submissions: 3
- Missing students: 1
- Late submissions: 0
- Review records: 2
- Grading-prep records: 2
- Draft grading records: 2
- Draft scores assigned: 0
- Student-facing records: 0
- Teacher-review-required draft records: 2

Completion statuses:

- `S0001`: submitted
- `S0002`: submitted
- `S0003`: missing

Output paths created:

- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/manifest.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/audit.jsonl`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/accepted_pdfs/S0001.pdf`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/accepted_pdfs/S0002_vectors_hw1.pdf`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/rejected_pdfs/S0002_bad_upload.txt`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/rejected_pdfs/not_a_pdf.txt`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/rejected_pdfs/unknown_student.pdf`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/drafts/acknowledgement_drafts.jsonl`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/drafts/resend_drafts.jsonl`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/drafts/reminder_drafts.jsonl`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/reports/submissions/p3_vectors_hw1_completion.csv`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/review/review_queue.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/review/grading_prep.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/review/review_summary.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/review/teacher_review_notes_template.csv`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/reports/submissions/p3_vectors_hw1_review_queue.csv`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/draft_grading/extraction_results.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/draft_grading/draft_grading_results.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/draft_grading/draft_grading_summary.json`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/output/submissions/p3_vectors_hw1/draft_grading/teacher_grading_review_packet.md`
- `.agent-runs/submission-phase-0-3-readiness-audit-2026-06-22/reports/submissions/p3_vectors_hw1_draft_grading_summary.csv`

## Phase 4 Blocker List

| Item | Classification | Notes |
| --- | --- | --- |
| Stable student identity model for email intake | blocker | Current identity is filename-derived. Phase 4 needs sender/message/roster matching with deterministic ambiguity handling. |
| Inbound email message model | blocker | Need local schema for sender, subject, received time, message ID, thread ID, body handling policy, and source connector/export metadata. |
| Attachment provenance model | blocker | Saved PDFs must retain message ID, attachment filename, attachment index, content hash, and quarantine/acceptance status. |
| Email duplicate/resend policy | blocker | Current duplicate logic is file-level only. Email intake needs policy for repeated messages, changed attachments, same attachment hash, and late resends. |
| Quarantine for ambiguous email submissions | blocker | Unknown sender, multiple possible students, missing PDF, multiple PDFs when not allowed, and unsafe message metadata should quarantine instead of entering Phase 1 as accepted. |
| Dry-run mode for email intake | blocker | Phase 4 should prove parsing, attachment saving, rejection, and draft creation without mutating durable private roots unless explicitly requested. |
| Message/thread ID tracking | should_fix_before_phase4 | This is closely tied to duplicate behavior and auditability. It can be part of the blocker work above. |
| Weak filename matching | should_fix_before_phase4 | Keep as fallback only. Prefer roster email/student mapping plus explicit attachment provenance. |
| Safe draft acknowledgement behavior for email intake | should_fix_before_phase4 | Existing draft records are safe, but Phase 4 should add email-specific draft types and tests confirming no send path exists. |
| Privacy gaps | can_defer | No Phase 0-3 privacy gap was found. Phase 4 must add fixture hygiene tests for synthetic email addresses/messages. |
| Output path accidentally becoming tracked | can_defer | Existing private roots and `.agent-runs/` are ignored. Add narrow allowlists only for intentional audit reports. |

## Recommended Phase 4 Scope

Phase 4 should be email intake only.

Include:

- Read inbound messages from local synthetic fixtures first; add a mailbox connector only if repo patterns and tests can keep credentials out of the suite.
- Parse sender, subject, received time, message ID, optional thread ID, and attachments.
- Save accepted PDF attachments into a private intake area under `data/submissions/` or a generated area under `output/submissions/`.
- Preserve attachment provenance through Phase 1 ingestion.
- Quarantine bad or ambiguous messages with machine-readable reasons.
- Produce acknowledgement/resend/reminder drafts only, with `send_allowed=false`.
- Update audit logs with message and attachment provenance.
- Keep tests credential-free and fixture-only.

Exclude:

- Outgoing email sending.
- SMTP, Gmail, Outlook, IMAP, or live mailbox behavior in tests.
- OCR.
- Automated scoring or final grades.
- Student-facing feedback.
- Manual scoring import.
- Real messages, real addresses, real rosters, or real PDFs.

## Tests And Verification

Targeted tests:

```bash
.venv/bin/python -m pytest -q tests/test_repo_hygiene.py tests/test_submission_models.py tests/test_submission_pdf_validation.py tests/test_submission_ingest.py tests/test_submission_review_queue.py tests/test_submission_grading_prep.py tests/test_submission_extraction.py tests/test_submission_draft_grading.py
```

Result: 42 passed, 5 PyMuPDF/SWIG deprecation warnings.

Fake end-to-end CLI workflow: passed. Counts are recorded above.

Additional verification:

```bash
git diff --check
```

Result: passed.

## Final Recommendation

GO WITH CAUTION.

Begin Phase 4 only as a local, fixture-backed inbound email intake layer with explicit message/attachment provenance, quarantine, and dry-run behavior. Do not add live sending, OCR, grading, final scoring, manual score import, or student-facing feedback.

Recommended exact next task: implement Phase 4 email-intake contracts and synthetic fixture tests for message parsing, attachment provenance, quarantine reasons, dry-run output, and non-sending draft acknowledgements.
