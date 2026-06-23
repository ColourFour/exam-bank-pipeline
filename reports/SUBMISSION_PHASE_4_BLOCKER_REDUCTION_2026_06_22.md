# Phase 4 Blocker Reduction

Date: 2026-06-22

Scope: contract and scaffold hardening before implementing inbound email intake. No full Phase 4 intake, live mailbox connector, attachment staging, OCR, grading, scoring, sending, or student-facing feedback was implemented.

## Blockers Addressed

- Stable student identity model: added deterministic student match result and candidate models, plus matching helpers for attachment filename, subject, sender email, and low-confidence body-preview fallback.
- Inbound email message model: added `InboundEmailMessage` with required message provenance fields and allowed statuses.
- Attachment provenance model: added `InboundEmailAttachment` and `EmailSubmissionProvenance`.
- Duplicate/resend policy: added pure duplicate/resend reason helper covering duplicate message IDs, duplicate attachment hashes, and resubmission detection.
- Quarantine reasons: added shared machine-readable reason constants.
- Dry-run behavior: added dry-run decision model and helper that reports what would happen without staging PDFs, creating submissions, writing durable manifests, or sending email.
- Filename matching fallback rule: documented that filename matching remains allowed but must be fallback/provenance-preserving when email metadata is available.
- Draft-only behavior: dry-run decisions only report `would_create_draft`; no send path was added.

## Blockers Remaining

- Full inbound email intake is still not implemented.
- No message fixture parser or CLI exists yet.
- No attachment staging into Phase 1 exists yet.
- No durable email intake manifest or audit writer exists yet.
- No teacher quarantine review workflow exists yet.
- No live mailbox connector exists, by design.

## Phase 4 Readiness

Phase 4 proper can now begin as a fixture-backed inbound email intake implementation. The next pass should parse synthetic message fixtures, run the new identity and dry-run policy helpers, write private quarantine/intake artifacts under ignored submission roots, and preserve provenance when a PDF is later staged for Phase 1.

Phase 4 must still exclude outgoing email, live credentials in tests, OCR, grading, final scores, manual score import, and student-facing feedback.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_repo_hygiene.py tests/test_submission_models.py tests/test_submission_email_models.py tests/test_submission_email_identity.py tests/test_submission_email_policy.py
```

Result: 31 passed, 5 PyMuPDF/SWIG deprecation warnings.

```bash
git diff --check
```

Result: passed.
