# Submission Privacy Boundaries

This document states the privacy boundary for assignment submissions before any local tracker implementation exists.

## Never Commit Real Student Data

The following data must never be committed:

- real student PDFs
- real rosters
- real student names paired with class or submission data
- real student IDs
- real student email addresses
- real grades
- real teacher feedback
- real submission audit logs
- real draft or sent student messages

Do not paste private student data into docs, tests, fixtures, JSON, CSV, Markdown, screenshots, reports, or examples.

## Fixture Requirements

Submission fixtures must be synthetic. Use:

- fake names
- fake IDs
- fake assignments
- reserved email domains such as `example.invalid`
- generated tiny PDFs or obviously synthetic PDF fixtures

Fixtures must not be adapted from a real roster, real mailbox, real feedback file, or real student submission.

## Ignored Private Roots

Private generated data should live only under ignored roots:

- `data/submissions/`
- `output/submissions/`
- `reports/submissions/`

These roots are for local teacher inputs, generated completion reports, validation summaries, draft messages, and audit logs. They are not durable public artifacts and must not be used by canonical exam-bank extraction or static exports.

## Asterion And Static Export Boundary

Asterion/static exports must not contain private student submission data. Submission records, rosters, student emails, draft messages, grades, and feedback must remain outside:

- canonical question-bank JSON
- canonical image assets
- Asterion catalog exports
- Asterion student-runtime exports
- Content Lab candidate exports
- topic packets
- public reports

## Message Boundary

Outgoing messages must remain drafts until a future controlled sending phase explicitly changes the contract. Drafts must require teacher review, must not be sent automatically, and must not include scores, grades, or evaluative feedback in Phase 1.

## Local-First Default

Local-first is the default until explicitly changed. Do not add cloud storage, databases, email intake, outgoing email, or messaging integrations without a future contract update and focused tests.
