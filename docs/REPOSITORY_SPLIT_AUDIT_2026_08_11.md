# Repository Split Audit — 2026-08-11

## Outcome

The former mixed repository is now three local Python repositories connected by
versioned JSON files and relative asset references:

| Repository | Responsibility | Produced boundary |
| --- | --- | --- |
| `exam-bank-pipeline` | PDF ingestion, question/mark-scheme extraction, normalization, topic/review sidecars, integrity and release tooling | Question v1 export |
| `../homework-ingest` | Classroom workspaces, email/filesystem intake, identity and assignment matching, submission status, normalization and teacher handoff | Submission v1 |
| `../autograder` | Source-neutral rubric scoring, B/M/A dependency policy, provider adapters, confidence/review flags and grading evaluations | GradeResult v1 |

There is no shared runtime package or network service. The repositories preserve
opaque identifiers and exchange ordinary JSON plus referenced files.

## Pre-deletion verification

Deletion from exam-bank began only after both sibling builds confirmed their
equivalents:

- Homework-ingest contains all intake-owned `submissions` modules, `emailing`,
  `classroom.py`, the dashboard, dedicated tests, synthetic fixtures, template,
  active contracts and submission history documents. Its former grading-prep
  flow is intentionally replaced by a normalized, review-gated handoff.
- Autograder contains equivalents for all eleven former `auto_grade` package
  files under `autograder.readiness`, all seven rubric-review scripts, their
  dedicated tests and docs, and isolated normalized-JSON compatibility adapters
  for the old BMA/draft result shapes.

Byte equality and recorded SHA-256 values were used for the frozen schemas:

| Contract | Producer namespace | SHA-256 |
| --- | --- | --- |
| `question.v1.schema.json` | `exam_bank.interchange.question` | `dfb5fcba79a13e12609e23945d8da09e79def153b848b7df3d9cb517e4bf7c5f` |
| `question-export.v1.schema.json` | `exam_bank.interchange.questions` | `ae835c24150fa5cc35c370959b9507d8431ac51f33d5d5377945ac56b032191d` |
| `submission.v1.schema.json` | `homework_ingest.interchange.submission` | `2cde34c8fc090a58a8952d55bf3903715831a1789025515018c1d593eef35f0a` |
| `grade-result.v1.schema.json` | `autograder.interchange.grade_result` | `851a94a84527c5dc475b3da942e9e21edae74eafa4f6ead3ee3406aef4619d0f` |

## Exam-bank deletions

The cleanup removes these subsystem-owned paths from exam-bank:

- `src/exam_bank/submissions/` (29 tracked files plus the untracked
  `email_models.py`, `email_reasons.py`, and `paths.py`)
- `src/exam_bank/emailing/`
- `src/exam_bank/classroom.py`
- `src/exam_bank/classroom_dashboard/`
- `src/exam_bank/auto_grade/`
- seven rubric-review scripts:
  `build_auto_grade_reviewer_packet.py`,
  `build_auto_grade_rubric_review_batch.py`,
  `build_auto_grade_rubric_review_queue.py`,
  `check_auto_grade_rubric_review_completion.py`,
  `extract_auto_grade_review_approval_template.py`,
  `promote_auto_grade_reviewed_rubrics.py`, and
  `validate_auto_grade_reviewed_rubrics.py`
- 33 dedicated `test_auto_grade_*`, `test_submission_*`, BMA, quiz-packet,
  classroom, dashboard, email, Mail.app and Outlook tests
- 19 files under `tests/fixtures/submissions/` and
  `templates/submissions/email_connector_config.template.json`
- 14 active/exclusive-history documents: the `AUTO_GRADING_*` and
  `SUBMISSION_*` contracts, the `docs/auto_grade/`
  guide, `docs/submission_tracking/`, and the three submission-only history
  audits dated 2026-06-22

In total this is 126 tracked deletions plus three untracked subsystem source
files. Of the tracked deletions, 52 are subsystem source/assets, seven are
scripts, 33 are tests, 19 are synthetic fixtures, 14 are docs, and one is a
configuration template.

A 4.3 MB untracked top-level `build/` cache still contained stale copies of the
moved packages. It was confirmed as generated wheel-build output and removed.
Root-anchored `/build/` and `/dist/` ignore rules now prevent that cache from
reappearing as visible repository content.

The command registry and legacy flat parser no longer expose classroom, email,
submission, quiz-packet, BMA or autograde commands. The package no longer ships
dashboard static assets.

The following mixed exam-bank couplings were also removed:

- the dedicated `auto_grade_eligible` release role and release example;
- auto-grade eligibility from protected storage-audit references;
- the auto-grade input/snapshot from the topic-routing baseline audit;
- the `auto_grade_eligibility_changed` field from Asterion release packaging.

The generic release-manifest implementation remains capable of binding an
arbitrary external JSON artifact when a caller explicitly chooses to do so; it
has no autograder-specific role or default.

### Preserved local pre-split release evidence

`manifests/releases/question_bank_release_manifest.v1.json` was subsequently
regenerated as exam-bank-only evidence. It binds the canonical bank, active
corpus manifest, and durable topic-routing sidecar without autograder roles.

The untracked local file
`manifests/releases/asterion_export_release_manifest.v1.json` remains preserved
pre-split evidence. It retains the retired `auto_grade_eligibility_changed`
field and its declared durable-sidecar hash no longer matches the current
sidecar. It is not authoritative tracked source and is not used as a current
default. Archive it or regenerate it explicitly after selecting current release
artifacts; silently rewriting it would discard local release evidence or claim
a new release without operator review.

## Retained boundaries and data safety

Exam-bank retains the authoritative Question schema, compatibility copies of
Submission and GradeResult schemas, the Question-export envelope, and the
`exam-bank data export-questions` / `validate-questions` commands. The packaged
Question schema is byte-identical to the top-level authoritative file so an
installed wheel does not depend on a source checkout.

Small helpers were duplicated into siblings instead of creating a fourth
repository. In particular, atomic JSON writing and the small MuPDF validation
helper are local to their owning projects. Paper/session parsing remains an
exam-bank concern; consumers treat `question_id` as opaque.

The cleanup deliberately does not read, move or delete these ignored private
or generated paths:

- `data/classes/`
- `data/submissions/`
- `output/submissions/`
- `reports/submissions/`
- `reports/email/`
- `.env`

Their legacy ignore rules remain with comments so an existing checkout cannot
accidentally expose pre-split private data. New homework runs belong in the
homework-ingest checkout; any manual migration of old private state is an
operator-controlled task outside this source refactor.

## Remaining interaction

1. Exam-bank exports a Question envelope and relative question/mark-scheme asset
   paths.
2. Homework-ingest optionally validates that export, normalizes private work to
   Submission records, and creates only review-gated autograder requests.
3. Autograder consumes Question, reviewed rubric and normalized Submission
   answer data, then writes GradeResult. It never imports email, PDF extraction,
   classroom or exam-bank layout code.

Missing reviewed rubrics or explicit answer mappings fail closed. Advisory
mark-event extraction is not grading authority.

## Historical documents

Mixed dated audits under `docs/history/` remain evidence of the pre-split state.
`docs/PROJECT_AUDIT_2026_06_30.md` carries an explicit historical banner. These
documents are not current command or ownership documentation.
