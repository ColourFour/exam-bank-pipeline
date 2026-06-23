# First Assignment Runbook

Use this for the first real assignment run. The goal is evidence collection, not automation.

## Prepare Inputs

Create private files under `data/submissions/<assignment_id>/`:

- `assignment.json`: assignment metadata matching the Phase 1 fixture shape.
- `roster.csv`: `student_id,class_id,display_name,email,active`.
- `email_connector_config.json`: copy `templates/submissions/email_connector_config.template.json` and fill in the assignment-specific scope.

Keep these files out of git.

## Mailbox Setup

Create a dedicated label, folder, or narrow search for the assignment. Do not scan the whole inbox. Tell students to include the assignment ID and student ID in the subject or PDF filename, for example:

```text
Subject: p3_quiz_2026_06_23 S0001
Filename: S0001_p3_quiz_2026_06_23.pdf
```

## Dry Run

```bash
.venv/bin/python scripts/import_live_email_submissions.py \
  --assignment data/submissions/p3_quiz_2026_06_23/assignment.json \
  --roster data/submissions/p3_quiz_2026_06_23/roster.csv \
  --connector-config data/submissions/p3_quiz_2026_06_23/email_connector_config.json \
  --submission-output-root output/submissions \
  --reports-root reports/submissions \
  --dry-run
```

Inspect:

- `output/submissions/<assignment_id>/live_email_import/live_email_dry_run.json`
- `reports/submissions/<assignment_id>_first_assignment_readiness.md`

## Apply

Apply only after reviewing the dry-run output and mailbox scope:

```bash
.venv/bin/python scripts/import_live_email_submissions.py \
  --assignment data/submissions/p3_quiz_2026_06_23/assignment.json \
  --roster data/submissions/p3_quiz_2026_06_23/roster.csv \
  --connector-config data/submissions/p3_quiz_2026_06_23/email_connector_config.json \
  --submission-output-root output/submissions \
  --reports-root reports/submissions \
  --apply
```

Review quarantine and intake artifacts under `output/submissions/<assignment_id>/email_intake/`, then inspect the completion CSV under `reports/submissions/`.

## Follow-Up Workflow

Build teacher review queue, draft grades, and outgoing approval queue only after intake is reviewed:

```bash
.venv/bin/python scripts/build_submission_review_queue.py --assignment-id <assignment_id>
.venv/bin/python scripts/build_submission_draft_grades.py --assignment-id <assignment_id>
.venv/bin/python scripts/build_outgoing_email_queue.py --assignment-id <assignment_id>
```

Approve outgoing messages by editing `output/submissions/<assignment_id>/outgoing_email/approval_template.csv` into a private approval CSV, then rebuild the outgoing queue with `--approval-csv`. Dry-run delivery before any future sending path.

## Privacy Checklist

- Do not commit real config, roster, student emails, submissions, message bodies, grades, feedback, or tokens.
- Keep all real artifacts under `data/submissions/`, `output/submissions/`, and `reports/submissions/`.
- Do not send outgoing email from this phase.
