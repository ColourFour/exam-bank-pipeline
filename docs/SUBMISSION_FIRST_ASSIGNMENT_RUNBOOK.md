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

## Classroom Dashboard

For the local browser workflow, start the dashboard from the repository root:

```bash
.venv/bin/python -m exam_bank.cli classroom \
  --host 127.0.0.1 \
  --port 8765
```

If port `8765` is already in use, choose another localhost port:

```bash
.venv/bin/python -m exam_bank.cli classroom \
  --host 127.0.0.1 \
  --port 8766
```

The dashboard reads class workspaces from `data/classes/`, submission artifacts from `output/submissions/`, and reports from `reports/submissions/`. Use the blue left sidebar to select a class. The email tab shows sent-recipient details when an assignment has already been sent; `Send now` may be used to intentionally resend after confirmation.

Equivalent CLI commands for the same local classroom workflow:

```bash
.venv/bin/python -m exam_bank.cli class-init \
  --class-id <class_id> \
  --roster path/to/roster.csv
```

```bash
.venv/bin/python -m exam_bank.cli class-add-assignment \
  --class-id <class_id> \
  --assignment-id <assignment_id> \
  --pdf path/to/assignment.pdf \
  --title "Quiz Review" \
  --send-at 2026-06-24T13:50:00+08:00 \
  --due-at 2026-06-26T11:55:00+08:00
```

Dry-run scheduled dispatch before any live send:

```bash
.venv/bin/python -m exam_bank.cli class-dispatch-due \
  --class-id <class_id> \
  --assignment-id <assignment_id> \
  --now 2026-06-24T13:50:00+08:00
```

Live dispatch requires the explicit `--send-live` flag:

```bash
.venv/bin/python -m exam_bank.cli class-dispatch-due \
  --class-id <class_id> \
  --assignment-id <assignment_id> \
  --now 2026-06-24T13:50:00+08:00 \
  --from teacher@example.edu \
  --send-live
```

Ingest uploaded or inbox submissions:

```bash
.venv/bin/python -m exam_bank.cli class-ingest-submissions \
  --class-id <class_id> \
  --assignment-id <assignment_id>
```

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
- Do not send outgoing email from legacy intake scripts. Classroom dashboard sends require explicit teacher confirmation; classroom CLI sends require `class-dispatch-due --send-live`.

## Verification

Run the focused classroom checks after changing dashboard layout, assignment email behavior, or classroom dispatch code:

```bash
.venv/bin/python -m pytest \
  tests/test_classroom.py \
  tests/test_classroom_dashboard.py
```
