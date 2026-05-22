# Reviewed Rubric Authoring Guide

Phase 2B review batches are worklists, not scoring evidence. A reviewer turns one batch candidate into an approved rubric only by editing a reviewed-rubrics draft entry and completing the checks below.

The live approved registry is `output/auto_grade/reviewed_rubrics.v1.json`. Do not point eligibility builds at an unapproved draft workspace. Draft workspaces may contain incomplete records; the promotion command selects only records that are already approved and validates them before writing the live registry.

## Source Checks

1. Open `canonical_question_artifact` from the batch candidate and confirm the question, parts, and total marks.
2. Open `canonical_mark_scheme_artifact` and confirm the mark scheme belongs to the same question.
3. Reconcile `total_marks` against both images. Set `rubric_total_verified: true` only after the total is confirmed. If the totals disagree, leave the rubric `needs_human_review` or mark it `blocked` and document the mismatch.

## Evidence Authoring

The draft `advisory_evidence` fields are extraction aids only. Do not treat them as accepted scoring text.

For each event:

- Rewrite the canonical mark-scheme evidence into `accepted_evidence`.
- Keep `mark_value` equal to the mark-scheme value for that event.
- Verify `mark_code` from the image. Allowed Phase 2B codes are `M`, `A`, `B`, `E`, `DM`, and `FT`; `unknown` cannot be approved.
- Add concise `common_errors` only when the mark scheme explicitly supports them.
- Add `alternative_methods` when the mark scheme gives valid alternatives or when the evidence has method variants that a reviewer intentionally accepts.
- Attach at least one `learning_target_ids` value to every approved event.

## Mark-Code Policy

- `M`: method mark. Describe the required method in `accepted_evidence`.
- `A`: accuracy mark. State the required final expression, value, or conclusion.
- `B`: independent mark. State the independent fact, graph feature, statement, or result.
- `E`: explanation mark. State the required explanation or justification.
- `DM`: dependent method mark. Complete `dependency` with the prior event or condition required before this mark can be awarded.
- `FT`: follow-through mark. Complete `follow_through_policy` with exactly what earlier error may be followed through and what remains required.

## Approval Fields

Set a rubric to `review_status: approved` only when all of these are true:

- The canonical question image has been checked.
- The canonical mark-scheme image has been checked.
- `rubric_total_verified` is `true`.
- Event mark values sum to `total_marks`.
- `reviewed_by` identifies the reviewer.
- `reviewed_at` is an ISO timestamp.
- Every event has accepted evidence, learning target IDs, and `review_status: approved`.
- Dependencies, follow-through, and alternative methods are documented where applicable.
- `safe_for_auto_grade_lab` is explicitly set.
- `safe_for_teacher_beta` is set only when teacher beta use is intended.
- `safe_for_student_self_check` remains `false` in Phase 2B.

Leave the rubric as `needs_human_review` when any evidence needs checking, any mark code is unresolved, dependencies or follow-through are unclear, learning targets are missing, or total marks do not reconcile. Use `blocked` when a canonical source mismatch or missing image prevents review.

## Approved Registry Workflow

1. Review candidate pages in `reports/auto_grade/reviewer_packets/review_batch_0001/`.
2. Edit the draft reviewed-rubrics workspace, for example `output/auto_grade/review_batches/reviewed_rubrics_draft_0001.v1.json`.
3. Validate the draft/workspace:

```bash
.venv/bin/python scripts/validate_auto_grade_reviewed_rubrics.py \
  --reviewed-rubrics output/auto_grade/review_batches/reviewed_rubrics_draft_0001.v1.json \
  --question-bank output/json/question_bank.json
```

4. Run the completion checker:

```bash
.venv/bin/python scripts/check_auto_grade_rubric_review_completion.py \
  --reviewed-rubrics output/auto_grade/review_batches/reviewed_rubrics_draft_0001.v1.json
```

5. Promote only approved rubrics into the live registry:

```bash
.venv/bin/python scripts/promote_auto_grade_reviewed_rubrics.py \
  --source-reviewed-rubrics output/auto_grade/review_batches/reviewed_rubrics_draft_0001.v1.json \
  --question-bank output/json/question_bank.json \
  --output output/auto_grade/reviewed_rubrics.v1.json \
  --mode replace-approved
```

6. Validate the live registry:

```bash
.venv/bin/python scripts/validate_auto_grade_reviewed_rubrics.py \
  --reviewed-rubrics output/auto_grade/reviewed_rubrics.v1.json \
  --question-bank output/json/question_bank.json
```

7. Rebuild eligible items from the live registry:

```bash
.venv/bin/python scripts/build_auto_grade_eligible_items.py \
  --question-bank output/json/question_bank.json \
  --reviewed-rubrics output/auto_grade/reviewed_rubrics.v1.json \
  --output output/auto_grade/eligible_items.v1.json
```

8. Validate eligible items:

```bash
.venv/bin/python scripts/validate_auto_grade_eligible_items.py \
  --eligible-items output/auto_grade/eligible_items.v1.json \
  --question-bank output/json/question_bank.json
```

9. Confirm that only `teacher_beta` changed. `student_self_check_beta` and `student_ready` must remain `0`.

The promotion summary is written to `reports/auto_grade/reviewed_rubrics_registry_summary.md`. Incomplete `needs_human_review` entries stay in the draft workspace and are counted as excluded from the live registry; they are not scoring or promotion candidates.

## Minimal Approved Shape

```json
{
  "rubric_id": "rr_11autumn21_q05",
  "source_question_id": "11autumn21_q05",
  "source_question_image_path": "p1/11autumn21/questions/q05.png",
  "source_mark_scheme_image_path": "p1/11autumn21/mark_scheme/q05.png",
  "source_mark_events_record_id": "11autumn21_q05",
  "paper": "11autumn21",
  "paper_family": "p1",
  "question_number": "5",
  "part_path": [],
  "total_marks": 5,
  "rubric_total_verified": true,
  "safe_for_auto_grade_lab": true,
  "safe_for_teacher_beta": true,
  "safe_for_student_self_check": false,
  "review_status": "approved",
  "reviewed_by": "reviewer name",
  "reviewed_at": "2026-05-21T00:00:00Z",
  "approval_scope": "teacher_beta",
  "events": [
    {
      "event_id": "rr_11autumn21_q05_e0001",
      "source_event_id": "11autumn21_q05_me0001",
      "part_path": [],
      "mark_code": "B",
      "mark_type": "independent_statement",
      "mark_value": 1,
      "dependency": "independent",
      "follow_through_policy": "none",
      "accepted_evidence": ["Human-reviewed evidence from the canonical mark-scheme image."],
      "common_errors": [],
      "alternative_methods": [],
      "learning_target_ids": ["9709_p1_topic_example"],
      "review_status": "approved",
      "review_notes": "Reviewed from canonical image."
    }
  ]
}
```
