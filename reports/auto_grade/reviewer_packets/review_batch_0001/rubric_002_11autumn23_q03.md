# Rubric 002: 11autumn23_q03

**Warning:** This candidate is not approved scoring evidence until reviewed_rubrics validation passes.

## Candidate Metadata

- question_id: `11autumn23_q03`
- paper: `11autumn23`
- paper_family: `p1`
- question_number: `3`
- part_path: `[]`
- total_marks: `3`
- proposed rubric_id: `rr_11autumn23_q03`
- detected mark codes: `A, B, M`
- risk flags: `none`
- blockers: `none`

## Canonical Images

- Question image: [p1/11autumn23/questions/q03.png](p1/11autumn23/questions/q03.png)
- Mark-scheme image: [p1/11autumn23/mark_scheme/q03.png](p1/11autumn23/mark_scheme/q03.png)

## Dependency And Follow-Through Notes

- Dependency flags: `none`
- Follow-through flags: `none`

## Advisory Mark Events

### Event 1: `11autumn23_q03_me0001`

- mark_code: `B`
- mark_code_raw: `B1`
- mark_type: `independent_statement`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
3 dV = 3x^{2} B1 SOI
```

### Event 2: `11autumn23_q03_me0002`

- mark_code: `M`
- mark_code_raw: `M1`
- mark_type: `method`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
ddVt = ddVx ddxt 3= 20^{2}0.01 M1 Correct use of chain rule with dV . x = 20 substituted into
```

### Event 3: `11autumn23_q03_me0003`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `medium`

Advisory text, not scoring evidence:

```text
12 A1
```

## Reviewer Checklist

- [ ] `canonical_question_image_verified`
- [ ] `canonical_mark_scheme_image_verified`
- [ ] `total_marks_reconciled`
- [ ] `advisory_events_rewritten_as_accepted_evidence`
- [ ] `mark_codes_verified`
- [ ] `dependencies_documented`
- [ ] `follow_through_documented`
- [ ] `alternative_methods_documented`
- [ ] `learning_target_ids_attached`
- [ ] `approval_scope_confirmed_teacher_beta_only`

## Approval Checklist

- [ ] Canonical question image verified.
- [ ] Canonical mark-scheme image verified.
- [ ] Total marks reconciled with event mark values.
- [ ] Every advisory event rewritten as human-reviewed accepted evidence.
- [ ] Mark codes verified and no `unknown` mark codes remain.
- [ ] Dependencies and follow-through policies documented where applicable.
- [ ] Learning target IDs assigned to every approved event.
- [ ] Student self-check and student-ready remain false/not present.

## JSON Fields Reviewer Must Complete

- `source_question_image_path`
- `source_mark_scheme_image_path`
- `reviewed_by`
- `reviewed_at`
- `review_status`
- `rubric_total_verified`
- `safe_for_auto_grade_lab`
- `safe_for_teacher_beta`
- `safe_for_student_self_check`
- `accepted_evidence` for every event
- `dependency` for every dependent event
- `follow_through_policy` for every follow-through event
- `alternative_methods`
- `learning_target_ids` for every approved event
- `review_notes`

## Draft Rubric Snapshot

```json
{
  "approval_scope": "none",
  "events": [
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "dV = 3x^{2}",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "3 dV = 3x^{2} B1 SOI",
        "raw_text": "3 dV = 3x^{2} B1 SOI"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_11autumn23_q03_e0001",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "B",
      "mark_type": "independent_statement",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "11autumn23_q03_me0001"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "ddVt \uf0e9\uf0ea\uf0eb= ddVx \uf0b4ddxt\uf0f9\uf0fa\uf0fb 3= \uf0b420^{2}\uf0b40.01",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "ddVt \uf0e9\uf0ea\uf0eb= ddVx \uf0b4ddxt\uf0f9\uf0fa\uf0fb 3= \uf0b420^{2}\uf0b40.01 M1 Correct use of chain rule with dV . x = 20 substituted into",
        "raw_text": "ddVt \uf0e9\uf0ea\uf0eb= ddVx \uf0b4ddxt\uf0f9\uf0fa\uf0fb 3= \uf0b420^{2}\uf0b40.01 M1 Correct use of chain rule with dV . x = 20 substituted into"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_11autumn23_q03_e0002",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "M",
      "mark_type": "method",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "11autumn23_q03_me0002"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "medium",
        "normalized_text": "12 A1",
        "raw_text": "12 A1"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_11autumn23_q03_e0003",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "11autumn23_q03_me0003"
    }
  ],
  "paper": "11autumn23",
  "paper_family": "p1",
  "part_path": [],
  "question_number": "3",
  "review_status": "needs_human_review",
  "reviewed_at": null,
  "reviewed_by": null,
  "rubric_id": "rr_11autumn23_q03",
  "rubric_total_verified": false,
  "safe_for_auto_grade_lab": false,
  "safe_for_student_self_check": false,
  "safe_for_teacher_beta": false,
  "source_mark_events_record_id": "11autumn23_q03",
  "source_mark_scheme_image_path": "p1/11autumn23/mark_scheme/q03.png",
  "source_question_id": "11autumn23_q03",
  "total_marks": 3
}
```
