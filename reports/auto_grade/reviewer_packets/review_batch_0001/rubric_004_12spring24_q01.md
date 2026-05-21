# Rubric 004: 12spring24_q01

**Warning:** This candidate is not approved scoring evidence until reviewed_rubrics validation passes.

## Candidate Metadata

- question_id: `12spring24_q01`
- paper: `12spring24`
- paper_family: `p1`
- question_number: `1`
- part_path: `[]`
- total_marks: `3`
- proposed rubric_id: `rr_12spring24_q01`
- detected mark codes: `A, B, M`
- risk flags: `none`
- blockers: `none`

## Canonical Images

- Question image: [p1/12spring24/questions/q01.png](p1/12spring24/questions/q01.png)
- Mark-scheme image: [p1/12spring24/mark_scheme/q01.png](p1/12spring24/mark_scheme/q01.png)

## Dependency And Follow-Through Notes

- Dependency flags: `none`
- Follow-through flags: `none`

## Advisory Mark Events

### Event 1: `12spring24_q01_me0001`

- mark_code: `B`
- mark_code_raw: `B1`
- mark_type: `independent_statement`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
1 Integrate to obtain −2x^{−}1 B1 OE
```

### Event 2: `12spring24_q01_me0002`

- mark_code: `M`
- mark_code_raw: `M1`
- mark_type: `method`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
Substitute limits correctly with clear indication seen that upper limit gives 0 M1 For integral of form −k x^{−}n, where k 0, n 0.
```

### Event 3: `12spring24_q01_me0003`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
Obtain ^{2}_{3} A1 WWW
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
        "answer_text": "Integrate to obtain \u22122x^{\u2212}1",
        "common_error_text": "",
        "condition_text": "oe",
        "confidence": "high",
        "normalized_text": "1 Integrate to obtain \u22122x^{\u2212}1 B1 OE",
        "raw_text": "1 Integrate to obtain \u22122x^{\u2212}1 B1 OE"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring24_q01_e0001",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "B",
      "mark_type": "independent_statement",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring24_q01_me0001"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "Substitute limits correctly with clear indication seen that upper limit gives 0",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "Substitute limits correctly with clear indication seen that upper limit gives 0 M1 For integral of form \u2212k x^{\u2212}n, where k \uf03e0, n \uf03e0.",
        "raw_text": "Substitute limits correctly with clear indication seen that upper limit gives 0 M1 For integral of form \u2212k x^{\u2212}n, where k \uf03e0, n \uf03e0."
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring24_q01_e0002",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "M",
      "mark_type": "method",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring24_q01_me0002"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "Obtain ^{2}_{3}",
        "common_error_text": "",
        "condition_text": "www",
        "confidence": "high",
        "normalized_text": "Obtain ^{2}_{3} A1 WWW",
        "raw_text": "Obtain ^{2}_{3} A1 WWW"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring24_q01_e0003",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring24_q01_me0003"
    }
  ],
  "paper": "12spring24",
  "paper_family": "p1",
  "part_path": [],
  "question_number": "1",
  "review_status": "needs_human_review",
  "reviewed_at": null,
  "reviewed_by": null,
  "rubric_id": "rr_12spring24_q01",
  "rubric_total_verified": false,
  "safe_for_auto_grade_lab": false,
  "safe_for_student_self_check": false,
  "safe_for_teacher_beta": false,
  "source_mark_events_record_id": "12spring24_q01",
  "source_mark_scheme_image_path": "p1/12spring24/mark_scheme/q01.png",
  "source_question_id": "12spring24_q01",
  "total_marks": 3
}
```
