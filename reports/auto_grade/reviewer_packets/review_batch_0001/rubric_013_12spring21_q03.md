# Rubric 013: 12spring21_q03

**Warning:** This candidate is not approved scoring evidence until reviewed_rubrics validation passes.

## Candidate Metadata

- question_id: `12spring21_q03`
- paper: `12spring21`
- paper_family: `p1`
- question_number: `3`
- part_path: `[]`
- total_marks: `4`
- proposed rubric_id: `rr_12spring21_q03`
- detected mark codes: `A, M`
- risk flags: `none`
- blockers: `none`

## Canonical Images

- Question image: [p1/12spring21/questions/q03.png](p1/12spring21/questions/q03.png)
- Mark-scheme image: [p1/12spring21/mark_scheme/q03.png](p1/12spring21/mark_scheme/q03.png)

## Dependency And Follow-Through Notes

- Dependency flags: `none`
- Follow-through flags: `none`

## Advisory Mark Events

### Event 1: `12spring21_q03_me0001`

- mark_code: `M`
- mark_code_raw: `M1`
- mark_type: `method`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
3 tan θ+ 2 sin θ = 3 tan θ−6 sin θ leading to 2 tan θ−8 sin θ[ = 0] M1 OE
```

### Event 2: `12spring21_q03_me0002`

- mark_code: `M`
- mark_code_raw: `M1`
- mark_type: `method`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
2 sin θ−8 sin θcosθ( = 0) leading to [2]sin θ(1−4 cos θ) [ = 0] M1
```

### Event 3: `12spring21_q03_me0003`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
cos θ = 1 A1 Ignore sin θ = 0
```

### Event 4: `12spring21_q03_me0004`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
θ = 75.5° only A1
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
        "answer_text": "tan \u03b8+ 2 sin \u03b8 = 3 tan \u03b8\u22126 sin \u03b8 leading to 2 tan \u03b8\u22128 sin \u03b8[ = 0]",
        "common_error_text": "",
        "condition_text": "oe",
        "confidence": "high",
        "normalized_text": "3 tan \u03b8+ 2 sin \u03b8 = 3 tan \u03b8\u22126 sin \u03b8 leading to 2 tan \u03b8\u22128 sin \u03b8[ = 0] M1 OE",
        "raw_text": "3 tan \u03b8+ 2 sin \u03b8 = 3 tan \u03b8\u22126 sin \u03b8 leading to 2 tan \u03b8\u22128 sin \u03b8[ = 0] M1 OE"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring21_q03_e0001",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "M",
      "mark_type": "method",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring21_q03_me0001"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "sin \u03b8\u22128 sin \u03b8cos\u03b8( = 0) leading to [2]sin \u03b8(1\u22124 cos \u03b8) [ = 0]",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "2 sin \u03b8\u22128 sin \u03b8cos\u03b8( = 0) leading to [2]sin \u03b8(1\u22124 cos \u03b8) [ = 0] M1",
        "raw_text": "2 sin \u03b8\u22128 sin \u03b8cos\u03b8( = 0) leading to [2]sin \u03b8(1\u22124 cos \u03b8) [ = 0] M1"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring21_q03_e0002",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "M",
      "mark_type": "method",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring21_q03_me0002"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "cos \u03b8 = 1",
        "common_error_text": "Ignore sin \u03b8 = 0",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "cos \u03b8 = 1 A1 Ignore sin \u03b8 = 0",
        "raw_text": "cos \u03b8 = 1 A1 Ignore sin \u03b8 = 0"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring21_q03_e0003",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring21_q03_me0003"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "\u03b8 = 75.5\u00b0 only",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "\u03b8 = 75.5\u00b0 only A1",
        "raw_text": "\u03b8 = 75.5\u00b0 only A1"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12spring21_q03_e0004",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12spring21_q03_me0004"
    }
  ],
  "paper": "12spring21",
  "paper_family": "p1",
  "part_path": [],
  "question_number": "3",
  "review_status": "needs_human_review",
  "reviewed_at": null,
  "reviewed_by": null,
  "rubric_id": "rr_12spring21_q03",
  "rubric_total_verified": false,
  "safe_for_auto_grade_lab": false,
  "safe_for_student_self_check": false,
  "safe_for_teacher_beta": false,
  "source_mark_events_record_id": "12spring21_q03",
  "source_mark_scheme_image_path": "p1/12spring21/mark_scheme/q03.png",
  "source_question_id": "12spring21_q03",
  "total_marks": 4
}
```
