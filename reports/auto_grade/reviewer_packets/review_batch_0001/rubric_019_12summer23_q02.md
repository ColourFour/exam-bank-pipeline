# Rubric 019: 12summer23_q02

**Warning:** This candidate is not approved scoring evidence until reviewed_rubrics validation passes.

## Candidate Metadata

- question_id: `12summer23_q02`
- paper: `12summer23`
- paper_family: `p1`
- question_number: `2`
- part_path: `[]`
- total_marks: `4`
- proposed rubric_id: `rr_12summer23_q02`
- detected mark codes: `A, B, M`
- risk flags: `none`
- blockers: `none`

## Canonical Images

- Question image: [p1/12summer23/questions/q02.png](p1/12summer23/questions/q02.png)
- Mark-scheme image: [p1/12summer23/mark_scheme/q02.png](p1/12summer23/mark_scheme/q02.png)

## Dependency And Follow-Through Notes

- Dependency flags: `none`
- Follow-through flags: `none`

## Advisory Mark Events

### Event 1: `12summer23_q02_me0001`

- mark_code: `B`
- mark_code_raw: `B1`
- mark_type: `independent_statement`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
2 [Coefficient of x^{4} = p =]1 5a^{2} B1 May be seen in an expansion or with x^{4}.
```

### Event 2: `12summer23_q02_me0002`

- mark_code: `B`
- mark_code_raw: `B1`
- mark_type: `independent_statement`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
[Coefficient of x^{2} = q =] 54a^{2} B1 May be seen in an expansion or with x^{2}.
```

### Event 3: `12summer23_q02_me0003`

- mark_code: `M`
- mark_code_raw: `M1`
- mark_type: `method`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
Equating their p + their q to 276 leading to an equation in a^{2} only M1 No x terms and no extra terms. If p and q are not identified
```

### Event 4: `12summer23_q02_me0004`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
a2 A1 CAO
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
        "answer_text": "[Coefficient of x^{4} = p =]1 5a^{2}",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "2 [Coefficient of x^{4} = p =]1 5a^{2} B1 May be seen in an expansion or with x^{4}.",
        "raw_text": "2 [Coefficient of x^{4} = p =]1 5a^{2} B1 May be seen in an expansion or with x^{4}."
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12summer23_q02_e0001",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "B",
      "mark_type": "independent_statement",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12summer23_q02_me0001"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "[Coefficient of x^{2} = q =] 54a^{2}",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "[Coefficient of x^{2} = q =] 54a^{2} B1 May be seen in an expansion or with x^{2}.",
        "raw_text": "[Coefficient of x^{2} = q =] 54a^{2} B1 May be seen in an expansion or with x^{2}."
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12summer23_q02_e0002",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "B",
      "mark_type": "independent_statement",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12summer23_q02_me0002"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "Equating their p + their q to 276 leading to an equation in a^{2} only",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "Equating their p + their q to 276 leading to an equation in a^{2} only M1 No x terms and no extra terms. If p and q are not identified",
        "raw_text": "Equating their p + their q to 276 leading to an equation in a^{2} only M1 No x terms and no extra terms. If p and q are not identified"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12summer23_q02_e0003",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "M",
      "mark_type": "method",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12summer23_q02_me0003"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "a\uf03d\uf0b12",
        "common_error_text": "",
        "condition_text": "cao",
        "confidence": "high",
        "normalized_text": "a\uf03d\uf0b12 A1 CAO",
        "raw_text": "a\uf03d\uf0b12 A1 CAO"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_12summer23_q02_e0004",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "12summer23_q02_me0004"
    }
  ],
  "paper": "12summer23",
  "paper_family": "p1",
  "part_path": [],
  "question_number": "2",
  "review_status": "needs_human_review",
  "reviewed_at": null,
  "reviewed_by": null,
  "rubric_id": "rr_12summer23_q02",
  "rubric_total_verified": false,
  "safe_for_auto_grade_lab": false,
  "safe_for_student_self_check": false,
  "safe_for_teacher_beta": false,
  "source_mark_events_record_id": "12summer23_q02",
  "source_mark_scheme_image_path": "p1/12summer23/mark_scheme/q02.png",
  "source_question_id": "12summer23_q02",
  "total_marks": 4
}
```
