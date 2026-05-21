# Rubric 007: 13autumn22_q01

**Warning:** This candidate is not approved scoring evidence until reviewed_rubrics validation passes.

## Candidate Metadata

- question_id: `13autumn22_q01`
- paper: `13autumn22`
- paper_family: `p1`
- question_number: `1`
- part_path: `[]`
- total_marks: `3`
- proposed rubric_id: `rr_13autumn22_q01`
- detected mark codes: `A, M`
- risk flags: `none`
- blockers: `none`

## Canonical Images

- Question image: [p1/13autumn22/questions/q01.png](p1/13autumn22/questions/q01.png)
- Mark-scheme image: [p1/13autumn22/mark_scheme/q01.png](p1/13autumn22/mark_scheme/q01.png)

## Dependency And Follow-Through Notes

- Dependency flags: `none`
- Follow-through flags: `none`

## Advisory Mark Events

### Event 1: `13autumn22_q01_me0001`

- mark_code: `M`
- mark_code_raw: `M1`
- mark_type: `method`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
1 8(1^{−} cos^{2}_{}) _{+} 6 cos^{}+ 1 = 0 M1 Expect 8 cos^{2}_{}−6 cos−9 = 0.
```

### Event 2: `13autumn22_q01_me0002`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
(4 cos+ 3)(2 cos−3)  = 0 A1 Factors or formula or completing square must be
```

### Event 3: `13autumn22_q01_me0003`

- mark_code: `A`
- mark_code_raw: `A1`
- mark_type: `accuracy`
- mark_value: `1`
- dependency: `independent`
- follow_through_policy: `none`
- advisory confidence: `high`

Advisory text, not scoring evidence:

```text
→cos= −0.75→=138.6° only, A1 AWRT, ignore solutions outside the given range,
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
        "answer_text": "8(1^{\u2212} cos^{2}_{\uf071}) _{+} 6 cos^{\uf071}+ 1 \uf05b= 0\uf05d",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "1 8(1^{\u2212} cos^{2}_{\uf071}) _{+} 6 cos^{\uf071}+ 1 \uf05b= 0\uf05d M1 Expect 8 cos^{2}_{\uf071}\u22126 cos\uf071\u22129 = 0.",
        "raw_text": "1 8(1^{\u2212} cos^{2}_{\uf071}) _{+} 6 cos^{\uf071}+ 1 \uf05b= 0\uf05d M1 Expect 8 cos^{2}_{\uf071}\u22126 cos\uf071\u22129 = 0."
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_13autumn22_q01_e0001",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "M",
      "mark_type": "method",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "13autumn22_q01_me0001"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "(4 cos\uf071+ 3)(2 cos\uf071\u22123) \uf05b = 0\uf05d",
        "common_error_text": "",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "(4 cos\uf071+ 3)(2 cos\uf071\u22123) \uf05b = 0\uf05d A1 Factors or formula or completing square must be",
        "raw_text": "(4 cos\uf071+ 3)(2 cos\uf071\u22123) \uf05b = 0\uf05d A1 Factors or formula or completing square must be"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_13autumn22_q01_e0002",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "13autumn22_q01_me0002"
    },
    {
      "accepted_evidence": [],
      "advisory_evidence": {
        "answer_text": "\uf05b\u2192cos\uf071= \u22120.75\u2192\uf071=\uf05d138.6\u00b0 only",
        "common_error_text": "ignore solutions outside the given range",
        "condition_text": "",
        "confidence": "high",
        "normalized_text": "\uf05b\u2192cos\uf071= \u22120.75\u2192\uf071=\uf05d138.6\u00b0 only, A1 AWRT, ignore solutions outside the given range,",
        "raw_text": "\uf05b\u2192cos\uf071= \u22120.75\u2192\uf071=\uf05d138.6\u00b0 only, A1 AWRT, ignore solutions outside the given range,"
      },
      "alternative_methods": [],
      "common_errors": [],
      "dependency": "independent",
      "event_id": "rr_13autumn22_q01_e0003",
      "follow_through_policy": "none",
      "learning_target_ids": [],
      "mark_code": "A",
      "mark_type": "accuracy",
      "mark_value": 1,
      "part_path": [],
      "review_notes": "",
      "review_status": "needs_human_review",
      "source_event_id": "13autumn22_q01_me0003"
    }
  ],
  "paper": "13autumn22",
  "paper_family": "p1",
  "part_path": [],
  "question_number": "1",
  "review_status": "needs_human_review",
  "reviewed_at": null,
  "reviewed_by": null,
  "rubric_id": "rr_13autumn22_q01",
  "rubric_total_verified": false,
  "safe_for_auto_grade_lab": false,
  "safe_for_student_self_check": false,
  "safe_for_teacher_beta": false,
  "source_mark_events_record_id": "13autumn22_q01",
  "source_mark_scheme_image_path": "p1/13autumn22/mark_scheme/q01.png",
  "source_question_id": "13autumn22_q01",
  "total_marks": 3
}
```
