# Auto-Grading Contract

Phase 1 defines contracts and deterministic readiness checks only. It does not grade student work, judge OCR or vision output, accept photo submissions, call LLMs, or produce student-facing feedback.

## Source Of Truth

Canonical question and mark-scheme images remain the source of truth for every item. The canonical question-bank export at `output/json/question_bank.json` is read-only input for auto-grade work and must not be mutated by any auto-grade builder or validator.

Generated text, mark events, topic routing, difficulty, and advisory evidence are sidecars. They can explain review needs, but they do not become scoring authority automatically. In particular, generated mark-event evidence is advisory unless a separate reviewed rubric contract explicitly promotes it.

## Role Ladder

Auto-grade eligibility is represented by one status:

- `blocked`: not usable for auto-grade review or student modes until blocking issues are fixed.
- `review_only`: suitable for human rubric authoring or review, but not for scoring.
- `teacher_beta`: may be used by teachers in a controlled beta only after a reviewed rubric validates.
- `student_self_check_beta`: may be used for limited student self-check only after reviewed rubric approval and student-mode gates validate.
- `student_ready`: may be used in student-facing workflows only after reviewed rubric approval, benchmark evidence, and all student-mode gates validate.

Phase 1 fails closed. No generated eligibility artifact may classify an item as `teacher_beta`, `student_self_check_beta`, or `student_ready` unless a reviewed rubric artifact exists and validates for that item.

## Artifacts

Current Phase 1 artifact:

- `output/auto_grade/eligible_items.v1.json`

Future artifacts:

- `output/auto_grade/reviewed_rubrics.v1.json`
- `output/auto_grade/benchmark_submissions.v1.json`
- future attempt sidecars under `output/auto_grade/`
- future score-event sidecars under `output/auto_grade/`

Reports belong under `reports/auto_grade/`.

Student submission images are not part of Phase 1 and must not be committed.

## Eligibility Expectations

Each eligible-item record must identify the canonical question image, canonical mark-scheme image, total marks, supported submission modes, supported grading mode, rubric id, learning-target ids, eligibility status, block reasons, and review metadata.

Required fail-closed rules:

- Missing canonical question image blocks the item.
- Missing canonical mark-scheme image blocks the item.
- Detectable unresolved total mismatches block the item.
- Missing or unreviewed rubric blocks grading modes, or at most leaves the item `review_only`.
- Advisory mark events alone are never enough for scoring eligibility.
- Failed or review-required topic routing must not block `review_only` by itself.
- Failed or review-required topic routing blocks student learning-target feedback readiness.
- Missing learning-target ids must be recorded as a reason blocking future student modes.
- Generated eligibility may classify records, but no item may become student-safe without a reviewed rubric.

## Validation Expectations

The validator must reject malformed artifacts, record-count mismatches, duplicate or orphan question ids, missing canonical artifact paths, missing files when existence checking is enabled, missing or invalid eligibility statuses, blocked records without block reasons, unsafe promotion without reviewed rubric approval, and total mismatches against reviewed rubric metadata when present.

Validation may emit warnings for review-only concerns, but any student-safe promotion without a reviewed rubric is an error.

## Fail-Closed Behavior

When evidence is missing, ambiguous, advisory-only, or not reviewed, the generated artifact must choose the safest status. For Phase 1, `student_ready` and `student_self_check_beta` counts are expected to be zero unless reviewed rubric artifacts already exist and validate.
