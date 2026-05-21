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

Phase 1 and Phase 2A fail closed. No generated eligibility artifact may classify an item as `teacher_beta`, `student_self_check_beta`, or `student_ready` unless a reviewed rubric artifact exists and validates for that item. In Phase 2A, validated reviewed rubrics can promote to `teacher_beta` at most.

## Artifacts

Current Phase 1 artifact:

- `output/auto_grade/eligible_items.v1.json`

Phase 2A artifacts:

- `output/auto_grade/reviewed_rubrics.v1.json`
- `output/auto_grade/rubric_review_queue.v1.json`

Phase 2B human review workflow artifacts:

- `output/auto_grade/review_batches/review_batch_0001.v1.json`
- `output/auto_grade/review_batches/reviewed_rubrics_draft_0001.v1.json`
- `docs/auto_grade/REVIEWED_RUBRIC_AUTHORING_GUIDE.md`

Future artifacts:

- `output/auto_grade/benchmark_submissions.v1.json`
- future attempt sidecars under `output/auto_grade/`
- future score-event sidecars under `output/auto_grade/`

Reports belong under `reports/auto_grade/`.

Student submission images are not part of Phase 1 or Phase 2A and must not be committed.

## Reviewed Rubrics

Phase 2A introduces `output/auto_grade/reviewed_rubrics.v1.json`. This artifact is the first scoring-rubric contract, but it is still limited to human-reviewed rubric metadata and teacher-beta readiness. It does not score student work, inspect student submissions, call OCR or vision systems, call LLMs, or produce student-facing feedback.

The reviewed-rubric artifact records source question id, canonical mark-scheme image path, optional source mark-events record id, paper metadata, part path, total marks, explicit total-mark verification, approval flags, reviewer identity, review timestamp, review status, approval scope, and reviewed mark events. Each reviewed event records mark code, mark type, mark value, dependency policy, follow-through policy, accepted evidence, common errors, alternative methods, learning-target ids, review status, and review notes.

The review queue artifact, `output/auto_grade/rubric_review_queue.v1.json`, converts advisory mark-event sidecars into human-review candidates. It ranks safer candidates first but does not exclude harder cases. The queue is planning evidence only. It is not approved scoring evidence.

Phase 2B introduces review-batch and draft-workspace artifacts for authoring the first human reviewed rubric gold set. A review batch is a bounded human worklist selected from the queue. A draft reviewed-rubrics workspace may copy advisory mark-event data into draft fields, but all entries must default to `review_status: "needs_human_review"` with no safety flags. These artifacts are not approved scoring evidence. They can affect eligibility only after a human intentionally completes required metadata, accepted evidence, total verification, and approval flags in an explicit reviewed-rubrics file that passes validation.

Phase 2A distinguishes these records:

- Advisory mark event: generated sidecar evidence extracted from mark schemes. It may help review, but it is never scoring authority by itself.
- Review candidate: a question placed in the rubric review queue with priority, blockers, and risk flags for a human reviewer.
- Reviewed rubric: a rubric entry with explicit human review metadata, verified totals, reviewed events, and approved review status.
- Teacher-beta rubric: a reviewed rubric that is safe for the auto-grade lab and explicitly safe for teacher beta. It may promote eligibility to `teacher_beta` only.
- Student-safe rubric: a future-phase rubric that would be safe for student self-check or student-ready use. Phase 2A forbids producing this status.

Approval metadata is required before any rubric can affect eligibility. An approved rubric must include `review_status: "approved"`, non-empty `reviewed_by`, non-empty `reviewed_at`, `rubric_total_verified: true`, and explicit safety scope. `safe_for_teacher_beta` requires `safe_for_auto_grade_lab`. Phase 2A rejects `safe_for_student_self_check` and rejects all `student_self_check_beta` or `student_ready` eligibility promotions.

Mark-code dependencies must be explicit. Dependent method marks such as `DM` require a dependency policy. Follow-through marks such as `FT` require a follow-through policy. Approved rubrics may use known mark codes including `M`, `A`, `B`, `E`, `DM`, and `FT`; `unknown` may appear in draft data but is rejected for approved rubrics.

Total marks must reconcile. For approved rubrics, event mark values must sum to the rubric total, and the rubric total must be verified against the canonical question and mark-scheme images. A mismatch fails validation and prevents promotion.

Learning-target mapping is required at event level for approved rubrics so later student-mode feedback can be reviewed against explicit curriculum targets. Missing, low-confidence, or review-required learning-target routing remains a student-mode blocker. It does not make advisory evidence scoring-safe.

Hard boundary: Phase 2A cannot create student-ready rubrics or student-safe statuses. No item may become `student_self_check_beta` or `student_ready` from Phase 2A artifacts.

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

The eligible-items validator must reject malformed artifacts, record-count mismatches, duplicate or orphan question ids, missing canonical artifact paths, missing files when existence checking is enabled, missing or invalid eligibility statuses, blocked records without block reasons, unsafe promotion without reviewed rubric approval, and total mismatches against reviewed rubric metadata when present.

The reviewed-rubrics validator must reject malformed top-level schema, duplicate rubric ids, duplicate event ids within a rubric, source question ids not in the question bank, missing source mark-scheme image paths, missing reviewer identity or timestamp for approved rubrics, unverified totals, event totals that do not match rubric totals, approved unknown mark codes, missing accepted evidence, missing dependency or follow-through policies, student-safe flags in Phase 2A, teacher-beta approval without lab safety, and advisory-only promotion attempts without review metadata.

Validation may emit warnings for review-only concerns, but any student-safe promotion is an error in Phase 2A.

## Fail-Closed Behavior

When evidence is missing, ambiguous, advisory-only, or not reviewed, the generated artifact must choose the safest status. For Phase 2A, `student_ready` and `student_self_check_beta` counts must remain zero. `teacher_beta` may be non-zero only when a valid reviewed rubric explicitly supports it.
