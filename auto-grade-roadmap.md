# Auto-Grade Roadmap

This roadmap defines the path from the current image-first exam-bank pipeline to a controlled auto-grading system where a student can submit a photo of a solution, receive an evidence-backed score, and understand the result through mark-code explanations first, then learning-target feedback.

The project is not ready for automated marking today. The right first milestone is a teacher/reviewer-safe auto-grading lab with strict abstention behavior. Student-facing self-checks come only after reviewed rubrics, benchmark submissions, verifier checks, and regression gates pass.

## Current Baseline

Measured from the current workspace on `2026-05-21`:

- Canonical export: `output/json/question_bank.json`, schema `exam_bank.question_bank` version 2.
- Records: `1301`.
- Canonical practice allowed by current Asterion export: `213 allow`, `1088 block`.
- Quick-check source allowed by current Asterion export: `53 allow`, `1230 block_until_reviewed`, `18 block`.
- Warmup generator source allowed by current Asterion export: `193 allow`, `1090 block_until_reviewed`, `18 block`.
- Mark-event sidecar: `1301` records, `1229` safe for advisory use, `0` safe for marking use.
- Mark-event extraction status: `1020 parsed`, `209 partial`, `72 review`.
- Mark-event validation: `ok=true`, `0` errors, `0` warnings.
- Content Lab candidates: `2416`, all currently `generation_gate.status=blocked_until_reviewed`.
- Dominant Content Lab blockers: missing source skill IDs, unreviewed mark events, unreviewed mappings/subparts, and question quality gates.
- Topic-routing sidecar: `safe_for_strict_filters=false`, with `153` failed records and `221` review-required records.
- Difficulty index: `1221` safe for teacher filtering, `0` safe for student sequencing.

Implication: the repository has a strong evidence substrate, but it intentionally blocks automated student scoring. Auto-grading must be introduced as a new audited contract, not by relaxing the existing advisory sidecars.

## North Star

Target student flow:

1. Student opens a trusted practice item.
2. Student writes a solution on paper.
3. Student uploads or photographs the solution.
4. The system identifies the item, checks image quality, transcribes relevant work, aligns work to official mark events, and scores only when evidence is sufficient.
5. The student sees mark-by-mark feedback:
   - `M`: method marks for valid process.
   - `A`: accuracy marks, often dependent on method.
   - `B`: independent result, statement, or fact marks.
   - `E`: explanation or communication marks where present.
   - `DM`: dependent method marks.
   - `FT`: follow-through marks from earlier work.
6. Later versions map each awarded or missed mark to learning targets, prerequisite gaps, and next practice recommendations.

The product promise should be: "This is an evidence-backed self-check, with clear confidence and review boundaries." It should not claim perfect examiner equivalence until the benchmarks prove it.

## Non-Negotiable Rules

- Canonical question and mark-scheme images remain the source of truth.
- `question_bank.json` is not mutated by grading runs.
- Existing mark-event evidence remains advisory until a new reviewed rubric contract approves a subset for scoring.
- No score is emitted without a reviewed rubric, source artifact integrity, submission-image quality checks, and a verifier pass.
- The grader must abstain rather than invent credit when work is unreadable, unmatched, incomplete, or outside supported method families.
- Every awarded mark must cite evidence from the student submission and the official mark event.
- Every withheld mark must cite the missing or contradicted evidence when possible.
- Student-facing output must expose uncertainty clearly.
- Learning-target feedback must use canonical topic/skill IDs only. Suggested new skills remain review-only.

## Proposed Architecture

Add an auto-grading layer beside the current extraction pipeline:

```text
question_bank.json
  + canonical images
  + reviewed mark-event rubric
  + reviewed skill/learning-target map
  + eligible item registry
        |
student solution image
        |
capture quality gate
        |
submission transcription and layout segmentation
        |
item/subpart alignment
        |
rubric event matcher
        |
scorer
        |
self-check verifier
        |
score event sidecar + student explanation
```

Primary future artifacts:

- `output/auto_grade/eligible_items.v1.json`
- `output/auto_grade/reviewed_rubrics.v1.json`
- `output/auto_grade/benchmark_submissions.v1.json`
- `output/auto_grade/submissions/<attempt_id>/submission_attempt.v1.json`
- `output/auto_grade/scoring/<attempt_id>.score_events.v1.json`
- `reports/auto_grade/*.md`

The auto-grade outputs should be attempt-level sidecars. They should never be merged into the canonical bank.

## Core Data Contracts

### Eligible Item Registry

Purpose: define which questions are even candidates for auto-grading.

Required fields:

- `question_id`
- `paper`, `paper_family`, `question_number`
- `canonical_question_artifact`
- `canonical_mark_scheme_artifact`
- `total_marks`
- `supported_submission_modes`, for example `photo_single_page`, `photo_multi_page`, `typed_work`
- `supported_grading_mode`, for example `whole_question`, `subpart`, `final_answer_only`, `worked_solution`
- `rubric_id`
- `learning_target_ids`
- `eligibility_status`: `blocked`, `review_only`, `teacher_beta`, `student_self_check_beta`, `student_ready`
- `block_reasons`
- `reviewed_by`
- `reviewed_at`

Initial inclusion rules:

- Asterion `canonical_practice=allow`.
- Question and mark-scheme image integrity pass.
- Mapping, validation, scope, and visual curation pass.
- Mark-scheme crop confidence is high.
- Reviewed mark-event rubric exists.
- Skill or learning-target mapping exists for the whole question or selected subpart.
- No unresolved total mismatch, unknown mark-code dependency, or serious mark-event review flag.

### Reviewed Rubric

Purpose: promote a human-reviewed subset of mark events into scoring rubrics.

Required fields:

- `rubric_id`
- `source_question_id`
- `source_mark_scheme_image_path`
- `total_marks`
- `events[]`
- event fields: `event_id`, `part_path`, `mark_code`, `mark_type`, `mark_value`, `dependency`, `follow_through_policy`, `accepted_evidence`, `common_errors`, `learning_target_ids`, `review_status`
- `rubric_total_verified=true`
- `safe_for_auto_grade_lab=true|false`
- `safe_for_student_self_check=true|false`

Promotion rule: generated mark events do not become a reviewed rubric by default. A human or audited review workflow must approve them.

### Submission Attempt

Purpose: store the student's uploaded work and all derived evidence.

Required fields:

- `attempt_id`
- `student_work_image_paths`
- `question_id`
- `capture_quality`
- `detected_regions`
- `transcription_candidates`
- `math_expression_candidates`
- `part_alignment`
- `warnings`
- `abstain_reasons`

No student submission images should be committed to git. Use ignored output roots or the downstream app's storage.

### Score Events

Purpose: record mark-level scoring decisions.

Required fields:

- `attempt_id`
- `question_id`
- `rubric_id`
- `score_total`
- `score_max`
- `confidence`
- `status`: `scored`, `partial_score`, `abstained`, `needs_review`
- `events[]`
- event fields: `rubric_event_id`, `mark_code`, `awarded`, `confidence`, `student_evidence`, `official_evidence`, `reason`, `dependency_result`, `verifier_status`
- `self_checks`
- `student_explanation`
- `teacher_debug`

Scoring output must separate student-facing explanation from teacher/debug evidence.

## Self-Check Layer

The auto-grader needs a verifier layer that can fail closed after the scorer proposes marks.

Minimum checks:

- Source check: question, mark scheme, rubric, and learning-target map all match the same `question_id`.
- Artifact check: canonical question and mark-scheme image files exist and hash correctly.
- Eligibility check: item status allows the requested use mode.
- Capture check: image is readable, not too blurred, not too cropped, and contains enough of the solution.
- Alignment check: submitted work aligns to the selected item and subpart.
- Rubric-total check: sum of rubric events equals official total.
- Dependency check: dependent `A`, `DM`, and `FT` marks obey rubric dependencies.
- Evidence check: no awarded mark lacks cited student evidence.
- Negative evidence check: no denied mark is contradicted by clear student evidence.
- Cross-scorer check: independent scorer and verifier agree within configured tolerance.
- Confidence check: low-confidence events cause partial scoring or abstention.
- Sum check: event totals equal the reported total.
- Regression check: score decisions remain stable on locked benchmark submissions.
- Privacy check: attempt output contains no unexpected personal data fields.

Student-facing scoring is allowed only when every required self-check passes.

## Phases And Gates

Soft targets assume one focused engineering stream. They are intentionally approximate.

### Phase 0: Current-State Readiness

Soft target: 1 week.

Goal: freeze a reliable baseline and choose the first auto-grade slice.

Work:

- Run the release validation checklist.
- Refresh audit, integrity, mark-event validation, Asterion export, topic-routing, and difficulty summaries.
- Define the first supported slice, preferably simple P1/P3 algebra/calculus items without heavy diagrams or many valid alternative methods.
- Decide whether the parallel project uses this CAIE bank directly or needs an adapter layer.

Pass gate:

- Full tests pass.
- Output integrity passes.
- Mark-event validation passes with `0` errors.
- First-slice candidate list exists with at least `50` candidate subparts or `25` whole questions.
- Every candidate has canonical question and mark-scheme artifacts.

Fail gate:

- Any candidate depends on unreviewed advisory evidence for scoring.
- Any candidate has failed mapping, failed validation, missing image artifacts, or unresolved mark-total mismatch.

### Phase 1: Auto-Grade Contracts And Harness

Soft target: 1 to 2 weeks.

Goal: create schemas and tests before building a scorer.

Work:

- Add an auto-grade contract doc.
- Add builders for `eligible_items.v1.json` and empty/fixture `benchmark_submissions.v1.json`.
- Add validation scripts that fail closed.
- Add fixture attempts for a tiny hand-authored set.
- Add command atlas entries only after commands exist.

Measurable goals:

- `eligible_items` builder can classify all `1301` records.
- At least `95%` of blocked records include actionable block reasons.
- Validation catches missing artifacts, mismatched IDs, bad totals, missing rubric approval, and unsafe use-mode promotion.

Pass gate:

- Targeted tests for eligibility and schema validation pass.
- Full tests pass.
- No canonical outputs are mutated.
- `eligible_items.v1.json` has `0` student-ready items until reviewed rubrics exist.

Fail gate:

- Any generated item is marked student-safe without a reviewed rubric.

### Phase 2: Reviewed Rubric Gold Set

Soft target: 2 to 4 weeks.

Goal: turn machine mark events into a small trusted rubric set.

Work:

- Build a review UI or review queue for mark events.
- Review mark codes, event order, dependencies, alternatives, follow-through rules, and accepted evidence.
- Attach canonical learning-target IDs at event or subpart level.
- Start with the least ambiguous items and explicitly exclude graph sketches, proof-heavy questions, and multi-method questions until later.

Measurable goals:

- `100` reviewed mark-event rubrics or `300` reviewed mark events, whichever comes first.
- `100%` reviewed rubrics have totals matching official marks.
- `100%` reviewed events have valid mark codes and dependency metadata.
- `95%` of reviewed events have at least one learning-target ID.
- Inter-reviewer agreement measured on at least `25` rubrics if a second reviewer is available.

Pass gate:

- Reviewed rubric validator passes.
- No reviewed rubric contains unresolved `unknown_mark_code`, total mismatch, or missing dependency policy.
- Spot-check report shows every awarded mark could be explained as M/A/B/E/DM/FT.

Fail gate:

- More than `2%` of reviewed rubrics require correction after audit.
- Any reviewed rubric total differs from the official mark total.

### Phase 3: Submission Photo Intake

Soft target: 3 to 5 weeks.

Goal: reliably ingest student solution photos before attempting scores.

Work:

- Add attempt ingestion under ignored output roots.
- Detect blur, glare, rotation, crop loss, low contrast, multiple pages, and handwriting density.
- Segment solution regions by item/subpart when possible.
- Produce transcription candidates using OCR/vision, with provenance and confidence.
- Support abstention when capture quality is insufficient.

Measurable goals:

- At least `200` benchmark submission images across `25` reviewed questions/subparts.
- Capture-quality classifier catches at least `95%` of intentionally bad fixture photos.
- Item/subpart alignment accuracy reaches `90%` on the first supported slice.
- Transcription includes provenance, confidence, and image-region links for `100%` of attempts.

Pass gate:

- Ingestion can process benchmark attempts without crashing.
- Low-quality inputs become `abstained` or `needs_review`, not scored.
- Attempt sidecars contain no canonical-bank mutations.

Fail gate:

- Any unreadable or wrong-question submission receives a normal score.

### Phase 4: Scorer V0, Mark-Code Feedback Only

Soft target: 4 to 8 weeks.

Goal: score reviewed rubrics in a lab setting and explain M/A/B/E marks.

Work:

- Implement rubric event matching against student evidence.
- Start with deterministic checks where possible: final numeric/algebraic answer matching, required method patterns, and exact/equivalent expressions.
- Use an LLM or vision model only as evidence extraction or judgment support, with structured output and verifier checks.
- Produce score-event sidecars with mark-by-mark award decisions.
- Keep the mode teacher/reviewer-only.

Measurable goals on the first benchmark set:

- Exact total score accuracy: at least `70%`.
- Within-one-mark score accuracy: at least `85%`.
- Mark-event decision precision: at least `80%`.
- High-confidence over-award rate: below `5%`.
- Abstention rate: below `35%` on supported, readable submissions.
- Explanation traceability: `100%` of awarded marks cite student evidence and official mark evidence.

Pass gate:

- Benchmark report meets all targets.
- Self-check verifier can force abstention.
- No score is produced for unsupported rubrics.

Fail gate:

- Any high-confidence score is emitted without cited evidence.
- Over-awards are concentrated in a mark type or topic without a mitigation plan.

### Phase 5: Verifier And Calibration Loop

Soft target: 4 to 6 weeks after Scorer V0.

Goal: make the grader aware of uncertainty and reduce confident wrong scores.

Work:

- Add scorer/verifier disagreement checks.
- Add dependency-specific tests for `A` after `M`, `DM`, and `FT`.
- Add alternative-method handling.
- Add calibration reports by topic, mark type, paper family, image quality, and rubric complexity.
- Introduce confidence bands: `high`, `medium`, `low`, `abstain`.

Measurable goals:

- Exact total score accuracy: at least `78%`.
- Within-one-mark score accuracy: at least `90%`.
- High-confidence over-award rate: below `3%`.
- High-confidence under-award rate: below `7%`.
- Verifier catches at least `80%` of known seeded scoring errors.
- Regression suite includes at least `500` scored attempts or synthetic variants.

Pass gate:

- Calibration report shows no unsupported topic or mark type is silently mixed into the headline score.
- Confidence bands are empirically meaningful: high-confidence attempts outperform medium-confidence attempts.

Fail gate:

- Verifier disagreement is ignored rather than causing abstention or review.

### Phase 6: Student Self-Check Beta

Soft target: 6 to 10 weeks after calibration.

Goal: expose auto-grading to students as a self-check on a narrow, reviewed slice.

Work:

- Add student-facing explanations with no teacher-debug leakage.
- Show score, confidence, awarded marks, missed marks, and "why" in plain language.
- Make retake/resubmit behavior explicit.
- Add teacher override and feedback capture.
- Log disagreement reports for continuous improvement.

Measurable goals:

- Student beta limited to at least `50` and at most `150` reviewed items.
- Within-one-mark accuracy remains at least `90%` on fresh beta audits.
- High-confidence over-award rate remains below `3%`.
- At least `95%` of beta scores have complete mark explanations.
- At least `90%` of student-visible missed-mark explanations are rated acceptable by teacher review on audit sample.
- Critical bug escape rate: `0` known cases where unsupported submissions receive confident final scores.

Pass gate:

- Teacher audit signs off on a randomized sample of beta attempts.
- Student-facing mode defaults to abstain when confidence or eligibility is insufficient.

Fail gate:

- Any blocked item becomes student-selectable for auto-grading.
- Any explanation invents unsupported mathematical content.

### Phase 7: Learning-Target Feedback

Soft target: 4 to 8 weeks after beta scoring stabilizes.

Goal: move from mark-code explanations to learning-target diagnosis.

Work:

- Map reviewed rubric events to canonical skills or learning targets.
- Distinguish primary target, prerequisite target, and exam-technique target.
- Convert missed marks into learning-target feedback.
- Feed results into practice recommendations only after topic/skill gates pass.
- Keep difficulty index advisory until a separate sequencing gate passes.

Measurable goals:

- `95%` of student-beta rubric events have reviewed learning-target IDs.
- Human agreement on primary learning target is at least `85%` on audit sample.
- Recommendation safety: `0` recommendations use invalid canonical IDs.
- For each suggested next step, the system can cite the missed mark event that caused it.

Pass gate:

- Learning-target feedback validates against canonical taxonomy and reviewed skill maps.
- Topic-routing sidecar is not used for strict targeting unless `safe_for_strict_filters=true`.

Fail gate:

- Feedback uses invented skills or broad AI suggestions as canonical targets.

### Phase 8: Expansion By Component And Method Family

Soft target: ongoing.

Goal: expand beyond the first slice without dropping quality.

Expansion order:

1. P1 algebra, functions, coordinate geometry, binomial, differentiation, integration.
2. P3 algebra, calculus, vectors, complex numbers where methods are structured.
3. S1 numeric probability/statistics items with clear final-answer checks.
4. M1 mechanics items with diagram and modeling dependencies.
5. Graph sketching, proof, explanation-heavy, and highly alternative-method items.

Per-slice pass gate:

- At least `50` reviewed rubrics or `150` reviewed mark events in the slice.
- Benchmark includes at least `5` attempts per rubric where practical.
- Within-one-mark accuracy at least `90%` for student self-check.
- High-confidence over-award rate below `3%`.
- Slice-specific abstention behavior documented.

Fail gate:

- New slice reduces aggregate beta accuracy below the current release threshold.

### Phase 9: Production Hardening

Soft target: after multiple beta slices pass.

Goal: make the system dependable as an ongoing product capability.

Work:

- Add versioned model, prompt, rubric, and benchmark manifests.
- Add cost and latency budgets.
- Add privacy retention policy for student images.
- Add teacher override workflows.
- Add continuous evaluation and drift monitoring.
- Add release gates for every grader model or prompt change.

Measurable goals:

- P95 scoring latency within downstream app requirement.
- Re-score determinism or bounded variance documented.
- Every production score links to grader version, rubric version, verifier version, and benchmark release.
- Every release candidate passes locked regression benchmarks.

Pass gate:

- A grader release cannot ship unless benchmark, verifier, privacy, and role-gate checks pass.

Fail gate:

- Any grader change can affect student scoring without a recorded benchmark comparison.

## Benchmark Strategy

The benchmark must be built before student release.

Required benchmark types:

- Clean correct solutions.
- Clean partially correct solutions.
- Common-error solutions from examiner reports and mark schemes.
- Illegible or low-quality photos.
- Wrong-question submissions.
- Alternative valid methods.
- Follow-through cases.
- Boundary cases where one mark depends on another.
- Blank or irrelevant submissions.

Core metrics:

- `exact_total_accuracy`
- `within_one_mark_accuracy`
- `event_precision`
- `event_recall`
- `high_confidence_over_award_rate`
- `high_confidence_under_award_rate`
- `abstention_rate`
- `wrong_question_detection_rate`
- `explanation_traceability_rate`
- `learning_target_agreement_rate`
- `regression_delta_vs_previous_release`

Minimum reporting slices:

- paper family
- topic or skill
- mark code
- image quality band
- rubric complexity
- handwritten vs typed
- single-page vs multi-page

## Initial Slice Recommendation

Start with a narrow slice that minimizes ambiguity:

- P1 and P3 only.
- Non-diagram or low-diagram-dependence questions.
- One or two subparts per item where possible.
- Mark schemes with clear `M`, `A`, and `B` events.
- Items with high-confidence crops and clean mapping.
- Items with reviewed skill mapping or easy skill-review path.

Avoid in the first student beta:

- graph sketches
- proof-heavy questions
- questions with many alternative methods
- table-heavy statistics questions
- mechanics diagram interpretation
- questions with unresolved follow-through complexity
- any item where the official mark scheme text is partial or review flagged

## Product Modes

### Review Lab

Internal mode for engineers and teachers. Can show debug evidence, scorer/verifier disagreement, raw transcripts, and rubric internals.

Allowed before student beta.

### Teacher Beta

Teacher-controlled scoring for selected items. Scores are marked as machine-assisted and reviewable.

Allowed after Phases 1 to 5 pass.

### Student Self-Check Beta

Student-facing feedback on a narrow reviewed slice. Must abstain aggressively and show confidence.

Allowed after Phase 6 gates pass.

### Student Ready

Broader student-facing use. Requires repeated beta success, stable benchmark results, privacy policy, monitoring, and release gates.

Not an early target.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Handwriting or photo quality hides correct work | False under-awards | Capture-quality gate, abstention, resubmission prompt |
| Model awards marks from hallucinated reasoning | False over-awards | Evidence citation, verifier, high-confidence over-award gate |
| Alternative valid methods are missed | False under-awards | Reviewed alternative-method rubric fields, teacher audit |
| Follow-through marks are mishandled | Wrong partial credit | Explicit dependency and FT policies in reviewed rubric |
| Topic or skill feedback is wrong | Bad learning recommendations | Canonical skill IDs only, reviewed learning-target mappings |
| Student trusts a low-confidence result too much | Product harm | Clear confidence labels, abstention-first UX |
| Benchmark overfits to neat solutions | Poor real-world performance | Include messy photos, common errors, and fresh beta audits |
| Existing advisory fields get over-promoted | Unsafe scoring | Separate auto-grade contract and validators |

## First Implementation Backlog

1. Create `docs/AUTO_GRADING_CONTRACT.md`.
2. Add `src/exam_bank/auto_grade/` with schema constants, eligibility builder, and validators.
3. Add `scripts/build_auto_grade_eligible_items.py`.
4. Add `scripts/validate_auto_grade_eligible_items.py`.
5. Add fixture tests for eligibility pass/block reasons.
6. Build a reviewed-rubric review queue from `question_bank.mark_events.v1.json`.
7. Define the first benchmark submission fixture schema.
8. Add `reports/auto_grade/` summaries for candidate counts, blockers, and next review targets.

## Release Rule

The auto-grader should move one role at a time:

```text
blocked
  -> review_only
  -> teacher_beta
  -> student_self_check_beta
  -> student_ready
```

No phase may skip a role. A higher role is allowed only when the lower role has measured evidence, reviewed artifacts, and passing regression gates.

