# Agent 1 Plan — Asterion Content Lab P3 70% Pass Loop

Task 5/27: Build an agentic loop that moves Asterion Content Lab P3 candidates toward a verified ≥70% pass rate on a representative P3 sample.

## 1. Current state

The previous iteration handoffs were built around OCR/readiness audit infrastructure. Keep the useful audit discipline, but change the goal. The next loop is not about OCR selection quality. It is about whether Asterion Content Lab candidate records can pass the gates required for safe teacher/content-lab use.

Known current constraints:

- Exam-bank remains image-first. Canonical question and mark-scheme PNGs are the source of truth.
- Asterion student runtime must not consume unreviewed Content Lab candidates.
- The Content Lab candidate export is an internal candidate queue, not a student-facing source of truth.
- Prior audit work found the bank has 1301 records, mapping/validation blockers, Asterion tier distributions, and 920 simple future-fillable subpart-mark records. Use those as leverage, not as proof of readiness.
- The key blocker class to attack now is likely schema/gate mismatch: the new exam-bank export carries gate authority through fields such as `source_skill_review_gate`, `mapping_review_gate`, and `mapping_review_satisfied`, while Asterion-side validation may still expect older reviewed skill-map or candidate-selection fields.

## 2. Iteration target

Create or modify the agent workflow so each loop can:

1. Audit `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json` or the current candidate export path.
2. Restrict the measurement to P3 candidates.
3. Build a deterministic, stratified P3 sample.
4. Validate each sampled candidate against the current Asterion Content Lab gates.
5. Identify blocker classes preventing candidates from passing.
6. Repair the highest-leverage blocker class without weakening gates.
7. Regenerate or revalidate the candidate export.
8. Report whether at least 70% of the sampled P3 candidates pass.

The target is a verified sample pass rate, not a claim that 70% of the whole bank is student-ready.

## 3. Definition of “pass”

A P3 Content Lab candidate counts as passing only if all of the following are true:

- It is a P3 candidate and can be traced to a valid canonical `question_id`.
- Canonical question image path exists.
- Canonical mark-scheme image path exists, unless the candidate is explicitly marked blocked because the mark scheme is missing.
- Source topic/skill mapping resolves to a reviewed Asterion P3 skill-map ID or an explicitly accepted bridge from exam-bank canonical skill ID to Asterion skill ID.
- Mapping review gate is satisfied by current accepted fields, not by a fake legacy shim.
- Validation status is pass or equivalent for the Content Lab gate being measured.
- It is not marked as blocked, quarantined, ambiguous, unsafe, review-only, or missing required evidence.
- If it uses mark events or subpart marks, the mark evidence is present and internally consistent.
- It does not rely on generated text, OCR text, topic guesses, or difficulty labels as canonical proof.

Optional stronger tiers may be reported separately:

- `teacher_preview_ready`: safe for teacher/content-lab review.
- `generation_seed_ready`: safe as a generation seed after reviewed gates pass.
- `student_runtime_ready`: do not use this tier unless there is an explicit student-runtime contract and full independent approval.

## 4. Non-goals

- Do not unlock candidates directly into student runtime.
- Do not lower trust gates to hit 70%.
- Do not treat text as source of truth.
- Do not invent curriculum IDs or infer skill mappings without reviewed evidence.
- Do not change unrelated OCR selection thresholds.
- Do not run broad topic/difficulty reruns unless they are directly required to repair Content Lab gates.
- Do not commit generated exports, PDFs, screenshots, or large output folders unless the human explicitly asks for a snapshot.
- Do not claim whole-bank readiness from a sample.

## 5. Required sample design

The sample must be deterministic and hard to game.

Minimum sample rules:

- Use only P3 candidates.
- Include all nine Asterion regions if eligible candidates exist: Algebra Vault, Logarithm Observatory, Trigonometry Spire, Argand Atrium, Calculus Cliffs, Integral Terraces, Vectors Gate, Iteration Forge, Differential Shrine.
- Include multiple paper/session families where possible.
- Include both likely-pass and likely-fail candidates.
- Include boundary cases: mapping review borderline, mark-event present but incomplete, simple subpart mark records, candidate records blocked by legacy schema mismatch, and missing mark-scheme cases.
- Persist the sample IDs so Agent 4 and Agent 5 review the same sample.

Recommended minimum: 100 P3 candidates if available. If fewer than 100 eligible P3 candidates are available, use all eligible candidates and clearly state the denominator.

Pass-rate formula:

```text
p3_content_lab_sample_pass_rate = passed_sample_records / total_sample_records
```

Target:

```text
p3_content_lab_sample_pass_rate >= 0.70
```

## 6. Files/modules likely involved

Agent 2 should inspect before changing anything:

- `README.md`
- `pyproject.toml`
- `output/json/question_bank.json`
- `output/json/question_bank.topic_routing.v1.json`
- `output/asterion/exports/latest/asterion_question_bank_v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Any existing Asterion export, validation, or Content Lab candidate modules under `src/exam_bank/`
- Any existing tests covering Asterion export contracts, topic routing, skill mapping, mark events, or Content Lab validation
- Prior audit/report scripts, especially readiness audit infrastructure that can be reused without dragging OCR scope into this iteration

Likely editable areas:

- A small deterministic Content Lab readiness audit script, probably under `scripts/`.
- A content-lab validation module if one already exists.
- Export/schema adapter code only if the mismatch is real and the code change is narrow.
- Tests for Content Lab candidate pass/fail gates and sample pass-rate reporting.

## 7. Loop structure

Each loop should use this sequence:

1. Baseline: run the Content Lab readiness audit against the current candidate export.
2. Classify: produce blocker counts by reason and by region.
3. Select target: choose the blocker class with the highest expected pass-rate improvement and lowest safety risk.
4. Patch: repair only that blocker class.
5. Rebuild/revalidate: regenerate the Asterion export or rerun validation, depending on repo structure.
6. Measure: rerun the same deterministic sample and a fresh full P3 blocker summary.
7. Guard: run focused tests, then full tests if the patch touches production validation/export logic.
8. Handoff: write a concise iteration note with changed files, validation commands, pass rate, blockers remaining, and whether another loop should run.

## 8. Priority blocker classes

Attack in this order unless repo evidence contradicts it:

1. Schema/gate mismatch between new exam-bank candidate fields and Asterion’s legacy validator expectations.
2. Reviewed Asterion P3 skill-map ID bridge from exam-bank canonical skill IDs.
3. Mapping review gate satisfaction (`mapping_review_gate`, `mapping_review_satisfied`, or equivalent).
4. Missing or inconsistent mark-scheme evidence.
5. Local mark-total/subpart mark blockers that can be filled from existing `mark_values_detected` or mark-event evidence.
6. Crop/path artifact blockers where canonical image files exist but exported paths or artifact-root resolution are wrong.
7. Quarantine ambiguous multi-topic candidates rather than promoting them.

## 9. Agent responsibilities

Agent 1: Own this plan and the success metric.

Agent 2: Implement the audit/repair pass.

Agent 3: Add tests and regression guards.

Agent 4: Run integration audit against generated outputs and repo hygiene.

Agent 5: Decide whether the loop succeeded, failed safely, or should continue.

## 10. Acceptance criteria

- A deterministic P3 Content Lab sample exists and is persisted.
- A baseline pass rate is reported.
- The implementation repairs at least one real blocker class or honestly reports why no safe repair was possible.
- Revalidation reports current sample pass rate.
- The sample pass rate reaches ≥70%, or the remaining blockers are reported in ranked order for the next loop.
- No student-runtime data path is changed unless explicitly approved.
- No trust gates are weakened.
- Tests pass.
- Generated artifacts are not accidentally tracked.

## 11. Stop conditions

Stop and report instead of forcing the target if any of these occur:

- Passing requires weakening gates or faking reviewed evidence.
- The sample can only hit 70% by excluding hard categories without documenting that exclusion.
- Candidate records cannot be traced to canonical question and mark-scheme images.
- The patch would make unreviewed generated content student-facing.
- Asterion skill mapping is invented rather than reviewed or explicitly bridged.
- Full tests fail.
- Export generation fails in a way that invalidates measurement.
- Generated large artifacts, secrets, or local-only files become tracked.

## 12. Required final summary for every loop

At the end of the loop, summarize in plain English:

- What was done.
- Why it was done.
- Files changed or created.
- Validation commands run and results.
- Baseline pass rate and final pass rate.
- What blocker class moved the metric.
- What remains blocked.
- Risks or concerns.
- Suggested next step.
