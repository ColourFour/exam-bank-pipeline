# Agent 5 Final Review — iteration_002

## Verdict

PASS for iteration 2 as a quantified difficulty contract.

This is not an OCR acceptance. OCR candidate-quality review remains deferred from iteration 1 and must carry forward to iteration 3.

## Reviewed Items

- Iteration 2 handoffs and prompts:
  - `agent_handoffs/iteration_002/agent1_plan.md`
  - `agent_handoffs/iteration_002/Prompt/agent5_prompt.md`
  - `agent_handoffs/iteration_002/agent2_impl_notes.md`
  - `agent_handoffs/iteration_002/agent3_tests.md`
  - `agent_handoffs/iteration_002/agent4_integration.md`
- Difficulty implementation:
  - `src/exam_bank/classification.py`
  - `src/exam_bank/classification_models.py`
  - `src/exam_bank/models.py`
  - `src/exam_bank/pipeline.py`
  - `src/exam_bank/exporters.py`
  - `src/exam_bank/audit.py`
  - `scripts/audit_difficulty.py`
- Difficulty and export tests:
  - `tests/test_classification.py`
  - `tests/test_output_contract.py`
  - `tests/test_audit.py`
  - existing DeepSeek compatibility coverage through the full suite
- Repo hygiene:
  - `git status --short`
  - generated/export-like tracked and untracked artifact checks

## Checks Run

```bash
.venv/bin/python -m pytest
```

Result:

```text
269 passed, 3 skipped in 137.15s (0:02:17)
```

Artifact hygiene checks showed no generated full-bank export, JSON output, images, CSVs, env files, or secret/key-looking files newly visible to git. The tracked `output` content remains `output/json/.gitkeep`; tracked PDFs are the existing input corpus.

## Acceptance Findings

- Full tests pass: PASS.
- Difficulty is no longer only a coarse label: PASS. Local classification now returns `difficulty_score`, `difficulty_band`, `difficulty_score_scale`, `difficulty_features`, `difficulty_review_flags`, and `difficulty_model_version` in addition to the existing coarse label/confidence/evidence fields.
- Numeric score is bounded, deterministic, documented, and exported: PASS. The score is an integer clamped to `0..100`, uses documented scale `0-100`, and has deterministic fixture coverage. Top-level JSON exports `difficulty_score` and `difficulty_band`; `notes` exports the score scale, features, flags, and model version.
- Labels remain compatible: PASS. Canonical labels remain `easy`, `average`, `difficult`; DeepSeek normalization compatibility is retained by the passing suite.
- Coarse label derives from the score: PASS. Score bands are `0-34 easy`, `35-69 average`, `70-100 difficult`; tests assert `difficulty == difficulty_band` and expected band derivation.
- Evidence/features explain the score: PASS. Features include marks, structure, cognitive demand, mathematical density, mixed-topic complexity, topic prior, and trust; the human-readable evidence string remains exported.
- Weak text, missing marks, or failed evidence lower confidence or add flags: PASS. Missing marks add `marks_missing_for_difficulty`; low text quality produces low difficulty confidence and `difficulty_uncertain`; low topic confidence reduces difficulty confidence.
- Generated artifacts are not tracked: PASS.

## Reasons To Reject Considered

- Reject if this were field-only plumbing with no scoring behavior. Not found: scoring uses feature contributions before deriving the band.
- Reject if the numeric score could escape the declared range. Not found: `_normalize_difficulty_score` clamps to `0..100`, and tests cover bounds.
- Reject if labels changed to incompatible user-facing aliases. Not found: canonical values remain unchanged.
- Reject if weak or missing evidence could still claim high-confidence difficulty. Not found in tested paths; missing marks and degraded text lower confidence and add review flags.
- Reject if generated bank artifacts were tracked. Not found.
- Reject if OCR thresholds or OCR candidate selection changed under this iteration. Not found; OCR remains outside iteration 2 scope.

## Blockers And Deferrals

No blockers for accepting the iteration 2 code contract.

Deferred:

- The existing full-bank export is stale and still lacks difficulty metadata for all 1301 records, so bank-level score distribution cannot yet be audited from `output/json/question_bank.json`.
- No psychometric calibration against real student performance data was attempted.
- OCR candidate-quality validation remains deferred exactly as in iteration 1.

## Recommended Iteration 3 Target

Iteration 3 should generate a fresh OCR-enabled, candidate-aware full-bank or representative export without committing generated artifacts, then run:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json
```

Carry forward Agent 5 iteration 1's OCR recommendation: validate real OCR candidate-selection quality on a fresh export, including OCR-selected records and high-margin rejected examples, before any OCR threshold tuning. In the same pass, audit the newly exported difficulty score distribution by paper family, topic, marks bucket, label, score range, confidence, and review flags.
