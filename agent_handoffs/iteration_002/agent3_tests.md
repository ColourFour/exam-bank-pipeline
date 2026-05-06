# Agent 3 Test Report — iteration_002

## Scope

Verified Agent 2's quantified difficulty implementation against the Agent 1 plan and Agent 3 prompt.

Reviewed implementation paths:

- `src/exam_bank/classification.py`
- `src/exam_bank/classification_models.py`
- `src/exam_bank/models.py`
- `src/exam_bank/pipeline.py`
- `src/exam_bank/exporters.py`
- `src/exam_bank/audit.py`
- `scripts/audit_difficulty.py`
- `tests/test_classification.py`
- `tests/test_output_contract.py`
- `tests/test_audit.py`
- `tests/test_deepseek_enrich.py`

## Tests Added Or Reviewed

Added one focused regression test in `tests/test_classification.py`:

- `test_numeric_difficulty_is_bounded_deterministic_and_drives_label`

This checks multiple public `classify_question` fixtures for deterministic score/features/flags, score bounds `0..100`, and exact label/band derivation from score thresholds:

- `0-34`: `easy`
- `35-69`: `average`
- `70-100`: `difficult`

Reviewed existing Agent 2 tests covering:

- direct low-mark routine questions remain easy
- linked/mixed/high-mark questions score higher
- proof-style high-mark questions can score difficult
- missing marks and degraded text lower confidence and add review flags
- export includes top-level and notes difficulty metadata
- audit reports difficulty distributions and missing metadata
- DeepSeek `medium -> average` and `hard -> difficult` normalization remains compatible

## Commands And Results

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_classification.py tests/test_output_contract.py tests/test_audit.py tests/test_deepseek_enrich.py
```

Result:

```text
66 passed in 0.50s
```

Full suite:

```bash
.venv/bin/python -m pytest
```

Result:

```text
269 passed, 3 skipped in 134.29s (0:02:14)
```

Audit command smoke check:

```bash
.venv/bin/python scripts/audit_difficulty.py --help
```

Result:

```text
exited 0 and printed --input / --json-output usage
```

## Required Check Findings

- Score bounded/deterministic: PASS. Added focused test confirms deterministic public-classifier score, features, flags, and `0..100` bounds across easy/average/difficult-style fixtures.
- Coarse label derives consistently from score: PASS. Added focused test confirms `difficulty == difficulty_band` and both match the configured score bands.
- Missing marks and weak/degraded text lower confidence or add review flags: PASS. Existing test confirms `marks_missing_for_difficulty`, low confidence, and uncertainty handling for degraded text.
- Direct low-mark routine questions remain easy: PASS. Existing test confirms a 3-mark routine trig question stays `easy` with score in `0..34`; routine differentiation also remains lower than linked mixed work.
- Linked/proof/mixed/high-mark questions score higher: PASS. Existing tests confirm linked mixed work scores above routine work, and proof-style high-mark complex-number work reaches `difficult`.
- Exported JSON includes difficulty fields: PASS. Existing output contract tests verify top-level `difficulty`, `difficulty_score`, `difficulty_band`, and explanatory `notes` fields.
- DeepSeek difficulty normalization remains compatible: PASS. Existing DeepSeek tests passed, including `medium -> average` and `hard -> difficult` normalization cases.

## Scope Guard

No OCR scoring, OCR threshold, candidate selection, DeepSeek provider behavior, topic taxonomy, mark-scheme mapping, crop detection, or full-bank export regeneration was introduced by my verification pass.

Agent 2's implementation appears scoped to deterministic difficulty scoring, data threading, JSON export, audit reporting, and focused tests. The implementation is not field-only plumbing: scoring behavior uses marks, structure, cognitive demand, mathematical density, mixed-topic complexity, topic prior, and trust modifiers before deriving the coarse label from the numeric score.

## Repo Hygiene

`git status --short` after verification shows existing modified/untracked files from Agent 2 plus my focused test/report additions:

```text
 M src/exam_bank/audit.py
 M src/exam_bank/classification.py
 M src/exam_bank/classification_models.py
 M src/exam_bank/exporters.py
 M src/exam_bank/models.py
 M src/exam_bank/pipeline.py
 M tests/test_audit.py
 M tests/test_classification.py
 M tests/test_output_contract.py
?? agent_handoffs/iteration_002/
?? scripts/audit_difficulty.py
```

I did not revert or overwrite edits by other agents. No generated full-bank export was committed.

## Verdict

PASS.

Agent 2's quantified difficulty implementation has meaningful deterministic scoring behavior, not just additive fields. Focused tests and the full suite pass.
