# Agent 3 Test Report - iteration_002 readiness audit coverage

## Scope

Added focused synthetic-fixture coverage for `scripts/audit_question_bank_readiness.py`.

This pass did not change extraction behavior, OCR selection logic, crop detection, mark-scheme mapping, topic classification, difficulty classification, export semantics, or generated full-bank outputs.

## Tests Added

Extended `tests/test_question_bank_readiness_audit.py` from 3 tests to 12 tests.

New coverage protects:

- Field resolution across top-level and `notes` fields, including top-only, notes-only, equal duplicates, conflicting duplicates, and missing values.
- Field disagreement reporting via `field_disagreement_count`, `field_disagreements_sample`, and `field_presence_report.csv`.
- Missing optional OCR/readiness fields, including missing `ocr_text`, `ocr_ran`, `ocr_selected`, score fields, candidate source/decision fields, rejected reasons, visual/text readiness fields, trust, and crop-confidence fields.
- Numeric score summaries with valid numeric scores, missing fields, nulls, and non-numeric score-like values. Missing/non-numeric values remain missing and are not converted to fake zeroes.
- OCR inactive/candidate-aware state, OCR selected state, OCR rejected state, suspicious OCR-selected output, and possible OCR false-negative output.
- Suspicious OCR-selected flags for missing question number, missing mark brackets, mapping fail, validation fail, low question crop confidence, and OCR much shorter than native text.
- Possible OCR false-negative flags for higher OCR score, high OCR trust with low selected text trust, OCR fixing merged/sparse native text, OCR preserving structure better than native, and a positive margin with only `ocr_not_clearly_better`.
- Readiness tier classification exactly as implemented by the audit script for Tier 0, Tier 2, Tier 3, Tier 4, and Tier 5.
- Hard blocker reasons for missing question ID, missing question image path, missing mark-scheme image path, mapping fail, validation fail, question/mark-scheme total mismatch, and missing artifacts under `--artifact-root`.
- Mapping and validation distributions for `pass`, `review`, `fail`, and missing.
- Subpart mark fillability reasons for simple-fillable marks, wrong mark sums, likely nested/count-mismatch cases, and already-populated subpart marks.
- Baseline comparison by stable `question_id`, including added/removed/shared records, exact vs normalized question-text changes, OCR-selection changes, status movement ordering, mapping/validation movement, and tier movement.
- Deterministic report output by rerunning the audit on the same tiny fixture and comparing generated JSON/CSV bytes.

All tests use small synthetic JSON fixtures under `tmp_path`. They do not require the full `output/json/question_bank.json`, OCR engines, APIs, network access, PDFs, rendered PNGs, DeepSeek, OpenAI, or external services.

## Tiny Refactors

None. The tests exercise the existing CLI/reporting boundary and generated reports directly.

## Commands Run

Passed:

```bash
.venv/bin/python -m py_compile scripts/audit_question_bank_readiness.py
```

Passed:

```bash
.venv/bin/python -m pytest tests/test_question_bank_readiness_audit.py -q
```

Result:

```text
12 passed in 0.13s
```

Passed:

```bash
.venv/bin/python -m pytest tests/test_audit.py tests/test_ocr.py tests/test_output_contract.py tests/test_extraction_structure.py -q
```

Result:

```text
34 passed in 0.09s
```

Passed:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
337 passed, 3 skipped in 112.07s (0:01:52)
```

## Remaining Gaps

- These tests protect the reporting layer with synthetic data; they intentionally do not prove OCR activation on the canonical 1301-record export.
- OCR activation/export disconnect remains an iteration_003 task.
- No OCR threshold tuning, crop recovery, mapping repair, topic/difficulty rerun, subpart mark promotion, run manifest work, or Asterion export slicing was attempted.

## Repo Hygiene

Generated audit reports under `output/audits/` remain ignored by git. This pass only extends focused tests and this handoff note.

## Verdict

PASS.

The readiness-audit reporting layer now has focused regression coverage for the highest-risk behaviors needed before iteration_003 investigates the OCR activation/export disconnect.

---

# Prior Agent 3 Test Report - iteration_002 difficulty scoring

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
