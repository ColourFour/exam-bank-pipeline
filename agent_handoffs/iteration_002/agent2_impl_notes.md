# Agent 2 Implementation Notes — iteration_002

## Scope implemented

- Added deterministic local numeric difficulty metadata while preserving canonical labels: `easy`, `average`, `difficult`.
- Threaded difficulty details through `DifficultyDecision`, `ClassificationResult`, `QuestionClassificationState`, `QuestionRecord`, pipeline record construction, and JSON export.
- Added focused scoring/export/audit tests.
- Added a small difficulty audit command following the existing audit/script pattern.

## Difficulty model contract

- `difficulty_score`: integer from `0` to `100`.
- `difficulty_score_scale`: `0-100`.
- `difficulty_band`: derived from score using:
  - `0-34`: `easy`
  - `35-69`: `average`
  - `70-100`: `difficult`
- `difficulty`: remains the canonical internal label and matches the derived band for local classification.
- `difficulty_model_version`: `local-difficulty-v1`.
- `difficulty_features`: structured feature groups for marks, structure, cognitive demand, mathematical density, mixed-topic complexity, topic prior, and trust.
- `difficulty_review_flags`: difficulty-specific flags such as `marks_missing_for_difficulty` and `difficulty_uncertain`.

## JSON export fields

Top-level question fields added:

- `difficulty`
- `difficulty_score`
- `difficulty_band`

`notes` fields added:

- `difficulty`
- `difficulty_confidence`
- `difficulty_evidence`
- `difficulty_uncertain`
- `difficulty_score`
- `difficulty_score_scale`
- `difficulty_features`
- `difficulty_review_flags`
- `difficulty_model_version`

## Schema compatibility decision

I kept `QUESTION_BANK_SCHEMA_VERSION = 2`. The export changes are additive and preserve existing field names and canonical label values.

## Audit command

Added:

```bash
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json
```

The audit reports label/band/confidence distributions, score summaries, score buckets, distributions by paper family/topic/marks bucket, review flag counts, and missing metadata counts.

## Tests run

```bash
.venv/bin/python -m pytest tests/test_classification.py tests/test_output_contract.py tests/test_audit.py
```

Result: `38 passed in 0.19s`.

```bash
.venv/bin/python -m pytest
```

Result: `268 passed, 3 skipped in 141.47s (0:02:21)`.

```bash
.venv/bin/python scripts/audit_difficulty.py --help
```

Result: exited `0` and printed the expected `--input` / `--json-output` usage.

## Files changed

- `agent_handoffs/iteration_002/agent2_impl_notes.md`
- `scripts/audit_difficulty.py`
- `src/exam_bank/audit.py`
- `src/exam_bank/classification.py`
- `src/exam_bank/classification_models.py`
- `src/exam_bank/exporters.py`
- `src/exam_bank/models.py`
- `src/exam_bank/pipeline.py`
- `tests/test_audit.py`
- `tests/test_classification.py`
- `tests/test_output_contract.py`

## Deferred work

- No calibration against student performance data was attempted.
- No full-bank regeneration was committed.
- OCR selection, OCR thresholds, DeepSeek behavior, topic taxonomy, mark-scheme mapping, crop detection, and adaptive trainer behavior were left unchanged.
