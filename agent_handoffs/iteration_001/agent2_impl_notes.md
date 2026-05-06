# Agent 2 Implementation Notes — iteration_001

## Scope

Implemented the approved measurement/reporting batch only. No extraction behavior changed, no OCR thresholds changed, and no tests were edited.

## Files Changed

- `src/exam_bank/audit.py`
  - Added reusable OCR candidate audit helpers.
  - Added stale/incomplete candidate metadata detection.
  - Added score summaries that ignore missing/null scores.
  - Added OCR rejected-reason aggregation.
  - Added suspicious OCR-selected and readiness-inflation risk report sections.
  - Added optional baseline comparison by `question_id`.
- `scripts/audit_ocr_candidates.py`
  - Added standalone report command because `tests/test_runtime_paths.py` intentionally limits the package CLI to `process` and `audit`.

## Command Added

```bash
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json
```

Optional baseline/output form:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py \
  --input output/json/question_bank.json \
  --baseline /path/to/baseline/question_bank.json \
  --json-output /tmp/ocr_candidate_audit.json
```

## Current Bank Report Summary

Command run:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output /tmp/ocr_candidate_audit.json
```

Result:

- Total records: `1301`
- Candidate metadata presence:
  - `text_candidate_source`: `0` present, `1301` missing
  - `text_candidate_decision`: `0` present, `1301` missing
  - `ocr_selected`: `0` present, `1301` missing
  - `native_text_score`: `0` present, `1301` missing
  - `ocr_text_score`: `0` present, `1301` missing
  - `selected_text_score`: `0` present, `1301` missing
- Data quality findings:
  - `candidate_metadata_missing_for_all_records:text_candidate_source,text_candidate_decision,ocr_selected,native_text_score,ocr_text_score,selected_text_score`
  - `stale_or_candidate_unaware_export`
- OCR-selected count: `0`
- Candidate source distribution: `missing: 1301`
- Candidate decision distribution: `missing: 1301`
- OCR rejected reasons: none reportable because candidate metadata is missing
- Score summaries: all candidate score counts are `0`; nulls were not treated as zeroes
- Text fidelity distribution:
  - `clean: 310`
  - `degraded: 938`
  - `unusable: 53`
- Text-only status distribution:
  - `fail: 1043`
  - `ready: 73`
  - `review: 185`
- Visual curation status distribution:
  - `fail: 420`
  - `ready: 217`
  - `review: 664`
- Question text trust distribution:
  - `high: 86`
  - `medium: 224`
  - `low: 938`
  - `unusable: 53`

## Baseline Comparison

No baseline/comparison export was found outside `output/`, matching Agent 1's plan. Baseline comparison is blocked by missing baseline, not by report failure.

## Suspicious Records and Readiness Inflation

No OCR-selected records can be sampled from the current export because all candidate-selection metadata is missing. Therefore:

- Suspicious OCR-selected records: none reportable from this export
- Readiness inflation risk records: none reportable from this export
- Representative OCR sample: unavailable until a fresh candidate-aware export exists

## Tests Run

Passed:

```bash
.venv/bin/python -m pytest tests/test_audit.py tests/test_ocr.py tests/test_output_contract.py -q
```

Result: `17 passed in 0.05s`

Passed:

```bash
.venv/bin/python -m pytest tests/test_ocr.py::test_ocr_enabled_success_returns_low_trust_for_math_heavy_text tests/test_ocr.py::test_ocr_enabled_failure_is_captured_without_crashing -q -vv
```

Result: `2 passed in 0.04s`

Passed:

```bash
.venv/bin/python -m pytest -q
```

Result: `259 passed, 3 skipped in 133.15s`

Order-specific test issue observed but not changed:

```bash
.venv/bin/python -m pytest tests/test_audit.py tests/test_runtime_paths.py tests/test_ocr.py tests/test_output_contract.py -q
```

This failed two OCR monkeypatch tests after `tests/test_runtime_paths.py` reset `exam_bank.*` modules. The same OCR tests pass alone, and the required full suite passes in repository order. I did not edit tests because Agent 2 is not approved to do so.

## Blockers

- `output/json/question_bank.json` is stale or candidate-unaware for OCR candidate-selection measurement.
- No baseline export is available for before/after comparison.
- No OCR-selected representative sample can be produced until the bank is regenerated with candidate metadata.

## Next-Loop Candidates

Evidence-based candidates for Agent 3/4/5 and iteration 2:

- Generate a fresh full-bank export with OCR enabled into a separate output path, then rerun `scripts/audit_ocr_candidates.py`.
- Preserve the current stale export as an explicit baseline only if the human wants before/after comparison against this state.
- If a durable package CLI command is desired, Agent 3 or the human should approve updating the runtime-front-door test contract first.
- Do not tune OCR thresholds until the fresh candidate-aware report shows actual OCR-selected records or high-margin rejected candidates.

## Agent 2 Fix Addendum After Agent 3 NEEDS FIX

Fixed the suspicious OCR-selected audit gap identified by Agent 3. This is reporting-only; no OCR scoring, extraction behavior, crop handling, DeepSeek behavior, topic classification, or trust/readiness logic changed.

Changed:

- `src/exam_bank/audit.py`
  - Added suspicious OCR-selected risk reasons for:
    - `missing_question_number`
    - `missing_marks`
    - `selected_with_hard_failure`

Commands run:

```bash
.venv/bin/python -m pytest tests/test_audit.py -q
```

Result: `6 passed in 0.02s`

```bash
.venv/bin/python -m pytest tests/test_ocr.py tests/test_output_contract.py -q
```

Result: `16 passed in 0.05s`

```bash
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output /tmp/ocr_candidate_audit_agent2_fix.json
```

Result: passed. Current export still has `1301` records with all OCR candidate metadata missing and remains `stale_or_candidate_unaware_export`.

```bash
.venv/bin/python -m pytest -q
```

Result: `264 passed, 3 skipped in 137.79s`

Remaining blocker: Agent 4/5 still need a fresh candidate-aware full-bank export before auditing real OCR selection behavior.
