# Iteration 004 - OCR Candidate Quality Guard

## Initial worktree state

The worktree was already dirty from earlier audit/reporting and iteration_003 work. Initial status before this iteration's OCR selector edit included:

```text
 M README.md
 M agent_handoffs/iteration_001/agent2_impl_notes.md
 M agent_handoffs/iteration_001/agent3_tests.md
 M agent_handoffs/iteration_002/agent3_tests.md
 M docs/ROADMAP.md
 M tests/test_ocr.py
?? ROADMAP.md
?? agent_handoffs/iteration_003/
?? scripts/audit_question_bank_readiness.py
?? tests/test_question_bank_readiness_audit.py
```

No pre-existing work was discarded or reverted.

## Scope

iteration_003 established that the canonical repo export at `output/json/question_bank.json` is still native-only by default, while OCR works when the pipeline is run with `--enable-ocr`. This iteration followed the recommended next step: audit OCR candidate quality on an OCR-enabled full-bank export and apply the smallest selector fix for a clear false-positive pattern.

## Files inspected

- `src/exam_bank/ocr.py`
- `tests/test_ocr.py`
- `output/audits/iteration_003_ocr_candidate/audit_summary.json`
- `output/audits/iteration_003_ocr_candidate/ocr_candidate_summary.json`
- `output/audits/iteration_003_ocr_candidate/ocr_suspicious_records.csv`
- `output/audits/iteration_003_ocr_candidate/possible_ocr_false_negatives.csv`
- `output_ocr_candidate/json/question_bank.json`

## Current observed OCR state

The canonical export remains native-only unless regenerated with OCR enabled:

```text
output/json/question_bank.json
record_count=1301
ocr_ran=0
ocr_text_nonblank=0
ocr_selected=0
ocr_engine=<blank>: 1301
```

The OCR-enabled reference export has populated OCR text and candidate metadata:

```text
output_ocr_candidate/json/question_bank.json
record_count=1301
ocr_ran=1301
ocr_text_nonblank=1301
ocr_engine=tesseract: 1301
```

Candidate selection metadata such as `ocr_selected`, `text_candidate_source`, and `text_candidate_decision` is preserved under `notes` and read by the audit layer.

## Root cause

The OCR candidate audit found five OCR-selected records where:

- native text contained the expected question number,
- OCR text omitted the expected question number,
- the question scope was not failed,
- OCR was selected because `_ocr_missing_question_number_is_tolerable(...)` allowed the missing number if the OCR otherwise looked structurally usable.

Affected selected false positives were:

- `12summer23_q08`
- `13summer23_q01`
- `32spring23_q07`
- `41autumn23_q05`
- `42summer21_q06`

This tolerance was appropriate for hard scope recovery, but too broad for clean or review scope records because it allowed OCR to replace native text while dropping the question anchor.

## Minimal fix applied

Updated `select_text_candidate(...)` in `src/exam_bank/ocr.py` so a missing expected question number can only be tolerated when `scope_quality_status == "fail"`.

Before:

```python
if _ocr_missing_question_number_is_tolerable(ocr, expected_mark_count, expected_subparts):
    reasons.append("ocr_missing_question_number_tolerated")
```

After:

```python
if scope_quality_status == "fail" and _ocr_missing_question_number_is_tolerable(ocr, expected_mark_count, expected_subparts):
    reasons.append("ocr_missing_question_number_tolerated")
```

This preserves the existing scope-failure recovery path while rejecting OCR missing the question number for clean/review scope records.

## Why the fix is safe

- It only tightens OCR candidate selection.
- It does not change OCR execution, OCR preprocessing, crop detection, mark-scheme mapping, readiness gates, or Asterion tier semantics.
- It does not make OCR text canonical by default.
- It preserves the existing test case where OCR may be selected when native scope failed.
- The full-bank OCR rerun showed no worsened baseline comparison records against the previous OCR-enabled candidate export.

## Before/after audit summary

Before, using `output_ocr_candidate/json/question_bank.json`:

```text
record_count=1301
ocr_ran_count=1301
ocr_selected_count=35
ocr_text_nonblank=1301
ocr_engine_distribution=tesseract: 1301
text_candidate_source=native: 1266, ocr: 35
text_candidate_decision=native_retained: 1266, ocr_selected: 35
suspicious_ocr_selected_count=25
selected_ocr_missing_expected_question_number=5
possible_ocr_false_negative_count=281
mapping_status=pass: 1280, fail: 21
validation_status=pass: 917, review: 370, fail: 14
hard_blocker_count=28
tiers=0: 28, 1: 359, 2: 625, 3: 16, 4: 37, 5: 236
```

After, using `output/iteration_004_ocr_guard_candidate/json/question_bank.json`:

```text
record_count=1301
ocr_ran_count=1301
ocr_selected_count=30
ocr_text_nonblank=1301
ocr_engine_distribution=tesseract: 1301
text_candidate_source=native: 1271, ocr: 30
text_candidate_decision=native_retained: 1271, ocr_selected: 30
suspicious_ocr_selected_count=20
selected_ocr_missing_expected_question_number=0
possible_ocr_false_negative_count=286
mapping_status=pass: 1280, fail: 21
validation_status=pass: 917, review: 370, fail: 14
hard_blocker_count=28
tiers=0: 28, 1: 359, 2: 624, 3: 14, 4: 38, 5: 238
```

Top OCR rejected reasons before:

```text
ocr_not_clearly_better=1097
ocr_lost_mark_brackets=83
page_furniture_or_header_text=82
ocr_missing_subpart_labels=17
ocr_missing_question_number=6
```

Top OCR rejected reasons after:

```text
ocr_not_clearly_better=833
ocr_missing_question_number=289
ocr_lost_mark_brackets=83
page_furniture_or_header_text=82
ocr_missing_subpart_labels=17
```

Score summaries:

```text
native_text_score mean=87.595 median=87 p25=66 p75=114 min=-12 max=172
ocr_text_score mean=82.931 median=82 p25=59 p75=109 min=-42 max=177
selected_text_score before mean=88.666 median=89 p25=66 p75=115 min=-12 max=172
selected_text_score after mean=88.537 median=89 p25=66 p75=115 min=-12 max=172
```

The baseline comparison from the OCR candidate audit changed exactly five records, all the missing-question-number OCR selections listed above. Changed fields were `ocr_selected`, `question_text`, `question_text_trust`, `text_candidate_source`, `text_fidelity_status`, and `text_only_status`. The comparison reported five improved records and zero worsened records.

## Tests added or updated

Updated `tests/test_ocr.py` with:

- `test_candidate_selection_rejects_missing_question_number_when_native_scope_is_clean`

Existing coverage still protects the allowed scope-failure recovery path:

- `test_candidate_selection_can_use_clean_ocr_when_native_scope_failed`

## Commands run

```text
.venv/bin/python scripts/audit_ocr_candidates.py --input output_ocr_candidate/json/question_bank.json --baseline output_ocr_candidate/triage/iteration_004/baseline_question_bank.json --json-output output/audits/iteration_003_ocr_candidate/ocr_candidate_summary.json
.venv/bin/python -m pytest tests/test_ocr.py -q
.venv/bin/python -m exam_bank.cli process --input input --output output/iteration_004_ocr_guard_candidate --enable-ocr
.venv/bin/python scripts/audit_question_bank_readiness.py --input output/iteration_004_ocr_guard_candidate/json/question_bank.json --baseline output_ocr_candidate/json/question_bank.json --artifact-root output/iteration_004_ocr_guard_candidate --out-dir output/audits/iteration_004_ocr_guard_candidate
.venv/bin/python scripts/audit_ocr_candidates.py --input output/iteration_004_ocr_guard_candidate/json/question_bank.json --baseline output_ocr_candidate/json/question_bank.json --json-output output/audits/iteration_004_ocr_guard_candidate/ocr_candidate_summary.json
```

Validation commands are recorded in the final response for this iteration.

## Generated output status

Generated full-bank OCR output and audit reports were written under ignored `output/` paths:

- `output/iteration_004_ocr_guard_candidate/`
- `output/audits/iteration_004_ocr_guard_candidate/`
- `output/audits/iteration_004_sample_13summer23/`

These generated outputs are for verification only and should not be committed.

## Remaining risks

- The canonical export remains native-only until intentionally regenerated with `--enable-ocr`.
- The fix increases `possible_ocr_false_negative_count` from 281 to 286 because five records now retain native text rather than OCR text that omitted the question number.
- Twenty suspicious OCR-selected records remain, mostly due to low question crop confidence.
- `13summer25_q05` remains a high-priority suspicious OCR selection because it is OCR-selected while mapping, validation, and scope are not pass/clean.
- This iteration does not evaluate OCR quality broadly and does not tune OCR thresholds.

## Recommended next iteration

Investigate the remaining suspicious OCR-selected records, starting with the single OCR-selected record that also has non-pass mapping/validation/scope status. After that, review the 20 low-crop-confidence OCR selections to decide whether crop confidence should become a selector guard or an audit-only warning.
