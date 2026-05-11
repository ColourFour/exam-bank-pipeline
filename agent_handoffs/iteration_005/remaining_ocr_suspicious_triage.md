# Iteration 005 - Remaining OCR Suspicious Triage

## Initial worktree state

Captured before edits:

```text
git status --short
# no output

git diff --name-only
# no output
```

The worktree was clean at the start of this iteration. No existing user work was discarded or overwritten.

## Files inspected

Required source, test, handoff, and docs files inspected:

- `src/exam_bank/ocr.py`
- `tests/test_ocr.py`
- `scripts/audit_question_bank_readiness.py`
- `tests/test_question_bank_readiness_audit.py`
- `agent_handoffs/iteration_003/ocr_activation_export_disconnect.md`
- `agent_handoffs/iteration_004/ocr_candidate_quality_guard.md`
- `README.md`
- `ROADMAP.md`

Required iteration_004 audit artifacts were present under `output/audits/iteration_004_ocr_guard_candidate/` rather than `output/audits/iteration_004/`:

- `audit_summary.md`
- `audit_summary.json`
- `ocr_suspicious_records.csv`
- `ocr_candidate_audit.csv`
- `possible_ocr_false_negatives.csv`
- `baseline_comparison.csv`

Additional generated JSON inspected:

- `output/json/question_bank.json`
- `output_ocr_candidate/json/question_bank.json`
- `output/iteration_004_ocr_guard_candidate/json/question_bank.json`
- `output/iteration_005_ocr_triage_candidate/json/question_bank.json`

## OCR-enabled audit source used

Before source:

```text
input JSON: output/iteration_004_ocr_guard_candidate/json/question_bank.json
audit dir:  output/audits/iteration_004_ocr_guard_candidate
baseline:   output_ocr_candidate/json/question_bank.json
```

After source:

```text
input JSON: output/iteration_005_ocr_triage_candidate/json/question_bank.json
audit dir:  output/audits/iteration_005
baseline:   output/iteration_004_ocr_guard_candidate/json/question_bank.json
```

The fresh OCR-enabled export was generated with:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output/iteration_005_ocr_triage_candidate --enable-ocr
```

The fresh readiness audit was generated with:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/iteration_005_ocr_triage_candidate/json/question_bank.json \
  --baseline output/iteration_004_ocr_guard_candidate/json/question_bank.json \
  --artifact-root output/iteration_005_ocr_triage_candidate \
  --out-dir output/audits/iteration_005
```

## Remaining suspicious OCR-selected records

Iteration_004 baseline had 20 suspicious OCR-selected records:

- `11autumn22_q10`
- `12spring22_q05`
- `12spring22_q08`
- `12spring22_q10`
- `12summer21_q02`
- `12summer21_q03`
- `13summer25_q05`
- `41summer21_q03`
- `41summer21_q04`
- `41summer22_q05`
- `42spring21_q02`
- `42summer22_q04`
- `43autumn21_q04`
- `43summer22_q05`
- `51autumn21_q01`
- `51summer21_q05`
- `51summer21_q07`
- `52spring21_q05`
- `52summer22_q05`
- `53summer21_q06`

After the iteration_005 guard, 19 suspicious OCR-selected records remain:

- `11autumn22_q10`
- `12spring22_q05`
- `12spring22_q08`
- `12spring22_q10`
- `12summer21_q02`
- `12summer21_q03`
- `41summer21_q03`
- `41summer21_q04`
- `41summer22_q05`
- `42spring21_q02`
- `42summer22_q04`
- `43autumn21_q04`
- `43summer22_q05`
- `51autumn21_q01`
- `51summer21_q05`
- `51summer21_q07`
- `52spring21_q05`
- `52summer22_q05`
- `53summer21_q06`

The only record removed from the suspicious selected-OCR set was `13summer25_q05`.

## Suspicious reason distribution

Iteration_004 suspicious OCR-selected audit flags:

| Audit flag | Count | Question IDs |
| --- | ---: | --- |
| `ocr_selected_with_low_question_crop_confidence` | 20 | `11autumn22_q10`, `12spring22_q05`, `12spring22_q08`, `12spring22_q10`, `12summer21_q02`, `12summer21_q03`, `13summer25_q05`, `41summer21_q03`, `41summer21_q04`, `41summer22_q05`, `42spring21_q02`, `42summer22_q04`, `43autumn21_q04`, `43summer22_q05`, `51autumn21_q01`, `51summer21_q05`, `51summer21_q07`, `52spring21_q05`, `52summer22_q05`, `53summer21_q06` |
| `ocr_selected_with_mapping_not_pass` | 1 | `13summer25_q05` |
| `ocr_selected_with_validation_not_pass` | 1 | `13summer25_q05` |
| `ocr_selected_with_uncertain_or_failed_scope` | 1 | `13summer25_q05` |

Iteration_005 remaining suspicious OCR-selected audit flags:

| Audit flag | Count | Question IDs |
| --- | ---: | --- |
| `ocr_selected_with_low_question_crop_confidence` | 19 | `11autumn22_q10`, `12spring22_q05`, `12spring22_q08`, `12spring22_q10`, `12summer21_q02`, `12summer21_q03`, `41summer21_q03`, `41summer21_q04`, `41summer22_q05`, `42spring21_q02`, `42summer22_q04`, `43autumn21_q04`, `43summer22_q05`, `51autumn21_q01`, `51summer21_q05`, `51summer21_q07`, `52spring21_q05`, `52summer22_q05`, `53summer21_q06` |

Required grouping details from the iteration_004 baseline:

| Group | Count | Question IDs |
| --- | ---: | --- |
| `mapping_status != pass` | 1 | `13summer25_q05` |
| `validation_status != pass` | 1 | `13summer25_q05` |
| `scope_quality_status != clean/pass/acceptable` | 1 | `13summer25_q05` |
| `question_crop_confidence == low` | 20 | all 20 iteration_004 suspicious records |
| `mark_scheme_crop_confidence == medium` | 5 | `12spring22_q08`, `13summer25_q05`, `42summer22_q04`, `52spring21_q05`, `52summer22_q05` |
| `mark_scheme_crop_confidence == low` | 0 | none |
| missing expected mark brackets | 0 | none |
| missing expected subparts | 0 | none |
| length much longer/shorter than native | 0 | none emitted by audit flags |
| page furniture / barcode / footer contamination | 0 audit flags | `13summer25_q05` manually contains header-like `UTE RR TT TR` in OCR text, but no audit furniture/barcode flag was emitted |
| next-question contamination | 0 | none |
| other audit flags | 0 | none beyond the four flags listed above |

Required grouping details after iteration_005:

| Group | Count | Question IDs |
| --- | ---: | --- |
| `mapping_status != pass` | 0 selected OCR | none |
| `validation_status != pass` | 0 selected OCR | none |
| `scope_quality_status != clean/pass/acceptable` | 0 selected OCR | none |
| `question_crop_confidence == low` | 19 | all 19 remaining suspicious records |
| `mark_scheme_crop_confidence == medium` | 4 | `12spring22_q08`, `42summer22_q04`, `52spring21_q05`, `52summer22_q05` |
| `mark_scheme_crop_confidence == low` | 0 | none |
| missing expected mark brackets | 0 | none |
| missing expected subparts | 0 | none |
| length much longer/shorter than native | 0 | none emitted by audit flags |
| page furniture / barcode / footer contamination | 0 audit flags | none |
| next-question contamination | 0 | none |
| other audit flags | 0 | none |

## 13summer25_q05 investigation

Iteration_004 values before the guard:

| Field | Value |
| --- | --- |
| `question_id` | `13summer25_q05` |
| `paper` | `13summer25` |
| `paper_family` | `p1` |
| `question_number` | `5` |
| `mapping_status` | `fail` |
| `validation_status` | `fail` |
| `scope_quality_status` | `review` |
| `question_crop_confidence` | `low` |
| `mark_scheme_crop_confidence` | `medium` |
| `native_text_score` | `-7` |
| `ocr_text_score` | `39` |
| `selected_text_score` | `39` |
| `text_candidate_source` | `ocr` |
| `text_candidate_decision` | `ocr_selected` |
| `text_candidate_decision_reasons` | `expected_mark_brackets_present`, `expected_question_number_present`, `native_score=-7`, `ocr_score=39`, `ocr_score_clear_margin`, `prompt_words_present`, `readable_prose_spacing`, `reasonable_text_length` |
| `ocr_selected` | `true` |
| `ocr_rejected_reasons` | `[]` |
| `question_text_trust` | `low` |
| `text_only_status` | `fail` |
| `visual_curation_status` | `fail` |
| `question_image_path` | `p1/13summer25/questions/q05.png` |
| `mark_scheme_image_path` | `p1/13summer25/mark_scheme/q05.png` |
| audit suspicious flags | `ocr_selected_with_low_question_crop_confidence`, `ocr_selected_with_mapping_not_pass`, `ocr_selected_with_uncertain_or_failed_scope`, `ocr_selected_with_validation_not_pass` |

Text fields before the guard, JSON-escaped to keep this handoff ASCII-only:

```text
question_text = "5 UTE RR TT TR Solve the equation 4sin@tan@ = 1+5cos\\u00e9 for \\u2014180\\u00b0 < 8 < 180\\u00b0. [6]"
ocr_text      = "5 UTE RR TT TR Solve the equation 4sin@tan@ = 1+5cos\\u00e9 for \\u2014180\\u00b0 < 8 < 180\\u00b0. [6]"
```

The selected OCR preserves the question number and mark bracket, but it is not clean: it contains a header/barcode-like prefix (`UTE RR TT TR`) and sits on a record already marked `mapping_status=fail`, `validation_status=fail`, `scope_quality_status=review`, and `question_crop_confidence=low`.

Decision: this was evidence of a missing selector guard, not a low-crop-confidence guard. OCR should not become the selected question text when mapping or validation is already non-pass unless the record is in the documented hard-scope recovery path. The record itself remains a crop/mapping hard-blocker candidate for a later extraction pass.

Iteration_005 values after the guard:

| Field | Value |
| --- | --- |
| `text_candidate_source` | `native` |
| `text_candidate_decision` | `native_retained` |
| `selected_text_score` | `-7` |
| `ocr_selected` | `false` |
| `ocr_rejected_reasons` | `ocr_mapping_status_not_pass`, `ocr_validation_status_not_pass` |
| `question_text_trust` | `low` |
| `text_only_status` | `fail` |
| `visual_curation_status` | `fail` |
| audit suspicious flags | none, because OCR is no longer selected |

After guard text fields:

```text
question_text = "5 Solve the equation 4sinitani = 1 + 5cosi for - 180\\u00b01i1180\\u00b0. [6]"
ocr_text      = "5 UTE RR TT TR Solve the equation 4sin@tan@ = 1+5cos\\u00e9 for \\u2014180\\u00b0 < 8 < 180\\u00b0. [6]"
```

The OCR text remains available as an OCR candidate, but it is no longer promoted to `question_text`.

## Low question-crop-confidence investigation

The 20 iteration_004 suspicious selections were all low question-crop-confidence. After removing `13summer25_q05`, the remaining 19 are all low-crop-confidence with otherwise pass/clean statuses:

- `mapping_status=pass`: 19/19
- `validation_status=pass`: 19/19
- `scope_quality_status=clean`: 19/19
- selected OCR missing expected question number: 0/19
- selected OCR missing expected mark brackets: 0/19
- selected OCR missing expected subparts: 0/19
- selected OCR next-question/page-furniture audit flags: 0/19

Representative examples:

| Question ID | Statuses | Crop confidence | Scores | Notes |
| --- | --- | --- | --- | --- |
| `11autumn22_q10` | mapping pass, validation pass, scope clean | question low, mark high | native 53, OCR 87, margin 34 | preserves question number, two mark brackets, and subparts |
| `12spring22_q08` | mapping pass, validation pass, scope clean | question low, mark medium | native 48, OCR 98, margin 50 | preserves marks/subparts; no contamination audit flags |
| `51summer21_q05` | mapping pass, validation pass, scope clean | question low, mark high | native 54, OCR 108, margin 54 | OCR contains table-like text noise but preserves structure and remains text/trust gated |

Low crop confidence is a visual/crop review signal, not by itself a selector false-positive signal in this evidence set. Hard-blocking low crop confidence would remove 19 structurally preserved OCR selections. A stronger-margin low-crop rule would be arbitrary from the observed margins: remaining low-crop selected margins range from 32 to 57, and several structurally preserved examples sit in the 32-39 range. Additional structure-preservation checks are already active for question number, mark brackets, expected subparts, page furniture/header text, and next-question contamination.

Decision: low question-crop-confidence remains audit-only for now.

## Selector guard decision

Applied Outcome A for `13summer25_q05`: add a narrow selector guard.

The selector now rejects OCR when `mapping_status` or `validation_status` is provided and is not `pass`, except when `scope_quality_status == "fail"` so the iteration_004 hard-scope/native-failure recovery path remains intact.

This preserves these principles:

1. OCR missing the expected question number remains rejected except for documented hard-scope recovery.
2. OCR still must preserve expected mark brackets and expected subparts.
3. OCR still rejects page furniture/header and next-question contamination when those audit/selector regexes catch them.
4. OCR selection no longer promotes text for non-pass mapping/validation records outside hard-scope recovery.
5. Readiness/trust gates are unchanged.
6. Image artifacts remain canonical.
7. The canonical export default remains unchanged; OCR is still enabled only with `--enable-ocr`.

## Minimal fix applied, if any

Changed:

- `src/exam_bank/ocr.py`
  - Added optional `mapping_status` and `validation_status` inputs to `select_text_candidate`.
  - Added explicit rejection reasons:
    - `ocr_mapping_status_not_pass`
    - `ocr_validation_status_not_pass`
  - Guard applies only when `scope_quality_status != "fail"` to preserve hard-scope recovery behavior.
- `src/exam_bank/pipeline.py`
  - Passed preliminary mapping and validation statuses into `select_text_candidate`.
- `tests/test_ocr.py`
  - Added a regression test where OCR otherwise has a clear score margin but is rejected for mapping/validation non-pass outside hard-scope recovery.
  - Strengthened the existing hard-scope recovery test by passing non-pass mapping/validation and asserting OCR remains selectable in that documented exception.

No audit/reporting code was changed. No README or ROADMAP update was needed because the handoff documents the narrow selector behavior and the default canonical export policy remains unchanged.

## Before/after metrics

| Metric | Iteration 004 OCR baseline | Iteration 005 after guard |
| --- | ---: | ---: |
| `record_count` | 1301 | 1301 |
| `ocr_ran_count` | 1301 | 1301 |
| OCR text nonblank count | 1301 | 1301 |
| `ocr_selected_count` | 30 | 29 |
| suspicious OCR-selected count | 20 | 19 |
| selected OCR missing expected question number count | 0 | 0 |
| selected OCR with mapping/validation/scope non-pass count | 1 | 0 |
| selected OCR with low question crop confidence count | 20 | 19 |
| possible OCR false negatives count | 286 | 287 |
| hard blocker count | 28 | 28 |
| mapping status | fail 21, pass 1280 | fail 21, pass 1280 |
| validation status | fail 14, pass 917, review 370 | fail 14, pass 917, review 370 |
| Asterion tiers | 0:28, 1:359, 2:624, 3:14, 4:38, 5:238 | 0:28, 1:359, 2:624, 3:14, 4:38, 5:238 |
| records changed versus iteration_004 OCR baseline | n/a for before | 1 |
| worsened records count | n/a for before | 0 |

The only changed record was `13summer25_q05`:

- `ocr_selected`: `true -> false`
- `text_candidate_source`: `ocr -> native`
- `text_candidate_decision`: `ocr_selected -> native_retained`
- `selected_text_score`: `39 -> -7`
- baseline comparison `tier_movement`: `unchanged`
- `text_regression_flags`: `[]`
- `status_inflation_flags`: `[]`
- `worsened_status_fields`: `[]`

The possible false-negative count increased by one because `13summer25_q05` now retains native text while OCR has a higher score. This is acceptable because the rejected OCR candidate is tied to failed mapping/validation, review scope, low question crop confidence, and visible header-like contamination.

## Tests added or updated

Updated:

- `tests/test_ocr.py`

Added/strengthened coverage:

- `test_candidate_selection_rejects_ocr_when_mapping_or_validation_not_pass_without_hard_scope_recovery`
- `test_candidate_selection_can_use_clean_ocr_when_native_scope_failed`

No tests require Tesseract, PDFs, generated images, APIs, or network access.

## Commands run

Inspection and setup:

```bash
git status --short
git diff --name-only
sed -n '1,260p' src/exam_bank/ocr.py
sed -n '260,620p' src/exam_bank/ocr.py
sed -n '1,260p' tests/test_ocr.py
sed -n '260,620p' tests/test_ocr.py
sed -n '1,260p' scripts/audit_question_bank_readiness.py
sed -n '260,620p' scripts/audit_question_bank_readiness.py
sed -n '620,1040p' scripts/audit_question_bank_readiness.py
sed -n '1370,1475p' scripts/audit_question_bank_readiness.py
sed -n '1720,1835p' scripts/audit_question_bank_readiness.py
sed -n '1,260p' tests/test_question_bank_readiness_audit.py
sed -n '260,620p' tests/test_question_bank_readiness_audit.py
sed -n '620,1100p' tests/test_question_bank_readiness_audit.py
sed -n '1,240p' agent_handoffs/iteration_003/ocr_activation_export_disconnect.md
sed -n '1,260p' agent_handoffs/iteration_004/ocr_candidate_quality_guard.md
sed -n '1,220p' README.md
sed -n '220,420p' README.md
sed -n '1,340p' ROADMAP.md
```

Structured CSV/JSON inspection used short `.venv/bin/python - <<'PY' ... PY` scripts to summarize:

- suspicious OCR selected flag counts
- selected OCR status distributions
- `13summer25_q05` before/after fields
- before/after audit metrics
- baseline comparison changed/regression rows

Generation and audit:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output/iteration_005_ocr_triage_candidate --enable-ocr
.venv/bin/python scripts/audit_question_bank_readiness.py --input output/iteration_005_ocr_triage_candidate/json/question_bank.json --baseline output/iteration_004_ocr_guard_candidate/json/question_bank.json --artifact-root output/iteration_005_ocr_triage_candidate --out-dir output/audits/iteration_005
```

Validation:

```bash
.venv/bin/python -m py_compile scripts/audit_question_bank_readiness.py
.venv/bin/python -m pytest tests/test_question_bank_readiness_audit.py -q
# 12 passed

.venv/bin/python -m pytest tests/test_ocr.py -q
# 16 passed

.venv/bin/python -m pytest tests/test_audit.py tests/test_ocr.py tests/test_output_contract.py tests/test_extraction_structure.py -q
# 37 passed

.venv/bin/python -m pytest -q
# 340 passed, 3 skipped in 112.48s

git diff --check
# passed

git status --short

git status --short --ignored output/audits/iteration_005 output/iteration_005_ocr_triage_candidate
# !! output/audits/
# !! output/iteration_005_ocr_triage_candidate/
```

## Remaining risks

- The canonical `output/json/question_bank.json` remains native-only until intentionally regenerated with `--enable-ocr`.
- Nineteen OCR-selected records remain suspicious only because `question_crop_confidence == low`; they remain review-gated by crop/readiness semantics.
- Low crop confidence is still not solved. It should be addressed through crop/scope recovery, not OCR selector tuning.
- `13summer25_q05` remains a hard-blocker/crop-mapping recovery target. The OCR candidate is retained as OCR metadata but not selected as `question_text`.
- The audit does not currently flag the `UTE RR TT TR` prefix as barcode/header contamination. This should be considered during crop/furniture recovery rather than by broad OCR regex tuning in this iteration.
- Possible OCR false negatives increased from 286 to 287. The added case is intentionally rejected because it is structurally risky and review/hard-blocked.

## Recommended next iteration

Run a crop/scope recovery iteration focused on 2024-2025 and hard-blocker cases, starting with `13summer25_q05`.

Recommended scope:

- Investigate weak question anchors, side-panel exclusion, and barcode/header remnants in 2024-2025 papers.
- Reduce mapping/validation hard blockers without using OCR text to hide failed crop or mapping evidence.
- Keep OCR selector thresholds unchanged unless a new, corpus-backed false-positive pattern appears.
- Preserve current readiness/trust semantics and image-first canonical policy.
