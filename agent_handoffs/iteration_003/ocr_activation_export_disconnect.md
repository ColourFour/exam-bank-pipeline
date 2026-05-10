# iteration_003 OCR Activation/Export Disconnect

## Initial worktree state

Captured before edits:

```text
 M README.md
 M agent_handoffs/iteration_001/agent2_impl_notes.md
 M agent_handoffs/iteration_001/agent3_tests.md
 M agent_handoffs/iteration_002/agent3_tests.md
 M docs/ROADMAP.md
?? ROADMAP.md
?? scripts/audit_question_bank_readiness.py
?? tests/test_question_bank_readiness_audit.py
```

`git diff --name-only` initially listed:

```text
README.md
agent_handoffs/iteration_001/agent2_impl_notes.md
agent_handoffs/iteration_001/agent3_tests.md
agent_handoffs/iteration_002/agent3_tests.md
docs/ROADMAP.md
```

Those files were treated as pre-existing work and were not reverted. `agent_handoffs/iteration_003/` did not exist before this pass and was created for this note.

## Files inspected

Required files inspected:

- `README.md`
- `ROADMAP.md`
- `config.yaml`
- `pyproject.toml`
- `scripts/audit_question_bank_readiness.py`
- `tests/test_question_bank_readiness_audit.py`
- `tests/test_ocr.py`
- `tests/test_output_contract.py`
- `tests/test_extraction_structure.py`
- `src/exam_bank/ocr.py`
- `src/exam_bank/pipeline.py`
- `src/exam_bank/models.py`
- `src/exam_bank/exporters.py`
- `src/exam_bank/trust.py`
- `src/exam_bank/extraction_structure.py`
- `agent_handoffs/iteration_001/agent2_impl_notes.md`
- `agent_handoffs/iteration_002/agent3_tests.md`
- `output/audits/iteration_001/audit_summary.md`
- `output/audits/iteration_001/audit_summary.json`
- `output/json/question_bank.json`
- `output/triage/iteration_004/baseline_question_bank.json`

Additional files inspected because they are on the OCR activation path:

- `src/exam_bank/cli.py`
- `src/exam_bank/config.py`
- `src/exam_bank/image_rendering.py`
- `src/exam_bank/pdf_extract.py`
- `tests/test_config.py`
- `tests/test_runtime_paths.py`
- `tests/test_sample_pipeline.py`
- `docs/ROADMAP.md`
- `.gitignore`
- `output_ocr_candidate/json/question_bank.json`

No required file was missing. The optional `output_ocr_candidate/json/question_bank.json` was present and was used only as OCR-enabled reference evidence.

## Current observed OCR state

Direct JSON inspection showed:

| Export | Records | `ocr_ran` | `ocr_text` nonblank | `ocr_engine` | `text_candidate_source` | `ocr_selected` |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `output/json/question_bank.json` | 1301 | 0 | 0 | blank: 1301 | native: 1301 | 0 |
| `output/triage/iteration_004/baseline_question_bank.json` | 1301 | 1301 | 1301 | tesseract: 1301 | native: 1272, ocr: 29 | 29 |
| `output_ocr_candidate/json/question_bank.json` | 1301 | 1301 | 1301 | tesseract: 1301 | native: 1266, ocr: 35 | 35 |

Top-level and `notes` duplicate OCR fields had no disagreements in all three checked exports. `ocr_text` is stored top-level in the current schema; no checked export had populated notes-only `ocr_text`.

## Root cause

The canonical repo export at `output/json/question_bank.json` is a no-OCR export generated with the default/config path where OCR is disabled:

```yaml
ocr:
  enabled: false
```

OCR code is present and is called when enabled:

- `src/exam_bank/cli.py` exposes `process --enable-ocr` and sets `config.ocr.enabled = True`.
- `src/exam_bank/pdf_extract.py` uses `config.ocr.enabled` for OCR fallback and sparse lower-region OCR.
- `src/exam_bank/image_rendering.py` calls `run_question_crop_ocr(...)` after rendering question crops.
- `src/exam_bank/pipeline.py` reads `render_result.ocr_text`, runs native/OCR candidate selection, and stores OCR/candidate fields on `QuestionRecord`.
- `src/exam_bank/exporters.py` preserves top-level OCR fields and diagnostic candidate fields under `notes`.

The disconnect is therefore not an exporter omission, model-default overwrite, silent OCR failure, notes/top-level field loss, or audit misread for the checked files. The current canonical path is simply native-only/no-OCR, while historical/reference and `output_ocr_candidate` paths are OCR-enabled.

The exact command to generate an OCR-enabled production-style export is:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```

To replace the canonical export intentionally, use `--output output` instead, but that would regenerate generated bank output and should be treated as generated data, not a code fix.

## Minimal fix applied

No extraction or exporter fix was appropriate.

Applied only a focused activation-path test:

- `tests/test_ocr.py::test_process_cli_enable_ocr_routes_enabled_config_to_pipeline`

This test mocks the pipeline boundary and verifies `process --enable-ocr` routes `config.ocr.enabled=True`, OCR language, Tesseract command, and output root into `process_inputs` without requiring PDFs, Tesseract, or full-bank artifacts.

Also reran the canonical readiness audit into ignored generated output:

- `output/audits/iteration_003/`

And ran an OCR-enabled reference audit against the existing candidate export:

- `output/audits/iteration_003_ocr_candidate/`

## Why the fix is safe

- No extraction logic changed.
- No OCR thresholds changed.
- No OCR engine changed.
- No crop, mapping, topic, difficulty, validation, readiness, or trust gates changed.
- No generated bank JSON was regenerated or committed.
- The added test is fully mocked and only protects the documented activation switch.

## Before/after audit summary

Canonical before was `output/audits/iteration_001/audit_summary.json`.

Canonical after was rerun with:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/iteration_003
```

| Metric | Before canonical | After canonical | OCR-enabled candidate smoke |
| --- | ---: | ---: | ---: |
| record_count | 1301 | 1301 | 1301 |
| ocr_ran count | 0 | 0 | 1301 |
| ocr_selected count | 0 | 0 | 35 |
| ocr_text nonblank count | 0 | 0 | 1301 |
| possible OCR false negatives | 0 | 0 | 281 |
| suspicious OCR-selected count | 0 | 0 | 25 |
| hard blocker count | 23 | 23 | 28 |

After canonical OCR/candidate distributions:

```text
ocr_engine_distribution: {"missing": 1301}
text_candidate_source_distribution: {"native": 1301}
text_candidate_decision_distribution: {"native_retained": 1301}
top_ocr_rejected_reasons:
  empty_ocr_text: 1301
  empty_text: 1301
  ocr_lost_mark_brackets: 948
  ocr_missing_question_number: 1301
  ocr_missing_subpart_labels: 968
native_text_score_summary: count=1301, min=-12, p25=66, median=87, p75=114, max=172, mean=87.649
ocr_text_score_summary: count=1301, min=-100, p25=-100, median=-100, p75=-100, max=-100, mean=-100
selected_text_score_summary: count=1301, min=-12, p25=66, median=87, p75=114, max=172, mean=87.649
mapping_status: {"fail": 20, "pass": 1281}
validation_status: {"fail": 9, "pass": 921, "review": 371}
Asterion tiers: {"0": 23, "1": 360, "2": 185, "3": 13, "4": 52, "5": 668}
```

OCR-enabled candidate smoke metrics:

```text
ocr_engine_distribution: {"tesseract": 1301}
text_candidate_source_distribution: {"native": 1266, "ocr": 35}
text_candidate_decision_distribution: {"native_retained": 1266, "ocr_selected": 35}
top_ocr_rejected_reasons:
  ocr_lost_mark_brackets: 83
  ocr_missing_question_number: 6
  ocr_missing_subpart_labels: 17
  ocr_not_clearly_better: 1097
  page_furniture_or_header_text: 82
native_text_score_summary: count=1301, min=-12, p25=66, median=87, p75=114, max=172, mean=87.595
ocr_text_score_summary: count=1301, min=-42, p25=59, median=82, p75=109, max=177, mean=82.931
selected_text_score_summary: count=1301, min=-12, p25=66, median=89, p75=115, max=172, mean=88.666
mapping_status: {"fail": 21, "pass": 1280}
validation_status: {"fail": 14, "pass": 917, "review": 370}
Asterion tiers: {"0": 28, "1": 359, "2": 625, "3": 16, "4": 37, "5": 236}
```

These OCR-enabled candidate metrics prove activation/export works, not OCR quality. The suspicious/false-negative counts require a separate OCR candidate-quality review.

## Tests added or updated

Updated:

- `tests/test_ocr.py`

Added:

- `test_process_cli_enable_ocr_routes_enabled_config_to_pipeline`

Existing tests already covered:

- OCR disabled by default.
- OCR success/failure result capture.
- model/export preservation of OCR fields.
- candidate decision field preservation.
- readiness audit field resolution and OCR risk reporting.

## Commands run

```bash
git status --short
git diff --name-only
```

Required source and artifact inspection used `sed`, `rg`, direct JSON summaries, and `stat`.

Required grep/rg checks covered:

```bash
rg "ocr_ran" -n .
rg "ocr_text" -n .
rg "ocr_selected" -n .
rg "text_candidate_source" -n .
rg "native_text_score" -n src tests scripts config.yaml pyproject.toml README.md ROADMAP.md agent_handoffs/iteration_001 agent_handoffs/iteration_002 docs
rg "ocr_text_score" -n src tests scripts config.yaml pyproject.toml README.md ROADMAP.md agent_handoffs/iteration_001 agent_handoffs/iteration_002 docs
rg "tesseract" -n src tests scripts config.yaml pyproject.toml README.md ROADMAP.md agent_handoffs/iteration_001 agent_handoffs/iteration_002 docs
rg "OCR" -n src tests scripts config.yaml pyproject.toml README.md ROADMAP.md agent_handoffs/iteration_001 agent_handoffs/iteration_002 docs
```

Audit reruns:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/iteration_003

.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output_ocr_candidate/json/question_bank.json \
  --baseline output_ocr_candidate/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output_ocr_candidate \
  --out-dir output/audits/iteration_003_ocr_candidate
```

Validation:

```bash
.venv/bin/python -m py_compile scripts/audit_question_bank_readiness.py
.venv/bin/python -m pytest tests/test_question_bank_readiness_audit.py -q
.venv/bin/python -m pytest tests/test_audit.py tests/test_ocr.py tests/test_output_contract.py tests/test_extraction_structure.py -q
.venv/bin/python -m pytest -q
git diff --check
git status --short
```

Validation results:

```text
py_compile passed
tests/test_question_bank_readiness_audit.py: 12 passed
focused audit/OCR/output/extraction set: 35 passed
full pytest: 338 passed, 3 skipped
git diff --check passed
```

Ignored output check:

```text
.gitignore:16:output/* output/audits/iteration_003/audit_summary.json
.gitignore:16:output/* output/audits/iteration_003_ocr_candidate/audit_summary.json
```

Final `git status --short` after validation:

```text
 M README.md
 M agent_handoffs/iteration_001/agent2_impl_notes.md
 M agent_handoffs/iteration_001/agent3_tests.md
 M agent_handoffs/iteration_002/agent3_tests.md
 M docs/ROADMAP.md
 M tests/test_ocr.py
?? ROADMAP.md
?? scripts/audit_question_bank_readiness.py
?? tests/test_question_bank_readiness_audit.py
?? agent_handoffs/iteration_003/
```

Generated audit outputs are ignored and do not appear in normal `git status --short`.

## Remaining risks

- `output/json/question_bank.json` remains native-only/no-OCR. Do not use it for OCR selection quality judgments.
- OCR-enabled candidate smoke output has `35` OCR-selected records, `25` suspicious OCR-selected records, and `281` possible false negatives. Those are audit leads, not confirmed quality defects.
- OCR-enabled candidate hard blockers differ from canonical no-OCR blockers, so compare OCR-enabled runs only to OCR-enabled baselines.
- The schema still lacks a top-level run manifest, so the exact command/profile that produced a JSON cannot be proven from the file alone.

## Recommended next iteration

Run an OCR candidate-quality audit on an OCR-enabled export, preferably `output_ocr_candidate/json/question_bank.json` or a freshly regenerated candidate from:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```

Then review `possible_ocr_false_negatives.csv` and `ocr_suspicious_records.csv` before any OCR threshold, crop, or selector changes. If OCR activation remains intentionally deferred for canonical `output/json`, move to hard blocker recovery with that limitation explicitly documented.
