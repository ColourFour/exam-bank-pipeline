# Agent 2 Implementation Notes - iteration_001

## Scope

Implemented the deterministic reporting/tooling pass requested by Agent 1. No extraction behavior, OCR selection thresholds, crop logic, mapping logic, trust gates, topic classification, difficulty classification, DeepSeek behavior, tests, or exported schema were changed.

## Required Repo State Verified

Inspected:

- `README.md`
- `pyproject.toml`
- `config.yaml`
- `src/exam_bank/ocr.py`
- `src/exam_bank/pipeline.py`
- `src/exam_bank/models.py`
- `src/exam_bank/exporters.py`
- `src/exam_bank/extraction_structure.py`
- `src/exam_bank/trust.py`
- `tests/test_ocr.py`
- `tests/test_output_contract.py`
- `tests/test_extraction_structure.py`
- `output/json/question_bank.json`
- previous handoffs/comparison reports under `agent_handoffs/iteration_001`, `output/triage`, and `output_ocr_candidate/triage`
- `agent_handoffs/iteration_001/Prompt/agent1_prompt.md`

Important verified state:

- The export is image-first. `exporters.py` writes contract fields at top level and diagnostic pipeline fields mostly under `notes`.
- Preferred field source in the new audit: top-level for export-contract/student-facing fields, `notes` for diagnostic fields. Disagreements are recorded.
- Current `output/json/question_bank.json` has schema `exam_bank.question_bank` version `2`, declared and actual record count `1301`.
- Current canonical export has candidate-selection metadata, but OCR did not run: `ocr_ran=false` for all `1301`, `text_candidate_source=native` for all `1301`, `ocr_selected=false` for all `1301`, and `ocr_text`/`ocr_engine` are blank.
- A baseline exists at `output/triage/iteration_004/baseline_question_bank.json` and was used for comparison.

## Files Changed

- `scripts/audit_question_bank_readiness.py`
  - New standalone deterministic full-bank audit/report command.
  - Reads top-level and `notes` fields through a documented source-of-truth helper.
  - Produces field presence, OCR candidate, suspicious OCR, possible false-negative, readiness tier, blocker, crop, mapping/validation, subpart mark, representative sample, baseline comparison, summary, and recommendation reports.
  - Supports optional `--baseline` and `--artifact-root`.
- `agent_handoffs/iteration_001/agent2_impl_notes.md`
  - Updated with this implementation and run evidence.

## Command Added

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/iteration_001
```

## Output Files Produced

`output/audits/iteration_001/` now contains:

- `audit_summary.md`
- `audit_summary.json`
- `field_presence_report.csv`
- `ocr_candidate_audit.csv`
- `ocr_suspicious_records.csv`
- `possible_ocr_false_negatives.csv`
- `readiness_tiers.csv`
- `hard_blockers.csv`
- `crop_quality_report.csv`
- `mapping_validation_report.csv`
- `subpart_marks_report.csv`
- `representative_review_sample.csv`
- `baseline_comparison.csv`
- `baseline_comparison_summary.json`
- `next_iteration_recommendations.md`

These are generated output artifacts under ignored `output/`; they should not be committed unless the human explicitly wants audit artifacts versioned.

## Current Bank Results

From `output/audits/iteration_001/audit_summary.json`:

- Records audited: `1301`
- OCR ran: `0`
- OCR selected: `0`
- Text candidate source: `native: 1301`
- Text candidate decision: `native_retained: 1301`
- OCR measurement blocker: `partial_ocr_candidate_fields_missing:ocr_engine,ocr_text`
- Suspicious OCR-selected records: `0` because no OCR was selected
- Possible OCR false negatives: `0` from this canonical export because OCR text is blank
- Mapping status: `pass: 1281`, `fail: 20`
- Validation status: `pass: 921`, `review: 371`, `fail: 9`
- Asterion highest tier counts: `0: 23`, `1: 360`, `2: 185`, `3: 13`, `4: 52`, `5: 668`
- Hard blocker count: `23`
- Simple future-fillable subpart mark records: `920`
- Missing top-level run metadata: `generated_at`, `run_id`, `pipeline_version`, `git_commit`, `model_versions`, `ocr_engine_version`, `input_manifest_sha256`, `artifact_root`, `qa_summary`

## Baseline Comparison

Baseline used:

```bash
output/triage/iteration_004/baseline_question_bank.json
```

Summary:

- Reliable comparison: `true`
- Records added: `0`
- Records removed: `0`
- Shared records: `1301`
- Asterion tier movement: `improved: 630`, `unchanged: 671`
- Main changed fields:
  - `ocr_text_score: 1301`
  - `text_only_status: 952`
  - `question_text_trust: 906`
  - `text_fidelity_status: 906`
  - `question_crop_confidence: 870`
  - `topic_confidence: 870`
  - `visual_curation_status: 553`
  - `question_text: 269`
  - `validation_status: 250`
  - `mapping_status: 161`
- Worsened status counts:
  - `mapping_status: 1`
  - `mark_scheme_crop_confidence: 11`
  - `text_fidelity_status: 1`

## Next-Iteration Recommendation

The limiting issue for OCR selection measurement is not threshold tuning. The current canonical export has no OCR text, so OCR selected/false-negative behavior cannot be judged from it.

Recommended next priorities from the audit:

1. Produce or select an OCR-enabled candidate export before tuning OCR selection.
2. Address mapping/validation hard blockers and contradictions.
3. Inspect crop/scope confidence limits on visual readiness.
4. Plan a subpart mark promotion sprint using the `920` simple-fillable records.
5. Add a future top-level run manifest for auditable freshness.
6. Delay topic/difficulty reruns until text fidelity and mapping blockers are reduced.

## Validation Commands

Passed:

```bash
.venv/bin/python -m py_compile scripts/audit_question_bank_readiness.py
```

Passed:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/iteration_001
```

Passed:

```bash
.venv/bin/python -m pytest tests/test_audit.py tests/test_ocr.py tests/test_output_contract.py tests/test_extraction_structure.py -q
```

Result: `34 passed in 0.11s`

Agent 3 still owns focused tests for this new audit/reporting layer.
