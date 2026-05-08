# Triage Workflow

Triage exists because full extraction quality cannot be judged from aggregate counts alone. A change can reduce one flag while silently worsening crops, mark-scheme pairing, or text trust. The triage loop freezes a baseline, samples a dominant failure cluster, records visual notes, reruns the pipeline, and compares current output back to the frozen baseline.

## Basic Loop

Create a deterministic sample:

```bash
.venv/bin/python -m exam_bank.cli triage-sample \
  --input output/json/question_bank.json \
  --sample-size 30
```

This writes a new iteration folder under `output/triage/`, for example:

```text
output/triage/iteration_004/
  baseline_question_bank.json
  summary.json
  sample.json
  index.html
  review.jsonl
```

Serve the gallery:

```bash
.venv/bin/python -m exam_bank.cli triage-serve \
  --iteration output/triage/iteration_004
```

Open the printed local URL, inspect the question crop and mark-scheme crop side by side, select a root cause, and save notes. Notes append to `review.jsonl`; they do not mutate the baseline.

After a full rerun, compare:

```bash
.venv/bin/python -m exam_bank.cli triage-compare \
  --iteration output/triage/iteration_004 \
  --current output/json/question_bank.json \
  --output output/triage/iteration_004/comparison.my-new-run.json
```

Use a clearly named comparison file. Do not overwrite historical comparisons unless the replacement is intentionally the same run.

## Choosing The Next Target

By default, `triage-sample` uses `--issue-set hard-failures` and `--target auto`, which samples the largest primary hard-failure issue.

Hard failures are records where one of these is failed:

- `notes.validation_status`
- `notes.mapping_status`
- `visual_curation_status`

Primary issue selection uses:

1. First validation flag if present.
2. Mapping failure reason if mapping failed.
3. Visual curation failure.
4. Review statuses.

Use the largest issue when it represents a real code path. Override the target only when the largest issue is already understood, blocked, or not actionable:

```bash
.venv/bin/python -m exam_bank.cli triage-sample \
  --input output/json/question_bank.json \
  --target question_scope_contaminated \
  --sample-size 30
```

## Preserving Baselines

`baseline_question_bank.json` is the frozen reference for that iteration. Do not delete it. Do not edit it. Do not regenerate it in place.

If a new run should become a baseline, create a new iteration:

```bash
.venv/bin/python -m exam_bank.cli triage-sample \
  --input output/json/question_bank.json \
  --iteration iteration_005 \
  --sample-size 30
```

Historical baselines under `output/triage/iteration_*` are part of the evidence chain.

## OCR And No-OCR Baselines

OCR-enabled runs must be compared against OCR-enabled baselines. No-OCR runs must be compared against no-OCR baselines.

Do not use a mixed OCR/no-OCR comparison as the main production score. OCR changes:

- `ocr_ran`
- `notes.text_source_profile`
- text candidate scores
- OCR rejection reasons
- visual/text trust gates
- review flags such as sparse merged OCR text

Mixed comparisons can still be useful for debugging layout or crop behavior, but the report must say that the comparison is not canonical for production OCR.

Current examples:

- `output/triage/iteration_003/comparison.math-repair-ocr.json` is OCR-to-OCR and is a canonical production-style comparison.
- `output/triage/iteration_004/comparison.layout-review-current.json` compares an OCR baseline against the current no-OCR export; it is useful but not a canonical OCR score.

## Interpreting Comparison Files

Important fields:

- `baseline_record_count`, `current_record_count`
- `baseline_hard_failure_count`, `current_hard_failure_count`
- `hard_failure_delta`
- `baseline_target_issue_count`, `current_target_issue_count`
- `target_issue_delta`
- `baseline_issue_counts`, `current_issue_counts`
- `issue_count_deltas`
- `improved_records`
- `worsened_records`

Negative deltas are improvements. Positive deltas are regressions or stricter gating and need explanation.

`worsened_records` is a sampled list of shared records where tracked statuses moved in the wrong direction. Treat any non-empty list as a review blocker unless the worsening is intentionally stricter and documented.

## Recording Review Notes

Use the root-cause selector in the triage gallery. Available root causes include:

- `question_crop_boundary`
- `mark_scheme_mapping`
- `paper_total_detection`
- `false_positive_validation_gate`
- `text_ocr_quality`
- `classification_only`
- `source_pdf_issue`
- `unknown`

Good notes include:

- What is visible in the crop.
- Whether the mark-scheme crop matches the question.
- Whether the issue is crop/scope, text fidelity, mark-scheme mapping, paper-total accounting, or classification only.
- The exact formula or layout that was corrupted, when text is the issue.
- Whether the current validation flag is correct or too strict.

Do not edit old JSONL lines. If a note is wrong, save a later note for the same `question_id`; consumers can use the latest note per question.

## What Counts As Real Improvement

Real improvement means:

- The target issue decreases.
- Overall hard failures do not increase unexpectedly.
- `worsened_records` is empty, or each worsening is intentionally stricter and documented.
- The reviewed sample visually confirms the fix.
- The fix adds or preserves regression tests.
- The comparison uses the correct OCR/no-OCR baseline profile.

Suspicious improvement means:

- Failure counts drop because flags disappeared without crop/text behavior improving.
- `text_only_status` or `question_text_trust` improves while `text_fidelity_status` remains degraded.
- Mapping passes despite wrong source PDF metadata.
- OCR text is selected while missing question numbers, marks, subparts, or mark brackets.
- A no-OCR output is used to claim production OCR progress.

## Example Current Commands

Compare current no-OCR output against iteration 004 for debugging:

```bash
.venv/bin/python -m exam_bank.cli triage-compare \
  --iteration output/triage/iteration_004 \
  --current output/json/question_bank.json \
  --output output/triage/iteration_004/comparison.current-no-ocr-debug.json
```

Run the quality gate against iteration 004:

```bash
.venv/bin/python scripts/quality_gate.py \
  --iteration output/triage/iteration_004 \
  --current output/json/question_bank.json \
  --require-target-improvement
```

For production-style OCR work, generate a separate OCR-enabled output path and compare it to an OCR baseline:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output_ocr_candidate \
  --enable-ocr

.venv/bin/python -m exam_bank.cli triage-compare \
  --iteration output/triage/iteration_004 \
  --current output_ocr_candidate/json/question_bank.json \
  --output output/triage/iteration_004/comparison.ocr-candidate.json
```

Do not replace `output/json/question_bank.json` until the comparison is understood.
