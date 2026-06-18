# Project Gates

Use these gates from the repository root. Pick the smallest command set that proves the selected iteration; do not run long extraction jobs unless the plan specifically requires regeneration.

## Fast Regression Gates

```bash
.venv/bin/python -m pytest -q -m "not integration and not rendering"
```

```bash
.venv/bin/python -m pytest -q -m rendering
```

```bash
.venv/bin/python -m pytest -q -m "integration"
```

```bash
.venv/bin/python -m pytest -q
```

## Output Integrity Gates

Fail fast on missing or inconsistent generated artifacts:

```bash
.venv/bin/python -m exam_bank.cli output-integrity-audit \
  --input output/json/question_bank.json \
  --artifact-root output \
  --output output/json/audit.current.integrity.json
```

Produce extraction-readiness, crop-quality, mapping, OCR-selection, and Asterion-tier reports:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --artifact-root output \
  --out-dir output/audits/current
```

Validate canonical output structure without rewriting it:

```bash
.venv/bin/python -m exam_bank.cli output-normalize-structure \
  --root output \
  --validate-only
```

Audit exact storage duplicates and asset-manifest state without deleting files:

```bash
.venv/bin/python scripts/audit_output_storage.py --dry-run
```

Validate asset references against the canonical asset manifest:

```bash
.venv/bin/python scripts/validate_asset_references.py
```

## Review-Pack Gates

Create a small deterministic triage sample for visual inspection:

```bash
.venv/bin/python -m exam_bank.cli triage-sample \
  --input output/json/question_bank.json \
  --sample-size 30
```

Serve the generated sample when the auditor needs to inspect images interactively:

```bash
.venv/bin/python -m exam_bank.cli triage-serve \
  --iteration output/triage/iteration_001
```

Compare a reviewed sample against the current question bank:

```bash
.venv/bin/python -m exam_bank.cli triage-compare \
  --iteration output/triage/iteration_001 \
  --current output/json/question_bank.json
```

## Targeted Test Files

For output-contract, readiness, audit, and Asterion asset-reference changes:

```bash
.venv/bin/python -m pytest \
  tests/test_audit.py \
  tests/test_output_contract.py \
  tests/test_asterion_export.py \
  tests/test_output_management.py \
  tests/test_question_bank_readiness_audit.py \
  -q
```

For image rendering and crop behavior:

```bash
.venv/bin/python -m pytest tests/test_image_rendering.py -q
```

For image alignment controller or metrics work, use the existing alignment-focused tests when present:

```bash
.venv/bin/python -m pytest tests/test_image_alignment_controller.py -q
```

## Manual Auditor Checklist

- Open sampled question PNGs and confirm each contains only the intended question number and no neighboring question content.
- Open sampled mark-scheme PNGs and confirm each belongs to the same paper identity and question number as the question record.
- Compare sampled PNG filenames, `question_id`, `paper`, `question_number`, `question_image_path`, `mark_scheme_image_path`, and asset-manifest entries.
- Check suspicious records rather than only clean records: missing image path, duplicate image path, low crop confidence, mapping `review` or `fail`, validation `review` or `fail`, weak anchors, stitched mark schemes, and unusual multi-page crops.
- Treat ambiguous samples as review-required, not clean.

## Missing Gates / TODO

- No dedicated single command was found for semantic PNG content validation that OCRs the rendered PNG and compares the visible question number to the record.
- No dedicated single command was found for orphan-image detection across all canonical subject-family roots independent of the storage audit.
- No dedicated single command was found for automatic neighboring-question crop detection across the full output set.
- No dedicated single command was found for building a review pack from only suspicious alignment/crop candidates; `triage-sample` is the closest existing review-pack command.
