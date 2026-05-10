# Agent 3 Tests - iteration_001

## Phase

Post-Agent-2 readiness-audit verification.

## Tests Added

Added focused tests in `tests/test_question_bank_readiness_audit.py` for the new `scripts/audit_question_bank_readiness.py` reporting layer.

Covered:

- The audit command writes the expected summary/report files.
- Top-level vs `notes` field disagreements are counted.
- Missing numeric scores are reported as missing, not converted to zero.
- Mapping fail plus validation pass is surfaced as a contradiction.
- Hard blockers include missing mark-scheme image/text blockers.
- Baseline comparison detects added, removed, and shared records by `question_id`.
- Subpart mark simple-fillable cases are detected from `mark_values_detected`.
- Suspicious OCR-selected records are written to `ocr_suspicious_records.csv`.
- Possible OCR false negatives are written to `possible_ocr_false_negatives.csv`.
- Missing artifact files become blockers when `--artifact-root` is provided.

## Commands Run

Passed:

```bash
.venv/bin/python -m pytest tests/test_question_bank_readiness_audit.py -q
```

Result: `3 passed in 0.05s`

Passed:

```bash
.venv/bin/python -m pytest tests/test_audit.py tests/test_question_bank_readiness_audit.py tests/test_ocr.py tests/test_output_contract.py tests/test_extraction_structure.py -q
```

Result: `37 passed in 0.13s`

Passed:

```bash
.venv/bin/python -m pytest -q
```

Result: `328 passed, 3 skipped in 109.38s`

## Scope Guard Findings

The Agent 3 pass only added tests and documentation handoff notes. No OCR scoring, OCR thresholds, extraction behavior, crop detection, mark-scheme mapping, DeepSeek behavior, topic classification, difficulty classification, or trust/readiness gates were changed.

The Agent 2 readiness audit remains reporting-only and writes generated artifacts under ignored `output/audits/`.

## Verdict

IMPLEMENTATION VERIFIED

The new readiness-audit script has focused regression coverage for its highest-risk behaviors. Full-suite execution is still recommended before merging a larger branch, but the reporting layer, existing audit helpers, OCR guards, output contract guards, and extraction-structure guards pass together.

## Next-Loop Notes

- Iteration 002 should now target hard blockers: mapping failures, validation failures, missing mark-scheme text/image paths, mapping/validation contradictions, and local mark-total mismatches.
- OCR selection tuning should remain deferred until an OCR-enabled candidate export is audited.
- Generated audit outputs under `output/audits/` should stay untracked unless the human explicitly chooses to version a report snapshot.
