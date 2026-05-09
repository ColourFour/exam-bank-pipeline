# Agent 1 Plan - Auto-Triage Iteration 002, 9709-Only Scope

Important update: `input/beyond` has been removed to focus explicitly on CAIE 9709. Park all 9231 expected-total work. Do not compare a 9709-only candidate against any baseline that included 9231 records.

## Decision

**3. Baseline scope is invalid; stop until a fresh 9709-only OCR baseline is frozen.**

The input tree is now 9709-only, but the current active OCR candidate and active auto-triage baseline are stale mixed-scope artifacts:

- `input` contains no `9231` or `input/beyond` PDFs.
- `output_ocr_candidate/json/question_bank.json` still has 1329 records and includes 28 `input/beyond/9231_*` records.
- `output_ocr_candidate/triage/iteration_001/baseline_question_bank.json` also has 1329 records and includes the same 28 `input/beyond/9231_*` records.
- `output/json/question_bank.json` has 1301 records and no 9231 records, but it is not usable as the OCR comparison baseline because `ocr_ran=false` for all records.

There are older clean 9709-only OCR baselines in `output/triage`:

- `output/triage/iteration_001/baseline_question_bank.json`: 1301 records, OCR-enabled, no 9231.
- `output/triage/iteration_003/baseline_question_bank.json`: 1301 records, OCR-enabled, no 9231.
- `output/triage/iteration_004/baseline_question_bank.json`: 1301 records, OCR-enabled, no 9231.

Those three have identical question-id scope to each other, but they were not frozen for this new `paper_total_mismatch` 9709-only cycle. Prefer creating a fresh `output_ocr_candidate/triage/iteration_002` baseline from a freshly rerun 9709-only OCR export before Agent 2 changes any code.

## Current 9709 Evidence From Stale Candidate

Ignoring the stale 9231 records, the remaining 9709 `paper_total_mismatch` records are grouped by paper:

| Records | Component | Family | Expected | Detected | Evidence |
| --- | --- | --- | --- | --- | --- |
| 12 | 12 | p1 | 75 | 74 | `question_mark_total_mismatch`, `cross_page_scope` |
| 11 | 32 | p3 | 75 | 72 | `question_mark_total_mismatch`, `cross_page_scope` |
| 11 | 32 | p3 | 75 | 78 | `question_mark_total_mismatch`, `cross_page_scope` |
| 11 | 33 | p3 | 75 | 74 | `question_mark_total_mismatch`, `cross_page_scope` |
| 11 | 33 | p3 | 75 | 67 | mark-scheme source points to `9709 Mathematics November 2021 Mark Scheme  12.pdf`, plus `question_subparts_incomplete` and `question_mark_total_mismatch` |
| 11 | 15 | p1 | 75 | 76 | `weak_question_anchor`, `question_scope_contaminated`, `anchor_or_boundary` |
| 11 | 31 | p3 | 75 | 71 | `anchor_or_boundary`, `cross_page_scope` |
| 10 | 12 | p1 | 75 | 72 | `question_mark_total_mismatch`, `question_subparts_incomplete`, `anchor_or_boundary` |
| 7 | 52 | p5 | 50 | 48 | `question_mark_total_mismatch`, `cross_page_scope` |
| 6 | 51 | p5 | 50 | 51 | `question_mark_total_mismatch`, `polluted_pass_requires_review`, `cross_page_scope` |
| 6 | 51 | p5 | 50 | 52 | `question_subparts_incomplete`, `question_scope_contaminated`, `anchor_or_boundary` |

No sampled 9709 mismatch group has missing expected total, unknown family, missing component, or malformed 9709 syllabus metadata. Current evidence points away from expected-total configuration and toward downstream extraction, mark-total parsing, subpart detection, crop boundary, and one mark-scheme pairing issue.

## Agent 2 Phase 0: Establish Clean Baseline

Agent 2 must not implement code changes until this phase passes.

1. Confirm input scope is 9709-only:

```bash
find input -maxdepth 2 -type f | rg '9231|beyond' || true
```

Expected: no output.

2. Rerun the OCR candidate from the current input tree:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```

3. Confirm the rerun is OCR-enabled and 9709-only:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status --input output_ocr_candidate/json/question_bank.json
```

And run this scope check:

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("output_ocr_candidate/json/question_bank.json")
data = json.loads(path.read_text())
records = data["questions"]
beyond = [
    record for record in records
    if "input/beyond" in record.get("notes", {}).get("source_pdf", "")
    or "9231" in record.get("notes", {}).get("source_pdf", "")
]
print("record_count", len(records))
print("declared_record_count", data.get("record_count"))
print("beyond_or_9231_count", len(beyond))
print("ocr_counts", dict(Counter(str(record.get("ocr_ran")) for record in records)))
if beyond:
    raise SystemExit("candidate still contains 9231/beyond records")
if any(record.get("ocr_ran") is not True for record in records):
    raise SystemExit("candidate is not fully OCR-enabled")
PY
```

4. Freeze a fresh 9709-only OCR baseline for this iteration before implementation:

```bash
.venv/bin/python -m exam_bank.cli triage-sample \
  --input output_ocr_candidate/json/question_bank.json \
  --output-root output_ocr_candidate/triage \
  --iteration iteration_002 \
  --issue-set hard-failures \
  --target paper_total_mismatch \
  --sample-size 30 \
  --seed 1
```

If `output_ocr_candidate/triage/iteration_002` already exists, do not overwrite it. Inspect it and prove it was created from the fresh 9709-only OCR candidate; otherwise choose the next unused iteration name and document the deviation.

5. Confirm baseline and candidate have matching record scope:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

baseline_path = Path("output_ocr_candidate/triage/iteration_002/baseline_question_bank.json")
candidate_path = Path("output_ocr_candidate/json/question_bank.json")
baseline = json.loads(baseline_path.read_text())["questions"]
candidate = json.loads(candidate_path.read_text())["questions"]
baseline_ids = {record["question_id"] for record in baseline}
candidate_ids = {record["question_id"] for record in candidate}
print("baseline_count", len(baseline), "candidate_count", len(candidate))
print("only_baseline", len(baseline_ids - candidate_ids))
print("only_candidate", len(candidate_ids - baseline_ids))
if baseline_ids != candidate_ids:
    raise SystemExit("baseline and candidate record scopes differ")
PY
```

## Files Agent 2 Should Inspect

- `src/exam_bank/pipeline.py`
  - `_PAPER_TOTALS_BY_SYLLABUS`
  - `_expected_paper_total`
  - `_paper_total_syllabus_code`
  - `_paper_total_check`
  - `_apply_paper_total_metadata`
  - `_paper_total_focus`
- `tests/test_pipeline_reconciliation.py`
- `src/exam_bank/document_metadata.py`
- `src/exam_bank/document_registry.py`
- `src/exam_bank/mark_schemes.py`
- `src/exam_bank/question_detection_patterns.py`
- `src/exam_bank/output_layout.py`
- `src/exam_bank/triage.py`
- `src/exam_bank/auto_triage.py`
- Current fresh candidate JSON: `output_ocr_candidate/json/question_bank.json`
- Fresh baseline JSON: `output_ocr_candidate/triage/iteration_002/baseline_question_bank.json`

## Files Agent 2 May Change

Metadata/reconciliation changes are allowed only if Phase 0 proves a 9709-only OCR baseline exists and the fresh candidate shows a metadata/reconciliation defect.

Permitted for a metadata-only fix:

- `src/exam_bank/pipeline.py`
- `tests/test_pipeline_reconciliation.py`

Do not change:

- OCR behavior
- crop generation
- question span detection
- mark scheme matching
- DeepSeek enrichment
- canonical `output/json/question_bank.json`
- image-first source-of-truth behavior
- validation gates
- hard-failure thresholds
- 9231 expected totals

If the evidence points to mark-scheme pairing, crop boundaries, question total parsing, subpart parsing, or contamination, Agent 2 must stop rather than forcing a metadata patch.

## Required Tests Before Implementation

If and only if a metadata/reconciliation defect is found, Agent 2 must write focused tests first in `tests/test_pipeline_reconciliation.py`.

Required invariants:

- `test_expected_paper_total_preserves_9709_totals` remains unchanged in behavior:
  - `9709 P1 = 75`
  - `9709 P3 = 75`
  - `9709 P4 = 50`
  - `9709 P5 = 50`
- Unsupported syllabuses do not inherit 9709 totals.
- Mixed record syllabuses do not inherit 9709 totals.
- Missing or malformed 9709 metadata does not produce a false pass.
- Unknown expected totals do not trigger rescan and do not add `paper_total_mismatch`.
- True 9709 mismatches still trigger rescan and still fail with `paper_total_mismatch`.

Do not add 9231 tests or 9231 totals in this iteration.

## Distinguishing Causes

Treat paper-total mismatch as paper-level, not question-level. Group records by `(source_pdf, mark_scheme_source_pdf, component, paper_family, expected_total, detected_total)` before drawing conclusions.

Metadata/reconciliation defect indicators:

- `source_pdf` is 9709 but `paper_total_expected` is `null`.
- `paper_family` is unknown or inconsistent with component.
- component/source paper code is missing or malformed.
- expected total differs from the existing 9709 rules.
- mixed or malformed syllabus metadata causes accidental 9709 fallback.

Question-level extraction or downstream defect indicators:

- expected total is correct (`75` or `50`) and detected total is off.
- `paper_total_focus_reason` includes `cross_page_scope`, `anchor_or_boundary`, or `recovery_stalled`.
- validation flags include `question_mark_total_mismatch`, `question_subparts_incomplete`, `question_scope_contaminated`, `weak_question_anchor`, or `polluted_pass_requires_review`.
- mark-scheme source PDF is wrong for the paper, as seen in the stale `33autumn25` group pointing at `9709 Mathematics November 2021 Mark Scheme  12.pdf`.

## Expected Outcome

Based on current stale-candidate evidence, this likely is **not** a metadata/reconciliation issue. The remaining 9709 records already have correct expected totals and valid 9709 metadata. After a clean baseline is frozen, Agent 2 should only continue with a metadata fix if the fresh 9709-only candidate contradicts that finding.

If the fresh candidate reproduces the same pattern, stop and recommend an extraction-focused iteration. The next target cluster should be selected from the fresh 9709-only OCR status output, likely one of:

- `question_mark_total_mismatch`
- `question_subparts_incomplete`
- `question_mark_total_missing`
- `question_scope_contaminated`
- mark-scheme pairing/source mismatch for the `33autumn25` group

## Verification Commands

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_pipeline_reconciliation.py
```

Full suite:

```bash
.venv/bin/python -m pytest
```

Diff hygiene:

```bash
git diff --check
```

Rerun 9709-only OCR candidate:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```

Candidate-safe status:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status --input output_ocr_candidate/json/question_bank.json
```

Candidate-safe comparison after a valid baseline is frozen and after any Agent 2 implementation:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-compare \
  --iteration agent_handoffs/auto_triage/iteration_002 \
  --baseline-triage output_ocr_candidate/triage/iteration_002 \
  --current output_ocr_candidate/json/question_bank.json \
  --output output_ocr_candidate/triage/iteration_002/comparison.auto-iteration-002.json \
  --test-status pass
```

Do not use any runbook command that points back to `output/json/question_bank.json` unless it is explicitly overridden to `output_ocr_candidate/json/question_bank.json`.

## Acceptance Criteria

- Baseline and candidate are both OCR-enabled.
- Baseline and candidate are both 9709-only.
- Baseline and candidate have matching question-id scope.
- Candidate contains zero `9231` or `input/beyond` records.
- 9231 totals are not added.
- 9709 expected-total behavior is unchanged.
- `hard_failure_count` does not increase.
- `worsened_record_count` remains `0`.
- `status_regressions` remains empty.
- Canonical `output/json/question_bank.json` is not overwritten.
- Any accepted improvement is supported by focused tests, full tests, and OCR-to-OCR comparison.

## Stop Conditions

Stop before implementation if:

- `output_ocr_candidate/json/question_bank.json` still includes 9231 or `input/beyond` records after rerun.
- The baseline includes 9231 or `input/beyond` records.
- Baseline and candidate question-id sets differ.
- Either baseline or candidate is not fully OCR-enabled.
- No fresh 9709-only OCR baseline exists for this iteration.

Stop after investigation if:

- 9709 expected totals are already correct.
- mismatch groups are explained by extraction, subpart parsing, mark-total parsing, crop boundary, contamination, or mark-scheme pairing.
- fixing the issue would require changing OCR, crop generation, question span detection, mark-scheme matching, validation gates, or hard-failure thresholds.

## Phase 0 Execution Result

Executed on 2026-05-09 after `input/beyond` was removed.

Fresh OCR candidate:

- Command: `.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr`
- Processed questions: 1301
- Processed papers: 148
- JSON: `output_ocr_candidate/json/question_bank.json`
- OCR profile: `ocr_enabled=true`, `ocr_counts={"true": 1301}`
- 9231/beyond records: 0
- Hard failures: 153
- Dominant cluster: `paper_total_mismatch` with 107 records

Fresh baseline:

- Command: `.venv/bin/python -m exam_bank.cli triage-sample --input output_ocr_candidate/json/question_bank.json --output-root output_ocr_candidate/triage --iteration iteration_002 --issue-set hard-failures --target paper_total_mismatch --sample-size 30 --seed 1`
- Baseline: `output_ocr_candidate/triage/iteration_002/baseline_question_bank.json`
- Record scope check: baseline 1301, candidate 1301, `only_baseline=0`, `only_candidate=0`
- Baseline-freeze comparison: `output_ocr_candidate/triage/iteration_002/comparison.baseline-freeze.json`
- Baseline-freeze deltas: `hard_failure_delta=0`, `target_issue_delta=0`, `worsened_records=[]`

Fresh 9709 `paper_total_mismatch` investigation:

- Mismatch records: 107
- Families: `p1=33`, `p3=55`, `p5=19`
- Expected totals: `75=88`, `50=19`
- Status: all `mismatch_after_rescan`
- Metadata anomalies: 0

Conclusion:

**2. No metadata fix available; remaining failures are extraction/mark-total issues.**

Do not make a metadata/reconciliation code change for this iteration. The fresh 9709-only candidate confirms that the remaining paper-total failures already have valid 9709 paper family/component metadata and unchanged expected totals. The evidence points to question-level extraction, subpart/mark-total parsing, crop-boundary contamination, and one mark-scheme pairing/source mismatch group.

Recommended next pass: hand off to an extraction-focused iteration targeting paper-total root causes by paper, starting with the sampled groups where `paper_total_focus_reason` includes `cross_page_scope`, `anchor_or_boundary`, `question_mark_total_mismatch`, `question_subparts_incomplete`, or the `33autumn25` mark-scheme source mismatch.
