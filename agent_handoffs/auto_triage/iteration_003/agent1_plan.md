# Agent 1 Plan - Auto-Triage Iteration 003, 9709 Extraction-Failure Decomposition

## Decision

**Safe bounded extraction fix available.**

Target one causal cluster, not the `paper_total_mismatch` metadata symptom:

> Mark-scheme crop/total extraction starts at a later subpart when the parent question anchor is missing, causing the paper total to use an incomplete mark-scheme total even when the question-paper crop has the complete mark total.

Representative source-truth case:

- `31autumn25_q09`
- Question crop: `output_ocr_candidate/p3/31autumn25/questions/q09.png`
- Mark-scheme crop: `output_ocr_candidate/p3/31autumn25/mark_scheme/q09.png`
- Question image shows parts `(a)` to `(d)` with marks `[2] + [2] + [1] + [3] = 8`.
- Current mark-scheme crop starts at `9(c)` and includes only `9(c)`/`9(d)`, so `mark_scheme_total_detected=4`.
- This makes the whole `31autumn25` paper report `paper_total_detected=71` instead of expected `75`.

This is preferable to a broad mark parsing rewrite because it is narrow, visible in image crops, and should improve one full paper group (11 records) without changing expected totals or validation gates.

## Scope Baseline

Use the frozen 9709-only OCR baseline from iteration 002:

- Baseline: `output_ocr_candidate/triage/iteration_002/baseline_question_bank.json`
- Candidate: `output_ocr_candidate/json/question_bank.json`
- Record count: 1301
- OCR enabled: `{"true": 1301}`
- 9231/beyond records: 0
- Baseline/candidate scope: `only_baseline=0`, `only_candidate=0`
- Baseline hard failures: 153
- Baseline `paper_total_mismatch`: 107

Do not use `output/json/question_bank.json` for this iteration.

## Grouped Summary

The 107 remaining `paper_total_mismatch` records are 11 paper-level groups:

| Paper | Records | Component | Family | Expected | Detected | Question Sum | Mark-Scheme Sum | Likely Root Cause |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `12autumn21` | 12 | 12 | p1 | 75 | 74 | 75 | 74 | incorrect mark-scheme/question mark total for q12; cross-page scope |
| `12autumn24` | 10 | 12 | p1 | 75 | 72 | 71 | 72 | bad anchor/end boundary; q1/q5 mark-total disagreements |
| `15autumn25` | 11 | 15 | p1 | 75 | 76 | 63 | 76 | q11 scope contamination/anchor boundary; mark scheme totals override bad question crop |
| `31autumn25` | 11 | 31 | p3 | 75 | 71 | 75 | 71 | mark-scheme crop starts at `9(c)`, missing `9(a)`/`9(b)` |
| `32spring22` | 11 | 32 | p3 | 75 | 72 | 75 | 72 | q2/q5 mark-scheme/question mark total disagreements; cross-page scope |
| `32summer23` | 11 | 32 | p3 | 75 | 78 | 75 | 78 | q5 mark total extraction disagreement |
| `33summer23` | 11 | 33 | p3 | 75 | 74 | 72 | 74 | q9/q11 mark total extraction disagreements |
| `33autumn25` | 11 | 33 | p3 | 75 | 67 | 75 | 67 | mark-scheme source mismatch: records point to `9709 Mathematics November 2021 Mark Scheme  12.pdf`; exact N25 P33 MS source appears absent |
| `51summer22` | 6 | 51 | p5 | 50 | 51 | 50 | 51 | q4 mark total extraction disagreement |
| `51summer25` | 6 | 51 | p5 | 50 | 52 | 37 | 52 | q5 incomplete subparts, q6 scope contamination |
| `52autumn23` | 7 | 52 | p5 | 50 | 48 | 45 | 48 | q5/q6 mark total extraction disagreements; cross-page scope |

No group has missing/unknown expected total, malformed component metadata, or missing 9709 paper-family metadata.

## Top Causal Clusters

1. **Mark-scheme total/crop extraction overrides correct question-paper totals**
   - Candidate papers where question-side totals already sum to expected but mark-scheme totals do not:
     `12autumn21`, `31autumn25`, `32spring22`, `32summer23`, `51summer22`.
   - Potential impact: up to 51 records, but broad changes are risky.

2. **Anchor/end-boundary or cross-page scope errors**
   - Signals: `cross_page_scope`, `anchor_or_boundary`, `question_start_uncertain`, `crop_split_prompt_regions`, `question_scope_contaminated`.
   - Papers include `12autumn24`, `15autumn25`, `51summer25`, `52autumn23`.
   - Potential impact is high, but a broad boundary rewrite is risky.

3. **Mark-scheme source mismatch**
   - `33autumn25` points to `input/mark_schemes/9709 Mathematics November 2021 Mark Scheme  12.pdf`.
   - The correct input file `input/mark_schemes/9709 Mathematics November 2025 Mark Scheme  33.pdf` is absent.
   - Potential target count: 11 records, but fixing by making the mark scheme unmatched may cause a mapping-status regression for `33autumn25_q01`, so this is not the recommended first implementation target.

## Recommended Agent 2 Target

Implement one bounded extraction fix:

**When rendering and totaling a mark-scheme block for a parent question, do not start at a later subpart such as `9(c)` if the question paper indicates earlier subparts exist. Backfill or widen the mark-scheme block to include the earlier same-parent rows/pages, bounded by the previous parent question and next parent question.**

Primary regression target:

- `input/question_papers/9709 Mathematics November 2025 Question Paper  31.pdf`
- `input/mark_schemes/9709 Mathematics November 2025 Mark Scheme  31.pdf`
- `31autumn25_q09`

Expected behavior after fix:

- `31autumn25_q09.notes.mark_scheme_total_detected == 8`
- `31autumn25_q09.notes.mapping_status == "pass"` or at least no worse than current
- `31autumn25` paper total becomes `75`
- `31autumn25` group no longer contributes 11 `paper_total_mismatch` records

Do not solve this by changing `_record_solution_marks` to prefer question totals over mark-scheme totals globally. That would hide mark-scheme extraction defects and could create false passes, especially for `33autumn25`.

## Files Agent 2 May Inspect

- `src/exam_bank/mark_schemes.py`
  - `render_mark_scheme_images`
  - `_detect_table_question_anchors`
  - `_anchor_for_question`
  - `_next_boundary_anchor`
  - `_table_regions_for_anchor`
  - `_mark_total_for_question_block`
  - `_table_rows_for_question_block`
  - `_detected_subparts_for_question`
- `src/exam_bank/mark_scheme_models.py`
- `src/exam_bank/question_detection.py`
- `src/exam_bank/question_detection_patterns.py`
- `src/exam_bank/pipeline.py` for how mark-scheme totals feed paper total metadata
- `tests/test_question_detection.py`
- `tests/test_sample_pipeline.py`
- `output_ocr_candidate/p3/31autumn25/questions/q09.png`
- `output_ocr_candidate/p3/31autumn25/mark_scheme/q09.png`

Optional inspection only:

- `src/exam_bank/mark_scheme_pairing.py` for the `33autumn25` source mismatch, but do not mix that fix into this iteration.

## Files Agent 2 May Change

Preferred narrow write set:

- `src/exam_bank/mark_schemes.py`
- `tests/test_question_detection.py`
- `tests/test_sample_pipeline.py`

If a small helper needs to be exposed for testing, keep it local to `mark_schemes.py`.

Do not change:

- expected 9709 paper totals
- 9231 behavior
- OCR enablement
- DeepSeek sidecar behavior
- canonical `output/json/question_bank.json`
- validation gates
- hard-failure thresholds
- image-first source-of-truth behavior

## Tests First

Agent 2 must write failing tests before implementation.

Required focused test with repo PDFs:

- Add a `tests/test_sample_pipeline.py` regression for N25 P31:
  - define `REPO_N25_P31_QP = Path("input/question_papers/9709 Mathematics November 2025 Question Paper  31.pdf")`
  - define `REPO_N25_P31_MS = Path("input/mark_schemes/9709 Mathematics November 2025 Mark Scheme  31.pdf")`
  - skip if either file is missing
  - run `process_sample(..., mark_scheme_pdf=REPO_N25_P31_MS)` with OCR disabled
  - assert q9 question subparts include `["a", "b", "c", "d"]`
  - assert q9 mark-scheme subparts include `["a", "b", "c", "d"]`
  - assert q9 `question_marks_total == 8`
  - assert q9 `markscheme_marks_total == 8`
  - assert q9 mapping is not regressed
  - assert first record paper total expected is `75`
  - assert final paper total is `75` and status is `matched` or `recovered_after_rescan`

Recommended focused unit test:

- Add or update a `tests/test_question_detection.py` mark-scheme helper test that simulates anchors where a parent question has only later visible subpart anchors (`9(c)`, `9(d)`) but the expected question subparts include earlier labels. The test should assert the chosen crop/row range can include earlier same-parent rows instead of starting at `9(c)`.

Do not write tests that simply assert `paper_total_mismatch` disappears without checking the underlying q9 mark-scheme crop/total evidence.

## Verification Commands

Agent 2/3 should run focused tests first:

```bash
.venv/bin/python -m pytest tests/test_sample_pipeline.py -k "n25_p31 or paper_total"
.venv/bin/python -m pytest tests/test_question_detection.py -k "mark_scheme"
```

Then full suite:

```bash
.venv/bin/python -m pytest
```

Diff hygiene:

```bash
git diff --check
```

Rerun the 9709-only OCR candidate:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```

Verify candidate scope:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status --input output_ocr_candidate/json/question_bank.json
```

Compare against the frozen 9709-only OCR baseline:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-compare \
  --iteration agent_handoffs/auto_triage/iteration_003 \
  --baseline-triage output_ocr_candidate/triage/iteration_002 \
  --current output_ocr_candidate/json/question_bank.json \
  --output output_ocr_candidate/triage/iteration_002/comparison.auto-iteration-003.json \
  --test-status pass
```

Do not use any command that routes back to `output/json/question_bank.json`.

## Acceptance Criteria

Accept only if:

- baseline and candidate remain 9709-only
- OCR remains enabled for all records
- record scope still matches baseline, or differences are fully explained
- `31autumn25_q09` mark-scheme crop/total evidence improves
- `31autumn25` paper total improves from `71` to `75`
- `paper_total_mismatch` decreases by at least the `31autumn25` group size if no other regressions occur
- `hard_failure_count` does not increase
- `worsened_record_count == 0`
- `status_regressions == []`
- no validation gates are suppressed
- canonical `output/json/question_bank.json` is not overwritten
- tests pass

## Stop Conditions

Stop before implementation if:

- Agent 2 cannot reproduce the `31autumn25_q09` evidence from the crops and PDFs.
- The mark-scheme rows for `9(a)`/`9(b)` cannot be located with confidence.
- The required fix expands into a broad mark-scheme table parser rewrite.
- The only way to reduce the mismatch is to prefer question totals globally or suppress validation flags.
- The fix would worsen mapping/validation/visual statuses elsewhere.
- Baseline/candidate scope stops matching.

Stop after implementation if:

- `31autumn25` does not improve.
- `worsened_record_count` is nonzero.
- hard failures increase.
- status regressions are reported.
- OCR/corpus scope is no longer 9709-only.

