# Agent 1 Plan — Exam Bank Pipeline iteration_001

## 1. Current repo state

The project is an image-first CAIE 9709 extraction pipeline: question PNG crops and mark-scheme PNG crops remain the student-facing source of truth, while extracted/native/OCR text supports search, metadata, trust gating, topic labeling, and review workflows.

Recent code has conservative OCR/native candidate selection in `src/exam_bank/ocr.py`, integration in `src/exam_bank/pipeline.py`, export metadata in `src/exam_bank/exporters.py`, and regression tests in `tests/test_ocr.py`. The selector only chooses OCR when it clears a score margin and avoids hard rejection reasons such as failed scope quality, missing question numbers, missing subparts, lost mark brackets, page furniture, and next-question contamination.

`output/json/question_bank.json` is present with `record_count: 1301`, but the current export appears stale for this feature: all records have blank/null candidate-selection metadata in notes (`text_candidate_source`, `text_candidate_decision`, `ocr_selected`, and score fields), even though `ocr_ran` is true. That means the next step is full-bank measurement/reporting and stale-export detection, not another scoring implementation pass.

## 2. Iteration target

Measure OCR candidate-selection behavior across the full exported bank, compare it with a baseline export when available, sample OCR-selected records for human judgment, and decide whether the current conservative selection is safe, useful, too conservative, or too aggressive.

## 3. Non-goals

- No broad refactors.
- No new OCR engine or OCR preprocessing variants.
- No DeepSeek behavior changes.
- No topic-classification changes except auditing topic changes caused by selected text.
- No crop/scope detection fixes.
- No mark-scheme mapping fixes.
- No adaptive trainer work.
- No schema-breaking changes.
- No change that makes OCR or extracted text the student-facing source of truth.
- No weakening trust gates, validation semantics, review/fail gates, or readiness gating.
- No committing generated PNGs, PDFs, `.env`, or large output folders.

## 4. Files/modules likely involved

Inspection-only:

- `README.md`
- `config.yaml`
- `pyproject.toml`
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

May be edited by Agent 2 or Agent 3:

- A small reporting script/module if one exists or is added, preferably under a tooling-appropriate path already used by the repo.
- `src/exam_bank/cli.py` only if Agent 2 chooses a durable report command and keeps it narrow.
- `tests/test_ocr.py` for candidate-selection/report input fixtures.
- `tests/test_output_contract.py` for export metadata guard coverage if needed.
- A focused new test file only if local patterns make that cleaner than expanding existing tests.

## 5. Full-bank measurement plan

Agent 2 should build or run a report against a full-bank export that contains candidate-selection metadata. First detect whether the export is stale by checking whether `notes.text_candidate_source`, `notes.text_candidate_decision`, `notes.ocr_selected`, `notes.native_text_score`, `notes.ocr_text_score`, and `notes.selected_text_score` are populated. If they are absent across the bank, report that explicitly and produce either a fresh full-bank export with OCR enabled or a clear blocker explaining why the fresh run could not complete.

Required summary counts:

- total records
- `ocr_selected` count
- `text_candidate_source` distribution
- top `text_candidate_decision` values
- top `ocr_rejected_reasons`
- native text score summary: count, min, p25, median, p75, max, mean
- OCR text score summary: count, min, p25, median, p75, max, mean
- selected text score summary: count, min, p25, median, p75, max, mean
- `text_fidelity_status` distribution
- `text_only_status` distribution
- `visual_curation_status` distribution
- `question_text_trust` distribution

The report should treat missing/null candidate fields as reportable data quality findings, not as zeroes.

## 6. Before/after comparison plan

Agent 2 should look for a previous baseline export before regenerating or comparing. Candidate locations include an explicitly documented previous run, a user-provided path, or a non-output baseline JSON if present. I found no baseline/comparison JSON outside `output/` during planning, so absence is likely and must be stated clearly.

If a baseline exists, compare records by `question_id` and report:

- question_text changes
- OCR selection changes
- `text_fidelity_status` changes
- `text_only_status` changes
- `visual_curation_status` changes
- `question_text_trust` changes
- topic changes
- `mapping_status` changes
- `validation_status` changes
- records that improved
- records that worsened

If no baseline exists, Agent 2 must still produce current-bank measurement and mark before/after comparison as blocked by missing baseline, not failed.

## 7. Representative sample plan

Agent 2 should produce a representative table of OCR-selected records. If no records are OCR-selected after a fresh candidate-aware export, sample high-margin OCR candidates that were rejected and explain why no selected sample exists.

Required columns:

- `question_id`
- `paper_family`
- old question_text, if baseline exists
- new question_text
- `ocr_text`
- `native_text_score`
- `ocr_text_score`
- `selected_text_score`
- decision reasons
- rejected reasons if any
- `text_only_status`
- `visual_curation_status`
- human judgment: good selection / questionable / bad selection

The sample should cover multiple paper families and include boundary cases near the selection threshold, not only obvious wins.

## 8. Risk checks

Agent 4 and Agent 5 should audit:

- OCR selected but lost marks, subparts, or question number.
- OCR selected but introduced diagram labels, page furniture, barcode/header text, or next-question text.
- OCR selected for records with `scope_quality_status: fail`.
- OCR selection inflated `text_only_status`, `visual_curation_status`, or `question_text_trust` without real text improvement.
- Topic changed unexpectedly because selected text changed.
- `text_fidelity_status`, `mapping_status`, or `validation_status` worsened without explanation.
- Score margins are too loose and permit false positives.
- Score margins are too strict and reject obvious OCR improvements.
- Existing generated export lacks candidate metadata, causing misleading measurement unless regenerated or flagged.

## 9. Agent 3 test plan

Agent 3 should focus on tests and guards for measurement/reporting, not new OCR behavior. Verify or add tests that:

- Candidate-selection metadata is present in exported notes for newly exported records.
- A report handles missing/null candidate fields as stale or incomplete export data.
- Score summaries ignore nulls instead of treating them as numeric values.
- Rejected-reason aggregation handles empty and multi-reason lists.
- OCR-selected records with math-heavy or degraded text do not automatically become `text_only_status: ready`.
- Existing candidate-selection tests still cover hard rejections for scope failure, page furniture, next-question contamination, missing mark brackets, and missing expected structure.

Do not test future OCR preprocessing, DeepSeek behavior, crop detection, mark-scheme mapping, or topic-classification tuning in this iteration.

## 10. Agent 2 implementation/reporting plan

Agent 2 should mostly build measurement/reporting, not tune scoring. A small durable report command or script is preferred if it can be added without broad CLI churn; a one-off documented report is acceptable if permanent tooling is premature.

Agent 2 should:

- Detect whether `output/json/question_bank.json` contains candidate-selection metadata.
- If stale, regenerate or request/use a fresh candidate-aware export with OCR enabled.
- Produce the full-bank summary counts from section 5.
- Produce before/after comparison if a baseline exists.
- Produce the representative OCR-selected sample table and manual judgments.
- Report whether any readiness/status fields appear inflated by OCR selection.
- Defer scoring threshold changes unless there is a tiny obvious bug with clear evidence.

Agent 2 must not tune candidate thresholds without explicit evidence and approval.

## 11. Acceptance criteria

- Full tests pass.
- Full-bank OCR candidate-selection measurement is produced from a candidate-aware export, or stale/missing metadata is explicitly reported as the blocker.
- Before/after comparison is produced if a baseline exists.
- Missing baseline is explicitly reported if no baseline exists.
- OCR-selected records are sampled and manually categorized, or the report explains why no OCR-selected records exist.
- No evidence of readiness inflation.
- No generated artifacts are accidentally tracked.
- Any recommended scoring changes are deferred unless tiny and clearly justified.

## 12. Stop conditions

- Generated output is accidentally tracked.
- Tests fail.
- Full pipeline or report command fails.
- Candidate metadata remains missing after a fresh export attempt.
- OCR-selected records show serious false positives.
- Many readiness statuses improve without real text improvement.
- Schema-breaking changes appear.
- Scope expands beyond OCR candidate-selection measurement.
