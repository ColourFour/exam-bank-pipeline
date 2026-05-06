# Agent 3 Tests — iteration_001

## Phase

Post-Agent-2 verification.

## Tests added or reviewed

- Added focused OCR candidate audit tests in `tests/test_audit.py`:
  - `test_ocr_candidate_audit_reports_missing_metadata_counts_and_statuses`
  - `test_ocr_candidate_audit_flags_stale_export_and_ignores_null_scores`
  - `test_ocr_candidate_audit_compares_baseline_and_status_movement`
  - `test_ocr_candidate_audit_flags_suspicious_selected_records_and_readiness_inflation`
  - `test_write_ocr_candidate_audit_writes_deterministic_json`
- Reviewed existing OCR/export guard coverage in `tests/test_ocr.py` and `tests/test_output_contract.py`.

## Commands run

- `.venv/bin/python -m pytest tests/test_audit.py -q`
  - Result: `1 failed, 5 passed`.
  - Failure: `test_ocr_candidate_audit_flags_suspicious_selected_records_and_readiness_inflation`.
  - Missing risk reasons: `missing_question_number`, `missing_marks`, `selected_with_hard_failure`.
- `.venv/bin/python -m pytest tests/test_ocr.py tests/test_output_contract.py -q`
  - Result: `16 passed`.
- `.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output /tmp/ocr_candidate_audit_agent3.json`
  - Result: passed.
  - Report confirms `1301` records and `stale_or_candidate_unaware_export`.
  - Candidate fields are missing on all records: `text_candidate_source`, `text_candidate_decision`, `ocr_selected`, `native_text_score`, `ocr_text_score`, `selected_text_score`.
- `.venv/bin/python -m pytest -q`
  - Result: `1 failed, 263 passed, 3 skipped`.
  - Same Agent 3 guard failure as above.
- `git status --short`
  - Result:
    - `M agent_handoffs/iteration_001/agent1_plan.md`
    - `M src/exam_bank/audit.py`
    - `M tests/test_audit.py`
    - `?? agent_handoffs/iteration_001/agent2_impl_notes.md`
    - `?? scripts/`

## Scope guard findings

Agent 2 stayed within the approved implementation scope for production behavior. The diff is limited to `src/exam_bank/audit.py` and `scripts/audit_ocr_candidates.py`; there is no OCR scoring/threshold tuning and no changes to crop detection, DeepSeek, topic classification, mark-scheme mapping, or adaptive trainer code.

Agent 2 did not edit tests. Agent 3 added the focused audit tests above.

## Repo hygiene findings

No generated report artifact is tracked; Agent 3 wrote the report to `/tmp/ocr_candidate_audit_agent3.json`. Existing input PDFs are present in the repo/worktree but were not introduced by this iteration. `agent_handoffs/` and `scripts/` remain untracked in the current status.

## Verdict

NEEDS FIX

The measurement/reporting implementation is mostly testable and the command runs, but the suspicious OCR-selected audit does not yet flag all Agent 1 required risks. It flags scope failure, degraded selected text, and possible page furniture, but not selected OCR records with missing question number, missing marks, or hard validation failure.

## Next-loop notes

- Agent 2 should add suspicious-risk reasons for `missing_question_number`, `missing_marks`, and `selected_with_hard_failure`, then rerun `tests/test_audit.py` and the full suite.
- Current `output/json/question_bank.json` remains stale/candidate-unaware, so Agent 4/5 cannot audit real OCR selection behavior until a fresh candidate-aware export exists.
- Baseline comparison is still blocked by missing baseline export, not by audit command failure.

## Phase

Post-Agent-2 narrow-fix verification.

## Tests added or reviewed

- Reviewed Agent 2's fix addendum in `agent_handoffs/iteration_001/agent2_impl_notes.md`.
- Re-ran the focused audit guard in `tests/test_audit.py`, including `test_ocr_candidate_audit_flags_suspicious_selected_records_and_readiness_inflation`.
- Reviewed the production diff shape for `src/exam_bank/audit.py`; the post-fix risk reasons are reporting-only and cover `missing_question_number`, `missing_marks`, and `selected_with_hard_failure`.

## Commands run

- `.venv/bin/python -m pytest tests/test_audit.py -q`
  - Result: `6 passed`.
- `.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output /tmp/ocr_candidate_audit_agent3_postfix.json`
  - Result: passed.
  - Report confirms `1301` records and `stale_or_candidate_unaware_export`.
  - Candidate fields are missing on all records: `text_candidate_source`, `text_candidate_decision`, `ocr_selected`, `native_text_score`, `ocr_text_score`, `selected_text_score`.
  - Baseline comparison remains unavailable with reason `missing_baseline`.
- `.venv/bin/python -m pytest tests/test_ocr.py tests/test_output_contract.py -q`
  - Result: `16 passed`.
- `.venv/bin/python -m pytest -q`
  - Result: `264 passed, 3 skipped`.
- `git status --short`
  - Result before appending this section:
    - `M agent_handoffs/iteration_001/agent1_plan.md`
    - `M src/exam_bank/audit.py`
    - `M tests/test_audit.py`
    - `?? agent_handoffs/iteration_001/agent2_impl_notes.md`
    - `?? agent_handoffs/iteration_001/agent3_tests.md`
    - `?? scripts/`

## Scope guard findings

Agent 2's narrow fix stayed within the approved reporting/audit scope. I found no OCR scoring or threshold changes and no changes to extraction behavior, crop detection, DeepSeek, topic classification, mark-scheme mapping, or adaptive trainer code.

Agent 2 did not edit tests. Agent 3's existing focused tests remain the only test changes in this iteration.

## Repo hygiene findings

No generated report artifact is tracked; the post-fix report was written to `/tmp/ocr_candidate_audit_agent3_postfix.json`. `agent_handoffs/` and `scripts/` remain untracked in the current worktree state.

## Verdict

IMPLEMENTATION VERIFIED

The narrow suspicious OCR-selected flag fix satisfies the Agent 3 guard. The report command runs, focused tests pass, the existing OCR/output guards pass, and the full suite passes.

## Next-loop notes

- Agent 4/5 can review the reporting implementation and tests, but they still cannot audit real OCR selection behavior from the current bank because `output/json/question_bank.json` is stale/candidate-unaware.
- Iteration 2 should prioritize generating a fresh full-bank export with OCR candidate metadata into a separate output path, then rerun `scripts/audit_ocr_candidates.py`.
- Baseline comparison remains blocked until a baseline export path is provided or preserved deliberately.
