# Agent 3 Tests — Asterion Content Lab P3 70% Pass Loop

Task 5/27: Verify the Content Lab candidate audit/repair pass with regression tests and prevent fake 70% success.

## 1. Phase

Post-Agent-2 verification for Asterion Content Lab P3 candidate readiness.

Agent 3 owns test integrity. Do not broaden into new extraction/OCR behavior, curriculum invention, or student UI work.

## 2. Test targets

Add or update focused tests for the reporting and validation layer created by Agent 2.

Required coverage:

- The Content Lab audit command writes the expected summary/report files.
- The audit identifies P3 candidates deterministically.
- The stratified sample is deterministic and persisted.
- The sample pass-rate formula is correct.
- `review_required`, `blocked_*`, and `fail` records do not count as pass.
- Missing candidate fields are reported as blockers, not silently ignored.
- New gate fields such as `source_skill_review_gate`, `mapping_review_gate`, and `mapping_review_satisfied` are accepted only when semantically valid.
- Legacy fields are not faked without provenance.
- Exam-bank canonical skill IDs must resolve to reviewed Asterion P3 skill IDs or an accepted reviewed bridge.
- Ambiguous or quarantined mappings remain blocked.
- Missing question image paths block candidates.
- Missing mark-scheme image paths block candidates unless explicitly expected and reported as blocked.
- Mark-total or subpart-mark contradictions block candidates.
- Generated explanations, OCR text, difficulty labels, or topic guesses cannot serve as canonical evidence.
- The audit reports `student_runtime_changed=false` unless an explicit approved runtime change occurred.
- The audit reports `trust_gates_weakened=false`; tests should fail if pass rate improves by lowering gates.
- Generated audit artifacts are written under ignored output paths.

## 3. Anti-gaming tests

Add tests that would catch bad behavior designed only to hit 70%:

- Sample excludes all failing regions without explanation.
- Sample size is silently reduced after failures.
- `review_required` is counted as pass.
- Missing Asterion skill IDs are auto-filled from topic labels.
- Missing mark evidence is treated as pass because image-first display exists.
- Candidate records are promoted while `validation_status=fail` or equivalent.
- Content Lab candidates are routed into student runtime export.

## 4. Suggested test files

Use existing repo patterns. Likely candidates:

- `tests/test_asterion_content_lab_readiness_audit.py`
- `tests/test_asterion_export_contract.py`
- `tests/test_content_lab_candidates.py`
- `tests/test_output_contract.py`

If these files do not exist, create the narrowest new test file and keep fixtures small.

## 5. Commands to run

Run focused tests first:

```bash
.venv/bin/python -m pytest tests/test_asterion_content_lab_readiness_audit.py -q
```

Run related export/content-lab tests:

```bash
.venv/bin/python -m pytest tests/test_asterion* tests/test_content_lab* tests/test_output_contract.py -q
```

Run full suite if Agent 2 touched production validation/export logic:

```bash
.venv/bin/python -m pytest -q
```

If actual test file names differ, run the closest matching files and report exact commands.

## 6. Scope guards

Do not change:

- OCR scoring or thresholds.
- Crop detection.
- DeepSeek enrichment.
- Topic/difficulty classifiers.
- Student runtime loading behavior.
- Reviewed skill map content unless the task explicitly includes reviewed mapping data.
- Generated candidate data by hand to make tests pass.

## 7. Required verdict

Agent 3 must end with one of:

- `IMPLEMENTATION VERIFIED`
- `IMPLEMENTATION VERIFIED WITH DEFERRALS`
- `REJECTED`

Do not use `IMPLEMENTATION VERIFIED` unless tests prove the pass-rate calculation and blocker handling cannot be gamed.

## 8. Required handoff output

Write:

```text
agent_handoffs/content_lab_iteration_002/agent3_tests.md
```

Include:

- Tests added/changed.
- Commands run and exact results.
- Baseline/final pass metric observed from audit output, if available.
- Any blocker categories still not covered by tests.
- Any suspicious pass-rate movement.
- Scope guard findings.
- Verdict.

## 9. Next-loop notes

If the 70% target is not met, identify the next highest-value blocker class from the tested report output. Do not suggest weakening gates.
