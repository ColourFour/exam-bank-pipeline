# Agent 4 Integration Audit — iteration_001

## 1. Verdict

PASS WITH DEFERRALS

The implementation is scoped to OCR candidate audit/reporting, the full suite passes, and the report command correctly detects that the current bank is stale/candidate-unaware. This iteration does not validate real OCR selection quality because `output/json/question_bank.json` has no populated candidate-selection metadata.

## 2. Scope compliance

Agent 2 stayed within the Agent 1 reporting plan after the Agent 3 fix. The production diff is limited to `src/exam_bank/audit.py` plus the new `scripts/audit_ocr_candidates.py` helper command.

I found no changes to OCR scoring/thresholds, extraction behavior, crop detection, DeepSeek enrichment, topic classification, mark-scheme mapping, or adaptive trainer code.

Agent 3's tests are meaningful for the reporting layer. They cover stale metadata detection, null score handling, rejected-reason aggregation, baseline comparison, deterministic JSON writing, suspicious OCR-selected risk reasons, and readiness-inflation risk surfacing.

## 3. Tests and commands run

- `git status --short`
  - `M agent_handoffs/iteration_001/agent1_plan.md`
  - `M src/exam_bank/audit.py`
  - `M tests/test_audit.py`
  - `?? agent_handoffs/iteration_001/agent2_impl_notes.md`
  - `?? agent_handoffs/iteration_001/agent3_tests.md`
  - `?? scripts/`
- `git diff --stat`
  - `agent_handoffs/iteration_001/agent1_plan.md`: 391-line rewrite already present from prior prompt/plan work
  - `src/exam_bank/audit.py`: 350 added lines
  - `tests/test_audit.py`: 250-line focused audit test expansion
- `.venv/bin/python -m pytest -q`
  - `264 passed, 3 skipped in 132.39s`
- `.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output /tmp/ocr_candidate_audit_agent4.json`
  - Passed and wrote the JSON report to `/tmp`.
- `git ls-files output`
  - Only `output/json/.gitkeep` is tracked.
- `git ls-files '.env' '*.env'`
  - No tracked env files found.

## 4. Output behavior findings

Current bank report:

- Total records: `1301`
- Candidate metadata presence:
  - `text_candidate_source`: `0` present, `1301` missing
  - `text_candidate_decision`: `0` present, `1301` missing
  - `ocr_selected`: `0` present, `1301` missing
  - `native_text_score`: `0` present, `1301` missing
  - `ocr_text_score`: `0` present, `1301` missing
  - `selected_text_score`: `0` present, `1301` missing
- Data quality findings:
  - `candidate_metadata_missing_for_all_records:text_candidate_source,text_candidate_decision,ocr_selected,native_text_score,ocr_text_score,selected_text_score`
  - `stale_or_candidate_unaware_export`
- OCR-selected count: `0`, but this is not a real selection result because metadata is missing.
- Text candidate source/decision counts: `missing: 1301`
- OCR rejected reasons: none reportable.
- Candidate score summaries: all score counts are `0`; null/missing scores are not treated as zero.
- Text fidelity distribution:
  - `clean: 310`
  - `degraded: 938`
  - `unusable: 53`
- Text-only status distribution:
  - `fail: 1043`
  - `ready: 73`
  - `review: 185`
- Visual curation status distribution:
  - `fail: 420`
  - `ready: 217`
  - `review: 664`
- Question text trust distribution:
  - `high: 86`
  - `medium: 224`
  - `low: 938`
  - `unusable: 53`
- Baseline comparison: unavailable with reason `missing_baseline`.

## 5. OCR-selection risk findings

No real OCR-selected quality audit can be performed from the current export. The report returns:

- `suspicious_ocr_selected_records: []`
- `readiness_inflation_risk_records: []`
- `representative_records: []`

These empty lists are expected for the stale export and should not be interpreted as evidence that OCR selection is safe. The implementation tests do verify that the report would flag selected OCR records with missing question number, missing marks, hard failures, degraded selected text, page furniture text, and readiness inflation risks.

## 6. Generated artifact / repo hygiene findings

No generated report artifact was written into the repo during this audit; Agent 4 output went to `/tmp/ocr_candidate_audit_agent4.json`.

No generated output folder is tracked except the existing `output/json/.gitkeep`. No `.env` files are tracked.

The new `scripts/` directory is untracked as a whole, which currently contains `scripts/audit_ocr_candidates.py`. That is source/tooling, not generated output, but Agent 5 should make sure only the script is included if this iteration is committed.

## 7. Blocking issues

- Real OCR candidate-selection behavior remains blocked by the stale/candidate-unaware `output/json/question_bank.json`.
- Baseline comparison remains blocked by missing baseline export.
- No representative OCR-selected sample can be reviewed until a fresh candidate-aware export exists.

These are blockers for validating extraction improvement, not blockers for accepting the reporting tool itself.

## 8. Non-blocking deferrals

- CLI polish can wait. The standalone script is acceptable for this loop because the package CLI contract intentionally stays narrow.
- Human judgment labels in `representative_records` remain `not_reviewed`; that is acceptable until a fresh candidate-aware export produces sample rows.
- Any OCR threshold/scoring tuning should be deferred until the report is run against fresh candidate metadata.

## 9. Recommendation for Agent 5

Agent 5 should pass the iteration as reporting infrastructure with explicit deferrals. Do not claim OCR selection quality has improved or been validated.

Agent 5 should require iteration_002 to produce a fresh OCR-enabled full-bank export in a separate path, rerun `scripts/audit_ocr_candidates.py`, and review actual OCR-selected samples before any scoring or readiness-gating changes.

## 10. Next-loop seed

Concrete evidence for `iteration_002`:

- Current export has `1301` records and all six candidate metadata fields missing across every record.
- Fresh export command candidate: `.venv/bin/python -m exam_bank.cli process --input input --output <separate-output-path> --enable-ocr`
- Audit command to rerun: `.venv/bin/python scripts/audit_ocr_candidates.py --input <fresh-output-path>/json/question_bank.json --json-output /tmp/ocr_candidate_audit_iteration_002.json`
- Preserve or nominate a baseline export before comparing; none exists in this iteration.
- Review OCR-selected examples manually before changing thresholds.
- Treat suspicious selected records with missing question number, missing marks, hard failures, page furniture, or readiness inflation as stop-condition evidence.
