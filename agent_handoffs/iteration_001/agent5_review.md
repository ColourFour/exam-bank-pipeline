# Agent 5 Final Review — iteration_001

## 1. Final verdict

ACCEPTED WITH DEFERRALS

This iteration is accepted as OCR candidate-selection reporting infrastructure only. It does not prove OCR selection quality improved, and it does not validate real OCR-selected records, because the available exported bank is stale or candidate-unaware.

## 2. What was reviewed

- `README.md`
- `pyproject.toml`
- `config.yaml`
- `agent_handoffs/iteration_001/Prompt/agent5_prompt.md`
- `agent_handoffs/iteration_001/agent1_plan.md`
- `agent_handoffs/iteration_001/agent2_impl_notes.md`
- `agent_handoffs/iteration_001/agent3_tests.md`
- `agent_handoffs/iteration_001/agent4_integration.md`
- `src/exam_bank/audit.py`
- `tests/test_audit.py`
- `scripts/audit_ocr_candidates.py`
- `output/json/question_bank.json`

Commands run:

```bash
git status --short
git diff --stat
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output /tmp/ocr_candidate_audit_agent5.json
.venv/bin/python -m pytest -q
```

Results:

- Full suite: `264 passed, 3 skipped`.
- OCR candidate audit command passed.
- Current export has `1301` records.
- All six OCR candidate metadata fields are missing on all `1301` records:
  - `text_candidate_source`
  - `text_candidate_decision`
  - `ocr_selected`
  - `native_text_score`
  - `ocr_text_score`
  - `selected_text_score`
- Audit data quality findings:
  - `candidate_metadata_missing_for_all_records:text_candidate_source,text_candidate_decision,ocr_selected,native_text_score,ocr_text_score,selected_text_score`
  - `stale_or_candidate_unaware_export`
- Baseline comparison is unavailable with reason `missing_baseline`.

## 3. Reasons to reject considered

- Tests fail: not observed. Full suite passed.
- Scope broadened into extraction behavior: not observed. The production behavior change is limited to audit/reporting helpers plus a standalone script.
- OCR scoring or thresholds changed: not observed.
- DeepSeek, crop detection, topic classification, mark-scheme mapping, or adaptive trainer behavior changed: not observed.
- Report hides stale metadata: not observed. The report explicitly flags the export as stale/candidate-unaware.
- Null scores treated as zeroes: not observed. Score summaries report count `0` with null summary values.
- Suspicious OCR-selected records are not flagged: initially found by Agent 3, then fixed and verified. The tests now cover missing question number, missing marks, hard failure, degraded selected text, page furniture, and readiness-inflation risks.
- Generated output or secrets tracked: not observed. `output/json/.gitkeep` is the only tracked output path; no tracked `.env` files were found.
- Schema-breaking production export changes: not observed in this iteration.

## 4. Blocking issues

No blockers for accepting the reporting infrastructure.

Blocking issues for validating extraction/OCR quality remain:

- `output/json/question_bank.json` is stale or candidate-unaware for OCR candidate-selection measurement.
- No baseline export exists for before/after comparison.
- No OCR-selected representative samples can be reviewed from the current export.
- Empty `suspicious_ocr_selected_records`, `readiness_inflation_risk_records`, and `representative_records` in this run are not evidence of safe OCR selection; they are a consequence of missing candidate metadata.

## 5. Non-blocking deferrals

- Generate a fresh OCR-enabled full-bank export before interpreting selection counts.
- Preserve or explicitly nominate a baseline export before comparing record changes.
- Keep the standalone script for now; package CLI integration can wait.
- Leave OCR threshold/scoring tuning untouched until fresh candidate-aware audit output shows actual selected and rejected candidate behavior.
- Human judgment labels remain `not_reviewed` until representative OCR-selected records exist.
- If committing this iteration, include `scripts/audit_ocr_candidates.py` intentionally and avoid committing generated `/output` artifacts.

## 6. Scores

Score out of 10:

- Correctness: 8
- Test integrity: 8
- Output honesty: 9
- Scope control: 9
- Maintainability: 8
- Usefulness for next iteration: 8

## 7. Acceptance rationale

The iteration gives the project a durable, test-covered way to audit OCR/native candidate-selection metadata and to identify stale exports instead of silently reporting misleading zeroes. The implementation is honest about missing data, provides status distributions from the current bank, supports optional baseline comparison, and includes risk sections for suspicious OCR-selected records and readiness inflation.

The key limitation is deliberate: this pass did not regenerate the bank and therefore cannot judge whether OCR selection is currently safe, too conservative, or too aggressive. That limitation is clearly surfaced by Agents 2, 3, 4, and this review.

## 8. Suggested next iteration

Recommended `iteration_002` target:

Generate a fresh OCR-enabled full-bank export into a separate output path, run `scripts/audit_ocr_candidates.py` against that export, and manually review the produced OCR-selected or high-margin rejected samples before changing any OCR scoring thresholds.

Suggested commands:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_iteration_002 --enable-ocr
.venv/bin/python scripts/audit_ocr_candidates.py --input output_iteration_002/json/question_bank.json --json-output /tmp/ocr_candidate_audit_iteration_002.json
```

Stop conditions for `iteration_002`:

- Full suite fails.
- Fresh export still lacks candidate metadata.
- OCR-selected records lose question numbers, subparts, or mark brackets.
- OCR-selected records include page furniture, headers, barcode fragments, diagram-label contamination, or next-question text.
- OCR selection occurs on hard scope, mapping, or validation failures.
- `text_only_status`, `visual_curation_status`, or `question_text_trust` improves without clear text-quality evidence.
- Generated exports, PDFs, or secrets become tracked.
- Any scoring change is attempted before reviewing fresh candidate-aware audit evidence.
