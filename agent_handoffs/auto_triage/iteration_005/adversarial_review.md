# Adversarial Review: Iteration 005 OCR Candidate Hard-Failure Claim

Verdict: **ACCEPTED WITH CAVEATS**

The hard-failure target is met under the revised validation policy. The final OCR candidate has **44 hard failures under revised policy** across **1301 questions / 148 papers**.

This does **not** prove actual extraction/mapping error count is below 50. The reduction consists of **0 extraction/mapping repairs** and **89 validation-policy demotions** from the frozen 133-hard-failure baseline:

- **62** `paper_total_mismatch` records moved from hard failure to review-only paper evidence.
- **27** matched-paper `question_mark_total_mismatch` records moved from hard failure/mapping-fail to review-only metadata.
- **0** records were improved by changing question text, mark-scheme text, image paths, page refs, crop outputs, or extracted mark totals in the final candidate JSON.

## Verification Commands

- `git diff -- src/exam_bank/pipeline.py`, `git diff -- tests/test_pipeline_reconciliation.py`, `git diff -- tests/test_sample_pipeline.py`, and `git diff --stat` were empty because the worktree is clean and the implementation is already committed at `b27cd8c`.
- I audited the committed change with `git diff HEAD~1..HEAD`.
- `.venv/bin/python -m pytest`: **314 passed, 3 skipped**.
- `git diff --check HEAD~1..HEAD`: **failed** on trailing whitespace in generated triage HTML:
  - `output_ocr_candidate/triage/iteration_004/index.html`
  - `output_ocr_candidate/triage/iteration_005/index.html`
- `public/data/question_bank.json`: no diff; no tracked file at that path in this checkout.

## 1. Diff Audit

Changed file classifications:

| File | Classification | Audit note |
|---|---|---|
| `README.md` | reporting/counting correction | Documentation/runbook status updates; partly stale because it still describes the OCR candidate as 133 hard failures. |
| `agent_handoffs/auto_triage/iteration_004/agent1_request.md` | reporting/counting correction | Generated handoff artifact. |
| `agent_handoffs/auto_triage/iteration_004/commands.json` | reporting/counting correction | Generated handoff commands. |
| `agent_handoffs/auto_triage/iteration_004/decision.json` | reporting/counting correction | Generated acceptance decision. |
| `agent_handoffs/auto_triage/iteration_004/metrics_after.json` | reporting/counting correction | Generated metrics. |
| `agent_handoffs/auto_triage/iteration_004/metrics_before.json` | reporting/counting correction | Generated metrics. |
| `agent_handoffs/auto_triage/iteration_004/selected_target.json` | reporting/counting correction | Generated target selection. |
| `agent_handoffs/auto_triage/iteration_005/agent1_request.md` | reporting/counting correction | Generated handoff artifact. |
| `agent_handoffs/auto_triage/iteration_005/commands.json` | reporting/counting correction | Generated handoff commands. |
| `agent_handoffs/auto_triage/iteration_005/decision.json` | reporting/counting correction | Generated acceptance decision. |
| `agent_handoffs/auto_triage/iteration_005/metrics_after.json` | reporting/counting correction | Generated metrics. |
| `agent_handoffs/auto_triage/iteration_005/metrics_before.json` | reporting/counting correction | Generated metrics. |
| `agent_handoffs/auto_triage/iteration_005/selected_target.json` | reporting/counting correction | Generated target selection. |
| `docs/AUTO_TRIAGE.md` | reporting/counting correction | Workflow docs for auto-triage. |
| `docs/PROJECT_REVIEW.md` | reporting/counting correction | Project status docs. |
| `docs/ROADMAP.md` | reporting/counting correction | Roadmap docs. |
| `docs/TRIAGE_WORKFLOW.md` | reporting/counting correction | Workflow docs. |
| `docs/TRUST_MODEL.md` | reporting/counting correction | Trust-model docs; no code behavior. |
| `output_ocr_candidate/json/question_bank.json` | validation-policy reclassification | 89 records changed only trust/status metadata. |
| `output_ocr_candidate/triage/iteration_004/baseline_question_bank.json` | reporting/counting correction | Frozen baseline artifact. |
| `output_ocr_candidate/triage/iteration_004/comparison.auto-iteration-004.json` | reporting/counting correction | Generated comparison. |
| `output_ocr_candidate/triage/iteration_004/index.html` | reporting/counting correction | Generated gallery; has trailing whitespace. |
| `output_ocr_candidate/triage/iteration_004/review.jsonl` | reporting/counting correction | Empty review artifact. |
| `output_ocr_candidate/triage/iteration_004/sample.json` | reporting/counting correction | Generated triage sample. |
| `output_ocr_candidate/triage/iteration_004/summary.json` | reporting/counting correction | Generated triage summary. |
| `output_ocr_candidate/triage/iteration_005/baseline_question_bank.json` | reporting/counting correction | Frozen baseline artifact. |
| `output_ocr_candidate/triage/iteration_005/comparison.auto-iteration-005.json` | reporting/counting correction | Generated comparison. |
| `output_ocr_candidate/triage/iteration_005/index.html` | reporting/counting correction | Generated gallery; has trailing whitespace. |
| `output_ocr_candidate/triage/iteration_005/review.jsonl` | reporting/counting correction | Empty review artifact. |
| `output_ocr_candidate/triage/iteration_005/sample.json` | reporting/counting correction | Generated triage sample. |
| `output_ocr_candidate/triage/iteration_005/summary.json` | reporting/counting correction | Generated triage summary. |
| `src/exam_bank/pipeline.py` | validation-policy reclassification | Adds selective paper-total hard-fail logic and matched-paper mark-total reconciliation. |
| `tests/test_pipeline_reconciliation.py` | test expectation update | Adds tests for revised policy and structural-failure preservation. |
| `tests/test_sample_pipeline.py` | test expectation update | Changes one sample expectation from hard fail to review under new policy. |

The `<50` reduction came from **status reclassification**, not extraction repair or mark-scheme mapping repair. A field-level comparison of `output_ocr_candidate/json/question_bank.json` against `HEAD~1` showed changed fields only in `notes.validation_flags`, `notes.validation_status`, `notes.mapping_status`, `notes.mapping_failure_reason`, `notes.review_flags`, `notes.topic_trust_status`, `notes.visual_curation_status`, top-level `visual_curation_status`, and `text_only_status`.

## 2. Test Honesty Audit

Changed sample: `tests/test_sample_pipeline.py::test_repo_n24_p12_paper_total_is_matched_without_hiding_question_failures`.

Old expectation:

- `q1.validation_status == "fail"`
- `"question_mark_total_mismatch" in q1.validation_flags`

New expectation:

- `q1.validation_status == "review"`
- `q1.markscheme_mapping_status == "pass"`
- `"question_mark_total_mismatch" not in q1.validation_flags`
- `"question_mark_total_review_only" in q1.review_flags`

This follows from a documented policy change: when the paper total exactly matches and the mark-scheme crop exists, `question_mark_total_mismatch` may be treated as review-only metadata rather than a mapping hard failure.

The old hard-failure behavior is still tested for structurally bad records:

- `test_paper_total_mismatch_hard_fails_structurally_suspect_record`
- `test_question_mark_total_mismatch_stays_hard_fail_when_paper_total_mismatches`
- `test_question_mark_total_mismatch_stays_hard_fail_with_structural_flag`

I do not mark this as a blocking test weakening. The tests prove the easier behavior and also retain explicit coverage that bad records remain hard failures.

## 3. Demotion Audit

Baseline paths used:

- Initial frozen baseline: `output_ocr_candidate/triage/iteration_004/baseline_question_bank.json`
- Iteration 005 baseline: `output_ocr_candidate/triage/iteration_005/baseline_question_bank.json`
- Final candidate: `output_ocr_candidate/json/question_bank.json`
- Comparison: `output_ocr_candidate/triage/iteration_005/comparison.auto-iteration-005.json`

Demotion totals:

- Initial frozen hard failures: **133**
- Demoted from initial baseline to final non-hard: **89**
- `paper_total_mismatch-only`: **62**
- matched-paper `question_mark_total_mismatch`: **27**
- other demotions: **0**

### paper_total_mismatch-only demotions (62)

| question_id | paper | prev val/map/scope | current val/map/scope | previous validation_flags | current validation_flags | q/ms totals | paper exp/det | demotion reason | safety |
|---|---|---|---|---|---|---|---|---|---|
| 12autumn21_q01 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q02 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q03 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q04 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q05 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q06 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q07 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q08 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q09 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q10 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 7/7 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 12autumn21_q11 | 12autumn21 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 11/11 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q01 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 5/5 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q02 | 15autumn25 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q03 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 3/3 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q04 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 7/7 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q06 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 7/7 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q07 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 8/8 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q08 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 7/7 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q09 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 7/7 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 15autumn25_q10 | 15autumn25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 9/9 | 75/76 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q01 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q03 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q04 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q06 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q07 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 7/7 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q08 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q09 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 9/9 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q10 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 10/10 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32spring22_q11 | 32spring22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 11/11 | 75/72 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q01 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q02 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 3/3 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q03 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q04 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q06 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 7/7 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q07 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 9/9 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q08 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q09 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 10/10 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q10 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 10/10 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 32summer23_q11 | 32summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 9/9 | 75/78 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33autumn25_q01 | 33autumn25 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/67 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q01 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q02 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 3/3 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q03 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 4/4 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q04 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q05 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q06 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q07 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q08 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 33summer23_q10 | 33summer23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 10/10 | 75/74 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer22_q01 | 51summer22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 50/51 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer22_q02 | 51summer22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 6/6 | 50/51 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer22_q03 | 51summer22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 9/9 | 50/51 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer22_q06 | 51summer22 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 50/51 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer25_q01 | 51summer25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 3/3 | 50/52 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer25_q02 | 51summer25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 7/7 | 50/52 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer25_q03 | 51summer25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 12/12 | 50/52 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 51summer25_q04 | 51summer25 | fail/pass/review | review/pass/review | paper_total_mismatch | - | 12/12 | 50/52 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 52autumn23_q01 | 52autumn23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 50/48 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 52autumn23_q02 | 52autumn23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 50/48 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 52autumn23_q03 | 52autumn23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 5/5 | 50/48 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 52autumn23_q04 | 52autumn23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 8/8 | 50/48 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |
| 52autumn23_q07 | 52autumn23 | fail/pass/clean | review/pass/clean | paper_total_mismatch | - | 10/10 | 50/48 | paper mismatch retained in review_flags; per-record q/ms totals match and no hard structural/mapping signal | safe as review |

### matched-paper question_mark_total_mismatch demotions (27)

| question_id | paper | prev val/map/scope | current val/map/scope | previous validation_flags | current validation_flags | q/ms totals | paper exp/det | demotion reason | safety |
|---|---|---|---|---|---|---|---|---|---|
| 11autumn21_q05 | 11autumn21 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 1/5 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 11summer25_q05 | 11summer25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 3/7 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 12autumn24_q01 | 12autumn24 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 1/5 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 12autumn25_q06 | 12autumn25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 5/9 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 12spring21_q09 | 12spring21 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 4/9 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 12summer24_q08 | 12summer24 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 4/10 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 12summer25_q10 | 12summer25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 5/10 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 13autumn22_q11 | 13autumn22 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 4/11 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 13autumn23_q04 | 13autumn23 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 4/7 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 13autumn25_q10 | 13autumn25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 5/9 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 13summer25_q10 | 13summer25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 3/11 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 32autumn25_q10 | 32autumn25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 4/9 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 32spring24_q01 | 32spring24 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 3/2 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 32spring24_q10 | 32spring24 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 10/11 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 33autumn21_q11 | 33autumn21 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 4/10 | 75/75 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 41autumn23_q06 | 41autumn23 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 4/9 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 41autumn24_q07 | 41autumn24 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 4/8 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 41summer21_q07 | 41summer21 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 6/11 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 41summer25_q04 | 41summer25 | fail/fail/review | review/pass/review | question_mark_total_mismatch | - | 3/6 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 42spring23_q06 | 42spring23 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 3/9 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 42summer24_q04 | 42summer24 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 3/7 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 42summer24_q06 | 42summer24 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 1/11 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 43autumn22_q06 | 43autumn22 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 3/10 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 51summer23_q04 | 51summer23 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 5/9 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 51summer24_q02 | 51summer24 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 3/7 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 53autumn23_q05 | 53autumn23 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 5/11 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |
| 53autumn23_q06 | 53autumn23 | fail/fail/clean | review/pass/clean | question_mark_total_mismatch | - | 5/10 | 50/50 | paper total matched; q/ms total mismatch retained as review-only metadata | manual review |

### Other demotions

None.

## 4. Old-Definition Recount

Final candidate hard-failure counts:

| Definition | Hard failures |
|---|---:|
| Revised policy | 44 |
| If `paper_total_mismatch` still hard-failed every affected record | 106 |
| If `question_mark_total_mismatch` still hard-failed every affected record | 71 |
| Both old strict rules together | 133 |

This is the critical distinction:

- **44 hard failures under revised validation policy.**
- **133 hard failures under the old strict rules.**
- Therefore, this does **not** prove actual extraction/mapping errors reduced below 50.

## 5. Scope Regression Audit

Earlier progress notes reportedly mentioned 18 records moving `scope_quality_status=clean -> review` due to `paper_total_focus_candidate`.

I could not find that rejected comparison or log in the saved artifacts. Searches under `agent_handoffs/auto_triage/iteration_004`, `agent_handoffs/auto_triage/iteration_005`, `output_ocr_candidate/triage/iteration_004`, and `output_ocr_candidate/triage/iteration_005` did not find the referenced 18-record regression.

Independent recounts:

| Comparison | clean -> review scope changes |
|---|---:|
| iteration 004 baseline -> iteration 005 baseline | 0 |
| iteration 005 baseline -> final candidate | 0 |
| iteration 004 baseline -> final candidate | 0 |

Scope counts stayed unchanged: `clean: 919`, `review: 376`, `fail: 6`.

Conclusion: preserving existing `scope_quality_status` was a no-op preservation in the final artifact, not evidence of a hidden scope regression. The specific 18-record rejected intermediate state remains unresolved because it is not present in saved logs.

## 6. Structural-Failure Preservation Audit

Current candidate structural signals:

| Signal | Current records | Hard failures | Status summary |
|---|---:|---:|---|
| `question_scope_contaminated` | 6 | 6 | all `fail/fail/fail/fail` |
| `question_subparts_incomplete` | 8 | 8 | all hard; mixed scope clean/review |
| `missing_terminal_mark_total` | 1 | 1 | `fail/fail/review/fail` |
| `mark_scheme_part_structure_mismatch` | 2 | 2 | both mapping fail |
| `weak_question_anchor` | 10 | 10 | all hard |
| `polluted_pass_requires_review` | 4 | 4 | all hard |

Current mapping-failure reasons:

- `question_mark_total_mismatch`: 14
- `weak_question_anchor`: 10
- `question_subparts_incomplete`: 6
- `question_scope_contaminated`: 6
- `mark_scheme_part_structure_mismatch`: 2
- `missing_terminal_mark_total`: 1

No structurally suspect record was demoted to review solely because paper totals matched. Structural demotions from the initial baseline to final candidate: **0**.

## 7. Manual Sample Audit

I manually inspected metadata for 10 demoted records: 5 paper-total demotions and 5 question-mark-total demotions. I checked image path presence, mark-scheme crop presence, page refs, mapping status, totals, and whether the mismatch looked metadata-only rather than missing evidence.

| question_id | Group | Image paths | Page refs | Mapping | Marks evidence | Audit judgement |
|---|---|---|---|---|---|---|
| `12autumn21_q01` | paper_total_mismatch | question and mark scheme images exist | question `[2,3]`, mark scheme `[6]` | pass | q/ms `4/4`, paper `75/74` | Safe as review; per-record evidence coherent, paper mismatch retained. |
| `12autumn21_q02` | paper_total_mismatch | question and mark scheme images exist | question `[3,4]`, mark scheme `[7]` | pass | q/ms `5/5`, paper `75/74` | Safe as review. |
| `12autumn21_q03` | paper_total_mismatch | question and mark scheme images exist | question `[4,5]`, mark scheme `[7,8]` | pass | q/ms `5/5`, paper `75/74` | Safe as review; stitched mark scheme present. |
| `12autumn21_q04` | paper_total_mismatch | question and mark scheme images exist | question `[5,6]`, mark scheme `[8]` | pass | q/ms `4/4`, paper `75/74` | Safe as review. |
| `12autumn21_q05` | paper_total_mismatch | question and mark scheme images exist | question `[6,7]`, mark scheme `[9]` | pass | q/ms `6/6`, paper `75/74` | Safe as review. |
| `11autumn21_q05` | question_mark_total_mismatch | question and mark scheme images exist | question `[7,8]`, mark scheme `[9,10]` | pass after demotion | q/ms `1/5`, paper `75/75`; question text visibly contains `[3]`, `[1]`, `[1]` | Review-only is plausible as mark-total aggregation metadata; not an extraction fix. |
| `11summer25_q05` | question_mark_total_mismatch | question and mark scheme images exist | question `[7,8]`, mark scheme `[15]` | pass after demotion | q/ms `3/7`, paper `75/75`; question text contains `[2]`, `[2]`, `[3]` | Needs manual review; metadata discrepancy not missing crop evidence. |
| `12autumn24_q01` | question_mark_total_mismatch | question and mark scheme images exist | question `[2,3]`, mark scheme `[6]` | pass after demotion | q/ms `1/5`, paper `75/75`; question text contains `[3]`, `[1]`, `[1]` | Needs manual review; detector is taking terminal/subpart marks incorrectly. |
| `12autumn25_q06` | question_mark_total_mismatch | question and mark scheme images exist | question `[8,9,10]`, mark scheme `[15,16]` | pass after demotion | q/ms `5/9`, paper `75/75`; question text contains `[2]`, `[1]`, `[1]`, `[5]` | Needs manual review; review status is appropriate. |
| `12spring21_q09` | question_mark_total_mismatch | question and mark scheme images exist | question `[14,15,16]`, mark scheme `[11]` | pass after demotion | q/ms `4/9`, paper `75/75`; question text contains `[3]`, `[2]`, `[4]` | Needs manual review; metadata aggregation discrepancy, not absent mapping evidence. |

No sampled demotion had missing question image paths or missing mark-scheme crops. The 27 question-mark-total demotions should remain manual-review records; they are not safe to describe as fixed extraction/mapping.

## Final Verdict Language

The hard-failure target is met under the revised validation policy.

The reduction consists of **0 extraction/mapping repairs** and **89 validation-policy demotions**.

This does **not** prove actual extraction error count is below 50. It proves only that the hard-failure count is 44 under the new policy that treats selected paper-total and mark-total mismatches as review-only evidence.

## Plain-English Summary For Blake

The count is below 50, but mostly because the definition changed. The new policy looks defensible: clean records in a paper-total-mismatched paper no longer all get punished, and structurally bad records still hard-fail. The mark-total mismatch demotions are more delicate: the images and mark-scheme crops are present, but the mark totals are still wrong metadata, so those records must remain review-only and should not be counted as fixed.

Do not claim the pipeline has fewer than 50 actual extraction/mapping errors. Claim: **44 hard failures under the revised validation policy; 133 under the old strict policy.**
