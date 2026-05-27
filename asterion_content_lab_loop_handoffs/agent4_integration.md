# Agent 4 Integration Audit — Asterion Content Lab P3 70% Pass Loop

Task 5/27: Independently audit Agent 2 and Agent 3 work before final review.

## 1. Verdict options

Use one of:

- `PASS`
- `PASS WITH DEFERRALS`
- `REJECTED`

Do not pass the iteration as a 70% success unless the persisted P3 sample reports `sample_pass_rate >= 0.70` and the safety checks below are clean.

## 2. What to inspect

Inspect the actual diff and generated reports.

Required repo checks:

```bash
git status --short
git diff --stat
git diff --name-only
git ls-files output
git ls-files '.env' '*.env'
```

Required source/output review:

- Agent 1 plan.
- Agent 2 implementation notes.
- Agent 3 tests handoff.
- Content Lab audit script/module.
- Content Lab candidate validator/export adapter changes.
- Test changes.
- `output/audits/content_lab_iteration_002/audit_summary.json`.
- `output/audits/content_lab_iteration_002/p3_sample_frame.csv`.
- `output/audits/content_lab_iteration_002/p3_sample_results.csv`.
- `output/audits/content_lab_iteration_002/blocker_breakdown.csv`.
- Current Asterion candidate export.

## 3. Commands to run

Run Agent 2’s audit command exactly as documented, then run tests.

Suggested commands, adjusted to actual paths:

```bash
.venv/bin/python scripts/audit_asterion_content_lab_readiness.py \
  --candidates output/asterion/exports/latest/asterion_content_lab_candidates_v1.json \
  --question-bank output/json/question_bank.json \
  --topic-routing output/json/question_bank.topic_routing.v1.json \
  --asterion-bank output/asterion/exports/latest/asterion_question_bank_v1.json \
  --artifact-root output \
  --sample-size 100 \
  --out-dir /tmp/content_lab_agent4_audit

.venv/bin/python -m pytest tests/test_asterion* tests/test_content_lab* tests/test_output_contract.py -q
.venv/bin/python -m pytest -q
```

If full suite is too slow or environment-blocked, run the broadest feasible test set and state the limitation clearly.

## 4. Integration checks

Verify:

- The sample denominator is stable and reported.
- The sample includes multiple regions and does not cherry-pick easy candidates.
- Pass rate is calculated from the persisted sample results, not from a different later sample.
- All `pass` records have canonical question image evidence.
- All `pass` records have required mark-scheme evidence.
- All `pass` records resolve to reviewed Asterion P3 skill IDs or accepted reviewed bridges.
- New gate fields are mapped honestly and with provenance.
- `review_required` and `blocked_*` are not counted as pass.
- No unreviewed generated text is used as canonical evidence.
- No ambiguous/quarantined mappings were promoted.
- Student runtime export remains unchanged unless explicitly approved.
- `asterion_question_bank_v1.json` is not polluted with Content Lab candidate-only data.
- Generated output remains untracked.

## 5. Red-line failures

Reject if any are observed:

- Gates were weakened to hit 70%.
- Asterion skill mappings were guessed.
- Missing mark-scheme evidence was ignored.
- Runtime student data now consumes unreviewed Content Lab candidates.
- Sample was manipulated after seeing failures.
- Tests fail.
- Generated artifacts or secrets are tracked.
- Final notes claim whole-bank readiness from a sample.

## 6. Required output behavior findings

Report:

- Total P3 candidates.
- Sample size.
- Regions covered.
- Baseline pass rate, if available.
- Final pass rate.
- Target met: true/false.
- Top remaining blockers.
- Any records that appear falsely promoted.
- Any records that should pass but are still blocked due to schema mismatch.

## 7. Required handoff output

Write:

```text
agent_handoffs/content_lab_iteration_002/agent4_integration.md
```

Include:

- Verdict.
- Commands run and exact results.
- Repo hygiene findings.
- Output behavior findings.
- Safety/risk findings.
- Whether the 70% target is actually supported.
- Recommended Agent 5 decision.

## 8. Recommendation rule

If the pass rate is below 70% but the blockers are honestly measured and a safe blocker class was repaired, recommend `PASS WITH DEFERRALS` for the loop infrastructure, not for the 70% target.

If the pass rate is ≥70% and safety checks are clean, recommend `PASS`.

If the pass rate is ≥70% because the sample or gates were compromised, recommend `REJECTED`.
