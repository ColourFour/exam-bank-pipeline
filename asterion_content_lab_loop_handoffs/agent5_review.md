# Agent 5 Final Review — Asterion Content Lab P3 70% Pass Loop

Task 5/27: Decide whether the Asterion Content Lab P3 candidate loop succeeded, failed safely, or needs another iteration.

## 1. Final verdict options

Use one of:

- `ACCEPTED — 70% TARGET MET`
- `ACCEPTED WITH DEFERRALS — LOOP INFRASTRUCTURE ONLY`
- `REJECTED`

Be strict. Do not mark the 70% target as met unless the evidence supports it.

## 2. Required evidence to review

Review:

- Agent 1 plan.
- Agent 2 implementation notes.
- Agent 3 test handoff.
- Agent 4 integration audit.
- The Content Lab audit summary JSON/MD.
- The persisted P3 sample frame and sample results.
- Blocker breakdowns.
- Source/test diffs.
- Relevant export files.

Run or verify:

```bash
git status --short
git diff --stat
.venv/bin/python -m pytest -q
```

Run the audit command again if Agent 4 did not already rerun it in a clean output directory.

## 3. Acceptance rule for 70% target

Only use `ACCEPTED — 70% TARGET MET` if all are true:

- Persisted P3 sample exists.
- Sample is deterministic and covers representative P3 regions/categories.
- Sample denominator is stated.
- `sample_pass_rate >= 0.70`.
- `review_required`, `blocked_*`, and `fail` records were not counted as pass.
- All passing records satisfy Content Lab candidate gate requirements.
- No student-runtime path was changed to consume unreviewed candidates.
- Trust gates were not weakened.
- Tests pass.
- Generated outputs and secrets are not tracked.
- Agent 4 found no red-line safety issues.

## 4. Acceptance rule for loop infrastructure only

Use `ACCEPTED WITH DEFERRALS — LOOP INFRASTRUCTURE ONLY` if:

- Audit/sample infrastructure is sound.
- Tests pass.
- At least one blocker class was honestly repaired or isolated.
- The target is not yet met, or Agent 4 could not verify the target.
- Remaining blockers are ranked clearly for the next loop.

This verdict should explicitly say the 70% target was not proven.

## 5. Rejection rule

Use `REJECTED` if:

- Tests fail.
- Sample or pass-rate calculation is unreliable.
- The implementation weakens gates.
- The implementation guesses curriculum/skill mappings.
- The implementation promotes unreviewed Content Lab candidates into student runtime.
- Generated files or secrets are tracked.
- Claims exceed evidence.
- The code change is too broad to audit.

## 6. Required review sections

Final review must include:

1. Final verdict.
2. What was reviewed.
3. Commands run and exact results.
4. Pass-rate evidence.
5. Reasons to accept/reject considered.
6. Blocking issues.
7. Non-blocking deferrals.
8. Repo hygiene findings.
9. Scores out of 10:
   - Correctness
   - Test integrity
   - Output honesty
   - Scope control
   - Maintainability
   - Usefulness for next iteration
10. Suggested next iteration.

## 7. Suggested next iteration logic

If target is not met, choose the next loop based on the largest honest blocker class:

- Schema/gate mismatch remains high: run another validator/export-adapter pass.
- Skill mapping blockers remain high: build reviewed bridge table from existing reviewed decisions only.
- Mark evidence blockers remain high: run subpart mark promotion sprint using existing simple-fillable evidence.
- Artifact path blockers remain high: repair path/root resolution.
- Ambiguous topic blockers remain high: quarantine; do not force them into pass.

## 8. Required final summary

End with a 250–1000 word plain-English explanation of:

- What the project now has.
- Whether the 70% target was actually met.
- What changed and why.
- What remains unsafe or blocked.
- How to rerun the audit.
- What the next loop should do.

Do not sugarcoat. A clean `not yet` is better than a fake pass.
