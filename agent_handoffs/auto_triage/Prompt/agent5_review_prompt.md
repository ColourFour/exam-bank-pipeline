# Agent 5 Adversarial Review Prompt

Review the completed iteration as if it might be a false improvement.

Look for:
- Flag suppression instead of extraction fixes.
- Broad validation or trust-gate loosening.
- OCR/no-OCR baseline mistakes.
- Overfitting to sampled records.
- Regressions hidden outside the target issue.
- Missing regression tests for reviewed examples.
- Deleted or altered triage baselines.

Final verdict must be one of:
- PASS
- PASS WITH RISKS
- BLOCKED

Include metrics before/after, the comparison path, `worsened_records` summary, and exact full-test result.
