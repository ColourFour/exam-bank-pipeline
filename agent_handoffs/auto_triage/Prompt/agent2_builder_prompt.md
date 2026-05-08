# Agent 2 Builder Prompt

Implement only Agent 1's scoped plan.

Rules:
- Add or update regression tests for the reviewed examples.
- Keep changes limited to the files named by Agent 1 unless new evidence requires a small documented expansion.
- Avoid unrelated cleanup and formatting churn.
- Preserve image-first behavior and trust gating.
- Do not make failures disappear by suppressing validation, review, text-fidelity, or topic-trust flags.
- Do not modify or delete frozen `output/triage` baselines.

Report:
- Files changed.
- Tests added or updated.
- Any deviation from Agent 1's plan and the evidence for it.
