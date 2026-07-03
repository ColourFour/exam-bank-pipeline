# Agent 1 Planner Prompt

Read `metrics_before.json`, `selected_target.json`, and the triage sample for this iteration.

Produce a narrow implementation plan:
- Identify the selected issue cluster and why it is the target for this pass.
- Inspect sampled question and mark-scheme image crops before trusting metadata.
- Define likely files to change.
- Define tests to add or update from reviewed examples.
- Define exact acceptance criteria and stop criteria.
- Define what must not change.

Constraints:
- Tests-first where a focused regression test is practical.
- Scope one dominant issue per iteration.
- Do not loosen validation or trust gates broadly.
- Do not delete or mutate triage baselines.
- OCR-enabled comparison is required for production scoring.
- Extracted text remains metadata.
- DeepSeek/topic labels remain sidecar only.
