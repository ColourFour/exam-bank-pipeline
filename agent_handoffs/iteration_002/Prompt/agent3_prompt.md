# Agent 3 Prompt — iteration_002

Read `agent_handoffs/iteration_002/agent1_plan.md` and Agent 2's implementation notes before testing.

Verify the quantified difficulty implementation:

- Score is bounded and deterministic.
- Coarse label derives consistently from the score.
- Missing marks and weak/degraded text lower confidence or add review flags.
- Direct low-mark routine questions remain easy.
- Linked, proof-style, mixed-topic, or high-mark questions score higher.
- Exported JSON includes difficulty fields for new records.
- Existing DeepSeek difficulty normalization remains compatible.

Run focused tests first, then the full suite. If the implementation only adds fields without meaningful scoring behavior, mark it NEEDS FIX.

Report:

- tests added or reviewed
- commands run and results
- scope guard findings
- repo hygiene findings
- verdict
