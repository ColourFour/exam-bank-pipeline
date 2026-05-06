# Agent 2 Prompt — iteration_002

Read `agent_handoffs/iteration_002/agent1_plan.md` before editing.

Implement the iteration 2 difficulty work only:

- Add deterministic numeric difficulty metadata while preserving the existing canonical labels (`easy`, `average`, `difficult`).
- Thread difficulty score/details through classification state and JSON export.
- Add focused tests for scoring behavior and export presence.
- Add a lightweight difficulty audit command only if it stays small and follows existing audit/script patterns.

Do not change OCR selection, OCR thresholds, DeepSeek provider behavior, topic taxonomy, mark-scheme mapping, crop detection, or adaptive trainer behavior.

Report:

- files changed
- difficulty score scale and fields added
- tests run
- any export/schema compatibility decisions
- remaining blockers or deferred calibration work
