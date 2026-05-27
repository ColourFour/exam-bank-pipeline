# Asterion Content Lab Agentic Loop Handoff Bundle

This bundle rewrites the five uploaded OCR/readiness handoffs into a loop aimed at Asterion Content Lab P3 candidate readiness.

Target:

```text
At least 70% of a deterministic, stratified P3 Content Lab sample passes Content Lab candidate validation gates.
```

Core guardrail:

```text
Do not weaken gates, invent curriculum mappings, or promote unreviewed Content Lab candidates into student runtime to hit the metric.
```

Recommended use:

1. Put these files under a new folder such as:

```text
agent_handoffs/content_lab_iteration_002/
```

2. Give `agent1_plan.md` to the planning agent.
3. Give `agent2_impl_notes.md` to the implementation agent.
4. Give `agent3_tests.md` after Agent 2 finishes.
5. Give `agent4_integration.md` after Agent 3 finishes.
6. Give `agent5_review.md` last.

The loop is designed to repeat. If the sample pass rate is below 70%, Agent 5 should identify the next blocker class and seed the next iteration rather than claiming success.
