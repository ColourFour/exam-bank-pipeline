# Agentic Loop Template

Reusable five-agent improvement loop for software projects.

The loop is designed around one rule:

> The repo should get better without becoming dirtier, larger, or harder to reason about.

This template does not assume a specific coding-agent CLI. Use it with Codex, Claude Code, Cursor agents, Aider, or manual agent calls. The important part is that each stage writes artifacts and the next stage reads only the repo plus previous artifacts.

## Agent roles

1. Planner: reviews the repo and selects one bounded improvement slice.
2. Test Designer: creates tests/checks that prove the plan was completed.
3. Coder: changes implementation only enough to pass the tests and satisfy the plan.
4. Adversarial Auditor: attacks the implementation and decides whether the plan was actually completed.
5. Governor: after 5 iterations, reviews the whole run and updates the Planner policy/backlog so the next cycle is smarter and cleaner.

## Recommended run shape

Each iteration should produce:

```text
.agent-runs/<run-id>/iteration-01/
  01-plan.json
  02-test-plan.json
  03-implementation-report.json
  04-audit-report.json
  repo-delta.patch
```

After 5 iterations:

```text
.agent-runs/<run-id>/governor-review.json
.agent-loop/config/planner-purpose.md
.agent-loop/BACKLOG.md
```

## Hard rules

- One iteration = one improvement slice.
- The planner may not request broad repo rewrites.
- The test agent may edit tests/checks only.
- The coder may not weaken or delete tests.
- The auditor may not edit product code.
- Generated artifacts stay under `.agent-runs/` and should be gitignored.
- No iteration starts unless the repo is clean or the runner creates an explicit dirty-state snapshot.
- Agent 5 updates the Planner policy, not the project blindly.

## Minimal manual workflow

```bash
node .agent-loop/scripts/new-run.mjs
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent planner
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent test-designer
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent coder
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent auditor
```

Repeat iterations 1–5, then:

```bash
node .agent-loop/scripts/build-packet.mjs --run latest --agent governor
```

The packet builder creates prompt packets in the run folder. Paste those into your chosen coding agent, or wire them into your preferred agent CLI.
