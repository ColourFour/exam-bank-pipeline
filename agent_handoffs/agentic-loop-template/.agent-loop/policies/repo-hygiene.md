# Repo Hygiene Policy

The loop must leave the repo easier to reason about than it found it.

## Default Behavior

- Prefer editing existing source, tests, and docs over creating new files.
- Capture `git status --short` in Agent 1's plan. The current project may have dirty generated or handoff files; agents must identify which dirty files are in scope and must not revert unrelated user changes.
- Keep run artifacts in `.agent-runs/`.
- Keep regenerated candidate outputs under ignored output roots such as `output/candidates/`, not in source directories.
- Do not commit caches, logs, screenshots, one-off reports, local databases, full generated banks, generated PDFs, or model outputs.
- Do not introduce new dependency stacks for small extraction-quality problems.
- Do not create a permanent framework for a one-time workflow.
- Prefer deterministic fixtures from reviewed triage samples over large synthetic fixture sets.

## Generated Output Rules

- Do not directly edit `output/json/question_bank.json` to make tests pass. Fix source logic or create a named candidate output for comparison.
- Do not overwrite frozen triage comparison outputs unless the replacement path and reason are explicit.
- Do not delete current canonical image trees or sidecars during cleanup. Use inventory and cleanup-plan commands first.
- Generated release payloads and topic packets remain ignored unless a tracked manifest/provenance file is intentionally updated.

## Bloat Budget

Per iteration default maximum:

- 8 changed files.
- 4 new files.
- 0 new dependencies.

Exceeding this budget requires explicit justification in the plan and auditor approval.
