# Repo Hygiene Policy

The loop must leave the repo cleaner than it found it.

## Default behavior

- Prefer editing existing files over creating new files.
- Delete obsolete files when replacing behavior.
- Keep generated outputs in `.agent-runs/`, not in source directories.
- Do not commit caches, logs, screenshots, one-off reports, local databases, or model outputs.
- Do not introduce new dependency stacks for small problems.
- Do not create a permanent framework for a one-time workflow.

## Bloat budget

Per iteration default maximum:

- 8 changed files.
- 4 new files.
- 0 new dependencies.

Exceeding this budget requires explicit justification in the plan and auditor approval.
