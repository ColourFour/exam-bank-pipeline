# Repo Hygiene Policy

The loop must leave the repo cleaner and more trustworthy than it found it, with generated exam-bank outputs protected from accidental churn.

## Default behavior

- Prefer editing existing tests, validators, or focused scripts over creating new frameworks.
- Keep generated agent artifacts in `.agent-runs/`, not in source, input, dataset, or canonical output directories.
- Keep suspicious-output review packs small and clearly scoped to the selected risk.
- Do not commit caches, logs, screenshots, one-off reports, local databases, or model outputs.
- Do not introduce new dependency stacks for small problems.
- Do not create a permanent framework for a one-time workflow.
- Do not add broad reports where a deterministic validation check, failing fixture, or small review pack is sufficient.
- Do not run long full-pipeline extraction by default; use targeted tests and audit commands from `.agent-loop/project-gates.md` unless the selected slice requires regeneration.
- Do not overwrite `output/json/question_bank.json`, canonical `output/<subject_family>/*.png`, or `output/json/asset_manifest.v1.json` without a manifest/diff and explicit acceptance criteria.
- Remove stale generated artifacts from git only when they are clearly generated, ignored by policy, and covered by a manifest or diff.

## Bloat budget

Per iteration default maximum:

- 8 changed files.
- 4 new files.
- 0 new dependencies.

Exceeding this budget requires explicit justification in the plan and auditor approval.

## Verification discipline

- Agent 2 should produce tests or validation checks that fail on the targeted bad output shape before Agent 3 changes extraction/alignment code, whenever practical.
- Agent 3 should keep implementation edits to the minimum needed to pass those checks.
- Agent 4 must sample the actual affected outputs when image mapping, crop quality, mark-scheme pairing, or contamination is in scope. Reading only reports or test summaries is not enough.
