# Exam-Bank Agentic Loop Template

Five-agent improvement loop for the CAIE 9709 exam-bank extraction pipeline.

The loop is designed around one current rule:

> Improve the image-first question bank in small, verified slices without weakening trust gates or confusing generated output with source truth.

The project is currently centered on three Agent 1 priorities, in this order:

1. Produce clear output images: question PNGs and mark-scheme PNGs must be readable, correctly cropped, correctly paired, and present at canonical paths.
2. Parse correct topic content: topic metadata must map to the CAIE 9709 taxonomy and reviewed decisions, and unsafe topic sidecars must fail closed.
3. Get each question's JSON data correct: `question_bank.json` records must carry the right identity, question number, image paths, mark-scheme data, subparts, status flags, and provenance fields.

Question and mark-scheme images are canonical. OCR/native text, AI labels, topic routing, mark events, difficulty labels, and Asterion projections are support metadata unless a reviewed contract explicitly promotes them.

## Agent Roles

1. Planner: reviews current evidence and selects one bounded improvement slice from the active priority order.
2. Test Designer: creates focused tests/checks that prove the selected image, topic, or JSON correctness issue was fixed.
3. Coder: changes implementation only enough to pass the tests and satisfy the plan.
4. Adversarial Auditor: attacks the implementation against visual evidence, topic evidence, JSON integrity, and repo hygiene.
5. Governor: after 5 iterations, reviews whether the loop improved the current priorities and updates Agent 1 policy/backlog.

## Evidence To Prefer

- Current architecture and roadmap: `ARCHITECTURE.md`, `README.md`, `ROADMAP.md`.
- Command map: `docs/COMMAND_ATLAS.md`.
- Canonical bank: `output/json/question_bank.json`.
- Canonical assets: `output/pm1/`, `output/pm3/`, `output/stats/`, `output/mechanics/`.
- Durable topic sidecar: `data/topic_routing/question_bank.topic_routing.v1.json`.
- Working topic sidecar: `output/json/question_bank.topic_routing.v1.json`.
- Taxonomy: `exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json`.
- Triage and auto-triage artifacts: `output/triage/`, `agent_handoffs/auto_triage/`.

Do not trust JSON metadata alone for image problems. Inspect the referenced PNG crops or a deterministic visual sample before planning or auditing image-related work.

## Recommended Run Shape

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

## Useful Verification Commands

Fast tests:

```bash
.venv/bin/python -m pytest -q -m "not integration and not rendering"
```

Full tests:

```bash
.venv/bin/python -m pytest -q
```

Current export audit:

```bash
.venv/bin/python -m exam_bank.cli audit \
  --input output/json/question_bank.json \
  --output output/json/audit.current.json
```

Output integrity audit:

```bash
.venv/bin/python -m exam_bank.cli output-integrity-audit \
  --input output/json/question_bank.json \
  --artifact-root output \
  --output output/json/audit.current.integrity.json
```

OCR-enabled candidate comparison is required before claiming production extraction improvement:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input/pastpapers/9709 \
  --output output/candidates/ocr/latest \
  --enable-ocr
```

## Hard Rules

- One iteration = one improvement slice.
- Agent 1 must select from the current priority order unless repo evidence proves a blocker is higher priority.
- The planner may not request broad rewrites, broad trust-gate loosening, or generated-output cleanup as a substitute for extraction quality.
- The test agent may edit tests/checks only.
- The coder may not weaken or delete tests.
- The auditor may not edit product code.
- Generated artifacts stay under `.agent-runs/` or ignored output candidate roots.
- Frozen triage baselines and durable sidecars must not be deleted or silently overwritten.
- Agent 5 updates the Planner policy and backlog; it does not blindly rewrite project contracts.

## Minimal Manual Workflow

```bash
node .agent-loop/scripts/new-run.mjs
node .agent-loop/scripts/check-clean-repo.mjs
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent planner
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent test-designer
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent coder
node .agent-loop/scripts/build-packet.mjs --run latest --iteration 1 --agent auditor
node .agent-loop/scripts/validate-artifacts.mjs --run latest --iteration 1
```

Repeat iterations 1-5, then:

```bash
node .agent-loop/scripts/build-packet.mjs --run latest --agent governor
```

The packet builder creates prompt packets in the run folder. Paste those into the chosen coding agent, or wire them into a local agent CLI.

When the template is run in place from `agent_handoffs/agentic-loop-template`, `build-packet.mjs` discovers the parent exam-bank project root and includes current project docs plus `git status --short` in the packet. This keeps Agent 1 aligned with the live repo rather than only the reusable loop files.

`check-clean-repo.mjs` writes `.agent-runs/<run-id>/dirty-state-snapshot.txt` when the loop config allows dirty starts. Use that snapshot to distinguish loop changes from pre-existing user changes.

`validate-artifacts.mjs` checks completed JSON artifacts against the loop schemas. Run it before handing work to the next agent and before writing any quality report.
