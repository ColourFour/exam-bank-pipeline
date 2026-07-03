# Definition of Done

An iteration is done only when:

1. The planned acceptance criteria are satisfied for exactly one priority lane: clear output images, correct topic content, or correct per-question JSON data.
2. New or changed tests/checks verify the behavior, or the report explains why a deterministic non-test verification is the right evidence.
3. Existing focused tests pass, and full-suite or skipped-test risk is explicitly justified.
4. Image-related work includes inspected visual evidence: before/after crops, a deterministic triage sample, or an artifact-integrity check that proves paths resolve.
5. Topic-related work validates taxonomy IDs, reviewed-decision behavior, sidecar safety metadata, or topic-packet inclusion/exclusion.
6. JSON-related work verifies representative `question_bank.json` records and the exact fields named in the plan.
7. OCR-enabled comparison is run before claiming production extraction improvement.
8. No validation, mapping, visual-curation, topic-safety, text-trust, or role-gate signal was loosened without documented evidence.
9. The repo has no unnecessary generated artifacts outside `.agent-runs/` or ignored candidate output roots.
10. The implementation report lists every changed file and why it changed.
11. The auditor either passes the iteration or identifies concrete follow-up fixes.
12. No new dependency, script, route, state model, data format, or abstraction was added without justification.
