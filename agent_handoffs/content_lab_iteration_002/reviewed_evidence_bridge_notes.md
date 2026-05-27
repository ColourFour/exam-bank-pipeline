# Reviewed Evidence Bridge Notes — Content Lab Loop 002

## Outcome

The deterministic P3 sample remains `3/100 = 3.00%`. The 70% target was not reached.

## Evidence Sources Inspected

- `data/review/p3_exact_skill_reviewed_decisions.v1.json`
- `data/review/p3_exact_skill_reviewed_mark_events.v1.json`
- `reports/review/content_lab_control_authority_0001.v1.json`
- `reports/review/p3_reviewed_mark_events_0001.v1.json`
- `reports/p3_exact_skill_review_queue.v1.json`
- `output/json/question_bank.mark_events.v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- `output/asterion/exports/latest/asterion_question_bank_v1.json`

## Evidence Coverage

Generated reports:

- `output/audits/asterion_content_lab_loop/iteration_002/reviewed_evidence_coverage.json`
- `output/audits/asterion_content_lab_loop/iteration_002/reviewed_evidence_coverage.md`
- `output/audits/asterion_content_lab_loop/iteration_002/candidate_evidence_join.csv`
- `output/audits/asterion_content_lab_loop/iteration_002/human_review_queue.csv`

Coverage summary:

- P3 Content Lab candidates: `721`
- Existing generation-satisfying reviewed exact-skill/source-skill evidence: `15`
- Existing generation-satisfying reviewed mark-event/subpart evidence: `3`
- Candidates where full reviewed evidence exists but is not surfaced: `0`
- Candidates failing because reviewed evidence is absent: `718`
- Candidates failing with ambiguous/non-generation-satisfying evidence: `3`
- Legacy schema/validator mismatch candidates: `2`
- Candidates requiring human review no matter code changes: `718`

## Repair Attempts

1. `blocked_mapping_review_gate`
   - Evidence source: reviewed exact-skill/source-skill decisions.
   - Result: no unsurfaced eligible reviewed mapping evidence found. Existing clean reviewed records are already reflected in the export/readiness path.

2. `blocked_skill_mapping`
   - Evidence source: reviewed exact-skill/source-skill decisions and candidate reviewed gate fields.
   - Repair: readiness audit now requires newer reviewed gate fields to be backed by the reviewed decision file before counting pass.
   - Result: no pass-rate increase; this prevents false positives.

3. `blocked_mark_events`
   - Evidence source: reviewed mark-event decisions.
   - Result: only 11 approved mark-event decisions exist, covering the same 3 currently passing candidates. Raw detected mark values remain human-review queue input only.

## Tests

- `.venv/bin/python -m pytest tests/test_asterion_content_lab_readiness_audit.py -q`
  - `4 passed`
- `.venv/bin/python -m pytest tests/test_asterion_content_lab_readiness_audit.py tests/test_asterion_export.py tests/test_output_contract.py -q`
  - `29 passed`
- `.venv/bin/python -m pytest -q`
  - `733 passed, 3 skipped`
