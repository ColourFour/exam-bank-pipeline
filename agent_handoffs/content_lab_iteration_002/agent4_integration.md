# Agent 4 Integration Audit — Content Lab Iteration 002

## Verdict

PASS WITH DEFERRALS

The loop infrastructure is sound, deterministic, and tested. The 70% target is not supported by the current candidate evidence.

## Commands Run

```bash
.venv/bin/python scripts/audit_asterion_content_lab_readiness.py \
  --candidates output/asterion/exports/latest/asterion_content_lab_candidates_v1.json \
  --question-bank output/json/question_bank.json \
  --topic-routing output/json/question_bank.topic_routing.v1.json \
  --asterion-bank output/asterion/exports/latest/asterion_question_bank_v1.json \
  --artifact-root output \
  --sample-size 100 \
  --out-dir output/audits/asterion_content_lab_loop/latest
```

Result: `3/100 (3.00%) target_met=False`

```bash
.venv/bin/python -m pytest tests/test_asterion_content_lab_readiness_audit.py tests/test_asterion_export.py tests/test_output_contract.py -q
```

Result: `27 passed in 0.54s`

```bash
.venv/bin/python -m pytest -q
```

Result: `731 passed, 3 skipped, 5 warnings in 130.57s`

## Output Behavior

- Total Content Lab candidates: `2432`
- P3 candidates: `721`
- Sample size: `100`
- Sample passed: `3`
- Sample failed: `97`
- Regions covered: all nine requested P3 regions.

Top sample blockers:

- `blocked_mark_events`: 97
- `review_required`: 97
- `blocked_skill_mapping`: 94
- `blocked_mapping_review_gate`: 82

The audit reports `legacy_candidate_selection_review_flag_missing` as an Asterion-side schema/validator concern for two generation-ready records.

## Repo Hygiene

`git ls-files output` lists only:

- `output/json/.gitkeep`
- `output/json/asset_manifest.v1.json`

`git ls-files '.env' '*.env'` returned no files.

Generated audit output remains ignored under `output/audits/...`.

## Safety Findings

- No student-runtime path was changed.
- No trust gate was weakened.
- No reviewed skill mapping was invented.
- No generated/OCR text was made canonical.

## Recommended Agent 5 Decision

Use `ACCEPTED WITH DEFERRALS — LOOP INFRASTRUCTURE ONLY`. The sample pass rate is far below 70%, but the measurement and reports are honest and identify the next blocker class.
