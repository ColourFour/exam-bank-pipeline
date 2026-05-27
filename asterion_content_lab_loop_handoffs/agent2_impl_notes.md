# Agent 2 Implementation Handoff — Asterion Content Lab P3 70% Pass Loop

Task 5/27: Build the deterministic audit/repair layer for Asterion Content Lab candidates and push the P3 stratified sample toward ≥70% pass.

## 1. Scope

Implement the smallest durable tooling and code changes needed to measure and improve Asterion Content Lab P3 candidate readiness.

This is not an OCR threshold pass. This is not a student-runtime promotion pass. The target is Content Lab candidate validation and teacher/content-lab readiness.

## 2. First action: establish baseline

Before changing code, run or create a deterministic audit against the current candidate export:

```bash
.venv/bin/python scripts/audit_asterion_content_lab_readiness.py \
  --candidates output/asterion/exports/latest/asterion_content_lab_candidates_v1.json \
  --question-bank output/json/question_bank.json \
  --topic-routing output/json/question_bank.topic_routing.v1.json \
  --asterion-bank output/asterion/exports/latest/asterion_question_bank_v1.json \
  --artifact-root output \
  --sample-size 100 \
  --out-dir output/audits/content_lab_iteration_002
```

If this exact script does not exist, create it or create the closest equivalent. Keep it deterministic, CLI-friendly, and reporting-first.

## 3. Required audit outputs

Write generated outputs under an ignored audit directory such as:

```text
output/audits/content_lab_iteration_002/
```

Required files:

- `audit_summary.md`
- `audit_summary.json`
- `p3_candidate_inventory.csv`
- `p3_sample_frame.csv`
- `p3_sample_results.csv`
- `blocker_breakdown.csv`
- `blocker_by_region.csv`
- `schema_gate_mismatch_report.csv`
- `skill_mapping_gap_report.csv`
- `mark_evidence_gap_report.csv`
- `artifact_path_gap_report.csv`
- `promotion_candidates.csv`
- `still_blocked_candidates.csv`
- `next_iteration_recommendations.md`

The summary JSON must include at least:

```json
{
  "total_candidates": 0,
  "p3_candidates": 0,
  "sample_size": 0,
  "sample_passed": 0,
  "sample_failed": 0,
  "sample_pass_rate": 0.0,
  "target_pass_rate": 0.7,
  "target_met": false,
  "top_blockers": [],
  "regions_covered": [],
  "student_runtime_changed": false,
  "trust_gates_weakened": false
}
```

## 4. Pass/fail classification

For each P3 candidate, classify:

- `pass`
- `fail`
- `review_required`
- `blocked_missing_evidence`
- `blocked_schema_mismatch`
- `blocked_skill_mapping`
- `blocked_mapping_review_gate`
- `blocked_mark_scheme`
- `blocked_mark_events`
- `blocked_artifact_path`
- `blocked_quarantine_ambiguous`

A record only counts toward the 70% pass target if it is `pass` under the Content Lab candidate gate. `review_required` is not pass.

## 5. Highest-leverage implementation target

After baseline, repair one or more high-leverage blocker classes. Start with schema/gate mismatch if present.

Likely repair direction:

- Teach the validator/export adapter to accept the new reviewed gate fields where they are semantically equivalent to the old required evidence.
- Do not fake old fields. Preserve both old and new fields if needed, but mark provenance clearly.
- If the new fields are not equivalent, do not promote the record. Report the mismatch.

Candidate fields to inspect and reconcile:

- `source_skill_review_gate`
- `mapping_review_gate`
- `mapping_review_satisfied`
- reviewed Asterion skill-map IDs
- exam-bank canonical skill IDs
- candidate selection/review metadata
- mark-event or subpart mark evidence

## 6. Skill mapping bridge

If the main blocker is that exam-bank canonical skill IDs do not resolve to Asterion reviewed P3 skill IDs, implement or generate a reviewed bridge table only when evidence exists.

Acceptable bridge sources:

- Existing reviewed P3 skill map.
- Existing reviewed decision file.
- Explicitly reviewed mapping sidecar.
- Existing deterministic mapping with review gate marked satisfied.

Unacceptable bridge sources:

- AI guess.
- Topic name similarity alone.
- Difficulty label.
- Unreviewed advisory text.

If bridge evidence is missing, write `skill_mapping_gap_report.csv` and stop that blocker class.

## 7. Mark evidence bridge

If records are blocked because mark values or mark events are not in the shape Asterion expects, inspect whether existing bank evidence is sufficient.

Safe promotions:

- Simple subpart marks detected from `mark_values_detected` and attached to the same canonical question/subpart.
- Mark-scheme image exists and mark total is consistent.
- No local total disagreement.

Unsafe promotions:

- Missing mark-scheme image.
- Question total and mark-scheme total disagree.
- Mark events refer to ambiguous subparts.
- Candidate uses generated explanations as proof.

## 8. Regeneration and validation

After patching, regenerate or revalidate the Asterion exports using the repo’s existing command. If no command is obvious, document what you found and run the narrowest available validation command.

Possible commands to discover or adapt:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/validate_asterion_exports.py
.venv/bin/python scripts/validate_content_lab_candidates.py
.venv/bin/python scripts/audit_asterion_content_lab_readiness.py ...
```

Do not invent a command in the final notes if it does not exist.

## 9. Tests to run before handoff

Run focused tests for touched code, then a broader safe subset. If production export/validation logic changes, run the full suite unless runtime is prohibitive.

Minimum examples:

```bash
.venv/bin/python -m py_compile scripts/audit_asterion_content_lab_readiness.py
.venv/bin/python -m pytest tests/test_asterion* tests/test_content_lab* tests/test_output_contract.py -q
.venv/bin/python -m pytest -q
```

If patterns do not exist, use the actual matching tests and report the exact commands.

## 10. Required handoff output

Write:

```text
agent_handoffs/content_lab_iteration_002/agent2_impl_notes.md
```

Include:

- Baseline P3 sample pass rate.
- Final P3 sample pass rate.
- Sample denominator and selection method.
- Files changed.
- Commands run.
- Top blockers before/after.
- Whether the 70% target was met.
- Why any candidates remain blocked.
- Whether student runtime was untouched.
- Whether any gates were weakened.

## 11. Acceptance criteria

- Audit tool/report exists and is deterministic.
- P3 sample frame is persisted.
- Baseline and final pass rates are reported.
- At least one real blocker class is repaired or honestly proven unsafe to repair.
- No student-runtime path is changed.
- No generated outputs are tracked.
- Tests pass.
- Final state is clear enough for Agent 3 and Agent 4 to verify independently.

## 12. Stop conditions

Stop and hand off a blocked report if:

- The candidate export is missing or cannot be parsed.
- P3 candidates cannot be identified deterministically.
- The only way to improve pass rate is to weaken gates.
- Reviewed Asterion skill mapping evidence is missing.
- Mark-scheme evidence is missing or inconsistent for promoted records.
- Tests fail after attempted repair.
- Student runtime begins consuming unreviewed candidates.

## 13. Required final summary

End with a 250–1000 word plain-English summary explaining what changed, why it changed, how the pass metric moved, how to rerun the validation, what risks remain, and what the next loop should attack.
