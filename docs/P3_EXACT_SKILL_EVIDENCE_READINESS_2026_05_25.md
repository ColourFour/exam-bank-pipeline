# P3 Exact-Skill Evidence Readiness: 2026-05-25

## Executive Verdict

`READY_FOR_ASTERION_CONTENT_LAB_REVIEW_DIAGNOSTICS`

`NOT_READY_FOR_ASTERION_RUNTIME_MASTERY`

`NOT_READY_FOR_GUARDIAN`

`NOT_READY_FOR_CANDIDATE_GENERATION`

`NOT_READY_FOR_SOURCE_BACKED_WORKED_EXAMPLES`

The P3 exact-skill workflow can be connected to Asterion now only as Content Lab, admin, or reviewer diagnostics. It is not ready to drive runtime mastery, Guardian, candidate generation, export permissions, or source-backed worked examples because there are still zero clean reviewed P3 exact-skill records.

No final runtime sidecar was created in this pass:

```text
output/asterion/exports/latest/p3_exact_skill_evidence_v1.json
```

The reviewed registry was not edited:

```text
data/review/p3_exact_skill_reviewed_decisions.v1.json
```

## Current Pipeline State

The workflow now has these staged artifacts:

- Reviewed-decision registry and validator: curated input only, fail-closed by default.
- Review queue: full P3 candidate triage with blocker and status diagnostics.
- Review batches: small human-review packets and safe decision templates.
- Visual and interactive packet: local HTML review UI with autosave/export and optional local save server.
- DE vs parametric/implicit guard: `dy/dx` alone is not enough; separation of variables stays a differential-equation conflict cue.
- Cross-topic awareness: primary/supporting topic context is modeled instead of treated as generic ambiguity.
- Ambiguity audit: explains where broad ambiguity moved after sharper statuses.
- Part-level decomposition: proposes or diagnoses part/subpart review scope where existing signals support it.
- Current-state audit: consolidates queue, reviewed-registry, ambiguity, and decomposition counts.

All machine-generated candidate fields remain review context. They are not curriculum authority.

## Fresh Output Audit

Fresh audit source:

```text
reports/p3_exact_skill_current_state_audit.v1.json
reports/p3_exact_skill_current_state_audit.md
```

Current queue counts:

- Total queue items: `749`
- `cross_topic_candidate`: `568`
- `split_needed_candidate`: `21`
- `conflict_candidate`: `126`
- `fallback_only`: `34`
- `ambiguous_candidate`: `0`
- `clean_candidate`: `0`
- Missing question asset count: `0`
- Missing mark-scheme asset count: `0`
- No candidate P3 skill count: `0`
- Advisory-only mark-event count: `749`
- Already-reviewed queue scopes: `3`

Current cross-topic status counts:

- `cross_topic_reviewable`: `602`
- `cross_topic_split_needed`: `21`
- `conflict_needs_review`: `126`

Current topic-routing alignment counts:

- `aligned`: `465`
- `supporting_topic`: `73`
- `conflicting`: `126`
- `unknown`: `85`

Current part-decomposition counts:

- Decomposition candidate count: `513`
- Part-level decomposition candidate count: `0`
- Subpart-level decomposition candidate count: `0`
- `already_part_scoped`: `513`
- `needs_manual_split`: `21`
- `insufficient_part_signal`: `81`
- `conflict_needs_review`: `126`
- `not_decomposable`: `8`

Current reviewed-registry route counts:

- `clean`: `0`
- `thin`: `1`
- `ambiguous`: `0`
- `blocked`: `1`
- `deferred`: `0`
- `review_needed`: `1`
- `fallback_only`: `0`

## Comparison Against Previous Known State

Previous known queue state:

- Total: `749`
- `cross_topic_candidate`: `568`
- `split_needed_candidate`: `21`
- `conflict_candidate`: `126`
- `fallback_only`: `34`
- `ambiguous_candidate`: `0`
- `clean_candidate`: `0`

Fresh state:

- Total: `749` to `749`, delta `0`
- `cross_topic_candidate`: `568` to `568`, delta `0`
- `split_needed_candidate`: `21` to `21`, delta `0`
- `conflict_candidate`: `126` to `126`, delta `0`
- `fallback_only`: `34` to `34`, delta `0`
- `ambiguous_candidate`: `0` to `0`, delta `0`
- `clean_candidate`: `0` to `0`, delta `0`

The part-level decomposition pass did not change the main queue status counts. That is expected and conservative. The improvement is diagnostic: it enriches review scope, identifies `513` already part-scoped decomposition candidates, and confirms that remaining whole-question cross-topic records need either manual split or stronger part signal before they can become clean reviewed evidence.

## Asterion Integration Recommendation

Can this be added to Asterion now?

Yes, but only as Content Lab, admin, or reviewer diagnostics.

Asterion can safely consume:

- Review status summaries.
- Candidate status.
- Cross-topic status.
- Split-needed, conflict, fallback, and blocker flags.
- Proposed part-level decomposition.
- Recommended review action.
- Canonical question and mark-scheme asset refs for reviewer inspection.
- Blocker diagnostics.

Asterion must not consume as authority:

- `suggested_source_skill_ids`.
- Candidate skills as mastery evidence.
- Advisory mark events as source-backed example evidence.
- OCR/native/advisory text labels.
- Browser review responses before registry validation.
- Cross-topic candidates as clean evidence.
- Decomposition candidates as reviewed part boundaries.

The correct integration boundary is a read-only diagnostics surface. It should help reviewers choose and resolve candidates, but it must not unlock any runtime teaching, mastery, Guardian, candidate-generation, export, or source-backed worked-example behavior.

## How This Helps Asterion Now

This workflow can make Asterion Content Lab better immediately by improving reviewer triage. It can prioritize candidates that are worth image inspection, avoid resurfacing known conflict or fallback-only items as promising, show why a candidate is blocked or split-needed, and make cross-topic questions less confusing by separating primary and supporting context.

The decomposition diagnostics also help teachers, admins, and reviewers inspect the correct scope. When a whole question is too broad, Asterion can show that part/subpart review is needed instead of pretending the whole question supports one exact skill.

This prepares the path to source-backed worked examples, but does not authorize them yet.

## What Is Needed To Make It Best

Stage A: Add Asterion review diagnostics only.

Stage B: Human-review 25 to 50 candidates and validate the reviewed registry.

Stage C: Build `p3_exact_skill_evidence_v1.json` exporter from validated reviewed decisions only.

Stage D: Add Asterion read-only display of clean evidence coverage by skill.

Stage E: Use clean records for source-backed worked-example planning.

Stage F: Only later consider mastery, Guardian, or generation permissions after enough clean, diverse, image-backed evidence exists and the permission model is explicitly reviewed.

## Risks And Safeguards

- `cross_topic_candidate` is a review cue, not evidence.
- Decomposition candidates are suggestions, not reviewed boundaries.
- Mark events remain advisory-only.
- Browser responses are notes, not validated reviewed decisions.
- No clean evidence should enter Asterion runtime until the reviewed registry validates.
- Part-level records may still rely on whole-question images when part crops are unavailable.
- Source-backed examples require reviewed evidence basis and stable canonical assets.
- The DE vs parametric/implicit boundary remains a known-risk area and should stay visibly flagged.

## Fresh Commands Run

```text
git diff --check
.venv/bin/python -m pytest tests/test_p3_exact_skill_reviewed_decisions.py
.venv/bin/python -m pytest tests/test_p3_exact_skill_review_queue.py
.venv/bin/python -m pytest tests/test_p3_exact_skill_review_batch.py
.venv/bin/python -m pytest tests/test_p3_exact_skill_visual_review.py
.venv/bin/python -m pytest tests/test_p3_exact_skill_ambiguity_audit.py
.venv/bin/python -m pytest tests/test_p3_exact_skill_part_decomposition.py
.venv/bin/python -m pytest tests/test_p3_exact_skill_current_state_audit.py
.venv/bin/python scripts/validate_p3_exact_skill_reviewed_decisions.py
.venv/bin/python scripts/build_p3_exact_skill_review_queue.py
.venv/bin/python scripts/build_p3_exact_skill_ambiguity_audit.py
.venv/bin/python scripts/build_p3_exact_skill_part_decomposition.py
.venv/bin/python scripts/build_p3_exact_skill_review_batch.py --batch-id batch_0001 --limit 25
.venv/bin/python scripts/build_p3_exact_skill_review_batch.py --batch-id batch_part_decomp_0001 --limit 25 --batch-purpose part_decomposition_review
.venv/bin/python scripts/build_p3_exact_skill_visual_review_packet.py --batch-id batch_0001
.venv/bin/python scripts/build_p3_exact_skill_visual_review_packet.py --batch-id batch_part_decomp_0001
.venv/bin/python scripts/audit_p3_exact_skill_current_state.py
```

All listed commands passed in this pass.
