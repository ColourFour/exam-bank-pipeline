# Manual Review Batch 0003 Conclusions

Batch 0003 is an adversarial exact-skill retagging and mark-event approval probe, not a broad promotion batch and not a generation-readiness batch.

## Summary

- Reviewed records: `14`
- Selection categories: `{"clean_control_mark_event_probe": 3, "deferred_exact_skill_boundary_probe": 1, "prior_ambiguous_retag_probe": 6, "prior_blocked_confirmation": 2, "thin_adjacent_part_probe": 2}`
- Exact-skill decisions: `{"approved_control_already_promoted": 3, "blocked": 2, "deferred": 1, "deferred_thin": 2, "retagged_not_promoted": 6}`
- Promoted exact-skill records: `0`
- Clean registry count before/after: `15` / `15`
- Mark-event approval probes: `3`
- Approved mark events: `0`
- Content Lab generation-ready before/after: `0` / `0`

## Source Files Inspected

- `reviewed_registry`: `data/review/p3_exact_skill_reviewed_decisions.v1.json`
- `batch_0001_conclusions`: `reports/manual_review_batch_0001_conclusions.v1.json`
- `batch_0002_conclusions`: `reports/manual_review_batch_0002_conclusions.v1.json`
- `batch_0003_manifest`: `data/review/p3_exact_skill_batches/batch_0003_manifest.v1.json`
- `batch_0003_decision_template`: `data/review/p3_exact_skill_batches/batch_0003_decision_template.v1.json`
- `batch_0003_review_packet`: `data/review/p3_exact_skill_batches/batch_0003_review_packet.md`
- `batch_0003_review_responses`: `data/review/p3_exact_skill_batches/batch_0003_review_responses.v1.json`
- `batch_0003_plan`: `reports/p3_exact_skill_batch_0003_plan.v1.json`
- `review_queue`: `reports/p3_exact_skill_review_queue.v1.json`
- `content_lab_candidates`: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`

## Mark-Event Findings

- No mark events were approved.
- All mark-event refs remain advisory-only.
- Required later workflow change: Add a reviewed mark-event decision artifact with event_id, part_path, source asset refs, reviewer identity, reviewed_at, reviewed/approved/rejected status, and validator checks that bind approved event IDs to a clean source-skill record before Content Lab generation can be enabled.

## Promoted Exact-Skill Records

- None

## Retagged But Not Promoted Records

- `32autumn23_q09_b`: `retagged_not_promoted`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_5_area_volume_applications, 9709_p3_3_5_trig_and_partial_fraction_integration`; reason: Part (b) asks for an exact area and the mark scheme integrates a trigonometric expression. The trig identity work is supporting method evidence, not a clean trig-identities source-skill record.
- `32summer23_q07_b`: `retagged_not_promoted`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_4_parametric_implicit_differentiation, 9709_p3_3_4_tangents_normals_stationary_points`; reason: The mark scheme differentiates an implicit curve and applies a tangent-gradient condition. Derivative rules are supporting mechanics, so a new narrow review would be needed before any clean promotion.
- `33autumn22_q10_b`: `retagged_not_promoted`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_8_separable_differential_equations`; reason: The logarithms occur during integration and constant evaluation inside a separable differential-equation solution. They cannot approve the log/exponential equations skill.
- `31autumn21_q04_whole`: `retagged_not_promoted`; mark event: `left_advisory_only`; evidence: `whole-question`; retag: `9709_p3_3_5_substitution_and_parts`; reason: The whole question requires a substitution and limit handling, including improper-limit structure. A broad standard-integration seed would hide the assessed method.
- `33autumn23_q07_b`: `retagged_not_promoted`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_4_parametric_implicit_differentiation, 9709_p3_3_4_tangents_normals_stationary_points`; reason: The proposed polynomial skill is not the assessed target in the selected subpart. Any future differentiation seed should be reviewed independently.
- `31summer23_q02_a`: `retagged_not_promoted`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_1_modulus_equations_inequalities`; reason: The canonical context is a modulus graph or linear inequality task, not parametric or implicit differentiation. Retagging needs a later dedicated modulus review.

## Blocked Records

- `31autumn21_q07_c`: `blocked`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `none`; reason: The log work belongs to the earlier differential-equation solution, while this subpart only asks for limiting behaviour. It is not clean log/exponential equation evidence.
- `33autumn22_q10_a`: `blocked`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_8_initial_conditions_models`; reason: The selected subpart does not assess log/exponential algebra. It is model/constant interpretation in a differential-equation question and should stay out of the registry.

## Thin Or Deferred Records

- `31summer24_q06_c`: `deferred_thin`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `none`; reason: Part (c) only rearranges to show a fixed-point formula. The actual iteration evidence is in promoted part (d), so this remains thin adjacent evidence.
- `31summer24_q09_a`: `deferred_thin`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `none`; reason: Part (a) is a one-mark scalar-product/perpendicularity check. The stronger vector-line intersection evidence remains promoted in part (b).
- `32spring24_q05_b`: `deferred`; mark event: `left_advisory_only`; evidence: `part-level`; retag: `9709_p3_3_9_argand_loci_geometry`; reason: Batch 0002 already flagged this as clean-looking but uncertain against a narrower Argand/loci skill. Batch 0003 keeps it out of the seed registry.

## Clean Controls Not Re-Promoted

- `33summer23_q11_b`: `approved_control_already_promoted`; mark event: `left_advisory_only`; evidence: `part-level plus mark-event probe`; retag: `none`; reason: The existing reviewed registry record already covers part (b) polar/exponential complex work. Batch 0003 does not duplicate the record or approve mark events.
- `31summer24_q06_d`: `approved_control_already_promoted`; mark event: `left_advisory_only`; evidence: `part-level plus mark-event probe`; retag: `none`; reason: Part (d) contains the substantive iterative calculation and is already in the reviewed registry; adjacent part (c) remains thin.
- `31summer24_q09_b`: `approved_control_already_promoted`; mark event: `left_advisory_only`; evidence: `part-level plus mark-event probe`; retag: `none`; reason: Part (b) is the substantive vector-line record already promoted in Batch 0002; part (a) remains thin.

## Generation Readiness

- Changed: `false`
- Reason: Reviewed source-skill decisions were kept separate from mark-event approval; no explicit reviewed mark-event path exists in this batch.

## Remaining Risks

- Several retags identify likely safer skill IDs but are not clean registry evidence until reviewed in a promotion-focused pass.
- Part-level reviews still rely on whole-question and whole mark-scheme images for canonical asset references.
- Mark-event refs remain useful for reviewer navigation but are not a generation gate authority.

## Suggested Next Steps

- Create a dedicated reviewed mark-event schema and validator before attempting Content Lab generation approval.
- Run a small promotion pass only for retagged records whose narrower skill IDs can be cleanly proven from canonical images.
- Keep thin adjacent parts as negative controls in future batches.
