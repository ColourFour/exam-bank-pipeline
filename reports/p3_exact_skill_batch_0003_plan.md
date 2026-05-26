# P3 Exact-Skill Batch 0003 Plan

This is an adversarial retagging and mark-event approval probe. It does not promote candidates, mark anything generation-ready, or change canonical assets.

## Summary

- Generated at: `2026-05-26T02:26:00Z`
- Selected items: `14`
- Count by category: `{"clean_control_mark_event_probe": 3, "deferred_exact_skill_boundary_probe": 1, "prior_ambiguous_retag_probe": 6, "prior_blocked_confirmation": 2, "thin_adjacent_part_probe": 2}`
- Count by proposed source skill: `{"9709_p3_3_1_polynomial_division_factor_remainder": 1, "9709_p3_3_2_log_exponential_equations": 3, "9709_p3_3_3_identities_compound_double_angle_equations": 1, "9709_p3_3_4_derivative_rules": 1, "9709_p3_3_4_parametric_implicit_differentiation": 1, "9709_p3_3_5_standard_integration": 1, "9709_p3_3_6_fixed_point_iteration": 2, "9709_p3_3_7_vector_lines": 2, "9709_p3_3_9_complex_arithmetic_polar_form": 2}`
- Prior ambiguous/blocked/thin probes: `11`
- Clean controls: `3`
- Content Lab generation-ready candidates: `0`

## Review Guardrails

- Do not promote broad integration labels where the narrower target is substitution, trigonometric integration, area, or improper-limit work.
- Do not treat log, trig, polynomial, or derivative support work as the exact assessed skill.
- Do not promote thin adjacent parts that are only context for a more substantial neighbouring part.
- Do not approve mark events from advisory refs unless the workflow has an explicit reviewed-mark-event schema and validator path.
- Do not change Content Lab generation readiness from this batch.

## Selected Items

### `p3_exact_skill_review_queue:v1:32autumn23_q09:32autumn23_q09_b`

- Category: `prior_ambiguous_retag_probe`
- Reason: Retest Batch 0001/0002 trig-identity ambiguity where area/integration is the assessed target and trig identities are method support.
- Proposed source skill: `9709_p3_3_3_identities_compound_double_angle_equations`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["supporting_method_confusion", "integration_trig_area_target", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31autumn21_q07:31autumn21_q07_c`

- Category: `prior_ambiguous_retag_probe`
- Reason: Retest Batch 0001 DE/log ambiguity where the selected part is limiting behaviour from a differential-equation solution.
- Proposed source skill: `9709_p3_3_2_log_exponential_equations`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["supporting_method_confusion", "de_log_context", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:32summer23_q07:32summer23_q07_b`

- Category: `prior_ambiguous_retag_probe`
- Reason: Retest Batch 0001 derivative-rules ambiguity where implicit differentiation and tangent conditions are the safer target.
- Proposed source skill: `9709_p3_3_4_derivative_rules`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["wrong_skill_routing", "implicit_vs_derivative_rules", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_a`

- Category: `prior_ambiguous_retag_probe`
- Reason: Retest Batch 0002 log/exponential ambiguity where the subpart identifies constants in a differential-equation model.
- Proposed source skill: `9709_p3_3_2_log_exponential_equations`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["supporting_method_confusion", "de_log_context", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_b`

- Category: `prior_ambiguous_retag_probe`
- Reason: Retest Batch 0002 log/exponential ambiguity where logarithms occur during separable differential-equation solving.
- Proposed source skill: `9709_p3_3_2_log_exponential_equations`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["supporting_method_confusion", "de_log_context", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31autumn21_q04:31autumn21_q04_whole`

- Category: `prior_ambiguous_retag_probe`
- Reason: Retest broad standard-integration routing where substitution, changed limits, and improper-limit structure need narrower treatment.
- Proposed source skill: `9709_p3_3_5_standard_integration`
- Scope: `whole_question`
- Related reviewed evidence exists: `False`
- Known risk flags: `["broad_integration_label", "retag_to_narrower_integration", "do_not_default_to_clean", "mixed_or_ambiguous_topic", "whole_question_review_scope", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:33autumn23_q07:33autumn23_q07_b`

- Category: `prior_blocked_confirmation`
- Reason: Confirm Batch 0001 blocked polynomial/remainder route on an implicit-differentiation stationary-tangent subpart.
- Proposed source skill: `9709_p3_3_1_polynomial_division_factor_remainder`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["wrong_skill_routing", "polynomial_support_only", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31summer23_q02:31summer23_q02_a`

- Category: `prior_blocked_confirmation`
- Reason: Confirm Batch 0002 blocked parametric/implicit route on a modulus graph or linear-inequality subpart.
- Proposed source skill: `9709_p3_3_4_parametric_implicit_differentiation`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["wrong_skill_routing", "parametric_implicit_absent", "do_not_default_to_clean", "mixed_or_ambiguous_topic", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_c`

- Category: `thin_adjacent_part_probe`
- Reason: Retest thin fixed-point adjacent part against promoted part (d); exact but not enough evidence as a standalone source example.
- Proposed source skill: `9709_p3_3_6_fixed_point_iteration`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["thin_adjacent_part", "adjacent_part_contamination", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_a`

- Category: `thin_adjacent_part_probe`
- Reason: Retest thin vector adjacent part against promoted part (b); one-mark scalar-product evidence remains too thin.
- Proposed source skill: `9709_p3_3_7_vector_lines`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["thin_adjacent_part", "adjacent_part_contamination", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:32spring24_q05:32spring24_q05_b`

- Category: `deferred_exact_skill_boundary_probe`
- Reason: Retest clean-looking complex evidence skipped in Batch 0002 because the narrower Argand/loci boundary was not resolved.
- Proposed source skill: `9709_p3_3_9_complex_arithmetic_polar_form`
- Scope: `part_level`
- Related reviewed evidence exists: `False`
- Known risk flags: `["narrower_skill_boundary", "do_not_default_to_clean", "part_level_scope_uses_whole_images", "mark_events_advisory_only"]`
- Mark-event approval probe: `False`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:33summer23_q11:33summer23_q11_b`

- Category: `clean_control_mark_event_probe`
- Reason: Already-promoted clean control: verify source-skill machinery still marks it safe while mark-event refs remain advisory-only.
- Proposed source skill: `9709_p3_3_9_complex_arithmetic_polar_form`
- Scope: `part_level`
- Related reviewed evidence exists: `True`
- Known risk flags: `["already_promoted_clean_control", "mark_events_advisory_only", "mark_event_approval_probe", "part_level_scope_uses_whole_images"]`
- Mark-event approval probe: `True`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_d`

- Category: `clean_control_mark_event_probe`
- Reason: Already-promoted clean control: compare against thin adjacent part (c) and probe whether event-level approval exists.
- Proposed source skill: `9709_p3_3_6_fixed_point_iteration`
- Scope: `part_level`
- Related reviewed evidence exists: `True`
- Known risk flags: `["already_promoted_clean_control", "mark_events_advisory_only", "mark_event_approval_probe", "part_level_scope_uses_whole_images"]`
- Mark-event approval probe: `True`
- Generation ready: `False`

### `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_b`

- Category: `clean_control_mark_event_probe`
- Reason: Already-promoted clean control: compare against thin adjacent part (a) and probe whether event-level approval exists.
- Proposed source skill: `9709_p3_3_7_vector_lines`
- Scope: `part_level`
- Related reviewed evidence exists: `True`
- Known risk flags: `["already_promoted_clean_control", "mark_events_advisory_only", "mark_event_approval_probe", "part_level_scope_uses_whole_images"]`
- Mark-event approval probe: `True`
- Generation ready: `False`
