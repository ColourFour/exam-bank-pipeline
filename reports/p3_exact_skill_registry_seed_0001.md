# P3 Exact-Skill Registry Seed 0001

This report records a narrow promotion of clean draft review responses from `batch_0001` into validated reviewed-decision registry records. It does not promote ambiguous or blocked drafts and does not make Content Lab candidates generation-ready.

## Selected Records

| Evidence ID | Draft response | Reviewed skill | Asset verification | Mark events | Candidate gate | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b` | `batch_0001:1:p3_exact_skill_review_queue:v1:33summer23_q11:33summer23_q11_b` | `9709_p3_3_9_complex_arithmetic_polar_form` | question `question_image:33summer23:33summer23_q11`, mark scheme `mark_scheme_image:33summer23:33summer23_q11` verified | 4 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (b) asks for z^3 in polar/exponential form and the mark scheme confirms argument and modulus work. |
| `p3_exact_skill_review:batch_0001_seed:31summer24_q04:31summer24_q04_b` | `batch_0001:25:p3_exact_skill_review_queue:v1:31summer24_q04:31summer24_q04_b` | `9709_p3_3_9_complex_arithmetic_polar_form` | question `question_image:31summer24:31summer24_q04`, mark scheme `mark_scheme_image:31summer24:31summer24_q04` verified | 2 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (b) divides complex numbers already expressed in polar form and the mark scheme confirms r and theta. |
| `p3_exact_skill_review:batch_0001_seed:32summer23_q06:32summer23_q06_c` | `batch_0001:7:p3_exact_skill_review_queue:v1:32summer23_q06:32summer23_q06_c` | `9709_p3_3_6_fixed_point_iteration` | question `question_image:32summer23:32summer23_q06`, mark scheme `mark_scheme_image:32summer23:32summer23_q06` verified | 3 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (c) explicitly uses the given iterative formula and the mark scheme awards iteration outputs and final accuracy. |
| `p3_exact_skill_review:batch_0001_seed:32autumn23_q06:32autumn23_q06_c` | `batch_0001:10:p3_exact_skill_review_queue:v1:32autumn23_q06:32autumn23_q06_c` | `9709_p3_3_6_fixed_point_iteration` | question `question_image:32autumn23:32autumn23_q06`, mark scheme `mark_scheme_image:32autumn23:32autumn23_q06` verified | 3 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (c) explicitly uses an iterative formula for the root and the mark scheme confirms the iteration sequence. |
| `p3_exact_skill_review:batch_0001_seed:33summer23_q06:33summer23_q06_b` | `batch_0001:16:p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_b` | `9709_p3_3_3_identities_compound_double_angle_equations` | question `question_image:33summer23:33summer23_q06`, mark scheme `mark_scheme_image:33summer23:33summer23_q06` verified | 6 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (b) solves a trig equation using the compound-angle form from part (a); the mark scheme confirms interval-controlled equation solving. |
| `p3_exact_skill_review:batch_0001_seed:33summer23_q09:33summer23_q09_b` | `batch_0001:5:p3_exact_skill_review_queue:v1:33summer23_q09:33summer23_q09_b` | `9709_p3_3_7_vector_lines` | question `question_image:33summer23:33summer23_q09`, mark scheme `mark_scheme_image:33summer23:33summer23_q09` verified | 17 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (b) uses a point on a 3D vector line and perpendicular projection; the mark scheme confirms vector-line and scalar-product work. |
| `p3_exact_skill_review:batch_0001_seed:32spring23_q05:32spring23_q05_b` | `batch_0001:4:p3_exact_skill_review_queue:v1:32spring23_q05:32spring23_q05_b` | `9709_p3_3_4_parametric_implicit_differentiation` | question `question_image:32spring23:32spring23_q05`, mark scheme `mark_scheme_image:32spring23:32spring23_q05` verified | 4 advisory-only matching refs | `blocked_until_reviewed`; reasons `mark_events_not_reviewed_or_approved, missing_source_skill_ids` | Part (b) uses parametric differentiation from part (a) to handle a normal; the mark scheme confirms dy/dx and normal-gradient work. |

## Skipped Records

| Question/subpart | Draft status | Candidate skill | Skip reason |
| --- | --- | --- | --- |
| `33autumn21_q10` / `33autumn21_q10_c` | `clean` | `9709_p3_3_6_fixed_point_iteration` | not_in_small_seed_subset |
| `32autumn23_q09` / `32autumn23_q09_b` | `ambiguous` | `9709_p3_3_3_identities_compound_double_angle_equations` | ambiguous_draft_response_not_promoted |
| `32autumn21_q11` / `32autumn21_q11_c` | `clean` | `9709_p3_3_6_fixed_point_iteration` | not_in_small_seed_subset |
| `33autumn23_q03` / `33autumn23_q03_whole` | `clean` | `9709_p3_3_1_polynomial_division_factor_remainder` | clean_whole_question_candidate_deferred_for_later_pass |
| `32summer22_q10` / `32summer22_q10_d` | `clean` | `9709_p3_3_9_complex_arithmetic_polar_form` | not_in_small_seed_subset |
| `33summer23_q04` / `33summer23_q04_whole` | `clean` | `9709_p3_3_4_parametric_implicit_differentiation` | clean_whole_question_candidate_deferred_for_later_pass |
| `31autumn21_q07` / `31autumn21_q07_c` | `ambiguous` | `9709_p3_3_2_log_exponential_equations` | ambiguous_draft_response_not_promoted |
| `32summer23_q07` / `32summer23_q07_b` | `ambiguous` | `9709_p3_3_4_derivative_rules` | ambiguous_draft_response_not_promoted |
| `31summer24_q01` / `31summer24_q01_whole` | `clean` | `9709_p3_3_1_binomial_rational_expansion` | clean_whole_question_candidate_deferred_for_later_pass |
| `31autumn23_q08` / `31autumn23_q08_d` | `clean` | `9709_p3_3_6_fixed_point_iteration` | clean_but_topic_alignment_unknown_deferred |
| `33autumn23_q07` / `33autumn23_q07_b` | `blocked` | `9709_p3_3_1_polynomial_division_factor_remainder` | blocked_draft_response_not_promoted |
| `32spring23_q10` / `32spring23_q10_c` | `clean` | `9709_p3_3_7_vector_lines` | not_in_small_seed_subset |
| `33autumn23_q06` / `33autumn23_q06_b` | `clean` | `9709_p3_3_3_identities_compound_double_angle_equations` | not_in_small_seed_subset |
| `33autumn23_q01` / `33autumn23_q01_whole` | `clean` | `9709_p3_3_2_log_exponential_equations` | clean_whole_question_candidate_deferred_for_later_pass |
| `32autumn23_q04` / `32autumn23_q04_b` | `clean` | `9709_p3_3_9_complex_arithmetic_polar_form` | not_in_small_seed_subset |
| `32spring23_q02` / `32spring23_q02_b` | `clean` | `9709_p3_3_9_complex_arithmetic_polar_form` | not_in_small_seed_subset |
| `31summer23_q04` / `31summer23_q04_b` | `clean` | `9709_p3_3_3_identities_compound_double_angle_equations` | clean_but_topic_alignment_unknown_deferred |
| `31summer24_q07` / `31summer24_q07_b` | `clean` | `9709_p3_3_9_complex_arithmetic_polar_form` | not_in_small_seed_subset |

## Gating Status

- Registry records after conversion: `10`.
- Clean validated registry records after conversion: `7`.
- Content Lab candidates made generation-ready: `0`.
- Mark-event gating remains active: `true`.
- Selected records keep `candidate_generation=false`; matching mark-event refs remain `advisory` / `advisory_only=true`.

## Schema Friction

- The current reviewed-decision validator can represent clean source-skill evidence separately from candidate-generation readiness.
- Part-level records still cite whole-question and whole mark-scheme images because canonical part crops are not available in this batch.
- Content Lab export is not regenerated from reviewed decisions in this pass, so missing_source_skill_ids remains in existing candidate gate reasons.
