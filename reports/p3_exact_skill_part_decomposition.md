# P3 Exact-Skill Part Decomposition

This report proposes part/subpart review candidates only. It is not reviewed evidence and does not replace canonical image inspection.

## Summary

- whole_question_cross_topic_count: `102`
- decomposition_candidate_count: `513`
- part_level_candidate_count: `0`
- subpart_level_candidate_count: `0`
- needs_manual_split_count: `21`
- insufficient_part_signal_count: `81`
- already_part_scoped_count: `513`
- conflict_needs_review_count: `126`
- not_decomposable_count: `8`

## Decomposition Status Counts

- `not_decomposable`: 8
- `already_part_scoped`: 513
- `insufficient_part_signal`: 81
- `conflict_needs_review`: 126
- `needs_manual_split`: 21

## Diagnostic Notes

- Queue items: `749`
- Already subpart-scoped queue items: `608`
- Whole-question queue items: `141`
- Items with part-labeled mark events: `608`
- Items with existing part-scoped sibling queue records: `608`
- Part labels can be inferred from queue scopes and mark-event `part_path` values.
- Part-level image crops are not created here; whole-question images are linked when no part crop exists.
- Skill/topic suggestions come from existing queue records only and remain review context.

## Representative Candidates

- `p3_part_decomp:v1:31summer22_q10:a` from `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_4_derivative_rules` | confidence `medium`
- `p3_part_decomp:v1:31summer22_q10:c` from `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:31summer22_q10:d` from `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_d` | status `already_part_scoped` | part `d` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q07:a` from `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q07:b` from `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q08:a` from `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q08:b` from `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q10:b` from `p3_exact_skill_review_queue:v1:32autumn23_q10:32autumn23_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_7_scalar_product_angles` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q06:a` from `p3_exact_skill_review_queue:v1:32spring23_q06:32spring23_q06_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q06:b` from `p3_exact_skill_review_queue:v1:32spring23_q06:32spring23_q06_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q08:a` from `p3_exact_skill_review_queue:v1:32spring23_q08:32spring23_q08_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q08:b` from `p3_exact_skill_review_queue:v1:32spring23_q08:32spring23_q08_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q11:a` from `p3_exact_skill_review_queue:v1:32spring23_q11:32spring23_q11_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:32spring24_q10:a` from `p3_exact_skill_review_queue:v1:32spring24_q10:32spring24_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:32summer22_q05:b` from `p3_exact_skill_review_queue:v1:32summer22_q05:32summer22_q05_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:32summer22_q05:c` from `p3_exact_skill_review_queue:v1:32summer22_q05:32summer22_q05_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:32summer23_q10:a` from `p3_exact_skill_review_queue:v1:32summer23_q10:32summer23_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_4_derivative_rules` | confidence `medium`
- `p3_part_decomp:v1:33autumn22_q10:a` from `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:33autumn22_q10:b` from `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:33autumn22_q10:c` from `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:33autumn22_q11:a` from `p3_exact_skill_review_queue:v1:33autumn22_q11:33autumn22_q11_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q09:a` from `p3_exact_skill_review_queue:v1:33autumn23_q09:33autumn23_q09_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q09:b` from `p3_exact_skill_review_queue:v1:33autumn23_q09:33autumn23_q09_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:33summer23_q10:a` from `p3_exact_skill_review_queue:v1:33summer23_q10:33summer23_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:33summer23_q10:b` from `p3_exact_skill_review_queue:v1:33summer23_q10:33summer23_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_1_partial_fractions` | confidence `medium`
- `p3_part_decomp:v1:31autumn21_q07:a` from `p3_exact_skill_review_queue:v1:31autumn21_q07:31autumn21_q07_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:31autumn21_q07:b` from `p3_exact_skill_review_queue:v1:31autumn21_q07:31autumn21_q07_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:31autumn21_q07:c` from `p3_exact_skill_review_queue:v1:31autumn21_q07:31autumn21_q07_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q04:a` from `p3_exact_skill_review_queue:v1:31summer24_q04:31summer24_q04_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q04:b` from `p3_exact_skill_review_queue:v1:31summer24_q04:31summer24_q04_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q06:a` from `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_6_root_location` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q06:c` from `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q06:d` from `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_d` | status `already_part_scoped` | part `d` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q06:e` from `p3_exact_skill_review_queue:v1:31summer24_q06:31summer24_q06_e` | status `already_part_scoped` | part `e` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q07:b` from `p3_exact_skill_review_queue:v1:31summer24_q07:31summer24_q07_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q09:a` from `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q09:b` from `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q09:c` from `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q09:d` from `p3_exact_skill_review_queue:v1:31summer24_q09:31summer24_q09_d` | status `already_part_scoped` | part `d` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q10:a` from `p3_exact_skill_review_queue:v1:31summer24_q10:31summer24_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_5_standard_integration` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q10:b` from `p3_exact_skill_review_queue:v1:31summer24_q10:31summer24_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_5_standard_integration` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q11:a` from `p3_exact_skill_review_queue:v1:31summer24_q11:31summer24_q11_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_8_separable_differential_equations` | confidence `medium`
- `p3_part_decomp:v1:31summer24_q11:b` from `p3_exact_skill_review_queue:v1:31summer24_q11:31summer24_q11_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_8_separable_differential_equations` | confidence `medium`
- `p3_part_decomp:v1:32autumn21_q11:a` from `p3_exact_skill_review_queue:v1:32autumn21_q11:32autumn21_q11_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_4_derivative_rules` | confidence `medium`
- `p3_part_decomp:v1:32autumn21_q11:c` from `p3_exact_skill_review_queue:v1:32autumn21_q11:32autumn21_q11_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q04:b` from `p3_exact_skill_review_queue:v1:32autumn23_q04:32autumn23_q04_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q06:c` from `p3_exact_skill_review_queue:v1:32autumn23_q06:32autumn23_q06_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:32autumn23_q09:b` from `p3_exact_skill_review_queue:v1:32autumn23_q09:32autumn23_q09_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q02:b` from `p3_exact_skill_review_queue:v1:32spring23_q02:32spring23_q02_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q05:b` from `p3_exact_skill_review_queue:v1:32spring23_q05:32spring23_q05_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_4_parametric_implicit_differentiation` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q10:b` from `p3_exact_skill_review_queue:v1:32spring23_q10:32spring23_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:32spring23_q10:c` from `p3_exact_skill_review_queue:v1:32spring23_q10:32spring23_q10_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:32spring24_q03:a` from `p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32spring24_q03:b` from `p3_exact_skill_review_queue:v1:32spring24_q03:32spring24_q03_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32spring24_q05:b` from `p3_exact_skill_review_queue:v1:32spring24_q05:32spring24_q05_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32spring24_q07:c` from `p3_exact_skill_review_queue:v1:32spring24_q07:32spring24_q07_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:32spring24_q09:a` from `p3_exact_skill_review_queue:v1:32spring24_q09:32spring24_q09_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_7_scalar_product_angles` | confidence `low`
- `p3_part_decomp:v1:32spring24_q09:b` from `p3_exact_skill_review_queue:v1:32spring24_q09:32spring24_q09_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_7_scalar_product_angles` | confidence `medium`
- `p3_part_decomp:v1:32summer22_q10:a` from `p3_exact_skill_review_queue:v1:32summer22_q10:32summer22_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32summer22_q10:b` from `p3_exact_skill_review_queue:v1:32summer22_q10:32summer22_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32summer22_q10:c` from `p3_exact_skill_review_queue:v1:32summer22_q10:32summer22_q10_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32summer22_q10:d` from `p3_exact_skill_review_queue:v1:32summer22_q10:32summer22_q10_d` | status `already_part_scoped` | part `d` | skills `9709_p3_3_9_complex_arithmetic_polar_form` | confidence `medium`
- `p3_part_decomp:v1:32summer23_q06:b` from `p3_exact_skill_review_queue:v1:32summer23_q06:32summer23_q06_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:32summer23_q06:c` from `p3_exact_skill_review_queue:v1:32summer23_q06:32summer23_q06_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:32summer23_q07:a` from `p3_exact_skill_review_queue:v1:32summer23_q07:32summer23_q07_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_4_derivative_rules` | confidence `medium`
- `p3_part_decomp:v1:32summer23_q07:b` from `p3_exact_skill_review_queue:v1:32summer23_q07:32summer23_q07_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_4_derivative_rules` | confidence `medium`
- `p3_part_decomp:v1:32summer23_q11:a` from `p3_exact_skill_review_queue:v1:32summer23_q11:32summer23_q11_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:33autumn21_q10:c` from `p3_exact_skill_review_queue:v1:33autumn21_q10:33autumn21_q10_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q06:a` from `p3_exact_skill_review_queue:v1:33autumn23_q06:33autumn23_q06_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q06:b` from `p3_exact_skill_review_queue:v1:33autumn23_q06:33autumn23_q06_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q07:b` from `p3_exact_skill_review_queue:v1:33autumn23_q07:33autumn23_q07_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_1_polynomial_division_factor_remainder` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q11:a` from `p3_exact_skill_review_queue:v1:33autumn23_q11:33autumn23_q11_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q11:b` from `p3_exact_skill_review_queue:v1:33autumn23_q11:33autumn23_q11_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:33autumn23_q11:c` from `p3_exact_skill_review_queue:v1:33autumn23_q11:33autumn23_q11_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_7_vector_lines` | confidence `medium`
- `p3_part_decomp:v1:33summer22_q10:a` from `p3_exact_skill_review_queue:v1:33summer22_q10:33summer22_q10_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:33summer22_q10:b` from `p3_exact_skill_review_queue:v1:33summer22_q10:33summer22_q10_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:33summer22_q10:c` from `p3_exact_skill_review_queue:v1:33summer22_q10:33summer22_q10_c` | status `already_part_scoped` | part `c` | skills `9709_p3_3_2_log_exponential_equations` | confidence `medium`
- `p3_part_decomp:v1:33summer23_q05:b` from `p3_exact_skill_review_queue:v1:33summer23_q05:33summer23_q05_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_6_fixed_point_iteration` | confidence `medium`
- `p3_part_decomp:v1:33summer23_q06:a` from `p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_a` | status `already_part_scoped` | part `a` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
- `p3_part_decomp:v1:33summer23_q06:b` from `p3_exact_skill_review_queue:v1:33summer23_q06:33summer23_q06_b` | status `already_part_scoped` | part `b` | skills `9709_p3_3_3_identities_compound_double_angle_equations` | confidence `medium`
