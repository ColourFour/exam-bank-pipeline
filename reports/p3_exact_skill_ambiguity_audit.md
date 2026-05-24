# P3 Exact-Skill Ambiguity Audit

This audit is a triage artifact, not reviewed evidence. It explains where broad ambiguity moved after sharper candidate statuses were applied.

## Summary

- Total queue items: 749
- Audited item count: 749
- Remaining ambiguous candidates: 0
- Audit group counts are not mutually exclusive: `true`

## Status Counts

- `fallback_only`: 34
- `cross_topic_candidate`: 568
- `conflict_candidate`: 126
- `split_needed_candidate`: 21

## Audit Groups

### `cross_topic_reviewable`

- Count: 602
- Automatically reclassifiable safely: `true`
- Recommended handling: Keep reviewable; ask reviewer to confirm primary skill and supporting context.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31summer22_q01:31summer22_q01_whole`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_a`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_c`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_d`, `p3_exact_skill_review_queue:v1:31summer24_q08:31summer24_q08_whole`, `p3_exact_skill_review_queue:v1:32autumn21_q02:32autumn21_q02_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_a`, `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_b`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_a`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_b`
- Candidate skills involved: `9709_p3_3_1_binomial_rational_expansion`, `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p3_3_2_linearising_log_relationships`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_3_rsin_rcos_form`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_substitution_and_parts`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`, `9709_p3_3_7_position_vectors_geometry`, `9709_p3_3_7_scalar_product_angles`, `9709_p3_3_7_vector_lines`, `9709_p3_3_8_initial_conditions_models`, `9709_p3_3_8_separable_differential_equations`, `9709_p3_3_9_argand_loci_geometry`, `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p3_3_9_complex_roots_polynomials`
- Topic-routing topics involved: `9709_p3_topic_algebra`, `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`, `9709_p3_topic_numerical_solution_of_equations`, `9709_p3_topic_trigonometry`, `9709_p3_topic_vectors`

### `cross_topic_split_needed`

- Count: 21
- Automatically reclassifiable safely: `true`
- Recommended handling: Route to part/subpart scope review before any clean decision.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31autumn22_q02:31autumn22_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn23_q02:31autumn23_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn24_q02:31autumn24_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn25_q07:31autumn25_q07_whole`, `p3_exact_skill_review_queue:v1:31summer21_q10:31summer21_q10_whole`, `p3_exact_skill_review_queue:v1:31summer25_q04:31summer25_q04_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q02:32autumn23_q02_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q05:32autumn23_q05_whole`, `p3_exact_skill_review_queue:v1:32spring22_q02:32spring22_q02_whole`, `p3_exact_skill_review_queue:v1:32spring23_q04:32spring23_q04_whole`
- Candidate skills involved: `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`, `9709_p3_3_8_separable_differential_equations`, `9709_p3_3_9_argand_loci_geometry`, `9709_p3_3_9_complex_arithmetic_polar_form`
- Topic-routing topics involved: `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`

### `topic_routing_candidate_mismatch`

- Count: 199
- Automatically reclassifiable safely: `false`
- Recommended handling: Show as review cue; reject only when method context is genuinely conflicting.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_a`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_b`, `p3_exact_skill_review_queue:v1:32spring23_q08:32spring23_q08_a`, `p3_exact_skill_review_queue:v1:32spring23_q08:32spring23_q08_b`, `p3_exact_skill_review_queue:v1:32spring24_q10:32spring24_q10_a`, `p3_exact_skill_review_queue:v1:32summer22_q05:32summer22_q05_b`, `p3_exact_skill_review_queue:v1:32summer22_q05:32summer22_q05_c`, `p3_exact_skill_review_queue:v1:32summer23_q10:32summer23_q10_a`, `p3_exact_skill_review_queue:v1:33autumn22_q03:33autumn22_q03_whole`, `p3_exact_skill_review_queue:v1:33autumn22_q10:33autumn22_q10_a`
- Candidate skills involved: `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_3_rsin_rcos_form`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_tangents_normals_stationary_points`, `9709_p3_3_5_area_volume_applications`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_substitution_and_parts`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`
- Topic-routing topics involved: `9709_p3_topic_algebra`, `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`, `9709_p3_topic_numerical_solution_of_equations`, `9709_p3_topic_trigonometry`, `9709_p3_topic_vectors`

### `de_vs_parametric_implicit_conflict`

- Count: 21
- Automatically reclassifiable safely: `true`
- Recommended handling: Treat as known-risk conflict unless canonical images clearly resolve it.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:32autumn21_q07:32autumn21_q07_whole`, `p3_exact_skill_review_queue:v1:32spring21_q04:32spring21_q04_a`, `p3_exact_skill_review_queue:v1:32spring23_q09:32spring23_q09_whole`, `p3_exact_skill_review_queue:v1:32summer23_q08:32summer23_q08_a`, `p3_exact_skill_review_queue:v1:32summer23_q08:32summer23_q08_b`, `p3_exact_skill_review_queue:v1:33autumn23_q08:33autumn23_q08_whole`, `p3_exact_skill_review_queue:v1:33summer21_q07:33summer21_q07_a`, `p3_exact_skill_review_queue:v1:33summer21_q07:33summer21_q07_b`, `p3_exact_skill_review_queue:v1:33summer22_q06:33summer22_q06_b`, `p3_exact_skill_review_queue:v1:31autumn22_q08:31autumn22_q08_a`
- Candidate skills involved: `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_trig_and_partial_fraction_integration`
- Topic-routing topics involved: `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`

### `multiple_candidate_skills`

- Count: 260
- Automatically reclassifiable safely: `false`
- Recommended handling: Prefer split review; avoid broad whole-question evidence.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31autumn21_q03:31autumn21_q03_a`, `p3_exact_skill_review_queue:v1:31autumn21_q03:31autumn21_q03_b`, `p3_exact_skill_review_queue:v1:31autumn21_q08:31autumn21_q08_a`, `p3_exact_skill_review_queue:v1:31autumn21_q08:31autumn21_q08_c`, `p3_exact_skill_review_queue:v1:31autumn21_q09:31autumn21_q09_a`, `p3_exact_skill_review_queue:v1:31autumn21_q10:31autumn21_q10_d`, `p3_exact_skill_review_queue:v1:31autumn22_q01:31autumn22_q01_a`, `p3_exact_skill_review_queue:v1:31autumn22_q01:31autumn22_q01_b`, `p3_exact_skill_review_queue:v1:31autumn22_q02:31autumn22_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn22_q07:31autumn22_q07_c`
- Candidate skills involved: `9709_p3_3_1_binomial_rational_expansion`, `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p3_3_2_linearising_log_relationships`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_3_rsin_rcos_form`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_tangents_normals_stationary_points`, `9709_p3_3_5_area_volume_applications`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_substitution_and_parts`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`, `9709_p3_3_7_position_vectors_geometry`, `9709_p3_3_7_scalar_product_angles`, `9709_p3_3_7_vector_lines`, `9709_p3_3_8_initial_conditions_models`, `9709_p3_3_8_separable_differential_equations`, `9709_p3_3_9_argand_loci_geometry`, `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p3_3_9_complex_roots_polynomials`
- Topic-routing topics involved: `9709_p3_topic_algebra`, `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`, `9709_p3_topic_numerical_solution_of_equations`, `9709_p3_topic_trigonometry`, `9709_p3_topic_vectors`

### `broad_whole_question_scope`

- Count: 21
- Automatically reclassifiable safely: `false`
- Recommended handling: Require scope decision before approving.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31autumn22_q02:31autumn22_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn23_q02:31autumn23_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn24_q02:31autumn24_q02_whole`, `p3_exact_skill_review_queue:v1:31autumn25_q07:31autumn25_q07_whole`, `p3_exact_skill_review_queue:v1:31summer21_q10:31summer21_q10_whole`, `p3_exact_skill_review_queue:v1:31summer25_q04:31summer25_q04_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q02:32autumn23_q02_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q05:32autumn23_q05_whole`, `p3_exact_skill_review_queue:v1:32spring22_q02:32spring22_q02_whole`, `p3_exact_skill_review_queue:v1:32spring23_q04:32spring23_q04_whole`
- Candidate skills involved: `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`, `9709_p3_3_8_separable_differential_equations`, `9709_p3_3_9_argand_loci_geometry`, `9709_p3_3_9_complex_arithmetic_polar_form`
- Topic-routing topics involved: `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`

### `advisory_only_mark_events`

- Count: 749
- Automatically reclassifiable safely: `false`
- Recommended handling: Use mark events only as context, not authority.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31summer22_q01:31summer22_q01_whole`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_a`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_c`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_d`, `p3_exact_skill_review_queue:v1:31summer24_q08:31summer24_q08_whole`, `p3_exact_skill_review_queue:v1:32autumn21_q02:32autumn21_q02_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_a`, `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_b`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_a`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_b`
- Candidate skills involved: `9709_p3_3_1_binomial_rational_expansion`, `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p3_3_2_linearising_log_relationships`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_3_rsin_rcos_form`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_tangents_normals_stationary_points`, `9709_p3_3_5_area_volume_applications`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_substitution_and_parts`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`, `9709_p3_3_7_position_vectors_geometry`, `9709_p3_3_7_scalar_product_angles`, `9709_p3_3_7_vector_lines`, `9709_p3_3_8_initial_conditions_models`, `9709_p3_3_8_separable_differential_equations`, `9709_p3_3_9_argand_loci_geometry`, `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p3_3_9_complex_roots_polynomials`
- Topic-routing topics involved: `9709_p3_topic_algebra`, `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`, `9709_p3_topic_numerical_solution_of_equations`, `9709_p3_topic_trigonometry`, `9709_p3_topic_vectors`

### `weak_candidate_skill_context`

- Count: 105
- Automatically reclassifiable safely: `true`
- Recommended handling: Deprioritize for clean batches; review only after stronger items or by targeted pass.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31autumn23_q06:31autumn23_q06_a`, `p3_exact_skill_review_queue:v1:31autumn23_q06:31autumn23_q06_b`, `p3_exact_skill_review_queue:v1:31summer23_q02:31summer23_q02_a`, `p3_exact_skill_review_queue:v1:31summer24_q05:31summer24_q05_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q01:32autumn23_q01_a`, `p3_exact_skill_review_queue:v1:32autumn24_q03:32autumn24_q03_whole`, `p3_exact_skill_review_queue:v1:32autumn25_q07:32autumn25_q07_b`, `p3_exact_skill_review_queue:v1:32spring21_q04:32spring21_q04_b`, `p3_exact_skill_review_queue:v1:32spring22_q06:32spring22_q06_whole`, `p3_exact_skill_review_queue:v1:32spring23_q05:32spring23_q05_a`
- Candidate skills involved: `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_3_rsin_rcos_form`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_tangents_normals_stationary_points`, `9709_p3_3_5_area_volume_applications`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_substitution_and_parts`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`
- Topic-routing topics involved: `9709_p3_topic_algebra`, `9709_p3_topic_complex_numbers`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`, `9709_p3_topic_trigonometry`

### `fallback_or_low_quality_context`

- Count: 747
- Automatically reclassifiable safely: `true`
- Recommended handling: Use visual inspection; do not rely on OCR/native text.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31summer22_q01:31summer22_q01_whole`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_a`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_c`, `p3_exact_skill_review_queue:v1:31summer22_q10:31summer22_q10_d`, `p3_exact_skill_review_queue:v1:31summer24_q08:31summer24_q08_whole`, `p3_exact_skill_review_queue:v1:32autumn21_q02:32autumn21_q02_whole`, `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_a`, `p3_exact_skill_review_queue:v1:32autumn23_q07:32autumn23_q07_b`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_a`, `p3_exact_skill_review_queue:v1:32autumn23_q08:32autumn23_q08_b`
- Candidate skills involved: `9709_p3_3_1_binomial_rational_expansion`, `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_partial_fractions`, `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p3_3_2_linearising_log_relationships`, `9709_p3_3_2_log_exponential_equations`, `9709_p3_3_3_identities_compound_double_angle_equations`, `9709_p3_3_3_rsin_rcos_form`, `9709_p3_3_4_derivative_rules`, `9709_p3_3_4_parametric_implicit_differentiation`, `9709_p3_3_4_tangents_normals_stationary_points`, `9709_p3_3_5_area_volume_applications`, `9709_p3_3_5_standard_integration`, `9709_p3_3_5_substitution_and_parts`, `9709_p3_3_5_trig_and_partial_fraction_integration`, `9709_p3_3_6_fixed_point_iteration`, `9709_p3_3_6_root_location`, `9709_p3_3_7_position_vectors_geometry`, `9709_p3_3_7_scalar_product_angles`, `9709_p3_3_7_vector_lines`, `9709_p3_3_8_initial_conditions_models`, `9709_p3_3_8_separable_differential_equations`, `9709_p3_3_9_argand_loci_geometry`, `9709_p3_3_9_complex_arithmetic_polar_form`, `9709_p3_3_9_complex_roots_polynomials`
- Topic-routing topics involved: `9709_p3_topic_algebra`, `9709_p3_topic_complex_numbers`, `9709_p3_topic_differential_equations`, `9709_p3_topic_differentiation`, `9709_p3_topic_integration`, `9709_p3_topic_logarithmic_and_exponential_functions`, `9709_p3_topic_numerical_solution_of_equations`, `9709_p3_topic_trigonometry`, `9709_p3_topic_vectors`

### `reviewed_registry_conflict`

- Count: 3
- Automatically reclassifiable safely: `false`
- Recommended handling: Reconcile against existing reviewed registry before changing any decision.
- Representative queue IDs: `p3_exact_skill_review_queue:v1:31summer21_q01:31summer21_q01_whole`, `p3_exact_skill_review_queue:v1:32spring21_q01:32spring21_q01_whole`, `p3_exact_skill_review_queue:v1:32spring21_q02:32spring21_q02_whole`
- Candidate skills involved: `9709_p3_3_1_modulus_equations_inequalities`, `9709_p3_3_1_polynomial_division_factor_remainder`, `9709_p3_3_2_log_exponential_equations`
- Topic-routing topics involved: `9709_p3_topic_algebra`
