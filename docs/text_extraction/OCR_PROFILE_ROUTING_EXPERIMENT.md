# OCR Profile Routing Experiment

Experimental fixture-only OCR/preprocessing comparison. Canonical images remain the source of truth.

## Scope

- Records: 36
- Fixture scope: frozen_bad_text_fixtures_only
- Production behavior unchanged: True
- Writes question_bank.json: False

## Profile Summary

| Profile | Avg score | Pass | Warn | Fail | Improved | Regressed | Runtime total | Runtime avg | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| formula_heavy | 91.25 | 22 | 1 | 13 | 13 | 10 | 18.97s | 0.53s | 0 |
| dense_algebra | 90.00 | 21 | 2 | 13 | 11 | 10 | 28.53s | 0.79s | 0 |
| padding_variant | 89.31 | 20 | 3 | 13 | 11 | 11 | 20.12s | 0.56s | 0 |
| grayscale_threshold | 89.31 | 21 | 2 | 13 | 12 | 11 | 20.43s | 0.57s | 0 |
| diagram_safe | 89.17 | 19 | 4 | 13 | 10 | 11 | 18.10s | 0.50s | 0 |
| table_preserving | 83.19 | 16 | 1 | 19 | 5 | 11 | 9.95s | 0.28s | 0 |
| baseline_current | 82.78 | 18 | 5 | 13 | 0 | 0 | 0.00s | 0.00s | 0 |

## Family/Layout Signals

| Group type | Group | Records | Best profile |
| --- | --- | ---: | --- |
| paper_family | p1 | 9 | formula_heavy |
| paper_family | p3 | 15 | formula_heavy |
| paper_family | p4 | 3 | grayscale_threshold |
| paper_family | p5 | 9 | table_preserving |
| layout_type | anchor_sensitive | 36 | formula_heavy |
| layout_type | diagram_safe | 3 | baseline_current |
| layout_type | formula_heavy | 10 | formula_heavy |
| layout_type | table_preserving | 7 | padding_variant |

## Routing Safety Summary

- Routing scope: report_only_candidate_routing
- Writes selected text: False
- Writes Asterion exports: False
- Treats profile output as canonical: False
- Safety rule: A profile is marked safely_better for a slice only when it has at least one improvement, positive average delta, and zero regressions versus baseline in that slice.

| Slice type | Slice | Records | Best safe profile | No-safe reason |
| --- | --- | ---: | --- | --- |
| failure_type | mark_bracket_recovery | 23 | none | all profiles regressed on at least one fixture in this slice |
| failure_type | question_anchor_recovery | 6 | none | all profiles regressed on at least one fixture in this slice |
| failure_type | symbol_heavy_cases | 24 | none | all profiles regressed on at least one fixture in this slice |
| layout_family | calculus_integrals | 6 | none | all profiles regressed on at least one fixture in this slice |
| layout_family | dense_algebra | 10 | none | all profiles regressed on at least one fixture in this slice |
| layout_family | diagrams_tables | 16 | none | all profiles regressed on at least one fixture in this slice |
| layout_family | trig_log_notation | 4 | none | all profiles regressed on at least one fixture in this slice |
| layout_family | vectors_matrices | 4 | none | all profiles regressed on at least one fixture in this slice |
| paper_family | P1 | 9 | none | all profiles regressed on at least one fixture in this slice |
| paper_family | P3 | 15 | none | all profiles regressed on at least one fixture in this slice |
| paper_family | P4 | 3 | grayscale_threshold | none |
| paper_family | P5 | 9 | table_preserving | none |

## Routing Slice Profile Detail

| Slice | Profile | Avg score | Avg delta | Min delta | Max delta | Improved | Regressed | Runtime total | Safety |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| failure_type:mark_bracket_recovery | formula_heavy | 87.17 | 5.43 | -40 | 80 | 7 | 9 | 4.39s | unsafe_regressions |
| failure_type:mark_bracket_recovery | dense_algebra | 85.43 | 3.70 | -40 | 80 | 6 | 9 | 5.46s | unsafe_regressions |
| failure_type:mark_bracket_recovery | padding_variant | 84.35 | 2.61 | -40 | 80 | 6 | 10 | 4.09s | unsafe_regressions |
| failure_type:mark_bracket_recovery | diagram_safe | 84.35 | 2.61 | -40 | 80 | 6 | 10 | 4.24s | unsafe_regressions |
| failure_type:mark_bracket_recovery | grayscale_threshold | 84.13 | 2.39 | -40 | 80 | 6 | 10 | 4.42s | unsafe_regressions |
| failure_type:mark_bracket_recovery | table_preserving | 83.04 | 1.30 | -40 | 80 | 4 | 8 | 4.28s | unsafe_regressions |
| failure_type:question_anchor_recovery | formula_heavy | 89.17 | 33.33 | -20 | 100 | 4 | 2 | 2.16s | unsafe_regressions |
| failure_type:question_anchor_recovery | dense_algebra | 86.67 | 30.83 | -20 | 100 | 4 | 2 | 3.01s | unsafe_regressions |
| failure_type:question_anchor_recovery | grayscale_threshold | 85.00 | 29.17 | -25 | 100 | 4 | 2 | 2.08s | unsafe_regressions |
| failure_type:question_anchor_recovery | diagram_safe | 85.00 | 29.17 | -25 | 100 | 4 | 2 | 2.13s | unsafe_regressions |
| failure_type:question_anchor_recovery | padding_variant | 85.00 | 29.17 | -25 | 100 | 4 | 2 | 2.17s | unsafe_regressions |
| failure_type:question_anchor_recovery | table_preserving | 69.17 | 13.33 | -20 | 100 | 1 | 1 | 2.15s | unsafe_regressions |
| failure_type:symbol_heavy_cases | formula_heavy | 86.88 | 5.00 | -40 | 80 | 9 | 10 | 5.42s | unsafe_regressions |
| failure_type:symbol_heavy_cases | dense_algebra | 85.21 | 3.33 | -40 | 80 | 8 | 10 | 6.46s | unsafe_regressions |
| failure_type:symbol_heavy_cases | padding_variant | 84.17 | 2.29 | -40 | 80 | 8 | 11 | 4.45s | unsafe_regressions |
| failure_type:symbol_heavy_cases | grayscale_threshold | 83.96 | 2.08 | -40 | 80 | 8 | 11 | 5.04s | unsafe_regressions |
| failure_type:symbol_heavy_cases | diagram_safe | 83.96 | 2.08 | -40 | 80 | 7 | 11 | 4.62s | unsafe_regressions |
| failure_type:symbol_heavy_cases | table_preserving | 80.00 | -1.88 | -40 | 80 | 4 | 10 | 4.76s | unsafe_regressions |
| layout_family:calculus_integrals | formula_heavy | 82.50 | 10.83 | -20 | 80 | 3 | 3 | 1.20s | unsafe_regressions |
| layout_family:calculus_integrals | dense_algebra | 76.67 | 5.00 | -20 | 80 | 3 | 3 | 1.76s | unsafe_regressions |
| layout_family:calculus_integrals | grayscale_threshold | 75.83 | 4.17 | -25 | 80 | 3 | 3 | 1.12s | unsafe_regressions |
| layout_family:calculus_integrals | padding_variant | 75.83 | 4.17 | -25 | 80 | 3 | 3 | 1.16s | unsafe_regressions |
| layout_family:calculus_integrals | table_preserving | 75.83 | 4.17 | -40 | 80 | 2 | 3 | 1.09s | unsafe_regressions |
| layout_family:calculus_integrals | diagram_safe | 75.00 | 3.33 | -25 | 80 | 2 | 3 | 1.19s | unsafe_regressions |
| layout_family:dense_algebra | formula_heavy | 87.00 | 14.50 | -20 | 60 | 5 | 3 | 1.98s | unsafe_regressions |
| layout_family:dense_algebra | dense_algebra | 82.50 | 10.00 | -20 | 60 | 4 | 3 | 2.37s | unsafe_regressions |
| layout_family:dense_algebra | table_preserving | 81.00 | 8.50 | -20 | 60 | 3 | 2 | 1.75s | unsafe_regressions |
| layout_family:dense_algebra | padding_variant | 80.00 | 7.50 | -25 | 60 | 4 | 4 | 1.77s | unsafe_regressions |
| layout_family:dense_algebra | diagram_safe | 80.00 | 7.50 | -25 | 60 | 4 | 4 | 1.79s | unsafe_regressions |
| layout_family:dense_algebra | grayscale_threshold | 80.00 | 7.50 | -25 | 60 | 4 | 4 | 2.20s | unsafe_regressions |
| layout_family:diagrams_tables | grayscale_threshold | 98.75 | 7.50 | -15 | 40 | 6 | 1 | 14.28s | unsafe_regressions |
| layout_family:diagrams_tables | formula_heavy | 98.75 | 7.50 | -15 | 40 | 6 | 1 | 14.40s | unsafe_regressions |
| layout_family:diagrams_tables | padding_variant | 98.44 | 7.19 | -15 | 40 | 5 | 1 | 16.30s | unsafe_regressions |
| layout_family:diagrams_tables | dense_algebra | 98.44 | 7.19 | -15 | 40 | 5 | 1 | 23.20s | unsafe_regressions |
| layout_family:diagrams_tables | diagram_safe | 98.12 | 6.88 | -15 | 40 | 4 | 1 | 14.13s | unsafe_regressions |
| layout_family:diagrams_tables | table_preserving | 84.06 | -7.19 | -40 | 0 | 0 | 3 | 5.58s | unsafe_regressions |
| layout_family:trig_log_notation | diagram_safe | 75.00 | -20.00 | -40 | 0 | 0 | 3 | 0.45s | unsafe_regressions |
| layout_family:trig_log_notation | table_preserving | 75.00 | -20.00 | -40 | 0 | 0 | 3 | 0.48s | unsafe_regressions |
| layout_family:trig_log_notation | padding_variant | 75.00 | -20.00 | -40 | 0 | 0 | 3 | 0.51s | unsafe_regressions |
| layout_family:trig_log_notation | dense_algebra | 75.00 | -20.00 | -40 | 0 | 0 | 3 | 0.77s | unsafe_regressions |
| layout_family:trig_log_notation | formula_heavy | 73.75 | -21.25 | -40 | 0 | 0 | 3 | 0.46s | unsafe_regressions |
| layout_family:trig_log_notation | grayscale_threshold | 73.75 | -21.25 | -40 | 0 | 0 | 3 | 0.61s | unsafe_regressions |
| layout_family:vectors_matrices | grayscale_threshold | 90.00 | -6.25 | -15 | 5 | 1 | 2 | 1.41s | unsafe_regressions |
| layout_family:vectors_matrices | diagram_safe | 90.00 | -6.25 | -15 | 5 | 1 | 2 | 1.41s | unsafe_regressions |
| layout_family:vectors_matrices | padding_variant | 90.00 | -6.25 | -15 | 5 | 1 | 2 | 1.43s | unsafe_regressions |
| layout_family:vectors_matrices | formula_heavy | 90.00 | -6.25 | -15 | 5 | 1 | 2 | 2.00s | unsafe_regressions |
| layout_family:vectors_matrices | dense_algebra | 90.00 | -6.25 | -15 | 5 | 1 | 2 | 2.52s | unsafe_regressions |
| layout_family:vectors_matrices | table_preserving | 83.75 | -12.50 | -35 | 0 | 0 | 2 | 1.59s | unsafe_regressions |
| paper_family:P1 | formula_heavy | 88.33 | 3.89 | -40 | 45 | 3 | 4 | 2.87s | unsafe_regressions |
| paper_family:P1 | dense_algebra | 86.67 | 2.22 | -40 | 45 | 3 | 3 | 3.68s | unsafe_regressions |
| paper_family:P1 | diagram_safe | 85.56 | 1.11 | -40 | 40 | 3 | 4 | 2.25s | unsafe_regressions |
| paper_family:P1 | padding_variant | 85.56 | 1.11 | -40 | 40 | 3 | 4 | 2.31s | unsafe_regressions |
| paper_family:P1 | grayscale_threshold | 85.56 | 1.11 | -40 | 40 | 3 | 4 | 2.34s | unsafe_regressions |
| paper_family:P1 | table_preserving | 81.11 | -3.33 | -40 | 25 | 1 | 2 | 2.41s | unsafe_regressions |
| paper_family:P3 | formula_heavy | 87.33 | 5.67 | -25 | 80 | 6 | 6 | 2.71s | unsafe_regressions |
| paper_family:P3 | dense_algebra | 85.67 | 4.00 | -20 | 80 | 5 | 7 | 3.58s | unsafe_regressions |
| paper_family:P3 | padding_variant | 84.67 | 3.00 | -25 | 80 | 5 | 7 | 2.51s | unsafe_regressions |
| paper_family:P3 | grayscale_threshold | 84.33 | 2.67 | -25 | 80 | 5 | 7 | 2.79s | unsafe_regressions |
| paper_family:P3 | diagram_safe | 84.33 | 2.67 | -25 | 80 | 4 | 7 | 2.56s | unsafe_regressions |
| paper_family:P3 | table_preserving | 80.67 | -1.00 | -40 | 80 | 3 | 8 | 2.44s | unsafe_regressions |
| paper_family:P4 | grayscale_threshold | 100.00 | 28.33 | 5 | 40 | 3 | 0 | 1.35s | safely_better |
| paper_family:P4 | formula_heavy | 100.00 | 28.33 | 5 | 40 | 3 | 0 | 1.42s | safely_better |
| paper_family:P4 | diagram_safe | 98.33 | 26.67 | 0 | 40 | 2 | 0 | 1.54s | safely_better |
| paper_family:P4 | padding_variant | 98.33 | 26.67 | 0 | 40 | 2 | 0 | 1.62s | safely_better |
| paper_family:P4 | dense_algebra | 98.33 | 26.67 | 0 | 40 | 2 | 0 | 2.54s | safely_better |
| paper_family:P4 | table_preserving | 58.33 | -13.33 | -40 | 0 | 0 | 1 | 1.44s | unsafe_regressions |
| paper_family:P5 | table_preserving | 97.78 | 11.11 | 0 | 100 | 1 | 0 | 3.67s | safely_better |
| paper_family:P5 | diagram_safe | 97.78 | 11.11 | 0 | 100 | 1 | 0 | 11.75s | safely_better |
| paper_family:P5 | formula_heavy | 97.78 | 11.11 | 0 | 100 | 1 | 0 | 11.96s | safely_better |
| paper_family:P5 | padding_variant | 97.78 | 11.11 | 0 | 100 | 1 | 0 | 13.67s | safely_better |
| paper_family:P5 | grayscale_threshold | 97.78 | 11.11 | 0 | 100 | 1 | 0 | 13.95s | safely_better |
| paper_family:P5 | dense_algebra | 97.78 | 11.11 | 0 | 100 | 1 | 0 | 18.74s | safely_better |

## Routing Slice Regressions

| Slice | Record | Profile | Delta | Introduced issues |
| --- | --- | --- | ---: | --- |
| failure_type:mark_bracket_recovery | 11summer23_q01 | dense_algebra | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 11summer23_q01 | diagram_safe | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 11summer23_q01 | formula_heavy | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 11summer23_q01 | grayscale_threshold | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 11summer23_q01 | padding_variant | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 11summer23_q01 | table_preserving | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 12autumn23_q06 | dense_algebra | -15 | score-only regression |
| failure_type:mark_bracket_recovery | 12autumn23_q06 | diagram_safe | -15 | score-only regression |
| failure_type:mark_bracket_recovery | 12autumn23_q06 | formula_heavy | -15 | score-only regression |
| failure_type:mark_bracket_recovery | 12autumn23_q06 | grayscale_threshold | -15 | score-only regression |
| failure_type:mark_bracket_recovery | 12autumn23_q06 | padding_variant | -15 | score-only regression |
| failure_type:mark_bracket_recovery | 12autumn23_q06 | table_preserving | -15 | score-only regression |
| failure_type:mark_bracket_recovery | 12spring21_q02 | diagram_safe | -5 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 12spring21_q02 | formula_heavy | -5 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 12spring21_q02 | grayscale_threshold | -5 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 12spring21_q02 | padding_variant | -5 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 31summer23_q09 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 31summer23_q09 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 31summer23_q09 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 31summer23_q09 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 31summer23_q09 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 31summer23_q09 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q03 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q03 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q03 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q03 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q03 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q03 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q04 | dense_algebra | -5 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q04 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q04 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q04 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32spring23_q04 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer21_q01 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer21_q01 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer21_q01 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer21_q01 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer21_q01 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer21_q01 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| failure_type:mark_bracket_recovery | 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:mark_bracket_recovery | 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:mark_bracket_recovery | 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:mark_bracket_recovery | 33autumn21_q04 | dense_algebra | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:mark_bracket_recovery | 33autumn21_q04 | diagram_safe | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:mark_bracket_recovery | 33autumn21_q04 | formula_heavy | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:mark_bracket_recovery | 33autumn21_q04 | grayscale_threshold | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:mark_bracket_recovery | 33autumn21_q04 | padding_variant | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:mark_bracket_recovery | 33autumn21_q04 | table_preserving | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:mark_bracket_recovery | 33autumn21_q05 | dense_algebra | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:mark_bracket_recovery | 33autumn21_q05 | diagram_safe | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:mark_bracket_recovery | 33autumn21_q05 | formula_heavy | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 33autumn21_q05 | grayscale_threshold | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:mark_bracket_recovery | 33autumn21_q05 | padding_variant | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:mark_bracket_recovery | 33autumn21_q05 | table_preserving | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:question_anchor_recovery | 31summer23_q09 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 31summer23_q09 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 31summer23_q09 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 31summer23_q09 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 31summer23_q09 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 31summer23_q09 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| failure_type:question_anchor_recovery | 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:question_anchor_recovery | 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| failure_type:question_anchor_recovery | 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:question_anchor_recovery | 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:symbol_heavy_cases | 11autumn23_q04 | dense_algebra | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 11autumn23_q04 | diagram_safe | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 11autumn23_q04 | formula_heavy | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 11autumn23_q04 | grayscale_threshold | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 11autumn23_q04 | padding_variant | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 11summer23_q01 | dense_algebra | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 11summer23_q01 | diagram_safe | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 11summer23_q01 | formula_heavy | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 11summer23_q01 | grayscale_threshold | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 11summer23_q01 | padding_variant | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 11summer23_q01 | table_preserving | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 12autumn23_q06 | dense_algebra | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 12autumn23_q06 | diagram_safe | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 12autumn23_q06 | formula_heavy | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 12autumn23_q06 | grayscale_threshold | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 12autumn23_q06 | padding_variant | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 12autumn23_q06 | table_preserving | -15 | score-only regression |
| failure_type:symbol_heavy_cases | 12spring21_q02 | diagram_safe | -5 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 12spring21_q02 | formula_heavy | -5 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 12spring21_q02 | grayscale_threshold | -5 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 12spring21_q02 | padding_variant | -5 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 31summer22_q09 | table_preserving | -35 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| failure_type:symbol_heavy_cases | 31summer23_q09 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 31summer23_q09 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 31summer23_q09 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 31summer23_q09 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 31summer23_q09 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 31summer23_q09 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q03 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q03 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q03 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q03 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q03 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q03 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q04 | dense_algebra | -5 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q04 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q04 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q04 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32spring23_q04 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer21_q01 | dense_algebra | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer21_q01 | diagram_safe | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer21_q01 | formula_heavy | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer21_q01 | grayscale_threshold | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer21_q01 | padding_variant | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer21_q01 | table_preserving | -20 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| failure_type:symbol_heavy_cases | 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:symbol_heavy_cases | 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:symbol_heavy_cases | 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| failure_type:symbol_heavy_cases | 33autumn21_q04 | dense_algebra | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:symbol_heavy_cases | 33autumn21_q04 | diagram_safe | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:symbol_heavy_cases | 33autumn21_q04 | formula_heavy | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:symbol_heavy_cases | 33autumn21_q04 | grayscale_threshold | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:symbol_heavy_cases | 33autumn21_q04 | padding_variant | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:symbol_heavy_cases | 33autumn21_q04 | table_preserving | -20 | expected_structural_requirement_missing:integral_sign |
| failure_type:symbol_heavy_cases | 33autumn21_q05 | dense_algebra | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:symbol_heavy_cases | 33autumn21_q05 | diagram_safe | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:symbol_heavy_cases | 33autumn21_q05 | formula_heavy | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 33autumn21_q05 | grayscale_threshold | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| failure_type:symbol_heavy_cases | 33autumn21_q05 | padding_variant | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:symbol_heavy_cases | 33autumn21_q05 | table_preserving | -20 | expected_structural_requirement_missing:theta_symbol |
| failure_type:symbol_heavy_cases | 33summer21_q07 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| layout_family:calculus_integrals | 31summer23_q09 | dense_algebra | -20 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 31summer23_q09 | diagram_safe | -20 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 31summer23_q09 | formula_heavy | -20 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 31summer23_q09 | grayscale_threshold | -20 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 31summer23_q09 | padding_variant | -20 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 31summer23_q09 | table_preserving | -20 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| layout_family:calculus_integrals | 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| layout_family:calculus_integrals | 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| layout_family:calculus_integrals | 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| layout_family:calculus_integrals | 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| layout_family:calculus_integrals | 33autumn21_q04 | dense_algebra | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:calculus_integrals | 33autumn21_q04 | diagram_safe | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:calculus_integrals | 33autumn21_q04 | formula_heavy | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:calculus_integrals | 33autumn21_q04 | grayscale_threshold | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:calculus_integrals | 33autumn21_q04 | padding_variant | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:calculus_integrals | 33autumn21_q04 | table_preserving | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:calculus_integrals | 33summer21_q07 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| layout_family:dense_algebra | 12spring21_q02 | diagram_safe | -5 | likely_math_symbol_loss |
| layout_family:dense_algebra | 12spring21_q02 | formula_heavy | -5 | likely_math_symbol_loss |
| layout_family:dense_algebra | 12spring21_q02 | grayscale_threshold | -5 | likely_math_symbol_loss |
| layout_family:dense_algebra | 12spring21_q02 | padding_variant | -5 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q03 | dense_algebra | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q03 | diagram_safe | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q03 | formula_heavy | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q03 | grayscale_threshold | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q03 | padding_variant | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q03 | table_preserving | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q04 | dense_algebra | -5 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q04 | diagram_safe | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q04 | grayscale_threshold | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q04 | padding_variant | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32spring23_q04 | table_preserving | -20 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| layout_family:dense_algebra | 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| layout_family:dense_algebra | 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| layout_family:dense_algebra | 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| layout_family:dense_algebra | 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| layout_family:diagrams_tables | 11autumn23_q04 | dense_algebra | -15 | score-only regression |
| layout_family:diagrams_tables | 11autumn23_q04 | diagram_safe | -15 | score-only regression |
| layout_family:diagrams_tables | 11autumn23_q04 | formula_heavy | -15 | score-only regression |
| layout_family:diagrams_tables | 11autumn23_q04 | grayscale_threshold | -15 | score-only regression |
| layout_family:diagrams_tables | 11autumn23_q04 | padding_variant | -15 | score-only regression |
| layout_family:diagrams_tables | 31summer22_q09 | table_preserving | -35 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| layout_family:diagrams_tables | 33summer21_q07 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| layout_family:diagrams_tables | 42autumn21_q01 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| layout_family:trig_log_notation | 11summer23_q01 | dense_algebra | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 11summer23_q01 | diagram_safe | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 11summer23_q01 | formula_heavy | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 11summer23_q01 | grayscale_threshold | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 11summer23_q01 | padding_variant | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 11summer23_q01 | table_preserving | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 33autumn21_q04 | dense_algebra | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:trig_log_notation | 33autumn21_q04 | diagram_safe | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:trig_log_notation | 33autumn21_q04 | formula_heavy | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:trig_log_notation | 33autumn21_q04 | grayscale_threshold | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:trig_log_notation | 33autumn21_q04 | padding_variant | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:trig_log_notation | 33autumn21_q04 | table_preserving | -20 | expected_structural_requirement_missing:integral_sign |
| layout_family:trig_log_notation | 33autumn21_q05 | dense_algebra | -20 | expected_structural_requirement_missing:theta_symbol |
| layout_family:trig_log_notation | 33autumn21_q05 | diagram_safe | -20 | expected_structural_requirement_missing:theta_symbol |
| layout_family:trig_log_notation | 33autumn21_q05 | formula_heavy | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 33autumn21_q05 | grayscale_threshold | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| layout_family:trig_log_notation | 33autumn21_q05 | padding_variant | -20 | expected_structural_requirement_missing:theta_symbol |
| layout_family:trig_log_notation | 33autumn21_q05 | table_preserving | -20 | expected_structural_requirement_missing:theta_symbol |
| layout_family:vectors_matrices | 11autumn23_q04 | dense_algebra | -15 | score-only regression |
| layout_family:vectors_matrices | 11autumn23_q04 | diagram_safe | -15 | score-only regression |
| layout_family:vectors_matrices | 11autumn23_q04 | formula_heavy | -15 | score-only regression |
| layout_family:vectors_matrices | 11autumn23_q04 | grayscale_threshold | -15 | score-only regression |
| layout_family:vectors_matrices | 11autumn23_q04 | padding_variant | -15 | score-only regression |
| layout_family:vectors_matrices | 12autumn23_q06 | dense_algebra | -15 | score-only regression |
| layout_family:vectors_matrices | 12autumn23_q06 | diagram_safe | -15 | score-only regression |
| layout_family:vectors_matrices | 12autumn23_q06 | formula_heavy | -15 | score-only regression |
| layout_family:vectors_matrices | 12autumn23_q06 | grayscale_threshold | -15 | score-only regression |
| layout_family:vectors_matrices | 12autumn23_q06 | padding_variant | -15 | score-only regression |
| layout_family:vectors_matrices | 12autumn23_q06 | table_preserving | -15 | score-only regression |
| layout_family:vectors_matrices | 31summer22_q09 | table_preserving | -35 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| paper_family:P1 | 11autumn23_q04 | dense_algebra | -15 | score-only regression |
| paper_family:P1 | 11autumn23_q04 | diagram_safe | -15 | score-only regression |
| paper_family:P1 | 11autumn23_q04 | formula_heavy | -15 | score-only regression |
| paper_family:P1 | 11autumn23_q04 | grayscale_threshold | -15 | score-only regression |
| paper_family:P1 | 11autumn23_q04 | padding_variant | -15 | score-only regression |
| paper_family:P1 | 11summer23_q01 | dense_algebra | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P1 | 11summer23_q01 | diagram_safe | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P1 | 11summer23_q01 | formula_heavy | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P1 | 11summer23_q01 | grayscale_threshold | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P1 | 11summer23_q01 | padding_variant | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P1 | 11summer23_q01 | table_preserving | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P1 | 12autumn23_q06 | dense_algebra | -15 | score-only regression |
| paper_family:P1 | 12autumn23_q06 | diagram_safe | -15 | score-only regression |
| paper_family:P1 | 12autumn23_q06 | formula_heavy | -15 | score-only regression |
| paper_family:P1 | 12autumn23_q06 | grayscale_threshold | -15 | score-only regression |
| paper_family:P1 | 12autumn23_q06 | padding_variant | -15 | score-only regression |
| paper_family:P1 | 12autumn23_q06 | table_preserving | -15 | score-only regression |
| paper_family:P1 | 12spring21_q02 | diagram_safe | -5 | likely_math_symbol_loss |
| paper_family:P1 | 12spring21_q02 | formula_heavy | -5 | likely_math_symbol_loss |
| paper_family:P1 | 12spring21_q02 | grayscale_threshold | -5 | likely_math_symbol_loss |
| paper_family:P1 | 12spring21_q02 | padding_variant | -5 | likely_math_symbol_loss |
| paper_family:P3 | 31summer22_q09 | table_preserving | -35 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| paper_family:P3 | 31summer23_q09 | dense_algebra | -20 | likely_math_symbol_loss |
| paper_family:P3 | 31summer23_q09 | diagram_safe | -20 | likely_math_symbol_loss |
| paper_family:P3 | 31summer23_q09 | formula_heavy | -20 | likely_math_symbol_loss |
| paper_family:P3 | 31summer23_q09 | grayscale_threshold | -20 | likely_math_symbol_loss |
| paper_family:P3 | 31summer23_q09 | padding_variant | -20 | likely_math_symbol_loss |
| paper_family:P3 | 31summer23_q09 | table_preserving | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q03 | dense_algebra | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q03 | diagram_safe | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q03 | formula_heavy | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q03 | grayscale_threshold | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q03 | padding_variant | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q03 | table_preserving | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q04 | dense_algebra | -5 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q04 | diagram_safe | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q04 | grayscale_threshold | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q04 | padding_variant | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32spring23_q04 | table_preserving | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer21_q01 | dense_algebra | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer21_q01 | diagram_safe | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer21_q01 | formula_heavy | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer21_q01 | grayscale_threshold | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer21_q01 | padding_variant | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer21_q01 | table_preserving | -20 | likely_math_symbol_loss |
| paper_family:P3 | 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| paper_family:P3 | 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| paper_family:P3 | 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| paper_family:P3 | 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| paper_family:P3 | 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| paper_family:P3 | 33autumn21_q04 | dense_algebra | -20 | expected_structural_requirement_missing:integral_sign |
| paper_family:P3 | 33autumn21_q04 | diagram_safe | -20 | expected_structural_requirement_missing:integral_sign |
| paper_family:P3 | 33autumn21_q04 | formula_heavy | -20 | expected_structural_requirement_missing:integral_sign |
| paper_family:P3 | 33autumn21_q04 | grayscale_threshold | -20 | expected_structural_requirement_missing:integral_sign |
| paper_family:P3 | 33autumn21_q04 | padding_variant | -20 | expected_structural_requirement_missing:integral_sign |
| paper_family:P3 | 33autumn21_q04 | table_preserving | -20 | expected_structural_requirement_missing:integral_sign |
| paper_family:P3 | 33autumn21_q05 | dense_algebra | -20 | expected_structural_requirement_missing:theta_symbol |
| paper_family:P3 | 33autumn21_q05 | diagram_safe | -20 | expected_structural_requirement_missing:theta_symbol |
| paper_family:P3 | 33autumn21_q05 | formula_heavy | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P3 | 33autumn21_q05 | grayscale_threshold | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| paper_family:P3 | 33autumn21_q05 | padding_variant | -20 | expected_structural_requirement_missing:theta_symbol |
| paper_family:P3 | 33autumn21_q05 | table_preserving | -20 | expected_structural_requirement_missing:theta_symbol |
| paper_family:P3 | 33summer21_q07 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| paper_family:P4 | 42autumn21_q01 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |

## Per-Record Best/Worst

| Record | Family | Layouts | Best | Best score | Worst | Worst score |
| --- | --- | --- | --- | ---: | --- | ---: |
| 12summer21_q03 | p1 | anchor_sensitive, formula_heavy, table_preserving | baseline_current | 100 | dense_algebra | 100 |
| 32autumn21_q04 | p3 | anchor_sensitive, formula_heavy | formula_heavy | 80 | dense_algebra | 75 |
| 33autumn21_q04 | p3 | anchor_sensitive, formula_heavy | baseline_current | 100 | grayscale_threshold | 80 |
| 33autumn21_q05 | p3 | anchor_sensitive | baseline_current | 100 | formula_heavy | 75 |
| 31summer22_q09 | p3 | anchor_sensitive, diagram_safe | formula_heavy | 100 | table_preserving | 60 |
| 32spring23_q04 | p3 | anchor_sensitive, formula_heavy | baseline_current | 80 | grayscale_threshold | 60 |
| 12summer23_q01 | p1 | anchor_sensitive, formula_heavy | formula_heavy | 80 | baseline_current | 55 |
| 13summer23_q01 | p1 | anchor_sensitive | baseline_current | 100 | dense_algebra | 100 |
| 33summer24_q03 | p3 | anchor_sensitive | padding_variant | 100 | baseline_current | 60 |
| 35summer25_q04 | p3 | anchor_sensitive | grayscale_threshold | 100 | baseline_current | 20 |
| 33autumn25_q07 | p3 | anchor_sensitive, formula_heavy | table_preserving | 100 | baseline_current | 40 |
| 13autumn25_q09 | p1 | anchor_sensitive | grayscale_threshold | 100 | table_preserving | 60 |
| 43autumn21_q06 | p4 | anchor_sensitive, table_preserving | grayscale_threshold | 100 | table_preserving | 60 |
| 12spring22_q08 | p1 | anchor_sensitive, formula_heavy | formula_heavy | 100 | table_preserving | 55 |
| 52spring22_q06 | p5 | anchor_sensitive | diagram_safe | 100 | baseline_current | 0 |
| 32summer23_q09 | p3 | anchor_sensitive, formula_heavy | baseline_current | 60 | padding_variant | 35 |
| 41summer23_q06 | p4 | anchor_sensitive | grayscale_threshold | 100 | table_preserving | 60 |
| 52spring21_q05 | p5 | anchor_sensitive, table_preserving | baseline_current | 100 | dense_algebra | 100 |
| 51summer21_q05 | p5 | anchor_sensitive | baseline_current | 100 | dense_algebra | 100 |
| 52summer21_q02 | p5 | anchor_sensitive | baseline_current | 80 | dense_algebra | 80 |
| 52summer22_q02 | p5 | anchor_sensitive, table_preserving | baseline_current | 100 | dense_algebra | 100 |
| 53summer22_q07 | p5 | anchor_sensitive | baseline_current | 100 | grayscale_threshold | 100 |
| 11autumn23_q04 | p1 | anchor_sensitive, diagram_safe | baseline_current | 95 | formula_heavy | 80 |
| 12autumn23_q06 | p1 | anchor_sensitive | baseline_current | 95 | table_preserving | 80 |
| 51autumn23_q05 | p5 | anchor_sensitive, table_preserving | baseline_current | 100 | dense_algebra | 100 |
| 42autumn21_q01 | p4 | anchor_sensitive | grayscale_threshold | 100 | table_preserving | 55 |
| 52autumn21_q01 | p5 | anchor_sensitive, table_preserving | baseline_current | 100 | dense_algebra | 100 |
| 31summer21_q01 | p3 | anchor_sensitive | baseline_current | 100 | table_preserving | 100 |
| 32summer21_q01 | p3 | anchor_sensitive | baseline_current | 100 | dense_algebra | 80 |
| 33autumn21_q02 | p3 | anchor_sensitive | baseline_current | 100 | dense_algebra | 100 |
| 32spring23_q03 | p3 | anchor_sensitive | baseline_current | 100 | dense_algebra | 80 |
| 11summer23_q01 | p1 | anchor_sensitive | baseline_current | 100 | dense_algebra | 60 |
| 12spring21_q02 | p1 | anchor_sensitive, formula_heavy | baseline_current | 100 | formula_heavy | 95 |
| 33summer21_q07 | p3 | anchor_sensitive, diagram_safe | grayscale_threshold | 100 | table_preserving | 55 |
| 52summer21_q04 | p5 | anchor_sensitive, table_preserving | baseline_current | 100 | padding_variant | 100 |
| 31summer23_q09 | p3 | anchor_sensitive, formula_heavy | baseline_current | 100 | dense_algebra | 80 |

## Improvements

| Record | Profile | Delta | Resolved issues |
| --- | --- | ---: | --- |
| 32autumn21_q04 | formula_heavy | +5 | likely_math_symbol_loss |
| 31summer22_q09 | grayscale_threshold | +5 | likely_math_symbol_loss |
| 31summer22_q09 | formula_heavy | +5 | likely_math_symbol_loss |
| 31summer22_q09 | diagram_safe | +5 | likely_math_symbol_loss |
| 31summer22_q09 | padding_variant | +5 | likely_math_symbol_loss |
| 31summer22_q09 | dense_algebra | +5 | likely_math_symbol_loss |
| 12summer23_q01 | grayscale_threshold | +5 | expected_structural_requirement_missing:point_4_5 |
| 12summer23_q01 | formula_heavy | +25 | expected_structural_requirement_missing:point_4_5, likely_math_symbol_loss |
| 12summer23_q01 | table_preserving | +25 | expected_structural_requirement_missing:point_4_5, likely_math_symbol_loss |
| 12summer23_q01 | diagram_safe | +5 | expected_structural_requirement_missing:point_4_5 |
| 12summer23_q01 | padding_variant | +5 | expected_structural_requirement_missing:point_4_5 |
| 12summer23_q01 | dense_algebra | +5 | expected_structural_requirement_missing:point_4_5 |
| 33summer24_q03 | grayscale_threshold | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 33summer24_q03 | formula_heavy | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 33summer24_q03 | table_preserving | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 33summer24_q03 | diagram_safe | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 33summer24_q03 | padding_variant | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 33summer24_q03 | dense_algebra | +35 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 35summer25_q04 | grayscale_threshold | +80 | expected_structural_requirement_missing:specific_mark_bracket, expected_structural_requirement_missing:x_greater_than_zero, missing_mark_bracket, missing_specific_mark_bracket |
| 35summer25_q04 | formula_heavy | +80 | expected_structural_requirement_missing:specific_mark_bracket, expected_structural_requirement_missing:x_greater_than_zero, missing_mark_bracket, missing_specific_mark_bracket |
| 35summer25_q04 | table_preserving | +80 | expected_structural_requirement_missing:specific_mark_bracket, expected_structural_requirement_missing:x_greater_than_zero, missing_mark_bracket, missing_specific_mark_bracket |
| 35summer25_q04 | diagram_safe | +80 | expected_structural_requirement_missing:specific_mark_bracket, expected_structural_requirement_missing:x_greater_than_zero, missing_mark_bracket, missing_specific_mark_bracket |
| 35summer25_q04 | padding_variant | +80 | expected_structural_requirement_missing:specific_mark_bracket, expected_structural_requirement_missing:x_greater_than_zero, missing_mark_bracket, missing_specific_mark_bracket |
| 35summer25_q04 | dense_algebra | +80 | expected_structural_requirement_missing:specific_mark_bracket, expected_structural_requirement_missing:x_greater_than_zero, missing_mark_bracket, missing_specific_mark_bracket |
| 33autumn25_q07 | grayscale_threshold | +60 | expected_structural_requirement_missing:specific_mark_bracket, missing_mark_bracket, missing_specific_mark_bracket |
| 33autumn25_q07 | formula_heavy | +60 | expected_structural_requirement_missing:specific_mark_bracket, missing_mark_bracket, missing_specific_mark_bracket |
| 33autumn25_q07 | table_preserving | +60 | expected_structural_requirement_missing:specific_mark_bracket, missing_mark_bracket, missing_specific_mark_bracket |
| 33autumn25_q07 | diagram_safe | +60 | expected_structural_requirement_missing:specific_mark_bracket, missing_mark_bracket, missing_specific_mark_bracket |
| 33autumn25_q07 | padding_variant | +60 | expected_structural_requirement_missing:specific_mark_bracket, missing_mark_bracket, missing_specific_mark_bracket |
| 33autumn25_q07 | dense_algebra | +60 | expected_structural_requirement_missing:specific_mark_bracket, missing_mark_bracket, missing_specific_mark_bracket |
| 13autumn25_q09 | grayscale_threshold | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 13autumn25_q09 | formula_heavy | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 13autumn25_q09 | diagram_safe | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 13autumn25_q09 | padding_variant | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 13autumn25_q09 | dense_algebra | +40 | expected_structural_requirement_missing:any_mark_bracket, missing_mark_bracket |
| 43autumn21_q06 | grayscale_threshold | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 43autumn21_q06 | formula_heavy | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 43autumn21_q06 | diagram_safe | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 43autumn21_q06 | padding_variant | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 43autumn21_q06 | dense_algebra | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 12spring22_q08 | grayscale_threshold | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 12spring22_q08 | formula_heavy | +45 | expected_structural_requirement_missing:question_number_start, likely_math_symbol_loss, question_number_not_at_start |
| 12spring22_q08 | diagram_safe | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 12spring22_q08 | padding_variant | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 12spring22_q08 | dense_algebra | +45 | expected_structural_requirement_missing:question_number_start, likely_math_symbol_loss, question_number_not_at_start |
| 52spring22_q06 | grayscale_threshold | +100 | expected_structural_requirement_missing:question_number_start, expected_structural_requirement_missing:subpart_order, expected_structural_requirement_missing:subpart_start, question_number_not_at_start, subpart_label_order_displaced |
| 52spring22_q06 | formula_heavy | +100 | expected_structural_requirement_missing:question_number_start, expected_structural_requirement_missing:subpart_order, expected_structural_requirement_missing:subpart_start, question_number_not_at_start, subpart_label_order_displaced |
| 52spring22_q06 | table_preserving | +100 | expected_structural_requirement_missing:question_number_start, expected_structural_requirement_missing:subpart_order, expected_structural_requirement_missing:subpart_start, question_number_not_at_start, subpart_label_order_displaced |
| 52spring22_q06 | diagram_safe | +100 | expected_structural_requirement_missing:question_number_start, expected_structural_requirement_missing:subpart_order, expected_structural_requirement_missing:subpart_start, question_number_not_at_start, subpart_label_order_displaced |
| 52spring22_q06 | padding_variant | +100 | expected_structural_requirement_missing:question_number_start, expected_structural_requirement_missing:subpart_order, expected_structural_requirement_missing:subpart_start, question_number_not_at_start, subpart_label_order_displaced |
| 52spring22_q06 | dense_algebra | +100 | expected_structural_requirement_missing:question_number_start, expected_structural_requirement_missing:subpart_order, expected_structural_requirement_missing:subpart_start, question_number_not_at_start, subpart_label_order_displaced |
| 41summer23_q06 | grayscale_threshold | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 41summer23_q06 | formula_heavy | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 41summer23_q06 | diagram_safe | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 41summer23_q06 | padding_variant | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 41summer23_q06 | dense_algebra | +40 | expected_structural_requirement_missing:question_number_start, question_number_not_at_start |
| 42autumn21_q01 | grayscale_threshold | +5 | likely_math_symbol_loss |
| 42autumn21_q01 | formula_heavy | +5 | likely_math_symbol_loss |
| 33summer21_q07 | grayscale_threshold | +5 | likely_math_symbol_loss |
| 33summer21_q07 | formula_heavy | +5 | likely_math_symbol_loss |
| 33summer21_q07 | padding_variant | +5 | likely_math_symbol_loss |
| 33summer21_q07 | dense_algebra | +5 | likely_math_symbol_loss |

## Regressions

| Record | Profile | Delta | Introduced issues |
| --- | --- | ---: | --- |
| 33autumn21_q04 | grayscale_threshold | -20 | expected_structural_requirement_missing:integral_sign |
| 33autumn21_q04 | formula_heavy | -20 | expected_structural_requirement_missing:integral_sign |
| 33autumn21_q04 | table_preserving | -20 | expected_structural_requirement_missing:integral_sign |
| 33autumn21_q04 | diagram_safe | -20 | expected_structural_requirement_missing:integral_sign |
| 33autumn21_q04 | padding_variant | -20 | expected_structural_requirement_missing:integral_sign |
| 33autumn21_q04 | dense_algebra | -20 | expected_structural_requirement_missing:integral_sign |
| 33autumn21_q05 | grayscale_threshold | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 33autumn21_q05 | formula_heavy | -25 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 33autumn21_q05 | table_preserving | -20 | expected_structural_requirement_missing:theta_symbol |
| 33autumn21_q05 | diagram_safe | -20 | expected_structural_requirement_missing:theta_symbol |
| 33autumn21_q05 | padding_variant | -20 | expected_structural_requirement_missing:theta_symbol |
| 33autumn21_q05 | dense_algebra | -20 | expected_structural_requirement_missing:theta_symbol |
| 31summer22_q09 | table_preserving | -35 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| 32spring23_q04 | grayscale_threshold | -20 | likely_math_symbol_loss |
| 32spring23_q04 | table_preserving | -20 | likely_math_symbol_loss |
| 32spring23_q04 | diagram_safe | -20 | likely_math_symbol_loss |
| 32spring23_q04 | padding_variant | -20 | likely_math_symbol_loss |
| 32spring23_q04 | dense_algebra | -5 | likely_math_symbol_loss |
| 32summer23_q09 | grayscale_threshold | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| 32summer23_q09 | formula_heavy | -5 | likely_math_symbol_loss |
| 32summer23_q09 | diagram_safe | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| 32summer23_q09 | padding_variant | -25 | likely_math_symbol_loss, subpart_label_order_displaced |
| 32summer23_q09 | dense_algebra | -20 | subpart_label_order_displaced |
| 11autumn23_q04 | grayscale_threshold | -15 | score-only regression |
| 11autumn23_q04 | formula_heavy | -15 | score-only regression |
| 11autumn23_q04 | diagram_safe | -15 | score-only regression |
| 11autumn23_q04 | padding_variant | -15 | score-only regression |
| 11autumn23_q04 | dense_algebra | -15 | score-only regression |
| 12autumn23_q06 | grayscale_threshold | -15 | score-only regression |
| 12autumn23_q06 | formula_heavy | -15 | score-only regression |
| 12autumn23_q06 | table_preserving | -15 | score-only regression |
| 12autumn23_q06 | diagram_safe | -15 | score-only regression |
| 12autumn23_q06 | padding_variant | -15 | score-only regression |
| 12autumn23_q06 | dense_algebra | -15 | score-only regression |
| 42autumn21_q01 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| 32summer21_q01 | grayscale_threshold | -20 | likely_math_symbol_loss |
| 32summer21_q01 | formula_heavy | -20 | likely_math_symbol_loss |
| 32summer21_q01 | table_preserving | -20 | likely_math_symbol_loss |
| 32summer21_q01 | diagram_safe | -20 | likely_math_symbol_loss |
| 32summer21_q01 | padding_variant | -20 | likely_math_symbol_loss |
| 32summer21_q01 | dense_algebra | -20 | likely_math_symbol_loss |
| 32spring23_q03 | grayscale_threshold | -20 | likely_math_symbol_loss |
| 32spring23_q03 | formula_heavy | -20 | likely_math_symbol_loss |
| 32spring23_q03 | table_preserving | -20 | likely_math_symbol_loss |
| 32spring23_q03 | diagram_safe | -20 | likely_math_symbol_loss |
| 32spring23_q03 | padding_variant | -20 | likely_math_symbol_loss |
| 32spring23_q03 | dense_algebra | -20 | likely_math_symbol_loss |
| 11summer23_q01 | grayscale_threshold | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 11summer23_q01 | formula_heavy | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 11summer23_q01 | table_preserving | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 11summer23_q01 | diagram_safe | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 11summer23_q01 | padding_variant | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 11summer23_q01 | dense_algebra | -40 | expected_structural_requirement_missing:theta_symbol, likely_math_symbol_loss |
| 12spring21_q02 | grayscale_threshold | -5 | likely_math_symbol_loss |
| 12spring21_q02 | formula_heavy | -5 | likely_math_symbol_loss |
| 12spring21_q02 | diagram_safe | -5 | likely_math_symbol_loss |
| 12spring21_q02 | padding_variant | -5 | likely_math_symbol_loss |
| 33summer21_q07 | table_preserving | -40 | expected_structural_requirement_missing:question_number_present, missing_question_number |
| 31summer23_q09 | grayscale_threshold | -20 | likely_math_symbol_loss |
| 31summer23_q09 | formula_heavy | -20 | likely_math_symbol_loss |
| 31summer23_q09 | table_preserving | -20 | likely_math_symbol_loss |
| 31summer23_q09 | diagram_safe | -20 | likely_math_symbol_loss |
| 31summer23_q09 | padding_variant | -20 | likely_math_symbol_loss |
| 31summer23_q09 | dense_algebra | -20 | likely_math_symbol_loss |
