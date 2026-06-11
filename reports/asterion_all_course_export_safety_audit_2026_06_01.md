# Asterion All-Course Export Safety Audit

Audit date: 2026-06-01

## Scope

This was an adversarial audit of the all-course Asterion export gates after the P1, M1, and S1 safe-count iteration. The audit asked whether the high P1/M1/S1 counts were genuinely safe for student learning runtime, or only safe for catalog/image practice.

## Files Inspected

- `src/exam_bank/asterion_export.py`
- `src/exam_bank/asterion_course_contract.py`
- `src/exam_bank/cli.py`
- `scripts/validate_asterion_all_course_export.py`
- `tests/test_asterion_export.py`
- `tests/test_asterion_course_contract.py`
- `docs/ASTERION_EXPORT_CONTRACT.md`
- `docs/AI_ASSISTED_ENRICHMENT.md`
- `docs/OUTPUT_STORAGE_CONTRACT.md`
- `docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`
- `docs/COMMAND_ATLAS.md`
- `exam_bank_taxonomy/canonical/topic_filter_maps/topic_filter_map_9709_p1_v1.json`
- `exam_bank_taxonomy/canonical/topic_filter_maps/topic_filter_map_9709_m1_v1.json`
- `exam_bank_taxonomy/canonical/topic_filter_maps/topic_filter_map_9709_s1_v1.json`
- `exam_bank_taxonomy/canonical/question_topic_assignments/question_topic_assignments_9709_p1_v1.json`
- `exam_bank_taxonomy/canonical/question_topic_assignments/question_topic_assignments_9709_m1_v1.json`
- `exam_bank_taxonomy/canonical/question_topic_assignments/question_topic_assignments_9709_s1_v1.json`
- `exam_bank_taxonomy/canonical/strict_filtering_reports/strict_topic_filtering_report_9709_p1_v1.json`
- `exam_bank_taxonomy/canonical/strict_filtering_reports/strict_topic_filtering_report_9709_m1_v1.json`
- `exam_bank_taxonomy/canonical/strict_filtering_reports/strict_topic_filtering_report_9709_s1_v1.json`

## Export Files Inspected

- `output/json/question_bank.json`
- `output/json/question_bank.topic_routing.v1.json`
- `output/json/question_bank.mark_events.v1.json`
- `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json`
- `output/asterion/exports/latest/asterion_question_bank_v1.json`
- `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- `output/asterion/exports/latest/asterion_student_runtime_safe_decisions_v1.json`
- `output/asterion/exports/latest/asterion_student_runtime_safe_candidates_v1.json`
- `output/asterion/exports/latest/asterion_all_course_export_validation_2026_06_01.json`

## Blocker Found

The pre-audit implementation collapsed image/advisory readiness into `student_runtime_safe`.

Specifically:

- `quality_gate.student_runtime_image_ok=true` caused `usage_roles.canonical_practice=allow`.
- `student_runtime_safe_for_record()` treated `canonical_practice=allow` as student-runtime safe.
- `review_status_for_record()` then projected those records to `reviewed`.
- `build_asterion_student_question_bank()` loaded them into `asterion_question_bank_v1.json`.

That meant mechanically clean advisory topic routes plus canonical images were enough to put P1/M1/S1 records into the student runtime export. That was too optimistic for learning runtime.

## Patch Applied

The export now separates safety levels:

- `catalog_visible`
- `image_practice_safe`
- `advisory_topic_filter_ok`
- `reviewed_topic_filter_safe`
- `learning_runtime_safe`

`student_runtime_safe` now follows `learning_runtime_safe`. P3 legacy runtime is preserved. P1/M1/S1 records can still be catalog-visible and image/advisory safe, but they no longer enter `asterion_question_bank_v1.json` until reviewed topic alignment exists.

The validator now reports P1/M1/S1 runtime targets as report-only by default and records the safety-level counts. It no longer blesses advisory topic coverage as reviewed learning-runtime coverage.

## Counts After Patch

| course | catalog records | image-practice-safe | advisory-topic-filter-ok | reviewed-topic-filter-safe | learning-runtime-safe | runtime export | explicit runtime-safe decision qids |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1 | 401 | 282 | 256 | 0 | 0 | 0 | 0 |
| p3 | 396 | 293 | 281 | 0 | 281 | 281 | 313 |
| m1 | 258 | 173 | 148 | 0 | 0 | 0 | 0 |
| s1 | 246 | 169 | 137 | 0 | 0 | 0 | 0 |

## Examples: P1 Records That Were Previously Safe-Looking

| id | paper | q | topic_id | image_practice_safe | advisory_topic_filter_ok | learning_runtime_safe | q_crop_ok | ms_crop_ok | route_conf | route_review_required | explicit_decision | evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| `12spring21_q01` | `12spring21` | 1 | `9709_p1_topic_series` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `12spring21_q02` | `12spring21` | 2 | `9709_p1_topic_quadratics` | true | true | false | false | true | medium | false | false | image existence and deterministic image/mark gates; advisory route |
| `12spring21_q03` | `12spring21` | 3 | `9709_p1_topic_trigonometry` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `12spring21_q04` | `12spring21` | 4 | `9709_p1_topic_quadratics` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `12spring21_q05` | `12spring21` | 5 | `9709_p1_topic_functions` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |

## Examples: M1 Records That Were Previously Safe-Looking

| id | paper | q | topic_id | image_practice_safe | advisory_topic_filter_ok | learning_runtime_safe | q_crop_ok | ms_crop_ok | route_conf | route_review_required | explicit_decision | evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| `42spring21_q01` | `42spring21` | 1 | `9709_m1_topic_momentum` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `42spring21_q02` | `42spring21` | 2 | `9709_m1_topic_newtons_laws_of_motion` | true | true | false | false | true | medium | false | false | image existence and deterministic image/mark gates; advisory route |
| `42spring21_q03` | `42spring21` | 3 | `9709_m1_topic_forces_and_equilibrium` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `42spring21_q04` | `42spring21` | 4 | `9709_m1_topic_kinematics_of_motion_in_a_straight_line` | true | true | false | false | false | medium | false | false | image existence and deterministic image/mark gates; advisory route |
| `42spring21_q05` | `42spring21` | 5 | `9709_m1_topic_newtons_laws_of_motion` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |

## Examples: S1 Records That Were Previously Safe-Looking

| id | paper | q | topic_id | image_practice_safe | advisory_topic_filter_ok | learning_runtime_safe | q_crop_ok | ms_crop_ok | route_conf | route_review_required | explicit_decision | evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| `52spring21_q02` | `52spring21` | 2 | `9709_s1_topic_probability` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `52spring21_q03` | `52spring21` | 3 | `9709_s1_topic_the_normal_distribution` | true | true | false | false | false | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `52spring21_q04` | `52spring21` | 4 | `9709_s1_topic_discrete_random_variables` | true | true | false | false | true | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `52spring21_q05` | `52spring21` | 5 | `9709_s1_topic_representation_of_data` | true | true | false | false | false | high | false | false | image existence and deterministic image/mark gates; advisory route |
| `52spring21_q06` | `52spring21` | 6 | `9709_s1_topic_permutations_and_combinations` | true | true | false | false | false | high | false | false | image existence and deterministic image/mark gates; advisory route |

## Audit Question Answers

1. Before the patch, yes: `student_runtime_image_ok` caused `student_runtime_safe=true`. After the patch, no for P1/M1/S1; it feeds `image_practice_safe` and, with route evidence, `advisory_topic_filter_ok`.
2. Before the patch, yes: `topic_route.filter_ok=true` could help put a record into `asterion_question_bank_v1.json`. After the patch, not for P1/M1/S1.
3. Before the patch, yes: P1/M1/S1 could be projected to `reviewed` without explicit reviewed decisions. After the patch, those records are `needs_review`.
4. P1/M1/S1 had inherited the all-course canonical-practice projection. After the patch, P3 legacy runtime is preserved and non-P3 records are separated.
5. Before the patch, yes: the 35% validator target blessed advisory coverage as runtime coverage. After the patch it is report-only unless explicitly enforced.
6. Records with weak/missing/ambiguous topic routes do not get `advisory_topic_filter_ok`; missing-topic records are reported separately.
7. After the patch, yes: valid-image records with unreviewed topics are `image_practice_safe` and/or `advisory_topic_filter_ok`, not `learning_runtime_safe`.
8. After the patch, yes: Asterion can inspect the five separate safety booleans and the `safety_levels` object.
9. After the patch, `asterion_question_bank_v1.json` contains only P3 records: 281. P1/M1/S1 runtime counts are 0.
10. P1/M1/S1 topic IDs are backed by canonical topic-map files whose `review_status` is `needs_review`, and the active AI routing sidecar reports `safe_for_strict_filters=false`. They are not reviewed learning-runtime topic alignments.

## Risks And Blockers

- Blocker resolved in code: non-P3 advisory/image records no longer enter `student_runtime_safe`.
- Remaining risk: `advisory_topic_filter_ok` may still be mistaken for reviewed topic safety by downstream consumers. Asterion must not use it for mastery, recommendation, generated checks, Field Guide links, Guardian gates, or skill flows.
- Remaining risk: canonical P1/M1/S1 topic maps exist, but their review status is `needs_review`; they are official-section-shaped but not sufficient for reviewed learning runtime.
- Remaining risk: P3 legacy runtime is preserved, including records without explicit `asterion_student_runtime_safe_decisions_v1.json` decisions. This is intentional compatibility, but should remain documented as P3-specific legacy behavior.

## Recommendations

1. Asterion should consume `asterion_question_bank_v1.json` only for current learning runtime. It now contains P3 only.
2. Asterion may consume `asterion_exam_bank_catalog_v1.json` for P1/M1/S1 catalog display and broad image practice using `catalog_visible=true` and `image_practice_safe=true`.
3. Asterion may use `advisory_topic_filter_ok=true` only for clearly labeled advisory topic browsing or internal QA, not student learning runtime.
4. Do not treat `topic_route.filter_ok` or `advisory_topic_filter_ok` as reviewed topic safety.
5. Add reviewed P1/M1/S1 topic-alignment decisions before enabling `reviewed_topic_filter_safe` or `learning_runtime_safe` for those courses.
6. Keep `safe_for_strict_filters=false` on the raw topic-routing sidecar as a fail-closed signal for direct sidecar consumers.

## Validation

- Regenerated Asterion catalog/runtime with `--topic-routing output/json/question_bank.topic_routing.v1.json`.
- Regenerated Content Lab candidates with mark events and topic routing.
- `PYTHONPATH=src:. .venv/bin/python scripts/validate_asterion_all_course_export.py` -> `ok: true`, with warnings that P1/M1/S1 learning-runtime targets are report-only and some records lack topic IDs.
- Focused tests: `PYTHONPATH=src:. .venv/bin/pytest tests/test_asterion_export.py tests/test_asterion_course_contract.py tests/test_script_path_defaults.py` -> 37 passed.
- Full suite: `PYTHONPATH=src:. .venv/bin/pytest` -> 770 passed, 3 skipped, 5 warnings.
- `git diff --check` -> clean.
