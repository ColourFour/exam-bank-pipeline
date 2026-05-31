# Architecture

This repository is an image-first CAIE 9709 exam-bank pipeline. Question crops and mark-scheme crops are canonical. JSON records, OCR/native text, topic routing, difficulty labels, mark events, Asterion projections, and Content Lab candidates are metadata over those images.

## Data Flow

1. Source PDFs under `input/` are processed into `output/json/question_bank.json` plus canonical images under `output/p*/<paper>/questions/` and `output/p*/<paper>/mark_scheme/`.
2. Sidecars add advisory evidence without mutating the canonical bank: topic routing, mark events, advisory examiner-report/grade-threshold links, difficulty indexes, and AI-assisted review/debug outputs.
3. `src/exam_bank/asterion_export.py` builds `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json`, the broad all-course static-site catalog with canonical image references, asset IDs, integrity metadata, quality gates, subparts, role gates, course fields, and blocked/review states preserved.
4. The same export path writes `output/asterion/exports/latest/asterion_question_bank_v1.json`, the reviewed/safe student-runtime subset derived from the catalog.
5. `src/exam_bank/asterion_export.py` also builds `asterion_content_lab_candidates_v1.json`, which is review-only candidate metadata. It is not student runtime.
6. `src/exam_bank/asterion_student_runtime_safe.py` audits P3 Content Lab candidates for an explicit student-runtime-safe candidate export. This remains P3-specific review infrastructure.

## Course Contract

The static 9709 study site should use the course-aware contract in `src/exam_bank/asterion_course_contract.py`.

Supported course IDs:

- `p1`: Pure Mathematics 1, sourced from paper family `p1`
- `p3`: Pure Mathematics 3, sourced from paper family `p3`
- `m1`: Mechanics 1, sourced from paper family `p4`
- `s1`: Probability & Statistics 1, sourced from paper family `p5`

The Asterion all-course catalog adds these fields to each record:

- `course_id`
- `component_name`
- `topic_id`, when already available
- `source_exam`
- `question_image_path`
- `mark_scheme_image_path`
- `student_runtime_safe`
- `review_status`

The helper layer filters by `course_id`, paper, and component name; returns empty arrays for scaffolded courses; and fails closed for invalid course IDs. P3 legacy runtime behavior is preserved by treating `usage_roles.canonical_practice=allow` as `student_runtime_safe=true` for P3 records. P1, M1, and S1 require explicit reviewed/safe promotion before they can enter student runtime.

`asterion_exam_bank_catalog_v1.json` may include reviewed, needs-review, blocked, and candidate-state records. Its metadata includes course and component counts for P1, P3, M1, and S1. `asterion_question_bank_v1.json` is the student-facing subset and should contain only reviewed/safe records.

## Student Runtime Boundary

Student-facing exam-bank pages should load only records with:

- `student_runtime_safe=true`
- `review_status=reviewed`
- valid `course_id` in `p1 | p3 | m1 | s1`
- canonical question and mark-scheme image paths where the image-first policy requires them

If P1, M1, or S1 has no reviewed records, the static site should show:

```text
No reviewed exam-bank records available yet.
```

Do not reuse P3 records for P1, M1, or S1 empty states.

## Advisory Boundaries

Question and mark-scheme images remain canonical. OCR text, native text, AI labels, topic-routing labels, difficulty labels, mark-event candidates, and broad skill mappings are advisory unless a separate reviewed contract explicitly promotes them.

Content Lab candidates are review material only. They may contain useful blocker diagnostics and source-artifact references, but they must not be loaded as student runtime records and must not create generated student-facing practice content.

## P3-Specific Assumptions

The current P3 runtime/review path still has P3-only region names, P3 exact-skill IDs, P3 reviewed-decision inputs, and P3 Content Lab audit outputs. Those assumptions are intentionally contained in P3 review modules. Future P1/M1/S1 work should add official topic maps first, then map existing exam-bank records where image-backed evidence exists.
