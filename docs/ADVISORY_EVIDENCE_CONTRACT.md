# Advisory Evidence Contract

This contract covers examiner-report and grade-threshold evidence generated under `output/advisory_evidence/`.

The sidecar is advisory evidence only. Canonical question images and mark-scheme images remain the source of truth. Advisory evidence must not overwrite `output/json/question_bank.json`, topic routing sidecars, Asterion exports, taxonomy files, canonical images, or existing AI sidecars.

## Build Order

Run the advisory pipeline in this order:

```text
inventory -> extract native text -> parse -> link -> signals -> validate -> reports -> final sidecar
```

Commands:

```bash
.venv/bin/python scripts/build_advisory_inventory.py
.venv/bin/python scripts/extract_advisory_text.py
.venv/bin/python scripts/parse_examiner_reports.py
.venv/bin/python scripts/parse_grade_thresholds.py
.venv/bin/python scripts/link_advisory_evidence.py
.venv/bin/python scripts/build_advisory_topic_evidence.py
.venv/bin/python scripts/build_examiner_report_difficulty.py
.venv/bin/python scripts/build_grade_threshold_context.py
.venv/bin/python scripts/validate_advisory_evidence.py
.venv/bin/python scripts/build_advisory_review_reports.py
.venv/bin/python scripts/build_advisory_evidence_sidecar.py
```

Every command that writes stage output also supports `--dry-run`, except validation, which reads existing outputs and exits non-zero when validation fails.

## Output Contract

Stage outputs:

- `output/advisory_evidence/inventory/examiner_report_inventory.json`
- `output/advisory_evidence/inventory/grade_threshold_inventory.json`
- `output/advisory_evidence/extracted_text/examiner_reports/*.json`
- `output/advisory_evidence/extracted_text/grade_thresholds/*.json`
- `output/advisory_evidence/parsed/examiner_reports/*.json`
- `output/advisory_evidence/parsed/grade_thresholds/*.json`
- `output/advisory_evidence/linking/examiner_report_question_links.json`
- `output/advisory_evidence/linking/grade_threshold_component_links.json`
- `output/advisory_evidence/predictions/advisory_topic_evidence.v1.json`
- `output/advisory_evidence/predictions/advisory_examiner_report_difficulty.v1.json`
- `output/advisory_evidence/predictions/advisory_grade_threshold_context.v1.json`
- `output/advisory_evidence/reports/*.md`

Final aggregate sidecar:

- `output/advisory_evidence/question_bank.advisory_evidence.v1.json`

Interim advisory files must not be written under `output/json/`.

## Identity Rules

Internal identity follows existing repo metadata:

- `session`: `March`, `MayJune`, or `November`
- `session_key`: for example `9709_2025_MayJune`
- `session_slug`: for readable filenames only, for example `june_2025`

Duplicate document identities are allowed only when they are explicit warnings in inventory output. Distinct source files use stable path-hashed output filenames.

Duplicate advisory evidence records are allowed only with an explicit validation warning. The final sidecar remains one record per `question_id`, so duplicate linked candidates, topic evidence, or difficulty evidence must be visible in validation and review reports before any downstream review-only use.

## Evidence Boundaries

Examiner reports may support topic hints, misconception/method evidence, common-error patterns, omission/no-response signals, item-level difficulty hints, and review prioritization.

Grade thresholds may support component/session context, relative paper difficulty context, coarse calibration, and review prioritization.

Grade thresholds must not directly prove individual-question difficulty. The validator fails advisory difficulty records that assign item difficulty from threshold-only evidence.

No advisory evidence should be used for strict topic filtering, mastery decisions, student-facing question text, or replacement mark schemes.

## Validation Rules

`scripts/validate_advisory_evidence.py` checks:

- advisory schemas are present and expected
- link statuses use the allowed enum
- linked candidate question IDs exist in `question_bank.json`
- predicted topic IDs are allowed existing topic IDs
- confidence, evidence, difficulty, and context labels use allowed enums
- grade-threshold context does not contain item-level difficulty
- duplicate linked candidates and duplicate advisory evidence records emit explicit warnings
- final sidecar records link to real question-bank records
- final sidecar records do not contain canonical replacement fields

Implementation passes must also run `git diff --check` and a protected-path no-mutation check for canonical outputs, exports, taxonomy files, canonical image folders, and existing AI sidecars.

## AI Policy

AI enrichment is intentionally outside this phase. Add a separate audited plan before using AI on unresolved advisory cases.
