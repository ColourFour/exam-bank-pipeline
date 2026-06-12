# Asterion Export Packaging PR 16

- Date: `2026-06-11`
- Release manifest: `reports/asterion_export_release_manifest_pr16_2026_06_11.json`
- Packaging command: `scripts/package_asterion_export_release.py`

## Existing Artifact Convention

The repository keeps generated output under ignored `output/` roots.

- `.gitignore` ignores `output/*`.
- `output/asterion/exports/latest/asterion_question_bank_v1.json` is ignored by `.gitignore:16:output/*`.
- `git ls-files -- output/asterion/exports/latest` prints nothing.
- `git status --short --ignored output/asterion/exports/latest` reports `!! output/asterion/`.

Tracked release evidence already lives under `reports/`, including PR 15 provenance reports. There is no existing convention that commits the large Asterion export JSON files.

## Chosen Packaging Approach

Keep `output/asterion/exports/latest/*.json` ignored/generated. Add a deterministic packaging command that verifies the durable topic-routing sidecar provenance, verifies the export artifact hashes against PR 15 expected provenance, records file sizes and counts, and writes a small tracked release manifest.

No large export bundle was created in this PR. The manifest is the handoff contract; deployment or Asterion must receive export files whose hashes match it.

## Files Included

| Role | Path | SHA-256 | Size bytes |
| --- | --- | --- | ---: |
| Durable topic sidecar | `data/topic_routing/question_bank.topic_routing.v1.json` | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` | n/a |
| Local topic sidecar | `output/json/question_bank.topic_routing.v1.json` | `e73559581b9cd5970d38496b1f6b334050a17789cc25f082eb5ccb94b1142e4e` | n/a |
| Asterion catalog | `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json` | `9ae6882ea8a35da7ccd09f514fb59ab3b1e0fe551e8539e3efe230bd23b71236` | 41293182 |
| Asterion student runtime | `output/asterion/exports/latest/asterion_question_bank_v1.json` | `e20bcb7649044194dcd3eb1988eaebfef35d39a8e25bd5f88584af25d9ebdabb` | 8810119 |
| Content Lab candidates | `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json` | `085f17a49e318a622c7b1cb5f3fc864146d2905f40a7e8225d77260f9f3a4a56` | 11943716 |
| Validator report | `/tmp/asterion_export_release_provenance_pr15_validation.json` | `9e3ffad9755bd487b08a902e0ff3aeae2c6575e56c77741432c222b479ddce38` | 10879 |

## Export Counts

| Metric | Count |
| --- | ---: |
| Catalog records | 1301 |
| Student runtime records | 279 |
| P3 runtime records | 279 |
| Non-P3 runtime records | 0 |
| Content Lab candidates | 2432 |
| Advisory topic-filter OK | 887 |
| Learning-runtime safe | 279 |

No P1, M1, or S1 material became student-facing.

## Validator Status

`scripts/validate_asterion_all_course_export.py` passed with `ok: true`.

Validator warnings remain report-only:

- Catalog has P1/M1/S1 records without `topic_id`; see `student_runtime_missing_topic_counts`.
- P1/M1/S1 learning-runtime targets are report-only; non-P3 image/advisory records are not reviewed learning runtime.

## Packaging Implementation

Added `src/exam_bank/asterion_release_bundle.py` and `scripts/package_asterion_export_release.py`.

The command:

- verifies durable topic sidecar provenance
- verifies local `output/json` topic sidecar matches the durable artifact
- verifies all three export JSON files exist
- computes SHA-256 and file sizes
- reads validator status and counts
- fails on SHA mismatch against expected PR 15 provenance unless `--refresh-expected` is supplied
- writes `reports/asterion_export_release_manifest_pr16_2026_06_11.json`

## Docs Updated

- `README.md`
- `docs/OUTPUT_STORAGE_CONTRACT.md`
- `docs/COMMAND_ATLAS.md`
- `docs/ASTERION_EXPORT_CONTRACT.md`

The documented workflow is:

1. restore/verify the durable sidecar
2. regenerate exports
3. validate exports
4. package the release manifest
5. hand deployment/Asterion the generated export files matching the manifest hashes

## Scope Confirmation

Asterion runtime behavior did not change. Auto-grade eligibility did not change. Runtime promotion remains a separate reviewed decision, and packaging does not promote records.

No provider calls were run. Topic routing was not rerouted. Topic-routing behavior, prompt text/version, taxonomy, reviewed decisions, Asterion app/runtime behavior, and auto-grade eligibility were not changed.

## Recommended Next PR

Define the deployment-side retrieval step for ignored export artifacts: either upload the three validated JSON files to an approved release storage location, or teach deployment to require files whose SHA-256 values match `reports/asterion_export_release_manifest_pr16_2026_06_11.json`.
