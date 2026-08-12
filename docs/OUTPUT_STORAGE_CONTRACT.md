# Output Storage Contract

The exam-bank pipeline is image-first. Canonical rendered question images and mark-scheme images are the source of truth. Native text, OCR text, topic labels, difficulty labels, advisory sidecars, reports, Asterion exports, Content Lab candidates, and review packets are metadata over those images.

## Canonical Asset Locations

Canonical image assets live under the flat subject-family output tree:

- `output/pm1/*.png`
- `output/pm3/*.png`
- `output/stats/*.png`
- `output/mechanics/*.png`

These paths are stable relative asset references. JSON exports should store these relative paths, not absolute local paths.

Older nested paths such as `output/p1/<paper>/questions/q01.png` and `output/p1/<paper>/mark_scheme/q01.png` are legacy compatibility inputs for normalization. They are not the active canonical output contract.

`output/json/question_bank.json` is the canonical metadata index for the current run. `output/json/asset_manifest.v1.json` is an index over canonical image files. The manifest is not a replacement source of truth; it records asset IDs, paths, SHA-256 hashes, sizes, and image dimensions for validation and lookup.

`output/json/question_bank.topic_routing.v1.json` is an ignored generated/local cache. The durable reviewed source is `data/topic_routing/question_bank.topic_routing.v1.json`. `manifests/releases/question_bank_release_manifest.v1.json` binds that sidecar to the exact canonical question bank by SHA-256, size, schema, count, and question-ID set; the adjacent `.sha256` file remains a compatibility guard. A downstream handoff should rebuild the same manifest as a multi-role bundle containing every consumed sidecar and projection, following [Question-Bank Release Manifest Contract](RELEASE_MANIFEST_CONTRACT.md). Verify the release manifest before Asterion export regeneration, and restore the local cache only for compatibility:

```bash
exam-bank topic verify-release
exam-bank topic restore-release
```

## Generated And Rebuildable Locations

The following locations are generated outputs, caches, review aids, or historical evidence:

- `output/asterion/exports/latest/*.json`
- `output/releases/`
- `output/topic_packets/`
- `output/candidates/ocr/`
- `output/codex_text_extraction_candidate*/`
- `output/codex_text_extraction_targeted/`
- `output/audits/`
- `output/run_status/`
- `output/archive/`
- `reports/`

Do not promote files from these locations to canonical evidence without regenerating or validating against canonical images.

`output/` is the only supported generated-output root. A plural `outputs/`
folder is treated as legacy scratch space, is ignored by Git, and should be
migrated into a run-specific folder below `output/` or removed after its
artifacts are verified as rebuildable.

### Run checkpoint retention

Run checkpoints can contain a full `questions.json` payload for every paper, so
they need bounded retention. The dry-run cleanup planner classifies them using
these conservative rules:

- keep the latest completed checkpoint;
- keep interrupted, failed, pending, or running checkpoints until they are
  resumed or explicitly resolved;
- keep any checkpoint containing a `.keep` or `PINNED` marker;
- classify older completed checkpoints for archive review.

The planner does not delete or move checkpoints. Review the plan, confirm that
no release manifest or active recovery command refers to a checkpoint, and
then archive or remove it through the normal quarantine-first cleanup process.

Asterion export release handoff is represented by a tracked manifest under `reports/`, not by committing the large generated JSON files. The manifest records the exact ignored export paths, SHA-256 values, byte sizes, validation status, and durable topic-sidecar provenance. Deployment or Asterion handoff must consume export files matching the manifest hashes.

## App Export And Reference Policy

Downstream JSON should reference canonical assets with one or both of:

- canonical relative paths such as `pm1/pm1_2021_s21_12_qp_q01_question.png`
- stable asset IDs from `output/json/asset_manifest.v1.json`

Asterion exports preserve path fields for runtime compatibility, but those fields should point at canonical relative paths. New consumers should prefer `canonical_question_asset_id`, `canonical_mark_scheme_asset_id`, and subpart/source artifact `*_asset_id` fields when available, resolving them through the manifest.

The Asterion all-course catalog (`asterion_exam_bank_catalog_v1.json`) carries course-aware fields for the static 9709 site: `course_id`, `component_name`, `topic_id`, `topic_route`, `question_image_path`, `mark_scheme_image_path`, `catalog_visible`, `image_practice_safe`, `advisory_topic_filter_ok`, `reviewed_topic_filter_safe`, `learning_runtime_safe`, `student_runtime_safe`, and `review_status`. Supported course IDs are `p1`, `p3`, `m1`, `s1`, and `s2`. Paper 4 maps to canonical family `mechanics` and course `m1`. Under the current syllabus Papers 5 and 6 share canonical storage family `stats` but split into `s1` and `s2`; before 2020 Paper 6 is S1 and legacy Paper 5 (M2) is unsupported. These fields do not change the canonical image policy. A student-visible learning-runtime page should load the reviewed/safe runtime export (`asterion_question_bank_v1.json`), resolve and display the canonical image references, and show an empty reviewed-record state when no `learning_runtime_safe=true` records exist for a course. Non-P3 course records may be available for catalog or image-practice use in the catalog, but they are not learning-runtime records until reviewed topic alignment exists.

## Copying And Embedding Policy

Copying canonical images into downstream folders is allowed only for portable bundles that cannot resolve repository-relative paths. The copy must be exact, rebuildable, and documented by the bundle manifest.

Embedding is expected for standalone PDFs such as topic packets. A PDF is a portable artifact and may physically contain rendered images. The PDF must keep a manifest with source image paths so it can be regenerated from canonical assets.

HTML review packets and JSON sidecars should reference canonical paths instead of copying images.

## Deletion And Quarantine Policy

Never delete canonical question or mark-scheme images as part of an automated cleanup.

Exact duplicates must be proven by SHA-256, not by filename. A non-canonical duplicate may be removed only after references are remapped or proven absent. Automated cleanup must default to dry-run. Quarantine apply mode must move candidates to a quarantine directory such as `output/_quarantine_storage_cleanup`, not permanently delete them. Hard-delete mode is an explicit exception that must write `reports/output_storage_delete_manifest.v1.json` before deleting, and may only delete allowlisted non-canonical exact duplicates.

Archive folders are manual-review or quarantine-only unless a report explicitly proves each file is an exact duplicate and no live JSON export references it.

## Asset ID And Path Policy

Asset IDs are stable lookup keys derived from asset kind, paper, and question ID, for example:

- `question_image:12spring21:12spring21_q01`
- `mark_scheme_image:12spring21:12spring21_q01`

Paths remain stable relative paths under the output artifact root. Absolute paths should not appear in committed or exported metadata.

## Validation Expectations

Before deleting, quarantining, or publishing downstream exports, run:

```bash
.venv/bin/exam-bank data inventory --root output --include-size
.venv/bin/exam-bank data cleanup-plan --root output --include-size
.venv/bin/python -m exam_bank.command data build-asset-manifest
.venv/bin/python -m exam_bank.command data audit-storage --dry-run
.venv/bin/python -m exam_bank.command data audit-storage --apply
.venv/bin/python -m exam_bank.command data validate-assets --strict-companion-inputs
.venv/bin/exam-bank extract integrity --input output/json/question_bank.json --artifact-root output
.venv/bin/exam-bank topic verify-release
.venv/bin/python -m pytest -q tests/test_asset_manifest_storage_audit.py tests/test_asterion_export.py tests/test_topic_routing.py
```

`--apply` moves only allowlisted exact duplicates into a recoverable quarantine.
Reserve `--apply-delete` for a separately reviewed deletion manifest.

Validation must confirm:

- all image paths referenced by `question_bank.json` exist
- all image paths referenced by the Asterion catalog, student-runtime export, and Content Lab candidates exist
- all export asset IDs resolve through `output/json/asset_manifest.v1.json`
- no copied image appears in non-canonical export folders unless explicitly allowlisted
- canonical image files remain present
- topic-routing and Content Lab sidecars still have valid counts and schema names
- the release manifest binds the exact question bank and durable topic-routing sidecar hashes, counts, and question-ID sets
- course-aware Asterion filters do not expose Content Lab candidates or invalid course IDs to student runtime
- Asterion release manifests match the ignored export artifact hashes before handoff
