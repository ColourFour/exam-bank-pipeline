# Archive Manifest: generated_cleanup_20260513T233456Z

Created: 2026-05-14

Archive path: `output/archive/generated_cleanup_20260513T233456Z`

Purpose: pre-deletion manifest for a generated-output archive identified by `docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md` as historical evidence of prior OCR, AI enrichment, topic-routing, text-confidence, and Asterion export runs.

No archived files were deleted, moved, or regenerated while creating this manifest.

## Inventory Summary

Approximate total size: 471M

File count: 2,662 files

Extension inventory:

| Extension | Count |
| --- | ---: |
| `.png` | 2,612 |
| `.json` | 45 |
| `.jsonl` | 5 |

Top-level inventory:

| Path | Approx. size | File count / contents | Classification | Appears reproducible? | Needed to explain previous OCR/AI/text-confidence runs? | Notes |
| --- | ---: | --- | --- | --- | --- | --- |
| `output/` | 29M | Archived JSON sidecars and batch caches under `output/json` | historical evidence | Partly. Deterministic wrapper files may be rerunnable, but AI/API results are not exactly reproducible. | Yes | Built-in cleanup plan reports this tree as `unknown/manual review`. Treat all files here as historical, not current, unless later explicitly promoted. |
| `output/json/` | 29M | 47 files grouped under 23 top-level entries, including AI-assisted sidecars, DeepSeek sidecar, topic-routing smoke sidecars, audit/status snapshots, failures, and batch caches | historical evidence | Partly. Run shells are reproducible; model responses and failure states are not exactly reproducible. | Yes | Explains prompt-version, validation, failure, and smoke/full-run behavior from 2026-05-13. |
| `output_ocr_candidate/` | 443M | OCR candidate question bank/export JSON plus image artifact trees | keep-until-reviewed | Mostly. Generated from input PDFs and OCR pipeline, but exact output depends on runtime/config/tool versions. | Yes | Built-in cleanup plan reports this as `keep: latest candidate`. Preserve until a reviewer confirms current output supersedes it. |
| `output_ocr_candidate/json/` | 38M | OCR candidate question bank and Asterion projections | historical evidence | Mostly. Downstream exports are rerunnable from the same candidate bank and sidecars; exact candidate bank depends on OCR/runtime state. | Yes | These JSONs are archived evidence only, not current exports. |
| `output_ocr_candidate/p1/` | 103M | 802 PNG image artifacts | duplicated generated asset | Mostly | Yes | Most files duplicate current `output/p1` artifact paths and checksums. |
| `output_ocr_candidate/p3/` | 104M | 802 PNG image artifacts | duplicated generated asset, with keep-until-reviewed exceptions | Mostly | Yes | 781 of 802 files duplicate current output by path and checksum. 21 archive-only files under `p3/33autumn25` and `p3/34autumn25` need review before deletion. |
| `output_ocr_candidate/p4/` | 97M | 516 PNG image artifacts | duplicated generated asset | Mostly | Yes | Current output contains matching paths and checksums for these artifacts. |
| `output_ocr_candidate/p5/` | 100M | 492 PNG image artifacts | duplicated generated asset | Mostly | Yes | Current output contains matching paths and checksums for these artifacts. |

No `baseline` files or folders were found. In particular, the inventory command found no frozen triage baselines under this archive.

## JSON Sidecar Classification

| Path or group | Approx. size | Classification | Appears reproducible? | Needed to explain previous OCR/AI/text-confidence runs? | Notes |
| --- | ---: | --- | --- | --- | --- |
| `output/json/audit.current.json` | 8K | historical evidence | Yes, if same input sidecars are available | Yes | Snapshot audit evidence; archived, not current. |
| `output/json/status.current.json` | 8K | unknown | Unknown | Possibly | Small status snapshot without manifest metadata in the quick inspection. Keep until reviewed. |
| `output/json/question_bank.deepseek.json` | 2.8M | historical evidence | Not exactly; model/API dependent | Yes | Older DeepSeek sidecar. |
| `output/json/question_bank.deepseek.failures.jsonl` | 64K, 11 lines | disposable run evidence | Not exactly | Yes | Failure evidence should be kept until the failed run is understood or documented elsewhere. |
| `output/json/question_bank.ai_assisted.v2.json` | 5.2M | historical evidence | Not exactly; model/API dependent | Yes | AI-assisted sidecar, schema v2, `record_count` 1301, run timestamp 2026-05-13T10:19:55Z. |
| `output/json/question_bank.ai_assisted.v2.batches/` | 180K, 3 files | disposable run evidence | Not exactly | Yes | Batch cache for the matching AI-assisted run. |
| `output/json/question_bank.ai_assisted.v2.failures.jsonl` | 2.7M, 25 lines | disposable run evidence | Not exactly | Yes | Failure evidence for the matching AI-assisted run. |
| `output/json/question_bank.ai_assisted.v2.full.json` | 3.9M | historical evidence | Not exactly | Yes | Full AI-assisted sidecar, schema v2, `record_count` 1301, run timestamp 2026-05-13T10:41:22Z. |
| `output/json/question_bank.ai_assisted.v2.full.batches/` | 144K, 3 files | disposable run evidence | Not exactly | Yes | Batch cache for the full run. |
| `output/json/question_bank.ai_assisted.v2.smoke.json` | 4.0M | disposable run evidence | Not exactly | Yes | Smoke run, batch status included `schema_validation_error`, run timestamp 2026-05-13T08:16:12Z. |
| `output/json/question_bank.ai_assisted.v2.smoke.batches/` | 20K, 1 file | disposable run evidence | Not exactly | Yes | Batch cache for failed smoke run. |
| `output/json/question_bank.ai_assisted.v2.smoke.failures.jsonl` | 808K, 10 lines | disposable run evidence | Not exactly | Yes | Failure evidence for smoke run. |
| `output/json/question_bank.ai_assisted.v2.smoke2.json` | 3.9M | disposable run evidence | Not exactly | Yes | Later smoke run with successful batch, run timestamp 2026-05-13T09:08:31Z. |
| `output/json/question_bank.ai_assisted.v2.smoke2.batches/` | 52K, 1 file | disposable run evidence | Not exactly | Yes | Batch cache for smoke2 run. |
| `output/json/question_bank.ai_assisted.v2.clean_smoke.json` | 52K | disposable run evidence | Not exactly | Yes | Clean smoke sidecar, `record_count` 10, run timestamp 2026-05-13T10:39:28Z. |
| `output/json/question_bank.ai_assisted.v2.clean_smoke.batches/` | 64K, 1 file | disposable run evidence | Not exactly | Yes | Batch cache for clean smoke run. |
| `output/json/question_bank.ai_assisted.v2.clean_smoke_after_fix.json` | 52K | historical evidence | Not exactly | Yes | Post-fix clean smoke sidecar, `record_count` 10, run timestamp 2026-05-13T11:15:41Z. |
| `output/json/question_bank.ai_assisted.v2.clean_smoke_after_fix.batches/` | 56K, 1 file | disposable run evidence | Not exactly | Yes | Batch cache for post-fix clean smoke run. |
| `output/json/question_bank.ai_assisted.v2.clean_full_after_fix.batches/` | 456K, 21 files | disposable run evidence | Not exactly | Yes | Per-paper/per-question batch cache for the clean full after-fix attempt. |
| `output/json/question_bank.ai_assisted.v2.clean_full_after_fix.failures.jsonl` | 140K, 5 lines | disposable run evidence | Not exactly | Yes | Failure evidence for clean full after-fix attempt. |
| `output/json/question_bank.topic_routing.v1.smoke.json` | 12K | disposable run evidence | Not exactly; model/API dependent | Yes | Topic-routing smoke run, `record_count` 10, run timestamp 2026-05-13T12:08:38Z. |
| `output/json/question_bank.topic_routing.v1.progress_smoke.json` | 12K | disposable run evidence | Not exactly; model/API dependent | Yes | Topic-routing progress smoke run, `record_count` 10, run timestamp 2026-05-13T12:43:09Z. |
| `output/json/question_bank.topic_routing.v1.failures.jsonl` | 4.0M, 153 lines | disposable run evidence | Not exactly | Yes | Failure evidence for previous topic-routing work. |
| `output_ocr_candidate/json/question_bank.json` | 12M | historical evidence | Mostly, but exact OCR/text confidence may depend on runtime state | Yes | Archived OCR candidate question bank, schema v2, `record_count` 1301. |
| `output_ocr_candidate/json/asterion_question_bank_v1.json` | 6.9M | historical evidence | Yes from matching candidate bank and sidecars | Yes | Archived Asterion projection, `record_count` 1301. |
| `output_ocr_candidate/json/asterion_content_lab_candidates_v1.json` | 19M | historical evidence | Yes from matching candidate bank and sidecars | Yes | Archived Content Lab candidate projection, `record_count` 2416. |

## Duplicate Artifact Check

The archived OCR candidate image tree contains 2,612 PNG files:

| Check | Count |
| --- | ---: |
| Archived PNG paths also present under current `output/` | 2,591 |
| Matching paths with different checksums | 0 |
| Archived PNG paths not present under current `output/` | 21 |

Archive-only PNG paths:

```text
p3/33autumn25/mark_scheme/q01.png
p3/33autumn25/mark_scheme/q02.png
p3/33autumn25/mark_scheme/q03.png
p3/33autumn25/mark_scheme/q04.png
p3/33autumn25/mark_scheme/q05.png
p3/33autumn25/mark_scheme/q06.png
p3/33autumn25/mark_scheme/q07.png
p3/34autumn25/mark_scheme/q01.png
p3/34autumn25/mark_scheme/q02.png
p3/34autumn25/mark_scheme/q03.png
p3/34autumn25/mark_scheme/q04.png
p3/34autumn25/mark_scheme/q05.png
p3/34autumn25/mark_scheme/q06.png
p3/34autumn25/mark_scheme/q07.png
p3/34autumn25/questions/q01.png
p3/34autumn25/questions/q02.png
p3/34autumn25/questions/q03.png
p3/34autumn25/questions/q04.png
p3/34autumn25/questions/q05.png
p3/34autumn25/questions/q06.png
p3/34autumn25/questions/q07.png
```

Classification: `keep-until-reviewed`. These 21 files may represent an older naming/layout state, a previously generated candidate, or a current-output gap. Do not delete them until their source documents and current output expectations are checked.

## Actionable Cleanup Recommendation

Recommendation date: 2026-05-14

This section converts the manifest into cleanup guidance only. No archive files were deleted, moved, compressed, regenerated, or rewritten during this recommendation pass.

| Artifact or group | Evidence class | Recommended action | Cleanup gate |
| --- | --- | --- | --- |
| `output_ocr_candidate/json/question_bank.json` | Formal OCR candidate evidence | Keep. Compress only as part of a full archive-preservation bundle after checksums are recorded. | Do not delete unless a later manifest proves an equal or better OCR candidate bank is preserved elsewhere. |
| `output_ocr_candidate/json/asterion_question_bank_v1.json` | Formal Asterion projection evidence | Keep. Regenerable from the matching candidate bank, but cheap to retain and useful for export provenance. | May be regenerated on demand only if the matching candidate bank and role gates remain available. |
| `output_ocr_candidate/json/asterion_content_lab_candidates_v1.json` | Formal Content Lab projection evidence | Keep. Regenerable from the matching candidate bank, but preserve this snapshot for historical export counts. | May be regenerated on demand only if the matching candidate bank and role gates remain available. |
| `output/json/audit.current.json` | Formal audit snapshot evidence | Keep. Small, useful for explaining the archived run state. | Delete only after its counts are represented in a retained manifest or audit report. |
| `output/json/status.current.json` | Unclassified status snapshot | Keep until reviewed. | Delete later only if it has no fields beyond retained audit/run evidence. |
| `output/json/question_bank.deepseek.json` | Formal legacy AI sidecar evidence | Keep, preferably compressed with other formal sidecars. | Not exactly reproducible because model/API output is involved. |
| `output/json/question_bank.ai_assisted.v2.clean_smoke_after_fix.json` | Formal post-fix AI smoke evidence | Keep. This fresh 10-record run reports `safe_to_use_for_asterion_export: true`. | Retain until a newer clean smoke/full sidecar supersedes it and the supersession is documented. |
| `output/json/question_bank.topic_routing.v1.smoke.json` and `progress_smoke.json` | Formal topic-routing smoke evidence | Keep for now. Both 10-record smoke sidecars report `safe_for_strict_filters: true`. | Delete later only after a current topic-routing sidecar or report captures the safe smoke evidence. |
| `output/json/question_bank.ai_assisted.v2.full.json` | Historical AI run evidence, not a formal export baseline | Compress or keep until reviewed. It records a successful latest 25-record attempt but remains mixed with stale records and `safe_to_use_for_asterion_export: false`. | Delete later only after the mixed/stale status and failure summary are documented elsewhere. |
| `output/json/question_bank.ai_assisted.v2.json`, `smoke.json`, `smoke2.json`, and `clean_smoke.json` | Disposable or superseded AI run artifacts | Delete later after summary. Keep until the next cleanup pass records which run each file represents. | Do not delete before retaining the post-fix clean smoke sidecar and a summary of failed/superseded runs. |
| `output/json/*.batches/` | Disposable run caches | Delete later, or compress temporarily if raw prompts/responses are still needed for debugging. | Safe only after the matching parent sidecar is either kept or summarized. |
| `output/json/*.failures.jsonl` | Disposable failure evidence | Summarize, then delete later. Compress instead if raw provider/parser failure payloads still matter. | Safe only after failure counts/reasons are captured in documentation or a retained report. |
| `output_ocr_candidate/p1/`, `p4/`, `p5/` | Duplicated generated image trees | Delete later. These trees fully duplicate current `output/` paths by checksum. | Run inventory and checksum comparison immediately before deletion. |
| Matching files under `output_ocr_candidate/p3/` | Duplicated generated image assets | Delete later. 781 of 802 archived `p3` PNGs duplicate current `output/` paths by checksum. | Keep the 21 archive-only exceptions until reviewed. |
| 21 archive-only `p3` PNGs under `33autumn25` and `34autumn25` | Keep-until-reviewed image exceptions | Keep. Move later to a small exception folder or restore/regenerate into current output only if review proves they are expected current artifacts. | Do not delete until source documents and current-output expectations are checked. |

Recommended move/compression sequence for a later cleanup pass:

1. Keep the full archive in place until the 21 `p3` exceptions are resolved.
2. If local disk pressure matters, compress the whole archive or a formal-evidence subset before deleting duplicate PNG trees. Record the archive hash and compression command in this manifest.
3. Move only formal evidence, if needed, into a durable evidence bundle that contains this manifest, OCR candidate JSON, Asterion exports, audit/status snapshots, retained AI/topic sidecars, and checksums.
4. Treat AI/API sidecars as evidence snapshots, not reproducible build products. Regeneration can create a new result, but it cannot prove byte-for-byte equivalence.
5. Treat duplicated PNG trees as reproducible/generated assets after current-output checksum equivalence is rechecked.

## Recommended Retention Policy

Archive retention should be conservative and evidence-first:

- Preserve current outputs separately from archive cleanup. Archive cleanup must not touch current `output/json/question_bank.json`, current `output/p1`, `output/p3`, `output/p4`, `output/p5`, current Asterion exports, or current sidecars.
- Retain formal evidence until either it is copied into a durable evidence bundle with checksums or a newer manifest explicitly supersedes it.
- Retain the 21 archive-only `p3` PNGs until a reviewer decides whether they are expected current artifacts, stale naming artifacts, or disposable candidates.
- Retain raw AI/OCR batch caches and failure JSONL files only while they are needed for debugging. After their parent sidecars and failure summaries are preserved, they become deletion candidates.
- Compress before deleting when exact model/OCR history may still matter. Prefer one compressed archive or a small evidence bundle over keeping duplicate image trees expanded indefinitely.
- Before any deletion pass, rerun `output-inventory`, rerun `output-cleanup-plan`, recheck duplicate PNG checksums, and record before/after archive hashes.

## Validation Commands Run

```bash
.venv/bin/python -m exam_bank.cli output-inventory --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4
.venv/bin/python -m exam_bank.cli output-cleanup-plan --root output/archive/generated_cleanup_20260513T233456Z --include-size --max-depth 4
```

Command results:

- Inventory found 1 question bank, 4 artifact trees, 0 run IDs, 0 triage iterations, and 0 frozen baselines.
- Cleanup plan was dry-run only.
- Cleanup plan classified `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate` as `keep: latest candidate`.
- Cleanup plan classified `output/archive/generated_cleanup_20260513T233456Z/output` as `unknown/manual review`.

## Retention Recommendation

Do not delete or move this archive yet.

Before deletion, complete a review pass that:

1. Confirms whether the 21 archive-only `p3` PNGs correspond to expected current documents.
2. Decides which AI/API sidecars are formal historical evidence and which are disposable run evidence after their failures are summarized elsewhere.
3. Confirms whether `status.current.json` has any unique diagnostic value.
4. Records any approved deletion in a follow-up cleanup recommendation before touching archived content.

## First Safe Cleanup Pass

Cleanup date: 2026-05-14

Status: completed. Raw disposable run caches and failure JSONL files listed below were removed after their summaries were recorded in this manifest.

Scope approved for this pass:

- Remove only archived run caches and archived failure JSONL files already classified as disposable run evidence.
- Preserve all current outputs, current Asterion exports, archived sidecar JSON snapshots, archived OCR candidate JSON, archived OCR candidate image trees, `status.current.json`, and the 21 archive-only `p3` PNG exceptions.
- Preserve every tracked file except this manifest update.

Pre-cleanup archive state:

| Metric | Value |
| --- | ---: |
| Archive size | 481,988 KiB |
| Archive file count | 2,662 |
| Archive content hash | `ff6624ec46b3384e73730a4e5635ac3cad40bfce90dc5c397da4d2e0aab25ced` |
| Disposable raw evidence selected | 8,224 KiB |

Failure summaries retained before deleting raw JSONL:

| Raw failure file | Lines | Provider/model/run | Error summary |
| --- | ---: | --- | --- |
| `output/json/question_bank.deepseek.failures.jsonl` | 11 | DeepSeek `deepseek-v4-flash`, prompt `v4`, 2026-05-07T11:34:54.525304+00:00 | 8 `parse_error` entries where the provider response did not contain text content; 3 `provider_error` API connection errors. |
| `output/json/question_bank.ai_assisted.v2.failures.jsonl` | 25 | DeepSeek `deepseek-v4-flash`, prompt `v4`, 2026-05-13T07:02:28.920750+00:00 | 25 `parse_error` entries: 11 invalid JSON responses, 10 unexpected `subpart_id: a`, and 4 invalid empty/null `subpart_id` values. |
| `output/json/question_bank.ai_assisted.v2.smoke.failures.jsonl` | 10 | DeepSeek `deepseek-v4-flash`, prompt `v4`, 2026-05-13T08:16:12.731404+00:00 | 10 `schema_validation_error` entries where `strict_filter_reason` was empty. |
| `output/json/question_bank.ai_assisted.v2.clean_full_after_fix.failures.jsonl` | 5 | DeepSeek `deepseek-v4-flash`, prompt `v4`, 2026-05-13T11:16:59.150205+00:00 | 4 `schema_validation_error` entries and 1 `taxonomy_validation_error`; schema issues were missing or mismatched required AI-assisted fields, plus one top-level shape error. |
| `output/json/question_bank.topic_routing.v1.failures.jsonl` | 153 | `deepseek-v4-flash`, 2026-05-13T12:29:09.664228+00:00 | 153 `schema_validation_error` entries, mainly `evidence_used` references to unavailable `ocr_text` or `question_text`. |

Batch-cache summaries retained before deleting raw cache directories:

| Raw cache directory | Files |
| --- | ---: |
| `output/json/question_bank.ai_assisted.v2.batches/` | 3 |
| `output/json/question_bank.ai_assisted.v2.full.batches/` | 3 |
| `output/json/question_bank.ai_assisted.v2.smoke.batches/` | 1 |
| `output/json/question_bank.ai_assisted.v2.smoke2.batches/` | 1 |
| `output/json/question_bank.ai_assisted.v2.clean_smoke.batches/` | 1 |
| `output/json/question_bank.ai_assisted.v2.clean_smoke_after_fix.batches/` | 1 |
| `output/json/question_bank.ai_assisted.v2.clean_full_after_fix.batches/` | 21 |

Removed archived disposable artifacts:

| Path | Removed item type |
| --- | --- |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.full.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.smoke.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.smoke2.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.clean_smoke.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.clean_smoke_after_fix.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.clean_full_after_fix.batches/` | Batch cache directory |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.deepseek.failures.jsonl` | Failure JSONL |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.failures.jsonl` | Failure JSONL |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.smoke.failures.jsonl` | Failure JSONL |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.ai_assisted.v2.clean_full_after_fix.failures.jsonl` | Failure JSONL |
| `output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.topic_routing.v1.failures.jsonl` | Failure JSONL |

Post-cleanup archive state:

| Metric | Value |
| --- | ---: |
| Archive size | 473,764 KiB |
| Archive file count | 2,626 |
| Archive content hash | `7de4553f5a25b7b5ebda93c3e2352a56fc12b354e6ab94141816847eea918a57` |
| Current output size | 937,312 KiB |

Post-cleanup validation:

- `output-inventory --root output --include-size --max-depth 4` passed and still found 1 current question bank, 8 artifact trees, 2 current Asterion exports, 0 frozen baselines, and no generated reports classified as safe-to-delete.
- `output-cleanup-plan --root output --include-size --max-depth 4` passed and still classified current `output/json/question_bank.json`, `output/p1`, `output/p3`, `output/p4`, and `output/p5` as `keep: canonical/current`; `output/archive` remains `unknown/manual review`.
- `output-integrity-audit --input output/json/question_bank.json --artifact-root output` passed with `ok: true`, 1,301 records, and only the known 11 missing mark-scheme companions for `9709_2025_November_33`.
- Protected current outputs and current Asterion exports were present after cleanup.
