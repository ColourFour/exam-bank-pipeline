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
