# Project audit — 2026-07-13

This record captures the repository baseline, the forward-only cleanup and
optimization work completed on 2026-07-13, the verification evidence, and the
remaining structural risks. Git history was not rewritten.

## Protected working state

The following pre-existing topic-difficulty work was kept out of the staged
audit changes:

- `src/exam_bank/topic_difficulty_review.py`
- ignored `data/review/topic_difficulty/**/topic_packet_difficulty_review.v2.json`
- ignored topic-difficulty reconciliation batches, decisions, and summaries
  under `data/review/topic_difficulty/`

The source edit remains unstaged. The ignored artifacts remain on disk. No
cleanup command deleted local corpus, generated, review, or private data.

## Before and after

| Measure | Baseline | 2026-07-13 result |
| --- | ---: | ---: |
| Tracked files | 9,563 | 525 |
| Tracked `tmp/` files | 6,494 | 0 |
| Tracked review files | 1,554 | 7 promoted canonical artifacts |
| Tracked source PDFs | 932 | 0 |
| Top-level Python scripts | 84 | 50 |
| Package Python files | 166 | 174 |
| Package source lines | 85,510 | 87,084 |
| Canonical questions | 3,548 | 3,548 |
| Missing mark-scheme paths | 292 | 0 |
| Foreign top-level mark-scheme labels | 75 | 0 |

The checked-out tree is 9,038 tracked files smaller, a 94.5% reduction. Seven
small PDF files remain only as submission fixtures under `tests/fixtures/`; no
source-corpus PDF remains tracked.

## Storage and provenance boundary

- `manifests/corpora/caie_9709.v1.json` is the authoritative 932-document
  corpus contract. Verification passed for all 932 local files with no missing,
  size-mismatched, or checksum-mismatched documents.
- Hydration supports missing-only restoration, offline reporting, ordered
  source/mirror fallback, checksum verification, quarantine, and explicit
  `--repair` replacement.
- `input/`, `tmp/`, generated `output/`, generated `reports/`, review runs, and
  visual evidence are untracked and ignored. Local copies were preserved with
  index-only cleanup.
- Seven promoted review artifacts live under `data/review/canonical/` with
  authority and source provenance. Promotion validates schema, source run,
  source artifact hash, reviewer/timestamp fields, and writes atomically.
- Sixteen exact taxonomy archive duplicates were removed. Their former paths,
  source commit, and checksums are recorded in
  `manifests/archive/taxonomy_exact_duplicates.v1.json`.
- Release manifests moved from generated `reports/` storage to
  `manifests/releases/`.

## Canonical product repair

The initial integrity audit found 292 records without mark-scheme image paths
and 75 records whose mark-scheme text contained a neighboring top-level
question. The failures clustered around legacy/table boundary segmentation and
formula fragments adjacent to the next question label.

Boundary detection now clamps legacy grid regions to anchors, trims adjacent
formula fragments and foreign top-level text, and validates saved crops. A
targeted, fail-closed repair regenerated only affected papers before the final
controlled audit. The final audit reports:

- `ok: true`
- 3,548 unique records
- 0 missing question paths or files
- 0 missing mark-scheme paths or files
- 0 foreign top-level mark-scheme labels
- 0 suspicious crop dimension or whitespace failures

The run manifest derives QA counts from the same implementation as the payload
gate and now records corpus-manifest hash, configuration hash, and pipeline
version. Required mark-scheme evidence remains fail-closed downstream.

## Command and process boundary

- `exam-bank <domain> <command>` is the only installed public entry point.
- Registration is lazy across extract, data, topic, Asterion, AI, triage,
  review, classroom, email, marks, autograde, and advisory domains.
- Every registered command has a passing `--help` smoke test. Former flat names
  are absent from the public registry.
- Thirty-four one-function script wrappers were removed after their workflows
  moved to namespaced commands. The generated command reference is checked in
  CI.
- Extraction resume identity includes source-document SHA-256, effective
  configuration, pipeline version, and OCR profile; stale batches cannot be
  silently reused.
- `exam-bank extract run --workers N` uses per-paper staging roots. The parent
  atomically promotes artifacts, deterministically merges JSONL diagnostics,
  sorts canonical records, and atomically writes final JSON. Debug overlay mode
  remains single-worker.

Shared image stitching, answer-rule geometry, component/family mapping,
requested-ID normalization, and mark-scheme mapping validation were extracted
from duplicate implementations. Mark-scheme label recognition is shared by
segmentation and integrity auditing.

## Verification evidence

All commands were run from the repository root on 2026-07-13.

| Gate | Result |
| --- | --- |
| Ruff imports/correctness | pass |
| Data-independent suite | 1,177 passed; 178 deselected; 0 failed |
| Full suite | 1,325 passed; 34 skipped; 0 failed |
| Corpus verification | 932/932 verified; `ok: true` |
| Output integrity audit | 3,548 records; `ok: true` |
| One-worker / two-worker equivalence | 2 real papers, 21 records; identical question payload, image hashes, and diagnostic hash |
| Repository policy | no tracked input, tmp, generated output/reports, review runs, or unexpected large files |

The original fast-suite baseline was 1,135 passed, 15 failed, and 177
deselected. Structural changes began only after the failing baseline contracts
and canonical integrity issues were repaired.

## Full packet pre-commit rehearsal

All 27 topic packets were rebuilt into an isolated output root after the final
code changes. The regenerated set preserves the product contract: 3,481
questions, 3,481 solutions, 1,464 approved records, 2,017 review-required
records, 4,134 topic placements, 67 intentional exclusions, and no missing
solutions. All 27 difficulty sidecars are complete with no pending reviews.

The rehearsal caught and repaired two process regressions before commit:

- difficulty projections were coupled to review-artifact storage paths instead
  of their semantic packet membership;
- a block split across two PDF pages recorded only its starting page, leaving
  the continuation outside the manifest's section range.

The final packet set contains 3,309 A4 portrait pages. All pages opened with
strict `pypdf`, rendered with PyMuPDF, and were classified by the visual-audit
batch. There are no blank, encrypted, unclassified, or missing-asset pages, and
all question/answer ranges are contiguous. High-resolution samples covered
section transitions, split blocks, and every repaired solution-path family.

The previous packet set contained 3,339 pages. The 30-page and 423,946,439-byte
reduction is expected: 78 solution paths now point to their corrected canonical
session assets, eliminating wrong or oversized legacy crops. Ordered question
IDs, packet membership, review sections, topic coverage, difficulty ranks, and
answer availability are unchanged.

Thirteen pre-existing seed-bug pages remain in the review evidence, including
source-level duplicated prompts, neighboring-question fragments, and Cambridge
barcode/anonymization furniture. They are not packet-renderer regressions and
are primarily routed to Review Required, but the corpus is therefore not
claimed to be visually defect-free.

## Remaining structural risks

The repository boundary, canonical corpus, public command contract, and safety
gates are green. The following deeper optimizations remain active roadmap work
and are not claimed complete by this audit:

1. Fifty substantial specialist scripts still contain product or audit logic
   and should move into package domain modules before `scripts/` can disappear.
2. The largest modules remain 4,040 lines (`topic_packets.py`), 3,785 lines
   (`mark_schemes.py`), 3,334 lines (`image_rendering.py`), and 2,469 lines
   (`pipeline.py`). Initial shared seams were extracted, but the planned full
   physical split is incomplete.
3. Corpus-backed segmentation coverage still uses locally hydrated papers for
   several modes; additional compact generated-PDF fixtures are needed to make
   every rendering regression execute in a data-free clone rather than skip.

These items are ordered in the root `ROADMAP.md`; dated measurements and
completed work remain in this history record.
