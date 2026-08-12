# Project Audit - 2026-06-30

> Historical pre-split snapshot. Classroom, homework/email intake, and grading
> paths described below moved to `../homework-ingest` and `../autograder` on
> 2026-08-11; their commands and modules are no longer part of exam-bank.

This audit reviews the repository against the intended product scope:

1. Extract PNG images of questions and link them to PNG images of mark schemes.
2. Create a file with extracted text, placement/routing, topic, difficulty, and other useful metadata.
3. Support class-assist workflows for assignment intake, return, and eventually teacher-reviewed autograding.

The short version: the project has a strong image-first extraction core and a lot of useful audit/review infrastructure, but it is currently carrying too much generated state, too many experimental side workflows, and a few important contract inconsistencies. The fastest path forward is to stabilize the core artifact contract, make tests deterministic, reduce local/generated bloat, then build downstream class-assist/autograding only on a smaller trusted subset.

## Audit Inputs

Commands and checks run during this audit:

- Repository inventory and size checks with `find`, `du`, `git ls-files`, and `git count-objects`.
- Current export inspection of `output/json/question_bank.json`.
- Built-in audit: `.venv/bin/python -m exam_bank.cli audit --input output/json/question_bank.json`.
- Built-in integrity audit: `.venv/bin/python -m exam_bank.cli output-integrity-audit --input output/json/question_bank.json --artifact-root output`.
- Readiness audit: `.venv/bin/python scripts/audit_question_bank_readiness.py --input output/json/question_bank.json --artifact-root output`.
- Topic packet preflight: `.venv/bin/python scripts/audit_topic_packet_preflight.py --question-bank output/json/question_bank.json --topic-routing data/topic_routing/question_bank.topic_routing.v1.json --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json`.
- Output inventory and cleanup plan into `/private/tmp/exam_bank_project_audit_20260630`.
- Storage duplicate audit: `.venv/bin/python -m exam_bank.command data audit-storage --dry-run`.
- Test suite: `.venv/bin/python -m pytest -q`.

No cleanup or destructive operation was performed.

## Current State

### Dataset/export status

The current canonical export is `output/json/question_bank.json`.

- Schema: `exam_bank.question_bank` version `2`.
- Generated at: `2026-06-28T12:37:42.691691+00:00`.
- Run ID: `20260628T123742Z-d4a6a86def19`.
- Records: `3548`.
- Year range: `2008-2025`.
- Family counts:
  - `pm1`: `1092`
  - `pm3`: `1053`
  - `stats`: `1148`
  - `mechanics`: `255`

The image-first contract is partly strong:

- Question image paths: `0` missing.
- Question image files: all referenced question images exist.
- Mark-scheme image paths: `928` missing.
- Mark-scheme image files: all nonblank referenced mark-scheme images exist.
- Only `11` missing mark-scheme images are explained by a known missing source companion (`9709_2025_November_33`).
- `917` missing mark-scheme images are unexpected.

Current quality counts from the export/run/audits:

- Validation status: `2632 pass`, `771 review`, `145 fail`.
- Mapping status in payload notes: `3364 pass`, `184 fail`.
- Mapping status in run manifest: `2604 pass`, `944 fail`.
- Scope quality: `3143 clean`, `388 review`, `17 fail`.
- Text fidelity: `3382 clean`, `154 degraded`, `12 unusable`.
- Visual curation: `401 ready`, `3014 review`, `133 fail`.
- Text-only status: `686 ready`, `2616 review`, `246 fail`.
- Question crop confidence: `676 high`, `2872 low`.
- Mark-scheme crop confidence: `2400 high`, `220 medium`, `928 low`.

The discrepancy between payload mapping counts and run-manifest mapping counts is important. It indicates that at least one audit/export surface is no longer measuring the same contract as the others.

### Integrity audit result

The current output integrity audit failed.

Passing checks:

- Record count matches declared count.
- Question IDs are present and unique.
- Paper/question pairs are unique.
- Image paths are unique across records.
- Image path roles match their fields.
- Question image paths are relative and exist.
- Nonblank mark-scheme image paths are relative and exist.
- Rendered crop whitespace check passes.

Failing checks:

- `missing_mark_scheme_paths_only_known_companions`: `917` unexpected missing mark-scheme image paths.
- `mark_scheme_text_has_no_foreign_question_labels`: `75` records appear to include mark-scheme text for a neighboring top-level question.
- `rendered_crop_dimensions_not_suspicious`: `3` suspiciously tall mark-scheme crops:
  - `42winter17_q02`
  - `33summer17_q04`
  - `32winter17_q05`

The first failure is the biggest current blocker for goal 1 and future grading.

### Text, topic, and difficulty state

OCR ran on all `3548` records.

- OCR selected for question text: `120` records (`3.38%`).
- Native text retained: `3428` records.
- Possible OCR false negatives: `229`.
- Suspicious OCR-selected records: `108`.

Text should still be treated as advisory:

- Question text role counts:
  - `readable_text`: `790`
  - `search_hint`: `2608`
  - `untrusted_math_text`: `150`
- Question text trust:
  - `high`: `790`
  - `medium`: `2592`
  - `low`: `154`
  - `unusable`: `12`

Topic/difficulty is useful, but not yet safe as a strict downstream truth source:

- Topic confidence: `2760 high`, `392 medium`, `396 low`.
- Topic trust status: `691 normal`, `223 review_required`, `2634 degraded_text`.
- Difficulty confidence: `3129 high`, `414 medium`, `5 low`.
- `1037` records have high difficulty confidence while marks/text/mapping evidence is degraded.

Topic packet preflight:

- Question-bank records: `3548`.
- Topic-routing records: `3549`.
- Invalid current major topics before normalization: `983`.
- Unresolved after normalization: `0`.
- Release candidates after normalization: `360`.
- Review-required candidates after normalization: `2883`.
- Blocked candidates after normalization: `305`.
- Warning: some records change expected packet family after topic-taxonomy normalization.

Interpretation: topic packets are valuable review/teacher material, but the current packet layer should not be treated as production-ready student routing.

### Class assist and autograding state

The class-assist layer is real but still local-first/draft-first.

Implemented or partially implemented:

- Local class workspace creation under `data/classes`.
- Roster CSV handling.
- Assignment PDF registration and scheduling.
- Message/reminder schedule generation.
- Dry-run and optional live email dispatch plumbing.
- Local submission ingest from assignment inbox folders.
- Completion CSV/report output.
- Submission acknowledgement, resend, and missing-reminder drafts.
- Browser classroom dashboard.
- Native-PDF answer-presence checks.
- Draft grading artifacts that require teacher review.
- Reviewed rubric and eligible-item infrastructure.

Current limitations:

- There is no `output/auto_grade/eligible_items.v1.json` in the current generated output.
- Autograding is intentionally blocked unless reviewed rubrics, mark events, valid image assets, and safe topic routing exist.
- The current extraction bank has `928` missing mark-scheme images, so broad autograding would be premature.
- Tests around P3 exact-skill and reviewed mark-event artifacts currently depend on missing generated reports/outputs.

This is the right safety posture. The next step is not "turn on autograding"; it is to define a small gold pilot set and run class-assist flows against only that set.

## Bloat

### Local disk bloat

Top-level local sizes:

- `output`: `3.4G`
- `.git`: `670M`
- `data`: `545M`
- `input`: `279M`
- `.venv`: `172M`
- `exam_bank_taxonomy`: `44M`
- `reports`: `13M`
- `src`: `7.8M`
- `tests`: `5.7M`
- `scripts`: `1.4M`

Largest generated output areas:

- `output/submissions/p3_quiz_2026_06_23`: `1.3G`
- `output/topic_packets`: `933M`
- `output/stats`: `289M`
- `output/pm3`: `227M`
- `output/pm1`: `223M`
- `output/run_status`: `179M`
- `output/mechanics`: `97M`
- `output/json`: `54M`
- `output/candidates/ocr`: `54M`
- `output/triage/iteration_001`: `46M`

Private/local data is also substantial:

- `data/submissions`: `317M`
- `data/classes`: `124M`
- `data/review`: `99M`
- `data/topic_routing`: `4.9M`

The output cleanup plan classifies only a few generated audit folders as obvious archive/delete candidates. It leaves canonical image trees, submissions, topic packets, and recompute folders for manual decision. That is conservative and appropriate.

Storage duplicate audit:

- Output file count: `10295`.
- Output size: `3.35 GiB`.
- Unreferenced files: `3198`.
- Duplicate groups: `943`.
- Duplicate files: `1914`.
- Duplicate wasted bytes: `87.27 MiB`.
- Estimated reclaimable bytes: `75.36 MiB`.

The duplicate cleanup is useful but not the main disk win. The main win is deciding which generated submissions/topic packets/run-status artifacts should be archived or deleted.

### Git/repository bloat

Tracked files:

- Total tracked files: `1492`.
- Git pack size: `485.96 MiB`.
- Loose object size: `183.31 MiB`.

Tracked file distribution:

- `input`: `932` tracked files.
- `src`: `158`.
- `tests`: `124`.
- `exam_bank_taxonomy`: `82`.
- `scripts`: `70`.
- `data`: `48`.
- `docs`: `42`.

Large tracked files include:

- Source PDFs under `input/pastpapers/9709`, especially examiner reports and question papers.
- Large review decision files under `data/review`.
- Duplicated taxonomy canonical/archive files under `exam_bank_taxonomy`.
- `output/json/asset_manifest.v1.json` (`~7.4M`) is tracked.

This is not just local bloat. The repository itself carries a lot of data. That may be acceptable for an offline/local tool, but if the goal is a maintainable product repo, source PDFs and large review artifacts should move to a data-release/cache strategy rather than git history.

### Code bloat

Approximate Python line counts:

- `src/exam_bank`: `78,581` lines.
- `tests`: `35,047` lines.
- `scripts`: `14,414` lines.

Largest source files:

- `src/exam_bank/mark_schemes.py`: `3702` lines.
- `src/exam_bank/deepseek_enrich.py`: `3518` lines.
- `src/exam_bank/topic_packets.py`: `2908` lines.
- `src/exam_bank/image_rendering.py`: `2329` lines.
- `src/exam_bank/asterion_export.py`: `2304` lines.
- `src/exam_bank/pipeline.py`: `2164` lines.
- `src/exam_bank/p3_exact_skill/review_batch.py`: `1953` lines.
- `src/exam_bank/question_detection.py`: `1784` lines.
- `src/exam_bank/content_lab_auto_review.py`: `1775` lines.
- `src/exam_bank/audit.py`: `1725` lines.

There are `95` files under `scripts/`, many of which are thin wrappers around package functions. The CLI exposes extraction, regeneration, audits, Asterion, Content Lab, AI, topic review, topic packets, triage, output cleanup, email, classroom, and assignment workflows from one command surface.

This is the biggest code organization issue: the project has a mature core, but the runtime surface does not clearly distinguish:

- Core extraction.
- Generated output management.
- Review/triage experiments.
- Asterion export.
- Topic packet generation.
- Class assist/submission workflows.
- Autograding research.

## Test Health

`python -m pytest -q` currently fails:

- `1107` passed.
- `34` skipped.
- `14` failed.

Failure categories:

- Missing generated/report fixtures:
  - `output/audits/asterion_content_lab_loop/iteration_003b/sample_results.csv`
  - `reports/manual_review_batch_0003_conclusions.v1.json`
  - `reports/p3_exact_skill_registry_seed_0003.v1.json`
  - `reports/p3_exact_skill_review_queue.v1.json`
  - `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Reviewed mark-event artifact no longer validates.
- Question crop duplicate-region counter regression.
- Runtime CLI test expects the older command surface and fails because class/email/topic-review commands are now exposed.
- Runtime archive test expects `archive/topic_pdfs_legacy/...`, but the archive path is absent.
- Bad-text fixture still uses old paper family labels such as `p1` while the current bank uses canonical labels such as `pm1`.

Running `.venv/bin/pytest -q` directly also fails during collection because some tests import `tests.*` and `scripts.*`, while the console entry point does not put the repo root on import path in the same way as `python -m pytest`. CI uses `python -m pytest`, so CI should collect, but the substantive 14 failures remain.

The project has enough tests to support cleanup, but the suite is currently not a reliable gate. Fixing that should be the first engineering task.

## What Is Working

The strongest pieces are:

- Image-first source-of-truth policy is consistently represented in docs and core code.
- `PaperIdentity` gives a central naming/ID contract for question IDs, paper IDs, and canonical image paths.
- The pipeline already emits schema-versioned question-bank JSON with run metadata.
- Question images are present and unique for all current records.
- There are multiple useful audit tools: integrity, readiness, topic packet preflight, output inventory, storage duplicate audit.
- Class-assist primitives exist and have safety boundaries: local private roots, draft-only outgoing messages, teacher-review grading.
- The code is heavily tested even though the current suite is not green.

## Main Risks

1. Mark-scheme image gaps are too large.
   - `928` missing mark-scheme image paths block goal 1 and future grading.
   - `917` are not explained by known missing source files.

2. Mapping/validation contract drift.
   - `707` records have missing mark-scheme image paths but `validation_status=pass`.
   - `760` records have missing mark-scheme image paths but `mapping_status=pass`.
   - The run manifest says `944` mapping failures, while payload notes say `184`.

3. Review/generated artifacts are mixed into test expectations.
   - Tests depend on ignored/generated output and report files.
   - This makes cleanup risky until tests are changed to use fixtures or explicit integration markers.

4. Topic/difficulty metadata is over-presented relative to trust.
   - Many records have high topic/difficulty confidence but degraded text or mapping evidence.
   - Topic packet release candidates are only `360` after normalization.

5. The command surface is too broad.
   - `exam_bank.cli` is now a combined product CLI, research CLI, admin CLI, email tool, and classroom tool.
   - This makes "what should I run?" harder than it needs to be.

6. Local and tracked data growth is not under enough policy control.
   - `output` is `3.4G`.
   - `input` source PDFs are tracked.
   - Review decision JSONs and taxonomy archives have grown large.

## Quickest Improvements

These are the highest leverage fixes that do not require rethinking the product.

### 1. Make tests green and deterministic

Target: a clean `python -m pytest -q`.

Work:

- Replace tests that read ignored generated artifacts with small committed fixtures, or mark them as integration/manual-output tests.
- Fix `test_prompt_crop_deduplicates_overlapping_visual_regions` so duplicate-graphic metadata reflects the actual dedupe behavior.
- Update `test_runtime_paths.py` to match the current CLI surface or intentionally split active commands into separate subcommands/groups.
- Either restore the expected archive fixture or change the archive test to assert the documented current state.
- Update text-fidelity fixtures from `p1/p3/...` family labels to canonical `pm1/pm3/stats/mechanics`.
- Repair or regenerate the reviewed mark-event fixture so validation passes.

Why first: a green test suite makes all cleanup and optimization safer.

### 2. Fix mark-scheme asset contract invariants

Target: no record can be "pass" for mapping/validation if its mark-scheme image path is missing, unless it is explicitly and narrowly allowed.

Work:

- Add an invariant in export/audit code:
  - blank `mark_scheme_image_path` implies a hard blocker or documented missing companion.
  - blank `mark_scheme_image_path` cannot coexist with `mapping_status=pass`.
  - blank `mark_scheme_image_path` should not coexist with `validation_status=pass` for image-first grading/release roles.
- Make run-manifest QA counts derive from the same payload fields consumers see, or add a test that prevents payload/run-manifest count drift.
- Update the readiness audit to read the current `run_manifest` shape; it currently reports missing run metadata even though the export has a top-level run manifest.

Why second: this prevents downstream code from treating incomplete records as ready.

### 3. Attack missing mark-scheme images by cluster

Target: reduce `928` missing mark-scheme image paths and `917` unexpected misses.

Start with:

- Missing mark-scheme image count by family:
  - `stats`: `266`
  - `pm1`: `264`
  - `pm3`: `247`
  - `mechanics`: `151`
- Missing mark-scheme image count by format:
  - `legacy`: `789`
  - `caie_2024_2025`: `139`
- Highest year/family clusters:
  - `2018 stats`: `49`
  - `2025 pm3`: `46`
  - `2023 pm1`: `42`
  - `2024 pm3`: `40`
  - `2023 pm3`: `40`
  - `2017 stats`: `38`
  - `2025 mechanics`: `37`
  - `2024 pm1`: `37`
  - `2019 stats`: `37`

Work:

- Fix one representative family/year cluster at a time.
- Add small PDF fixtures for each segmentation mode that fails.
- Re-run integrity audit after each cluster.
- Keep broad regeneration separate from code fixes.

Why third: it directly improves goal 1 and unlocks autograding eligibility later.

### 4. Clean local generated bloat with a manifest

Target: reclaim space without damaging canonical current output.

Safe first pass:

- Delete local `__pycache__` and `.DS_Store` files.
- Keep `output/json/question_bank.json`.
- Keep canonical image trees unless regenerating from scratch.
- Keep frozen triage baseline `output/triage/iteration_001/baseline_question_bank.json`.
- Archive or delete old generated audit/report folders listed by the cleanup plan:
  - `output/audits/topic_confidence_recovery_readiness`
  - `output/audits/topic_confidence_rescoring`
- Decide whether `output/submissions/p3_quiz_2026_06_23` and `data/submissions/p3_quiz_2026_06_23` are current private evidence or disposable run artifacts.
- Decide whether generated topic packet PDFs should be kept locally, regenerated on demand, or archived outside the repo.

Why fourth: it reduces noise immediately but should not happen before identifying the few outputs tests still wrongly depend on.

### 5. Split the command surface

Target: make the main workflows obvious.

Suggested grouping:

- `exam_bank.cli process`, `regenerate-*`, `audit`, `output-integrity-audit`: core extraction and validation.
- `exam_bank.cli topic-*`, `topic-packets`: metadata/review projections.
- `exam_bank.cli asterion-*`: Asterion export.
- `exam_bank.cli class-*`, `quiz-packet`, `grade-quiz-bma`: class assist.
- `exam_bank.auto_grade.*`: rubric/autograding readiness.
- Keep one-off historical scripts under `scripts/legacy` or convert them to package CLIs only if still active.

Why fifth: this reduces conceptual bloat without deleting useful work.

## Most Valuable Improvements

### A. Stabilize the core extraction contract

This is the highest-value improvement because every other workflow depends on it.

Definition of done:

- Integrity audit passes.
- `question_bank.json` payload and `run_manifest.qa_summary` agree.
- No unexpected missing mark-scheme image paths.
- No validation pass for missing required image assets.
- No mark-scheme text with foreign top-level question labels.
- Suspicious crop dimensions are either fixed or explicitly reviewed.

### B. Build a small gold pilot dataset

There are currently `350` Tier 5 records in the readiness audit. Use those as the first trusted pilot set.

Use this pilot set for:

- Assignment generation.
- Class-assist PDF sending.
- Submission intake.
- Teacher review packet generation.
- Manual grading or answer-presence checks.
- Later reviewed-rubric beta.

Do not start by trying to make all `3548` records student/autograde ready.

### C. Make metadata useful but honest

The metadata file should be explicit about trust.

Recommended export:

- Keep `question_bank.json` as canonical image-backed master.
- Add a lightweight derived index for classroom/search use, for example `output/json/question_index.v1.json` or CSV.
- Include:
  - `question_id`
  - paper/session/component/family
  - question and mark-scheme image paths
  - extracted text
  - text role/trust/fidelity
  - topic and topic trust
  - difficulty and difficulty confidence
  - validation/mapping/visual status
  - autograde eligibility status

This gives the class-assist layer one clean intake file without pretending all metadata is equally reliable.

### D. Keep autograding teacher-reviewed and opt-in

Autograding should be built only after the asset contract is stable.

Near-term grading path:

- Use answer-presence checks as non-scoring teacher signals.
- Generate teacher review packets.
- Create reviewed rubric candidates only for gold pilot questions.
- Validate mark events and reviewed rubrics.
- Generate `eligible_items.v1.json`.
- Allow teacher-beta grading only where eligibility passes.

Student-facing autograding should remain blocked until teacher-beta evidence is stable.

## Suggested Roadmap

### Phase 0 - Stabilize and clean

Duration: 1-2 focused passes.

Goals:

- Fix test suite.
- Fix audit/contract drift.
- Decide generated output retention.
- Remove local caches and stale outputs with a manifest.

Acceptance:

- `python -m pytest -q` passes.
- Integrity audit either passes or has only documented allowlisted failures.
- Cleanup plan reviewed before any deletion.

### Phase 1 - Mark-scheme reliability

Goals:

- Reduce unexpected missing mark-scheme images from `917` toward zero.
- Fix the `75` foreign-question mark-scheme text cases.
- Fix or review the `3` suspiciously tall crops.

Acceptance:

- Mark-scheme image path exists for every record with a source companion.
- Missing mark-scheme cases are only known source-file gaps.
- Mapping/validation counts are consistent across payload, run manifest, and audits.

### Phase 2 - Trusted metadata/index layer

Goals:

- Produce a class/search-friendly metadata index.
- Keep trust fields prominent.
- Normalize topic-routing family/topic contracts.

Acceptance:

- One clear metadata artifact for consumers.
- Topic packet preflight release candidates increase meaningfully.
- No strict filter is enabled unless sidecar metadata says it is safe.

### Phase 3 - Class-assist pilot

Goals:

- Use a small gold pilot set for assignments.
- Exercise roster, send schedule, local/email intake, completion summary, and teacher review.
- Keep real student data ignored/local.

Acceptance:

- One assignment can be generated, sent/dry-run, ingested, summarized, and reviewed.
- No student-facing grades are sent automatically.
- Artifacts live under private ignored roots.

### Phase 4 - Teacher-beta autograding

Goals:

- Reviewed rubrics for gold pilot questions.
- Valid mark events.
- Eligible-items generation.
- Teacher-only draft scoring.

Acceptance:

- `eligible_items.v1.json` has a nonzero teacher-beta subset.
- Draft grading always requires teacher review.
- Student-facing autograde remains off until separately approved.

## Bottom Line

This project is not a small script anymore. It is a data pipeline plus review system plus classroom tool. The extraction core is valuable and close to the right shape. The main issue is not lack of capability; it is that generated outputs, review experiments, and downstream tools are all living beside the core without enough gates.

The most valuable next move is to make the current image/mark-scheme contract reliable and testable. After that, the fastest product win is a small trusted class-assist pilot, not broad autograding.
