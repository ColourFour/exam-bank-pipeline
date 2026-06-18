# CAIE 9709 Exam-Bank Extraction Pipeline

This repository builds an image-first CAIE 9709 exam-bank dataset from question-paper PDFs and mark-scheme PDFs. The active architecture is a unified historical + modern pipeline covering the 2008-2025 corpus through one recursive extraction entrypoint.

The pipeline scans PDFs, detects top-level questions, renders question crops, renders matching mark-scheme crops, pairs those artifacts, exports structured JSON, and writes review/audit sidecars. The canonical question PNGs and mark-scheme PNGs are the source of truth. Native PDF text, OCR text, AI enrichment, topic routing, readiness tiers, and Asterion projections are advisory metadata unless a specific consumer role gate says otherwise.

The Asterion handoff is course-aware for the static 9709 study site. Supported course IDs are `p1` (Pure Mathematics 1), `p3` (Pure Mathematics 3), `m1` (Mechanics 1), and `s1` (Probability & Statistics 1). Extraction asset folders use the canonical `PaperIdentity` subject families described below; Asterion consumers should use `src/exam_bank/asterion_course_contract.py` rather than inferring course IDs from asset paths.

## Unified Dataset

The pipeline is designed for a single CAIE 9709 dataset across both source eras:

- Legacy papers: `2008-2020`
- Modern papers: `2021-2025`
- Source tree: `input/pastpapers/9709/<year>/{question_papers,mark_schemes,exam_reports,grade_thresholds}/`
- Canonical export: `output/json/question_bank.json`
- Canonical assets: `output/<subject_family>/<subject_family>_<year>_<session>_<component>_<qp|ms>_qNN_<question|markscheme>.png`

Canonical extraction subject-family mapping:

- `pm1`: Pure Mathematics 1 (`p1`, component family 1)
- `pm3`: Pure Mathematics 3 (`p3`, component family 3)
- `stats`: canonical statistics asset family (`p4` and `p6` in `PaperIdentity`)
- `mechanics`: canonical mechanics asset family (`p5` in `PaperIdentity`)

`PaperIdentity` is a required dependency for canonical paper IDs, question IDs, mark-scheme pairing, and asset paths. It normalizes session/year/component metadata and prevents question and mark-scheme assets from being written under ambiguous or mismatched names.

Run-specific counts live in the export `run_manifest` and `output/run_status/<run_id>/run_manifest.json`. The dated audit baseline is still [`docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md`](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md), but it is historical evidence, not the live count source.

Current important generated artifacts:

- Canonical question bank: `output/json/question_bank.json`
- Canonical image trees: `output/pm1/*.png`, `output/pm3/*.png`, `output/stats/*.png`, and `output/mechanics/*.png`
- Canonical asset index: `output/json/asset_manifest.v1.json`
- Strict topic-routing sidecar working copy: `output/json/question_bank.topic_routing.v1.json`
- Durable refreshed topic-routing sidecar source: `data/topic_routing/question_bank.topic_routing.v1.json`
- Mark-event evidence sidecar: `output/json/question_bank.mark_events.v1.json`
- Advisory evidence sidecar: `output/advisory_evidence/question_bank.advisory_evidence.v1.json`
- Difficulty index sidecar: `output/json/question_bank.difficulty_index.v1.json`
- Asterion all-course catalog: `output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json`
- Asterion student runtime question bank: `output/asterion/exports/latest/asterion_question_bank_v1.json`
- Content Lab candidates: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`
- Topic packets: `output/topic_packets/`
- Text-fidelity review state: `data/review/text_fidelity_review_state.json`

Generated outputs live under ignored output roots. Commit source, tests, docs, schemas, and intentional fixtures, not full generated banks.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

OCR-enabled runs require Tesseract:

```bash
brew install tesseract
.venv/bin/python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
tesseract --version
```

AI sidecar commands require a DeepSeek API key:

```bash
set -a; source .env; set +a
```

Deterministic advisory-evidence commands use local examiner-report and grade-threshold PDFs under `input/` and do not require provider credentials.

## Command Atlas

Use [`docs/COMMAND_ATLAS.md`](docs/COMMAND_ATLAS.md) as the current command map. It covers standard and OCR extraction, resume behavior, audits, mark-event evidence, advisory evidence, difficulty index generation, Asterion and Content Lab projections, topic packets, topic routing, AI enrichment, AI sidecar audit, auto-triage, output inventory, cleanup planning, export summary diffs, taxonomy generation safety, and test commands.

AI-heavy workflows are long-running and sidecar-only. They require provider credentials and must not be treated as canonical extraction truth.

Fast local test loop:

```bash
.venv/bin/python -m pytest -q -m "not integration and not rendering"
```

Full validation, including integration and rendering regressions:

```bash
.venv/bin/python -m pytest -q
```

CI intentionally runs the full suite with `python -m pytest`.

Backfill or refresh the unified input manifest from PastPapers.co:

```bash
.venv/bin/python -m exam_bank.cli ingress pastpapers-co \
  --min-year 2008 \
  --max-year 2025 \
  --output exam_bank_input.jsonl
```

Run full pipeline extraction for the unified 2008-2025 source tree:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input/pastpapers/9709 \
  --output output \
  --enable-ocr
```

This `process` command is the implemented run-full-pipeline entrypoint: it recursively scans the legacy and modern folders, pairs question papers with mark schemes through the document registry and `PaperIdentity`, renders question crops and mark-scheme crops, writes `question_bank.json`, and emits run-status manifests. Use `--resume`, `--run-id`, and `--force-rerun` for long unified runs.

## Asset Generation Pipeline

Asset generation is shared across legacy and modern papers:

1. The document registry recursively scans source PDFs and groups question papers, mark schemes, examiner reports, and grade thresholds by syllabus, year, session, and component.
2. `PaperIdentity` normalizes the grouped metadata into canonical `paper_id` and `question_id` values.
3. Question detection finds top-level question spans in each question paper.
4. The renderer writes one canonical question crop per detected top-level question.
5. Mark-scheme pairing uses the same identity to find the matching mark scheme and render the corresponding mark-scheme crop.
6. Export normalization writes canonical relative asset paths into `question_bank.json` and sidecars.
7. Asset validation and storage audits verify canonical paths, exact duplicates, and manifest references.

The canonical asset schema is flat by subject family: `output/pm1/*.png`, `output/pm3/*.png`, `output/stats/*.png`, and `output/mechanics/*.png`. Older `output/p1/<paper>/questions/` style paths are legacy compatibility inputs for normalization, not the active output contract.

## Output Layout

`process --output <root>` writes the canonical output layout:

```text
<root>/
  json/question_bank.json
  pm1/
  pm3/
  stats/
  mechanics/
  run_status/<run_id>/
```

Recommended generated-output layout:

```text
output/
  json/question_bank.json
  pm1/
  pm3/
  stats/
  mechanics/
  candidates/ocr/<run_id>/
  triage/iteration_###/
  audits/<audit_id>/
  asterion/exports/<run_id-or-latest>/
  asterion/reports/<run_id-or-latest>/
  run_status/<run_id>/
```

Long-running commands write status files by default:

```text
output/run_status/<run_id>/run_status.json
output/run_status/<run_id>/batch_status.jsonl
output/run_status/<run_id>/run_manifest.json
```

Use `--no-progress` for quiet terminal output while still writing status files. Use `--status-dir`, `--run-id`, `--resume`, and `--force-rerun` where supported for resumable runs.

Inventory generated roots:

```bash
.venv/bin/python -m exam_bank.cli output-inventory \
  --root output \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

Create a dry-run cleanup plan:

```bash
.venv/bin/python -m exam_bank.cli output-cleanup-plan \
  --root output \
  --write output/output_cleanup_plan.md
```

Audit exact storage duplicates and write the canonical asset manifest:

```bash
.venv/bin/python -m exam_bank.cli output-normalize-structure \
  --root output \
  --validate-only
.venv/bin/python scripts/audit_output_storage.py --dry-run
.venv/bin/python scripts/validate_asset_references.py
```

Storage policy lives in [`docs/OUTPUT_STORAGE_CONTRACT.md`](docs/OUTPUT_STORAGE_CONTRACT.md). Exact duplicates are identified by SHA-256, and cleanup apply mode quarantines files instead of deleting them.
Hard-delete cleanup is opt-in only:

```bash
.venv/bin/python scripts/audit_output_storage.py --apply-delete
```

This writes `reports/output_storage_delete_manifest.v1.json` before deleting allowlisted non-canonical exact duplicates from generated/candidate/archive/cache-style paths.

Cleanup planning does not delete or move files.

## JSON Contract

`output/json/question_bank.json` is a versioned document with `schema_name`, `schema_version`, `record_count`, `run_manifest`, and `questions`.

Important record fields include:

- `question_id`, `paper`, `paper_family`, `question_number`
- `question_image_path`, `question_image_paths`, `canonical_question_artifact`
- `mark_scheme_image_path`, `mark_scheme_image_paths`
- `question_text`, `question_text_role`, `question_text_trust`
- `ocr_ran`, `ocr_engine`, `ocr_text`, `ocr_text_trust`
- `visual_required`, `visual_reason_flags`, `visual_curation_status`, `text_only_status`
- `mark_scheme_text`, `question_solution_marks`, `subparts`, `subparts_solution_marks`
- `topic`, `difficulty`, `difficulty_score`, `difficulty_band`
- `notes.validation_status`, `notes.mapping_status`, `notes.text_fidelity_status`
- `notes.topic_trust_status`, `notes.text_source_profile`

Consumers should prefer top-level export-contract fields when present and use `notes` for pipeline diagnostics. If a top-level field and `notes` disagree, use the documented consumer contract for that field rather than assuming either text source is canonical.

## OCR And Text

Native PDF text extraction and OCR are evidence sources, not ground truth. The selector keeps native text unless OCR is clearly better and rejects OCR candidates that lose question numbers, expected subparts, mark brackets, or scope.

OCR-enabled exports should be compared against OCR-enabled baselines. No-OCR runs can still isolate layout and crop regressions, but they change the text-source profile and can distort readiness/audit movement.

Recent text-extraction work adds advisory review infrastructure rather than changing the source-of-truth policy:

- bad-text fixture reports under `docs/text_extraction/`
- crop/context warning audits
- OCR profile experiments
- advisory normalized text candidate contracts
- text-fidelity review queues and review state

These tools are for review, triage, and future candidate layers. They do not make extracted text canonical.

## Triage And Review

Create and inspect a deterministic hard-failure triage sample:

```bash
.venv/bin/python -m exam_bank.cli triage-sample \
  --input output/json/question_bank.json \
  --sample-size 30

.venv/bin/python -m exam_bank.cli triage-serve \
  --iteration output/triage/iteration_001

.venv/bin/python -m exam_bank.cli triage-compare \
  --iteration output/triage/iteration_001 \
  --current output/json/question_bank.json
```

Do not delete frozen triage baselines such as `output*/triage/iteration_*/baseline_question_bank.json`. Do not overwrite comparison files unless the replacement has a clearly named path and reason.

## Asterion And Sidecars

Asterion export files are downstream projections, not a replacement for canonical images. Consumers must honor the role gates in [`docs/ASTERION_EXPORT_CONTRACT.md`](docs/ASTERION_EXPORT_CONTRACT.md) and must not treat the all-course catalog as globally student-facing safe.

Course-aware consumers should use the centralized helpers in `src/exam_bank/asterion_course_contract.py`. The helpers map `p1 -> p1`, `p3 -> p3`, `p4 -> m1`, and `p5 -> s1`; filter by course, paper, or component; return empty arrays for scaffolded courses; and exclude Content Lab candidate payloads from student runtime. The Asterion catalog includes `course_id`, `component_name`, `question_image_path`, `mark_scheme_image_path`, `student_runtime_safe`, and `review_status` fields on each record, plus top-level course and component metadata for all four courses.

`asterion_exam_bank_catalog_v1.json` is the broad static-site catalog sidecar. It may include reviewed, needs-review, blocked, and candidate-state records for P1, P3, M1, and S1 when those records exist in the canonical question bank. It is the right input for review tools, catalog counts, and future course/component mapping work.

`asterion_question_bank_v1.json` is the student-facing runtime subset. It is derived from the catalog and contains only records where `student_runtime_safe=true` and `review_status=reviewed`. Legacy P3 canonical-practice behavior is preserved by mapping P3 `usage_roles.canonical_practice=allow` to `student_runtime_safe=true`. P1, M1, and S1 records do not become student runtime just because their historical Asterion role is `allow`; they need an explicit reviewed/safe promotion. If a course has no reviewed records, show `No reviewed exam-bank records available yet.` Do not backfill P1/M1/S1 with P3 questions.

Strict Asterion topic filters should use `output/json/question_bank.topic_routing.v1.json` only when it has been restored or verified from `data/topic_routing/question_bank.topic_routing.v1.json`, matches `data/topic_routing/question_bank.topic_routing.v1.sha256`, `metadata.run_summary.safe_for_strict_filters=true`, and the individual record is not review-required. The `output/json/` sidecar path is ignored/local; run `.venv/bin/python -m exam_bank.topic_routing_artifact restore` or `.venv/bin/python -m exam_bank.topic_routing_artifact verify` before regenerating Asterion exports. See [`docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`](docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md).

`output/asterion/exports/latest/*.json` files are ignored/generated release payloads. The tracked handoff evidence is a release manifest under `reports/`, created with `scripts/package_asterion_export_release.py`. The release packaging workflow is: restore/verify the durable topic sidecar, regenerate exports, validate exports, package the release manifest, then hand deployment the generated export files whose SHA-256 values match the manifest. Packaging does not change runtime behavior, runtime promotion, or auto-grade eligibility.

Mark-event evidence (`question_bank.mark_events.v1.json`) and advisory evidence (`output/advisory_evidence/question_bank.advisory_evidence.v1.json`) are deterministic support sidecars. They can support review, filtering, and difficulty evidence, but they must not replace canonical images, official mark-scheme crops, strict topic routing, or Asterion role gates. See [`docs/MARK_EVENTS_CONTRACT.md`](docs/MARK_EVENTS_CONTRACT.md) and [`docs/ADVISORY_EVIDENCE_CONTRACT.md`](docs/ADVISORY_EVIDENCE_CONTRACT.md).

The difficulty index sidecar (`question_bank.difficulty_index.v1.json`) is an advisory sort/filter aid. Its `difficulty_index_0_100` field is not a psychometric score, and its current version does not enable student-facing sequencing. See [`docs/DIFFICULTY_INDEX_CONTRACT.md`](docs/DIFFICULTY_INDEX_CONTRACT.md).

Broad AI enrichment sidecars such as `question_bank.deepseek.json` and `question_bank.ai_assisted.v2*.json` are review/debug evidence until a separate audit approves a narrower use.

Content Lab exports are review material only. `asterion_content_lab_candidates_v1.json` may help reviewers inspect blockers, mark-event evidence, and future generation prerequisites, but those candidates must not be loaded as student exam-bank records or used to create student-facing content.

## Topic Packets

Topic packets are image-first downstream projections. They build printable PDFs from canonical question and mark-scheme crops, not reconstructed OCR/native/AI text.

The normal packet workflow generates broad CAIE 9709 major-topic packets by paper family and topic:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --reviewed-decisions data/review/topic_bank_reviewed_decisions.v1.json \
  --artifact-root output \
  --strict-syllabus
```

Preview the full pass without writing PDFs or manifests:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --reviewed-decisions data/review/topic_bank_reviewed_decisions.v1.json \
  --artifact-root output \
  --strict-syllabus \
  --dry-run
```

Default outputs are written under `output/topic_packets/<paper_family>/<major_topic>/` with `topic_packet.pdf` and `manifest.json`, plus `output/topic_packets/topic_packet_summary.json`. Add `--split-question-answer-pdfs` when legacy `questions.pdf` and `answers.pdf` side outputs are needed. Weak text/OCR/topic/crop signals are warnings, not blockers; invalid topics, missing question images, mapping failures, and validation failures remain hard exclusions unless explicitly included for review.

## Downstream Use

Downstream student-facing apps should not blindly load all records. Treat records as image-ready, metadata-ready, or fully trusted only when the relevant mapping, validation, artifact, crop, text, topic, and role gates pass.

Records with failed mapping, failed validation, failed scope, missing image paths, degraded/unusable text, review-required topic routing, or blocked Asterion roles should stay in teacher/reviewer workflows until resolved.

## More Docs

- [Command atlas](docs/COMMAND_ATLAS.md)
- [Architecture](ARCHITECTURE.md)
- [Release validation checklist](docs/RELEASE_VALIDATION_CHECKLIST.md)
- [Current audit baseline](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md)
- [Asterion export contract](docs/ASTERION_EXPORT_CONTRACT.md)
- [Topic routing sidecar contract](docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md)
- [Mark-event evidence contract](docs/MARK_EVENTS_CONTRACT.md)
- [Advisory evidence contract](docs/ADVISORY_EVIDENCE_CONTRACT.md)
- [Difficulty index contract](docs/DIFFICULTY_INDEX_CONTRACT.md)
- [AI-assisted enrichment](docs/AI_ASSISTED_ENRICHMENT.md)
- [Triage workflow](docs/TRIAGE_WORKFLOW.md)
- [Auto-triage workflow](docs/AUTO_TRIAGE.md)
- [Trust model](docs/TRUST_MODEL.md)
- [Roadmap](ROADMAP.md)
