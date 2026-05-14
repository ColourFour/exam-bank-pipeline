# CAIE 9709 Exam-Bank Extraction Pipeline

This repository builds an image-first CAIE 9709 exam-bank dataset from question-paper PDFs and mark-scheme PDFs.

The pipeline scans PDFs, detects top-level questions, renders question crops, renders matching mark-scheme crops, pairs those artifacts, exports structured JSON, and writes review/audit sidecars. The canonical question PNGs and mark-scheme PNGs are the source of truth. Native PDF text, OCR text, AI enrichment, topic routing, readiness tiers, and Asterion projections are advisory metadata unless a specific consumer role gate says otherwise.

## Current Baseline

The current project-state baseline is [`docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md`](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md). Use that report for measured counts, current generated-output names, known risks, and test status instead of copying count snapshots into this README.

As of that audit, dated `2026-05-14`, the current canonical export is `output/json/question_bank.json` from run `20260513T070200Z-56d469c1dd52`. It is OCR-enabled: OCR ran for the canonical export and OCR was selected over native text for a small subset of records. This supersedes older README text that described the current export as native-only.

Current important generated artifacts:

- Canonical question bank: `output/json/question_bank.json`
- Canonical image trees: `output/p*/<paper>/questions/*.png` and `output/p*/<paper>/mark_scheme/*.png`
- Strict topic-routing sidecar: `output/json/question_bank.topic_routing.v1.json`
- Asterion projection: `output/asterion/exports/latest/asterion_question_bank_v1.json`
- Content Lab candidates: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`

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

## Command Atlas

Use [`docs/COMMAND_ATLAS.md`](docs/COMMAND_ATLAS.md) as the current command map. It covers standard and OCR extraction, resume behavior, audits, Asterion and Content Lab projections, topic routing, AI enrichment, AI sidecar audit, output inventory, cleanup planning, and test commands.

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

Standard full extraction:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output
```

## Output Layout

`process --output <root>` writes the compatibility layout:

```text
<root>/
  json/question_bank.json
  p1/ p3/ p4/ p5/
  run_status/<run_id>/
```

Recommended generated-output layout:

```text
output/
  json/question_bank.json
  p1/ p3/ p4/ p5/
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

Asterion export files are downstream projections, not a replacement for canonical images. Consumers must honor the role gates in [`docs/ASTERION_EXPORT_CONTRACT.md`](docs/ASTERION_EXPORT_CONTRACT.md) and must not treat the full projection as globally student-facing safe.

Strict Asterion topic filters should use `output/json/question_bank.topic_routing.v1.json` only when `metadata.run_summary.safe_for_strict_filters=true` and only for records that are not review-required. See [`docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md`](docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md).

Broad AI enrichment sidecars such as `question_bank.deepseek.json` and `question_bank.ai_assisted.v2*.json` are review/debug evidence until a separate audit approves a narrower use.

## Downstream Use

Downstream student-facing apps should not blindly load all records. Treat records as image-ready, metadata-ready, or fully trusted only when the relevant mapping, validation, artifact, crop, text, topic, and role gates pass.

Records with failed mapping, failed validation, failed scope, missing image paths, degraded/unusable text, review-required topic routing, or blocked Asterion roles should stay in teacher/reviewer workflows until resolved.

## More Docs

- [Command atlas](docs/COMMAND_ATLAS.md)
- [Release validation checklist](docs/RELEASE_VALIDATION_CHECKLIST.md)
- [Current audit baseline](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md)
- [Asterion export contract](docs/ASTERION_EXPORT_CONTRACT.md)
- [Topic routing sidecar contract](docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md)
- [AI-assisted enrichment](docs/AI_ASSISTED_ENRICHMENT.md)
- [Triage workflow](docs/TRIAGE_WORKFLOW.md)
- [Auto-triage workflow](docs/AUTO_TRIAGE.md)
- [Trust model](docs/TRUST_MODEL.md)
- [Roadmap](ROADMAP.md)
