# CAIE 9709 Exam-Bank Extraction Pipeline

This repository builds an image-first CAIE 9709 exam-bank dataset from question-paper PDFs and mark-scheme PDFs.

The pipeline scans PDFs, detects top-level questions, renders question crops, renders matching mark-scheme crops, pairs those artifacts, exports structured JSON, and produces triage artifacts for repeatable review. The image crops are the source of truth. Extracted text is metadata for search, routing, topic labeling, review, enrichment, and downstream app behavior; it is trust-gated and must not be treated as a perfect transcription of the paper.

## Current Status

Measured on `output/json/question_bank.json` on 2026-05-08:

- Corpus: `1301` question records from `148` question-paper PDFs.
- Inputs: `148` question-paper PDFs and `147` mark-scheme PDFs. The registry reports one missing companion mark scheme: `9709 Mathematics November 2025 Question Paper  33.pdf`.
- Paper families: `p1: 401`, `p3: 396`, `p4: 258`, `p5: 246`.
- Images: `1301` question image paths exist; one referenced mark-scheme image file is missing (`12autumn21_q12`).
- Current export OCR state: `ocr_ran=False` for all `1301` records; `notes.text_source_profile=native_pdf` for all records.
- Current validation: `pass: 833`, `review: 320`, `fail: 148`.
- Current mark-scheme mapping (`notes.mapping_status`): `pass: 1233`, `fail: 68`.
- Current text fidelity: `clean: 1245`, `degraded: 48`, `unusable: 8`.
- Current scope quality: `clean: 919`, `review: 376`, `fail: 6`.
- Current visual curation: `ready: 663`, `review: 574`, `fail: 64`.
- Current text-only status: `ready: 217`, `review: 976`, `fail: 108`.
- Source-pairing audit: `11` records, all `33autumn25`, point at a `12autumn21` mark-scheme source; `10` fail mapping and `1` currently passes. Treat these as not student-ready until fixed.

The latest comparison against the iteration 004 frozen baseline is `output/triage/iteration_004/comparison.layout-review-current.json`: hard failures moved from `259` to `148` (`-111`), target `question_scope_contaminated` moved from `114` to `4` (`-110`), and `worsened_records` is `0`. This comparison uses the current no-OCR export, so it is useful for layout debugging but is not the canonical production OCR score.

The latest OCR-to-OCR comparison is `output/triage/iteration_003/comparison.math-repair-ocr.json`: hard failures moved from `385` to `259` (`-126`), target `polluted_pass_requires_review` moved from `126` to `3` (`-123`), and `worsened_records` is `0`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

OCR-enabled production-style runs require Tesseract:

```bash
brew install tesseract
.venv/bin/python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
tesseract --version
```

## Run The Pipeline

The supported extraction front door is:

```bash
python -m exam_bank.cli process --input input --output output
```

Production-style runs should enable OCR so OCR candidate metadata and hybrid text profiles are populated:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output --enable-ocr
```

`--input` is scanned recursively. The usual layout is:

```text
input/
  question_papers/
  mark_schemes/
  mappings/        # optional
```

The canonical JSON export is:

```text
output/json/question_bank.json
```

The paper-first image tree is:

```text
output/
  p1/
    12spring21/
      questions/q01.png
      mark_scheme/q01.png
  p3/
  p4/
  p5/
```

Paper folders use `{component}{season}{yy}`, for example `12spring21`, `33summer24`, and `53autumn25`.

## What The JSON Contains

`output/json/question_bank.json` is a versioned document with `schema_name`, `schema_version`, `record_count`, and `questions`.

Important record fields include:

- `question_id`, `paper`, `paper_family`, `question_number`
- `question_image_path`, `question_image_paths`, `canonical_question_artifact`
- `mark_scheme_image_path`, `mark_scheme_image_paths`
- `page_refs`
- `question_text`, `question_text_role`, `question_text_trust`
- `ocr_ran`, `ocr_engine`, `ocr_text`, `ocr_text_trust`
- `visual_required`, `visual_reason_flags`, `visual_curation_status`, `text_only_status`
- `mark_scheme_text`, `question_solution_marks`, `subparts`, `subparts_solution_marks`
- `topic`, `difficulty`, `difficulty_score`, `difficulty_band`
- `notes.validation_status`, `notes.validation_flags`
- `notes.mapping_status`, `notes.mapping_failure_reason`
- `notes.scope_quality_status`, `notes.text_fidelity_status`, `notes.text_fidelity_flags`
- `notes.topic_trust_status`, `notes.text_source_profile`
- `notes.review_flags`, `notes.extraction_quality_flags`

Some trust fields currently live under `notes` rather than as top-level fields. Consumers should read the top-level field when present and otherwise use `notes`.

## OCR And Text

Native PDF text extraction and OCR are both evidence sources, not ground truth. The selector keeps native text unless OCR is clearly better and avoids OCR with hard rejection reasons such as missing question numbers, missing expected subparts, lost mark brackets, page furniture, or scope failure.

OCR-enabled exports must be compared only against OCR-enabled baselines. No-OCR runs can be useful for isolating layout and crop regressions, but they change the text-source profile and can make hard-failure counts look better or worse for reasons unrelated to the production profile.

## Triage Loop

Use triage to freeze a baseline, sample the largest failure cluster, review examples visually, rerun the full pipeline, and compare movement:

```bash
python -m exam_bank.cli triage-sample --input output/json/question_bank.json --sample-size 30
python -m exam_bank.cli triage-serve --iteration output/triage/iteration_001
python -m exam_bank.cli triage-compare --iteration output/triage/iteration_001 --current output/json/question_bank.json
```

`triage-sample` creates:

```text
output/triage/iteration_001/
  baseline_question_bank.json
  summary.json
  sample.json
  index.html
  review.jsonl
```

Do not delete frozen baselines. Do not overwrite comparison files unless the new output has a clearly named comparison path.

## Audits

General visual/text trust audit:

```bash
python -m exam_bank.cli audit --input output/json/question_bank.json
```

OCR candidate audit:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json
```

Difficulty metadata audit:

```bash
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json
```

Quality gate against a frozen triage iteration:

```bash
.venv/bin/python scripts/quality_gate.py \
  --iteration output/triage/iteration_004 \
  --current output/json/question_bank.json \
  --require-target-improvement
```

## DeepSeek Sidecar

DeepSeek/topic enrichment is secondary sidecar metadata. It does not change extraction and does not mutate `question_bank.json`.

```bash
export DEEPSEEK_API_KEY=...
python -m exam_bank.deepseek_enrich \
  --input output/json/question_bank.json \
  --output output/json/question_bank.deepseek.json \
  --limit 25
```

Current sidecar evidence in `output/json/question_bank.deepseek.json` has `1301` enrichment entries, with `1246` marked `final_review_required=True`, `44` marked `False`, and `11` without that field because provider failures were logged. Treat this as review/enrichment evidence, not canonical truth.

## Tests

Run the suite with:

```bash
.venv/bin/python -m pytest
```

Latest pre-documentation run in this review: `293 passed, 3 skipped in 111.54s`.

Coverage is strongest around document classification, PDF layout extraction, question detection, crop-region behavior, mark-scheme mapping, OCR candidate selection, trust derivation, output contract, triage comparison, DeepSeek sidecar behavior, and representative sample-pipeline regressions. Gaps remain around full-corpus runtime assertions, source-pairing mismatch gates, visual pixel-level crop review, and a trusted-subset export.

## Downstream Use

Downstream student-facing apps should not blindly load all records. Use trust tiers:

- Image-ready: question crop exists, mark-scheme crop exists, mapping passes, scope is acceptable, and visual curation is ready.
- Metadata-ready: image-ready plus clean text fidelity, trusted topic status, marks, and subparts.
- Fully trusted practice item: image, mapping, marks, text, topic, and scope all pass, ideally with review or strong automated confidence.

Records with failed mapping, failed validation, failed scope, missing image files, source-pairing mismatch, degraded/unusable text, or review-required topic status should stay in teacher/reviewer workflows until resolved.

## More Docs

- [Project review](docs/PROJECT_REVIEW.md)
- [Triage workflow](docs/TRIAGE_WORKFLOW.md)
- [Trust model](docs/TRUST_MODEL.md)
- [Roadmap](docs/ROADMAP.md)
