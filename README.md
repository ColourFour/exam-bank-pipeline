# CAIE 9709 Exam-Bank Extraction Pipeline

## Project Overview

This repository builds an image-first CAIE 9709 exam-bank dataset from question-paper PDFs and mark-scheme PDFs.

The pipeline scans PDFs, detects top-level questions, renders question crops, renders matching mark-scheme crops, pairs those artifacts, exports structured JSON, and produces triage artifacts for repeatable review. The image crops are the source of truth. Extracted text is metadata for search, routing, topic labeling, review, enrichment, and downstream app behavior; it is trust-gated and must not be treated as a perfect transcription of the paper.

The active runtime path is the extraction pipeline plus evidence-gated triage. DeepSeek enrichment is a separate sidecar step that does not mutate `question_bank.json`. Topic labels, OCR text, and extracted text are secondary metadata in the export; none of them override image evidence.

Rendered question PNGs and rendered mark-scheme PNGs remain the student-facing/canonical source of truth. Native PDF text, OCR text, topic labels, difficulty labels, trust flags, crop confidence, validation fields, mapping fields, and readiness fields are support metadata for search, validation, review, and future adaptive practice.

## Current Audited State

Measured on `output/json/question_bank.json` on 2026-05-10 with:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/iteration_001
```

- Corpus: `1301` question records.
- Paper families: `p1: 401`, `p3: 396`, `p4: 258`, `p5: 246`.
- Schema: `exam_bank.question_bank` version `2`, with declared and actual record count both `1301`.
- The schema/export model is candidate-aware and can carry OCR candidate fields, crop confidence, validation, mapping, and readiness fields. Historical/uploaded OCR-enabled schema v2 JSONs may populate fields such as `ocr_ran`, `ocr_text`, `text_candidate_source`, crop confidence, validation, mapping, and readiness metadata.
- Current canonical export is candidate-aware but not OCR-enabled: `ocr_ran=false` for all records, `text_candidate_source=native` for all records, `text_candidate_decision=native_retained` for all records, and `ocr_selected=false` for all records.
- OCR selection quality, OCR false positives, and OCR false negatives cannot be meaningfully judged from the current canonical export because `ocr_text` and `ocr_engine` are blank for all records.
- Current validation: `pass: 921`, `review: 371`, `fail: 9`.
- Current mark-scheme mapping (`notes.mapping_status`): `pass: 1281`, `fail: 20`.
- Current text fidelity: `clean: 1284`, `degraded: 17`.
- Current visual curation: `ready: 681`, `review: 611`, `fail: 9`.
- Current text-only status: `ready: 228`, `review: 1047`, `fail: 26`.
- Current Asterion readiness tiers: Tier 0 `23`, Tier 1 `360`, Tier 2 `185`, Tier 3 `13`, Tier 4 `52`, Tier 5 `668`.
- Current hard blockers: `23`. Dominant blockers are mapping failures, validation failures, missing mark-scheme text/image paths, and local question/mark-scheme total mismatches.
- Subpart marks are not yet promoted: `968` records have subparts with all subpart marks null, and `920` records appear simple-fillable from detected mark brackets.
- New exports include a top-level `run_manifest` with `generated_at`, `run_id`, `pipeline_version`, `git_commit`, `model_versions`, `ocr_engine_version`, `input_manifest_sha256`, `artifact_root`, and `qa_summary`.

The latest readiness audit outputs are written under `output/audits/iteration_001/`. This folder is generated output and is ignored by git.

The latest baseline comparison used `output/triage/iteration_004/baseline_question_bank.json`: `1301` records matched by `question_id`, no records were added or removed, Asterion tier movement was `improved: 630`, `unchanged: 671`, and the only worsened status counts were `mapping_status: 1`, `mark_scheme_crop_confidence: 11`, and `text_fidelity_status: 1`.

The older OCR-enabled candidate at `output_ocr_candidate/json/question_bank.json` remains useful as historical evidence, but `iteration_001` proves the audit/reporting layer, not OCR-selection quality. The current ordering is: focused audit tests, OCR activation/export-disconnect investigation, hard blockers, 2024-2025 layout/crops, then subpart marks and canonical text work.

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

For comparison or review candidates, write OCR-enabled output to a separate folder instead of replacing the current canonical export:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```

The preferred candidate home for new runs is:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output/candidates/ocr/latest \
  --enable-ocr
```

`output_ocr_candidate/` remains a supported compatibility root for historical commands and comparisons.

Long-running commands write live status files and show a built-in text progress bar by default. Standard process runs write:

```text
output/run_status/<run_id>/run_status.json
output/run_status/<run_id>/batch_status.jsonl
output/run_status/<run_id>/run_manifest.json
```

Use `--no-progress` to silence terminal updates while still writing status files. Use `--status-dir`, `--run-id`, `--resume`, and `--force-rerun` for resumable batch runs.

Strict topic routing is a narrow DeepSeek pass that writes a compact sidecar without mutating `question_bank.json`. Progress is visible by default:

```bash
set -a; source .env; set +a

.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output output/json/question_bank.topic_routing.v1.json \
  --model deepseek-v4-flash \
  --status-dir output/run_status
```

Resume the same sidecar with:

```bash
.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output output/json/question_bank.topic_routing.v1.json \
  --model deepseek-v4-flash \
  --status-dir output/run_status \
  --resume
```

Add `--no-progress` only for quiet mode; status files are still written.

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

## Generated Output Layout

The current compatibility contract keeps `process --output <root>` working exactly as before: the selected output root contains paper artifact folders, `json/question_bank.json`, and `run_status/<run_id>/`. New `question_bank.json` manifests include an `output_layout` block that labels the profile, such as `canonical`, `ocr-candidate`, or `legacy`.

Recommended layout for new generated output:

```text
output/
  json/question_bank.json                 # canonical compatibility path
  p1/ p3/ p4/ p5/                         # canonical compatibility artifacts
  current/                                # pointer/copy area for future canonical promotion
  runs/<run_id>/                          # archival canonical run roots
  candidates/ocr/<run_id>/                # OCR candidate run roots
  triage/iteration_###/
    baseline_question_bank.json           # frozen; never edit in place
    summary.json
    sample.json
    review.jsonl
    index.html
    comparisons/
  audits/<audit_id>/
  asterion/exports/<run_id-or-latest>/
  asterion/reports/<run_id-or-latest>/
```

Canonical image artifacts are the `p*/<paper>/questions/*.png` and `p*/<paper>/mark_scheme/*.png` files under the output root that produced the JSON. `question_bank.json`, OCR text, topic labels, readiness tiers, audit CSVs, and Asterion JSONs are metadata or downstream projections.

Frozen triage baselines must not be deleted, moved, or overwritten: `output*/triage/iteration_*/baseline_question_bank.json`. Generated reports, candidate folders, run-status folders, and inventories can be regenerated, but archive or review them first when they support an active comparison.

Git should ignore generated output roots and local inventory reports: `output/`, `output_ocr_candidate/`, `repo_file_inventory.txt`, `generated_output_inventory.txt`, `output_inventory.*`, and `output_cleanup_plan.md`. Commit only source, tests, docs, schemas, and intentionally small fixtures or snapshots.

Inventory generated output roots:

```bash
.venv/bin/python -m exam_bank.cli output-inventory \
  --root output \
  --root output_ocr_candidate \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

Create a dry-run cleanup plan:

```bash
.venv/bin/python -m exam_bank.cli output-cleanup-plan \
  --root output \
  --root output_ocr_candidate \
  --write output/output_cleanup_plan.md
```

Cleanup planning does not delete or move files.

### Where Did My Files Go?

- Canonical compatibility JSON: `output/json/question_bank.json`.
- Canonical compatibility crops: `output/p*/<paper>/questions/` and `output/p*/<paper>/mark_scheme/`.
- OCR candidates: historical `output_ocr_candidate/` or preferred `output/candidates/ocr/<run_id>/`.
- Triage baselines: `output*/triage/iteration_*/baseline_question_bank.json`.
- New triage comparisons: `output*/triage/iteration_*/comparisons/`.
- Readiness audits: `output/audits/<audit_id>/`.
- Asterion exports: `output*/asterion/exports/latest/` unless `--output` is supplied.
- Run status: `<selected-output-root>/run_status/<run_id>/`.

## What The JSON Contains

`output/json/question_bank.json` is a versioned document with `schema_name`, `schema_version`, `record_count`, `run_manifest`, and `questions`.

The top-level `run_manifest` records export provenance and corpus QA rollups:

- `generated_at`, `run_id`, `pipeline_version`, `git_commit`
- `model_versions`, `ocr_engine_version`
- `input_manifest_sha256`, `artifact_root`
- `qa_summary` counts for paper families, validation status, mapping status, text fidelity, curation status, OCR usage, and missing artifact paths

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

The export contract supports OCR candidate fields, but the current canonical audited export has `ocr_ran=0`, `ocr_selected=0`, `text_candidate_source=native: 1301`, and blank `ocr_text`. OCR-selection quality cannot yet be evaluated from that export.

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

## Auto-Triage Loop

Auto-triage wraps the manual triage loop with corpus metrics, selected-target planning, agent handoff files, runbook generation, and an acceptance decision. It does not edit code or data by itself.

Measure a corpus:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status \
  --input output/candidates/ocr/latest/json/question_bank.json
```

Create the next handoff iteration until the hard-failure target is met:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-plan \
  --input output/candidates/ocr/latest/json/question_bank.json \
  --handoff-root agent_handoffs/auto_triage \
  --candidate-output output/candidates/ocr/latest \
  --target-max-hard-failures 100 \
  --sample-size 30
```

Print the runbook for a handoff created by the planner:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-runbook \
  --input output/candidates/ocr/latest/json/question_bank.json \
  --candidate-output output/candidates/ocr/latest \
  --handoff-root agent_handoffs/auto_triage
```

For older or non-default handoffs, pass `--iteration` and `--baseline-triage` explicitly so generated comparison paths match the intended frozen baseline.

After focused implementation, full tests, and an OCR-enabled rerun, compare and write the decision:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-compare \
  --iteration agent_handoffs/auto_triage/iteration_003 \
  --baseline-triage output_ocr_candidate/triage/iteration_002 \
  --current output_ocr_candidate/json/question_bank.json \
  --output output_ocr_candidate/triage/iteration_002/comparisons/comparison.auto-iteration-003.json \
  --test-status pass
```

Accepted auto-triage iterations require passing tests, OCR-enabled current and baseline outputs, a hard-failure or selected-target decrease, no `worsened_records`, no major unrelated status regression, and no broad validation or trust-gate loosening without extraction evidence.

## Audits

General visual/text trust audit:

```bash
python -m exam_bank.cli audit --input output/json/question_bank.json
```

Current output integrity audit:

```bash
.venv/bin/python -m exam_bank.cli output-integrity-audit --input output/json/question_bank.json --artifact-root output
```

This fails on duplicate IDs, duplicate `(paper, question_number)` pairs, missing question images, missing nonblank mark-scheme images, absolute image paths, or unexpected missing mark-scheme paths. The only documented missing mark-scheme companion is `9709_2025_November_33`.

OCR candidate audit:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json
```

Full readiness and extraction-layer audit:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/iteration_001
```

No-baseline variant:

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --artifact-root output \
  --out-dir output/audits/manual
```

`output/audits/` is generated output under ignored `output/` and should stay uncommitted unless a human explicitly chooses to version a report snapshot.

The readiness audit writes:

- `audit_summary.md` and `audit_summary.json`: human and machine summaries.
- `field_presence_report.csv`: field presence, source policy, and measurement blockers.
- `ocr_candidate_audit.csv`: per-record native/OCR/selected candidate metrics.
- `ocr_suspicious_records.csv`: selected-OCR risk cases, if any.
- `possible_ocr_false_negatives.csv`: cases where OCR looks better but was not selected, if measurable.
- `readiness_tiers.csv`: per-record Asterion tier and blocker/review reasons.
- `hard_blockers.csv`: records with hard blocker reasons.
- `crop_quality_report.csv`: crop/readiness summaries by paper, family, profile, and season.
- `mapping_validation_report.csv`: mapping, validation, mark-scheme, and contradiction checks.
- `subpart_marks_report.csv`: missing/simple-fillable/nested subpart mark analysis.
- `representative_review_sample.csv`: deterministic review sample.
- `next_iteration_recommendations.md`: generated next-step recommendations.
- `baseline_comparison.csv` and `baseline_comparison_summary.json`: only when `--baseline` is supplied.

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

## Asterion Readiness

Asterion handoff should be tiered, not all-or-nothing. The conservative projection commands write under `output*/asterion/exports/latest/` by default:

```bash
.venv/bin/python -m exam_bank.cli asterion-export \
  --input output/json/question_bank.json

.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates \
  --input output/json/question_bank.json
```

The current consumer contract for the question-bank projection and Content Lab candidate projection is documented in [`docs/ASTERION_EXPORT_CONTRACT.md`](docs/ASTERION_EXPORT_CONTRACT.md). Asterion consumers must honor the role gates in those exports and must not treat the full projection as globally student-facing safe.

Future Asterion-facing exports should be separated into slices such as:

```text
asterion_gold.json
asterion_multimodal.json
question_bank_master.json
```

Until tiered exports are implemented, downstream users should treat image artifacts as canonical unless a record is explicitly text-ready.

Topic routing for Asterion should use the strict DeepSeek topic-routing sidecar, not the broad AI-assisted v2 enrichment sidecar. The current sidecar target is:

```text
output/json/question_bank.topic_routing.v1.json
```

This sidecar is intentionally narrow: it records only canonical parent-topic routing from `exam_bank_taxonomy/canonical/`, with `primary_topic_id`, `topic_distribution`, confidence, review flags, and the limited text evidence actually sent to the model. It does not contain difficulty, skills, subtopics, rationales, Content Lab metadata, or Asterion readiness decisions. Records marked `review_required=true` must not enter strict Asterion topic filters.

Normal full run:

```bash
set -a; source .env; set +a

.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output output/json/question_bank.topic_routing.v1.json \
  --model deepseek-v4-flash \
  --status-dir output/run_status
```

Resume:

```bash
.venv/bin/python -m exam_bank.cli topic-route-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --output output/json/question_bank.topic_routing.v1.json \
  --model deepseek-v4-flash \
  --status-dir output/run_status \
  --resume
```

Use `question_bank.ai_assisted.v2*.json` only as review/debug evidence for now. The broad v2 attempt includes a larger task surface and is not the recommended Asterion strict-filter input unless a separate audit explicitly approves it.

The readiness audit currently classifies records as:

- Tier 0 - Hard blocker: any hard blocker, including missing IDs or image/text paths, mapping or validation failure, local question/mark-scheme total disagreement, unusable marks, scope failure, truncation/contamination flags, or missing artifact files when `--artifact-root` is supplied.
- Tier 1 - Master/review only: no hard blocker, but not enough evidence for Tier 2.
- Tier 2 - Asterion multimodal candidate: no hard blockers, mapping pass, validation pass, question image path, mark-scheme image path, mark-scheme text, and question/mark-scheme totals agreeing when both are present.
- Tier 3 - Asterion visual-ready: Tier 2 plus visual curation ready, high question and mark-scheme crop confidence, and clean scope status.
- Tier 4 - Asterion text-ready: Tier 2 plus text-only ready, high question-text trust, clean text fidelity, question number present in selected text, expected subparts present, mark brackets present, and `visual_required=false`.
- Tier 5 - Gold pilot set: implemented as Tier 3 plus Tier 4, or Tier 3 visual-required records with clean text fidelity and no contradiction flags.

## Quality Gates

The high-level gates currently audited are:

- `mapping_status`
- `validation_status`
- question image path present
- mark-scheme image path present
- mark-scheme text present where required
- question and mark-scheme totals agree when both are present
- question and mark-scheme crop confidence
- visual curation status
- text-only status
- question text trust
- artifact path existence when `--artifact-root` is supplied

## DeepSeek Sidecar

DeepSeek/topic enrichment is secondary sidecar metadata. It does not change extraction and does not mutate `question_bank.json`.

Current recommendation:

- Use `output/json/question_bank.topic_routing.v1.json` for audited Asterion topic routing after a successful `topic-route-ai` run.
- Treat `output/json/question_bank.deepseek.json` as the older broad suggestion sidecar.
- Treat `output/json/question_bank.ai_assisted.v2*.json` as broad review/debug evidence until it passes a separate audit. Do not promote broad v2 failures, suggestions, skills, subtopics, rationales, or difficulty fields into strict Asterion filters.

```bash
export DEEPSEEK_API_KEY=...
python -m exam_bank.deepseek_enrich \
  --input output/json/question_bank.json \
  --output output/json/question_bank.deepseek.json \
  --limit 25
```

Current sidecar evidence in `output/json/question_bank.deepseek.json` has `1301` enrichment entries, with `1246` marked `final_review_required=True`, `44` marked `False`, and `11` without that field because provider failures were logged. Treat this as review/enrichment evidence, not canonical truth.

The broad AI-assisted v2 pass enriches against the active canonical topic, subtopic, and skill IDs under `exam_bank_taxonomy/canonical/`. It preserves v1 sidecar evidence where useful, batches by paper, supports resume/retry, and computes deterministic difficulty percentiles within each paper family. This is currently a review/debug path, not the preferred Asterion topic-routing path:

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python -m exam_bank.cli enrich-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --existing-sidecar output/json/question_bank.deepseek.json \
  --output output/json/question_bank.ai_assisted.v2.json \
  --component p3 \
  --limit 20 \
  --resume \
  --status-dir output/run_status \
  --model deepseek-v4-flash \
  --include-subparts \
  --recompute-difficulty
```

Strict product filters should use only audited canonical IDs from approved sidecars. For topic filters, use non-review-required records from `question_bank.topic_routing.v1.json`. For broad v2, suggestions for new subtopics or skills are review-only, and `strict_filter_candidates` must be audited before any Asterion use. See `docs/AI_ASSISTED_ENRICHMENT.md` for the current sidecar policy.

## Tests

Run the full suite with:

```bash
.venv/bin/python -m pytest
```

Current focused validation set for the readiness-audit documentation pass:

```bash
.venv/bin/python -m pytest tests/test_audit.py tests/test_ocr.py tests/test_output_contract.py tests/test_extraction_structure.py -q
```

Focused reporting tests for `scripts/audit_question_bank_readiness.py` are expected as the Agent 3 gate. In the current worktree they live in `tests/test_question_bank_readiness_audit.py`; include them in validation when that file is present:

```bash
.venv/bin/python -m pytest tests/test_question_bank_readiness_audit.py -q
```

Latest full-suite run in this review:

```bash
.venv/bin/python -m pytest -q
```

Result: `337 passed, 3 skipped in 112.07s`.

Coverage is strongest around document classification, PDF layout extraction, question detection, crop-region behavior, mark-scheme mapping, OCR candidate selection, trust derivation, output contract, triage comparison, auto-triage planning/decision gates, DeepSeek sidecar behavior, and representative sample-pipeline regressions. Gaps remain around full-corpus runtime assertions, source-pairing mismatch gates, visual pixel-level crop review, and a trusted-subset export.

## Near-Term Roadmap

The immediate roadmap is extraction-readiness work, not downstream app features:

- `iteration_001`: audit/reporting layer completed by Agent 2.
- `iteration_002`: Agent 3 focused tests for the reporting script.
- `iteration_003`: resolve why the current canonical export has `ocr_ran=0`, blank `ocr_text`, and `text_candidate_source=native: 1301`.
- `iteration_004`: fix hard blockers, especially mapping failures, validation failures, missing mark-scheme text/images, validation/mapping contradictions, and local mark-total mismatches.
- `iteration_005`: dedicated 2024-2025 layout/crop recovery.
- `iteration_006`: crop metadata and artifact QA.
- `iteration_007`: promote subpart marks, including nested part paths.
- `iteration_008`: canonical text candidate system after OCR activation/export state is resolved.
- `iteration_009`: run manifest and export contract.
- `iteration_010`: tiered Asterion exports.
- `iteration_011`: topic/difficulty rerun only after crop, text, mapping, and mark-structure quality improve.

See `ROADMAP.md` for the detailed iteration targets.

## Downstream Use

Downstream student-facing apps should not blindly load all records. Use trust tiers:

- Image-ready: question crop exists, mark-scheme crop exists, mapping passes, scope is acceptable, and visual curation is ready.
- Metadata-ready: image-ready plus clean text fidelity, trusted topic status, marks, and subparts.
- Fully trusted practice item: image, mapping, marks, text, topic, and scope all pass, ideally with review or strong automated confidence.

Records with failed mapping, failed validation, failed scope, missing image files, source-pairing mismatch, degraded/unusable text, or review-required topic status should stay in teacher/reviewer workflows until resolved.

## More Docs

- [Project review](docs/PROJECT_REVIEW.md)
- [Triage workflow](docs/TRIAGE_WORKFLOW.md)
- [Auto-triage workflow](docs/AUTO_TRIAGE.md)
- [Trust model](docs/TRUST_MODEL.md)
- [Roadmap](ROADMAP.md)
