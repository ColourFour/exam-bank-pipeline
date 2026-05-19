# Command Atlas

This is the operator-facing command map for the current project. Commands are copy-pasteable from the repository root after installing the dev environment:

```bash
source .venv/bin/activate
```

Long-running commands write status under `output/run_status/` unless a command-specific output root is used. AI workflows are sidecar-only: they must not overwrite canonical extraction truth, and their outputs remain advisory until separately audited and role-gated.

## Summary

| Workflow | Command family | Category | Runtime | Mutates outputs |
| --- | --- | --- | --- | --- |
| Standard full run | `process` | Standard | Long | Yes |
| OCR-enabled run | `process --enable-ocr` | OCR | Long | Yes |
| Resume/cache-aware run | `process --resume` | Standard/OCR | Depends on cache | Yes |
| Audit | `audit`, `output-integrity-audit`, readiness script | Audit-only | Fast to medium | Optional report writes |
| OCR candidate audit | `scripts/audit_ocr_candidates.py` | Audit-only/OCR | Fast | Optional report write |
| Difficulty audit | `scripts/audit_difficulty.py` | Audit-only | Fast | Optional report write |
| Mark-event sidecar | `scripts/build_mark_events.py`, `scripts/validate_mark_events.py` | Advisory sidecar | Fast | Sidecar/report writes |
| Asterion export | `asterion-export` | Standard projection | Fast to medium | Yes |
| Content Lab candidates | `asterion-content-lab-candidates` | Standard projection | Fast to medium | Yes |
| Topic packets | `topic-packets` | Standard projection | Fast to medium | Yes |
| Topic routing | `topic-route-ai` | AI-heavy, audit/sidecar | Long-running | Sidecar only |
| AI enrichment | `enrich-ai` | AI-heavy, audit/sidecar | Long-running | Sidecar only |
| AI sidecar audit | `ai-sidecar-audit` | Audit-only | Fast | No |
| Output inventory | `output-inventory` | Audit-only | Fast to medium | Optional report writes |
| Output cleanup plan | `output-cleanup-plan` | Audit-only | Fast to medium | Optional report writes |
| Full tests | `pytest` | Test | About 2 minutes observed | No |
| Fast local tests | `pytest -m "not integration and not rendering"` | Test | Fast | No |
| Rendering tests | `pytest -m rendering` | Test | Fast | No |
| Integration/sample-pipeline tests | `pytest -m "integration"` | Test | About 2 minutes observed | No |
| Targeted tests | `pytest <files>` | Test | Fast | No |

## Extraction

### Standard Full Run

Purpose: build the canonical image-first question bank from PDFs without OCR.

Input: `input/`, `config.yaml`

Output: `output/json/question_bank.json`, `output/p*/...` image artifacts, `output/run_status/<run_id>/`

Category/runtime: standard, long

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output \
  --config config.yaml
```

### OCR-Enabled Run

Purpose: build the production-style OCR-enabled export. OCR text is evidence, not ground truth.

Input: `input/`, `config.yaml`, local Tesseract installation

Output: `output/json/question_bank.json`, image artifacts, OCR metadata, `output/run_status/<run_id>/`

Category/runtime: OCR, long

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output \
  --config config.yaml \
  --enable-ocr
```

### OCR Candidate Run

Purpose: create an OCR-enabled candidate without replacing the canonical `output/json/question_bank.json`.

Input: `input/`, `config.yaml`, local Tesseract installation

Output: `output/candidates/ocr/latest/json/question_bank.json`, candidate image artifacts, candidate run status

Category/runtime: OCR, long

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output/candidates/ocr/latest \
  --config config.yaml \
  --enable-ocr
```

### Resume/Cache-Aware Run

Purpose: resume a previous extraction and skip completed batches. Add `--enable-ocr` when resuming an OCR-enabled output.

Input: `input/`, existing `output/run_status/<run_id>/`

Output: updated output root and run status

Category/runtime: standard or OCR, depends on completed batches

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output \
  --config config.yaml \
  --resume
```

OCR-enabled resume:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output \
  --config config.yaml \
  --enable-ocr \
  --resume
```

## Audits

### Current Export Audit

Purpose: summarize visual-first text trust and curation readiness.

Input: `output/json/question_bank.json`

Output: terminal JSON, optionally `output/json/audit.current.json`

Category/runtime: audit-only, fast

```bash
.venv/bin/python -m exam_bank.cli audit \
  --input output/json/question_bank.json \
  --output output/json/audit.current.json
```

### Output Integrity Audit

Purpose: fail fast on missing or inconsistent generated artifacts.

Input: `output/json/question_bank.json`, image artifacts under `output/`

Output: terminal JSON, optionally `output/json/audit.current.integrity.json`

Category/runtime: audit-only, fast

```bash
.venv/bin/python -m exam_bank.cli output-integrity-audit \
  --input output/json/question_bank.json \
  --artifact-root output \
  --output output/json/audit.current.integrity.json
```

### Full Readiness Audit

Purpose: produce extraction-readiness, OCR-selection, crop-quality, mapping, and Asterion-tier reports.

Input: `output/json/question_bank.json`; optional baseline when available

Output: CSV, Markdown, and JSON reports under `output/audits/current/`

Category/runtime: audit-only, medium

```bash
.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --artifact-root output \
  --out-dir output/audits/current
```

### OCR Candidate Audit

Purpose: audit native/OCR text candidate selection metadata and optional baseline movement.

Input: current or candidate `question_bank.json`; optional baseline `question_bank.json`

Output: terminal JSON, optionally `output/audits/current/ocr_candidate_audit.json`

Category/runtime: audit-only/OCR, fast

```bash
.venv/bin/python scripts/audit_ocr_candidates.py \
  --input output/json/question_bank.json \
  --json-output output/audits/current/ocr_candidate_audit.json
```

Candidate comparison:

```bash
.venv/bin/python scripts/audit_ocr_candidates.py \
  --input output/candidates/ocr/latest/json/question_bank.json \
  --baseline output/json/question_bank.json \
  --json-output output/audits/current/ocr_candidate_comparison.json
```

### Difficulty Audit

Purpose: audit deterministic difficulty labels, scores, bands, and review flags.

Input: `output/json/question_bank.json`

Output: terminal JSON, optionally `output/audits/current/difficulty_audit.json`

Category/runtime: audit-only, fast

```bash
.venv/bin/python scripts/audit_difficulty.py \
  --input output/json/question_bank.json \
  --json-output output/audits/current/difficulty_audit.json
```

### Mark-Event Evidence Sidecar

Purpose: build deterministic advisory mark-event evidence from existing official mark-scheme text and image links. This is not student marking and does not replace canonical mark-scheme crops.

Input: `output/json/question_bank.json`, mark-scheme artifacts under `output/`

Output: `output/json/question_bank.mark_events.v1.json`, `output/reports/mark_events_audit.md`, `output/reports/mark_events_review_queue.json`, optional validation report

Category/runtime: advisory sidecar, fast

```bash
.venv/bin/python scripts/build_mark_events.py \
  --question-bank output/json/question_bank.json \
  --artifact-root output \
  --out output/json/question_bank.mark_events.v1.json \
  --report output/reports/mark_events_audit.md \
  --review-queue output/reports/mark_events_review_queue.json
```

```bash
.venv/bin/python scripts/validate_mark_events.py \
  --question-bank output/json/question_bank.json \
  --mark-events output/json/question_bank.mark_events.v1.json \
  --artifact-root output \
  --output output/json/question_bank.mark_events.validation.v1.json
```

Contract: [Mark-Event Evidence Contract](MARK_EVENTS_CONTRACT.md). The sidecar is advisory-only; `safe_for_marking_use` must remain false for generated records.

## Downstream Projections

### Asterion Export

Purpose: write the conservative Asterion-safe question-bank projection.

Input: `output/json/question_bank.json`, artifacts under `output/`, optional skill-map sidecar

Output: `output/asterion/exports/latest/asterion_question_bank_v1.json`

Category/runtime: standard projection, fast to medium

```bash
.venv/bin/python -m exam_bank.cli asterion-export \
  --input output/json/question_bank.json \
  --artifact-root output
```

### Content Lab Candidates

Purpose: write Asterion Content Lab candidate metadata without generating student-facing content.

Input: `output/json/question_bank.json` or the Asterion export, artifacts under `output/`

Output: `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`

Category/runtime: standard projection, fast to medium

```bash
.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates \
  --input output/json/question_bank.json \
  --artifact-root output
```

### Topic Packets

Purpose: generate image-first printable CAIE 9709 major-topic packets from canonical question and mark-scheme crops. OCR/native/AI text is not used as student-facing question or answer content.

Input: `output/json/question_bank.json`, `exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json`, artifacts under `output/`

Output: `output/topic_packets/<paper_family>/<major_topic>/topic_packet.pdf`, `manifest.json`, and `output/topic_packets/topic_packet_summary.json`

Category/runtime: standard projection, fast to medium

Default behavior: generate broad major-topic packets for P1, P3, P4, and P5 using A4 portrait compact printable layout. Questions flow first, answers/mark schemes start in a separate section at the end, and multiple small problems are placed on a page when they fit. The packet taxonomy validates the paper family and major topic only. Subtopic IDs may remain in the taxonomy for future use, but they are not used by the normal packet generation pass.

Compatibility: add `--split-question-answer-pdfs` when legacy `questions.pdf` and `answers.pdf` side outputs are needed.

Layout options: `--page-size a4|letter`, `--orientation portrait|landscape`, `--layout compact|one-per-page`, and `--answer-placement end|inline`. Use `--layout one-per-page --answer-placement inline` for the previous paired page-heavy packet ordering.

Release packets are quality-first: records must have `mapping_status=pass`, `validation_status=pass`, `scope_quality_status=clean`, `question_crop_confidence=high`, and `visual_curation_status=ready`. Records with valid topics but risky visual status are written under `output/topic_packets/review_required/...` with review reasons. Weak topic/text signals can remain manifest warnings when the visual source is release-safe.

Dry run:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --artifact-root output \
  --dry-run \
  --strict-syllabus
```

Full generation:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --artifact-root output \
  --strict-syllabus \
  --pdf-profile print \
  --page-size a4 \
  --layout compact \
  --answer-placement end
```

Targeted major-topic generation:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --artifact-root output \
  --paper-family p3 \
  --topic integration \
  --strict-syllabus
```

Permissive review runs can include hard-failure categories explicitly:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --artifact-root output \
  --strict-syllabus \
  --include-mapping-failures \
  --include-validation-failures
```

## AI Sidecars

AI-heavy workflows are long-running and sidecar-only. They require provider credentials, write resumable status, and must not be treated as canonical extraction truth.

```bash
set -a; source .env; set +a
```

### Topic Routing

Purpose: produce strict parent-topic routing metadata against canonical taxonomy IDs.

Input: `output/json/question_bank.json`, `exam_bank_taxonomy/canonical/`, provider credentials

Output: `output/json/question_bank.topic_routing.v1.json`, `output/run_status/<run_id>/`

Category/runtime: AI-heavy, long-running, sidecar-only

```bash
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

### AI Enrichment

Purpose: write broad AI-assisted taxonomy, subpart, and difficulty metadata for review/debug sidecars.

Input: `output/json/question_bank.json`, `exam_bank_taxonomy/canonical/`, optional historical DeepSeek sidecar, provider credentials

Output: `output/json/question_bank.ai_assisted.v2.json`, failure logs/status when configured

Category/runtime: AI-heavy, long-running, sidecar-only

```bash
.venv/bin/python -m exam_bank.cli enrich-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --existing-sidecar output/archive/generated_cleanup_20260513T233456Z/output/json/question_bank.deepseek.json \
  --output output/json/question_bank.ai_assisted.v2.json \
  --model deepseek-v4-flash \
  --include-subparts \
  --recompute-difficulty \
  --resume \
  --status-dir output/run_status
```

### AI Sidecar Audit

Purpose: audit AI-assisted sidecar freshness, failures, prompt-version mix, and Asterion usability.

Input: AI-assisted sidecar JSON

Output: terminal JSON

Category/runtime: audit-only, fast

```bash
.venv/bin/python -m exam_bank.cli ai-sidecar-audit \
  --input output/json/question_bank.ai_assisted.v2.json
```

## Output Management

### Output Inventory

Purpose: inventory generated output roots without modifying files.

Input: generated output roots

Output: Markdown and JSON inventory reports, or terminal Markdown

Category/runtime: audit-only, fast to medium

```bash
.venv/bin/python -m exam_bank.cli output-inventory \
  --root output \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

Include the archived generated cleanup root when planning cleanup evidence:

```bash
.venv/bin/python -m exam_bank.cli output-inventory \
  --root output \
  --root output/archive/generated_cleanup_20260513T233456Z \
  --include-size \
  --max-depth 4 \
  --write output/output_inventory.md \
  --json output/output_inventory.json
```

### Output Cleanup Plan

Purpose: create a dry-run cleanup classification. This command does not delete or move files.

Input: generated output roots

Output: Markdown and optional JSON cleanup plan

Category/runtime: audit-only, fast to medium

```bash
.venv/bin/python -m exam_bank.cli output-cleanup-plan \
  --root output \
  --write output/output_cleanup_plan.md
```

Include archived generated-output evidence:

```bash
.venv/bin/python -m exam_bank.cli output-cleanup-plan \
  --root output \
  --root output/archive/generated_cleanup_20260513T233456Z \
  --include-size \
  --max-depth 4 \
  --write output/output_cleanup_plan.md \
  --json output/output_cleanup_plan.json
```

## Taxonomy Generation

These scripts rewrite canonical taxonomy files unless `--dry-run` is used. Use `--dry-run` for inspection and pre-cleanup validation.

### Content Lab Candidates As Taxonomy Input

Purpose: generate candidate CAIE 9709 skill maps, question-skill mappings, and coverage reports from the current bank and Asterion/Content Lab projections.

Input: `output/json/question_bank.json`, `output/asterion/exports/latest/asterion_question_bank_v1.json`, `output/asterion/exports/latest/asterion_content_lab_candidates_v1.json`

Output: canonical skill maps, mappings, coverage reports, indexes, logs

Category/runtime: standard generator, medium; mutating unless `--dry-run`

```bash
.venv/bin/python scripts/generate_skill_maps.py \
  --question-bank output/json/question_bank.json \
  --asterion-question-bank output/asterion/exports/latest/asterion_question_bank_v1.json \
  --content-lab-candidates output/asterion/exports/latest/asterion_content_lab_candidates_v1.json \
  --dry-run
```

### Topic Filter Maps

Purpose: generate strict topic filter maps from skill maps and question-skill mappings.

Input: canonical skill-map index, Asterion export, current question bank

Output: canonical topic filter maps, topic assignments, strict filtering reports

Category/runtime: standard generator, medium; mutating unless `--dry-run`

```bash
.venv/bin/python scripts/generate_topic_filter_maps.py \
  --skill-map-index exam_bank_taxonomy/canonical/indexes/skill_map_index_v1.json \
  --asterion-question-bank output/asterion/exports/latest/asterion_question_bank_v1.json \
  --legacy-question-bank output/json/question_bank.json \
  --dry-run
```

## Tests

Pytest markers are opt-in filters only. Plain `pytest`, `python -m pytest`, and `python -m pytest -q` still run the full suite. CI uses the full unfiltered command.

### Full Tests

Purpose: run the full regression suite.

Input: source and tests

Output: pytest result

Category/runtime: test, about 2 minutes observed

```bash
.venv/bin/python -m pytest -q
```

### Fast Local Tests

Purpose: run the fast local loop while excluding slower integration/sample-pipeline and rendering groups.

Input: source and tests, excluding `integration` and `rendering` markers

Output: pytest result

Category/runtime: test, fast

```bash
.venv/bin/python -m pytest -q -m "not integration and not rendering"
```

### Rendering Tests

Purpose: run image and crop rendering behavior tests separately.

Input: source and rendering tests

Output: pytest result

Category/runtime: test, fast

```bash
.venv/bin/python -m pytest -q -m rendering
```

### Integration and Sample-Pipeline Tests

Purpose: run slower repository sample-pipeline PDF regressions separately.

Input: source, repository PDF fixtures, and sample-pipeline tests

Output: pytest result

Category/runtime: test, about 2 minutes observed

```bash
.venv/bin/python -m pytest -q -m "integration"
```

### Targeted Command-Surface Tests

Purpose: verify runtime path behavior and script help/dry-run safety.

Input: targeted test files

Output: pytest result

Category/runtime: test, fast

```bash
.venv/bin/python -m pytest \
  tests/test_runtime_paths.py \
  tests/test_generator_cli_safety.py \
  -q
```

### Targeted Audit/Output Tests

Purpose: verify audit, Asterion, output-management, and readiness surfaces after documentation or command-surface changes.

Input: targeted test files

Output: pytest result

Category/runtime: test, fast to medium

```bash
.venv/bin/python -m pytest \
  tests/test_audit.py \
  tests/test_output_contract.py \
  tests/test_asterion_export.py \
  tests/test_output_management.py \
  tests/test_question_bank_readiness_audit.py \
  -q
```

## Verified Help Commands

These help surfaces were checked while creating this atlas:

```bash
.venv/bin/python -m exam_bank.cli --help
.venv/bin/python -m exam_bank.cli process --help
.venv/bin/python -m exam_bank.cli audit --help
.venv/bin/python -m exam_bank.cli output-integrity-audit --help
.venv/bin/python -m exam_bank.cli asterion-export --help
.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --help
.venv/bin/python -m exam_bank.cli topic-route-ai --help
.venv/bin/python -m exam_bank.cli enrich-ai --help
.venv/bin/python -m exam_bank.cli ai-sidecar-audit --help
.venv/bin/python -m exam_bank.cli output-inventory --help
.venv/bin/python -m exam_bank.cli output-cleanup-plan --help
.venv/bin/python scripts/audit_ocr_candidates.py --help
.venv/bin/python scripts/audit_difficulty.py --help
.venv/bin/python scripts/audit_question_bank_readiness.py --help
.venv/bin/python scripts/generate_skill_maps.py --help
.venv/bin/python scripts/generate_topic_filter_maps.py --help
.venv/bin/python -m pytest --help
```
