# Project Review

## Executive Summary

This project is an image-first CAIE 9709 extraction pipeline. It turns question-paper PDFs and mark-scheme PDFs into paper-organized question crops, mark-scheme crops, structured JSON records, trust statuses, validation flags, mapping statuses, text-fidelity statuses, and triage artifacts. A new auto-triage layer now measures corpus health, selects the next hard-failure target, creates agent handoff iterations, prints runbooks, and records evidence-gated acceptance decisions.

The current exported corpus in `output/json/question_bank.json` contains `1301` records across `148` paper instances. The current export is a no-OCR export: `ocr_ran=False` for all `1301` records and `notes.text_source_profile=native_pdf` for all records. Production-style runs should use `--enable-ocr`; the latest frozen OCR-style baseline is `output/triage/iteration_004/baseline_question_bank.json`.

The pipeline is improving. The strongest measured OCR-to-OCR comparison is `output/triage/iteration_003/comparison.math-repair-ocr.json`, where hard failures moved from `385` to `259` and the target issue `polluted_pass_requires_review` moved from `126` to `3`, with no worsened records. The latest current-output comparison is `output/triage/iteration_004/comparison.layout-review-current.json`, where hard failures moved from `259` to `148` and `question_scope_contaminated` moved from `114` to `4`, also with no worsened records. That comparison is useful but not a canonical OCR score because the current export is no-OCR.

The current OCR candidate in `output_ocr_candidate/json/question_bank.json` is OCR-enabled for all `1301` records and has `133` hard failures. Its latest accepted auto-triage comparison is `output_ocr_candidate/triage/iteration_002/comparison.auto-iteration-003.json`, where hard failures moved from `153` to `133`, `paper_total_mismatch` moved from `107` to `86`, and `worsened_records` stayed empty.

The trust posture is appropriately conservative. Images remain the source of truth; extracted text is metadata. Current text fidelity is `clean: 1245`, `degraded: 48`, `unusable: 8`, but `visual_required=True` for `978` records in the current no-OCR export, so clean text does not mean text-only student readiness.

Still unsolved: math text remains fragile, OCR can corrupt math, native PDF extraction flattens notation, diagrams and answer-space furniture are hard, paper-total mismatches remain the dominant current hard-failure cluster, and one source-pairing defect is visible in the current export: `11` `33autumn25` records point at a `12autumn21` mark scheme because the `33` November 2025 mark scheme is absent; one of those records currently has `mapping_status: pass`.

## Project Purpose

The end goal is to build a reliable CAIE 9709 exam-bank dataset that can support student practice and training workflows without losing the original paper evidence.

The project should:

- Preserve source-paper visual truth through question and mark-scheme image crops.
- Extract text and metadata for search, routing, topic labeling, enrichment, and review.
- Keep text trust-gated because mathematical notation is difficult to reconstruct from PDFs and OCR.
- Allow local and DeepSeek/topic enrichment without corrupting the base extraction evidence.
- Provide repeatable triage loops so improvement claims are measured against frozen baselines.

## Current Architecture

Inputs live under `input/question_papers/` and `input/mark_schemes/`. The supported CLI entrypoint is:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output --enable-ocr
```

The processing flow is:

1. `document_registry` recursively scans source files and pairs companion documents by parsed syllabus/session/component metadata.
2. `pdf_extract` uses PyMuPDF to extract text lines and visual boxes. It reconstructs visual line order and can merge OCR blocks for sparse regions when OCR is enabled.
3. `question_detection` scores question-number anchors, detects top-level question spans, handles subpart continuity, filters page furniture, and assigns validation flags.
4. `image_rendering` renders prompt-first question crops from original PDF pixels, separates text and figures, stitches multi-region crops, and runs crop OCR when enabled.
5. `extraction_structure` normalizes body text, separates diagram-like text, extracts math-heavy lines, and flags likely text corruption.
6. `mark_schemes` detects mark-scheme answer tables, anchors question blocks, renders mark-scheme crops, detects subparts and mark totals, and returns mapping status/failure reason.
7. `pipeline` builds `QuestionRecord` objects, applies validation refinement, selects native/OCR text candidates, derives trust statuses, classifies topics/difficulty, and applies paper-total metadata.
8. `exporters` writes the schema-versioned JSON bank and paper-first image paths.
9. `triage` creates deterministic samples, serves a local visual review gallery, captures `review.jsonl`, and compares current exports against frozen baselines.
10. `auto_triage` builds status reports, plans handoff iterations, writes shared agent prompts, generates runbooks, and accepts/rejects completed passes using tests plus OCR-enabled comparison evidence.

DeepSeek enrichment is a separate sidecar step (`output/json/question_bank.deepseek.json`). It should be treated as secondary metadata.

## Data Contract

`output/json/question_bank.json` has:

- `schema_name`: currently `exam_bank.question_bank`
- `schema_version`: currently `2`
- `record_count`: current value `1301`
- `questions`: one object per detected top-level question

Key record fields:

- `question_id`: deterministic paper/question id such as `12spring21_q03`
- `paper`: paper instance id such as `12spring21`
- `paper_family`: `p1`, `p3`, `p4`, or `p5`
- `question_number`: top-level question number
- `question_image_path` / `question_image_paths`: rendered source-paper crop
- `mark_scheme_image_path` / `mark_scheme_image_paths`: rendered mark-scheme crop
- `page_refs`: source question and mark-scheme pages
- `question_text`: selected text candidate, useful as metadata only
- `question_text_role`, `question_text_trust`, `visual_required`, `visual_reason_flags`
- `ocr_ran`, `ocr_engine`, `ocr_text`, `ocr_text_trust`, `ocr_failure_reason`
- `subparts`, `subparts_solution_marks`, `question_solution_marks`
- `topic`, `difficulty`, `difficulty_score`, `difficulty_band`
- `notes.source_pdf`, `notes.mark_scheme_source_pdf`
- `notes.mapping_status`, `notes.mapping_failure_reason`
- `notes.validation_status`, `notes.validation_flags`
- `notes.scope_quality_status`
- `notes.text_source_profile`
- `notes.text_fidelity_status`, `notes.text_fidelity_flags`
- `notes.topic_trust_status`
- `notes.review_flags`, `notes.extraction_quality_flags`

Internal code still uses names such as `markscheme_mapping_status` and `markscheme_failure_reason` on `QuestionRecord`. The exported JSON currently exposes those as `notes.mapping_status` and `notes.mapping_failure_reason`.

## Iteration History

### iteration_001

- Basis: frozen baseline was OCR-enabled (`ocr_ran=True` for all `1301`; `hybrid: 898`, `native_pdf: 403`).
- Target issue: `question_scope_contaminated`.
- Summary file: `output/triage/iteration_001/summary.json`.
- Baseline hard failures: `500`.
- Baseline target issue count: `159`.
- Review notes: `36` entries, `30` unique questions. Latest unique root causes were `question_crop_boundary: 12`, `text_ocr_quality: 11`, `false_positive_validation_gate: 5`, `mark_scheme_mapping: 1`, `paper_total_detection: 1`.
- Canonical OCR comparison: `output/triage/iteration_001/comparison.ocr-rerun.json`.
- OCR hard failures: `500 -> 385` (`-115`).
- OCR target issue: `159 -> 117` (`-42`).
- OCR worsened records: `3`: `11autumn25_q05`, `32summer25_q10`, `33summer25_q06`.
- No-OCR comparisons also exist: `comparison.json`, `comparison.clean-rerun.json`, and `comparison.no-ocr.json`, each reporting `500 -> 258` (`-242`) and target `159 -> 117` (`-42`) with no worsened records. These are useful debugging artifacts, not production OCR scores.

### iteration_002

- Basis: no-OCR baseline (`ocr_ran=False` for all `1301`; `native_pdf: 1301`).
- Target issue: `question_scope_contaminated`.
- Summary file: `output/triage/iteration_002/summary.json`.
- Baseline hard failures: `258`.
- Baseline target issue count: `117`.
- Review notes: `0`.
- Comparison file: none found under `output/triage/iteration_002/`.
- Canonical production comparison: none. This iteration is a frozen sample/baseline artifact, not a completed measured improvement pass.

### iteration_003

- Basis: OCR-enabled baseline (`ocr_ran=True` for all `1301`; `hybrid: 890`, `native_pdf: 402`, `ocr: 9`).
- Target issue: `polluted_pass_requires_review`.
- Summary file: `output/triage/iteration_003/summary.json`.
- Comparison file: `output/triage/iteration_003/comparison.math-repair-ocr.json`.
- Hard failures: `385 -> 259` (`-126`).
- Target issue: `126 -> 3` (`-123`).
- Worsened records: `0`.
- Primary improvement: math-layout repair reduced polluted pass failures without increasing worsened tracked statuses.
- Review notes: `6` entries, with `text_ocr_quality: 4` and `unknown: 2`.
- Canonical for production comparison: yes, because baseline and current comparison were OCR-enabled.

### iteration_004

- Basis: OCR-enabled baseline from after iteration 003 (`ocr_ran=True` for all `1301`; `hybrid: 890`, `native_pdf: 402`, `ocr: 9`).
- Target issue: `question_scope_contaminated`.
- Summary file: `output/triage/iteration_004/summary.json`.
- Comparison file: `output/triage/iteration_004/comparison.layout-review-current.json`.
- Hard failures: `259 -> 148` (`-111`).
- Target issue: `114 -> 4` (`-110`).
- Worsened records: `0`.
- Review notes: `30` entries. Root causes: `question_crop_boundary: 15`, `text_ocr_quality: 10`, `unknown: 5`.
- Canonical for production comparison: no, because current `output/json/question_bank.json` is no-OCR while the frozen baseline is OCR-enabled.

### Auto-Triage Handoffs

- Shared prompts live under `agent_handoffs/auto_triage/Prompt/` for supervisor, planner, builder, test gatekeeper, integration, and adversarial review roles.
- Handoff iterations live under `agent_handoffs/auto_triage/iteration_*` and store `metrics_before.json`, `selected_target.json` when available, generated commands, `metrics_after.json`, and `decision.json`.
- Accepted auto-triage decision `agent_handoffs/auto_triage/iteration_001/decision.json` targeted `paper_total_mismatch`, used OCR-enabled current and baseline outputs, moved the selected target by `-28`, and had `worsened_record_count: 0`.
- Accepted auto-triage decision `agent_handoffs/auto_triage/iteration_003/decision.json` targeted `paper_total_mismatch`, used OCR-enabled current and baseline outputs, moved hard failures by `-20`, moved the target by `-21`, and had `worsened_record_count: 0`.
- Auto-triage is an evidence gate, not an implementation agent. It does not change extraction code or promote records by itself.

### Failure Cluster Shift

The dominant failure cluster shifted from scope contamination and polluted OCR/math-derived pass failures toward paper-total and mark-total mismatches:

- iteration 001 baseline: `question_scope_contaminated: 159`, `paper_total_mismatch: 134`, `polluted_pass_requires_review: 125`.
- iteration 003 baseline: `polluted_pass_requires_review: 126`, `question_scope_contaminated: 117`, `paper_total_mismatch: 107`.
- iteration 004 current comparison: `paper_total_mismatch: 107`, `question_mark_total_mismatch: 27`, `weak_question_anchor: 9`, `question_scope_contaminated: 4`, `question_subparts_incomplete: 1`.

This is a healthy movement pattern: earlier broad crop/text pollution failures have been reduced, leaving more specific accounting and pairing failures.

## What Is Working Well

- Image-first preservation is strong: current output has `1301` question image paths and all question image files exist.
- Paper-first output layout is practical and stable: `output/p1`, `output/p3`, `output/p4`, and `output/p5` contain paper instance folders.
- Deterministic question IDs and paper IDs make comparison possible.
- Mark-scheme crop rendering and mapping exist for nearly every record, with explicit pass/fail status and failure reasons.
- Validation statuses and flags are not cosmetic; they drive visual, text-only, scope, topic, and downstream trust decisions.
- Triage sampling is deterministic and creates frozen baselines, review galleries, review notes, and comparison files.
- Worsened-record reporting is available and has caught regressions.
- Auto-triage now turns the triage loop into a repeatable handoff with status metrics, selected targets, generated commands, test-status requirements, OCR-mode checks, and acceptance decisions.
- OCR candidate scoring is conservative: current no-OCR audit shows candidate metadata is populated but OCR was not selected because OCR text was empty.
- Recent math-layout repair reduced `polluted_pass_requires_review` by `123` in an OCR-to-OCR comparison.
- The test suite is broad and currently passes.
- DeepSeek enrichment is separated into a sidecar and final review gates remain conservative.

## What Is Not Working Well Yet

- Math text is still hard. Fractions, powers, inverse trig notation, vectors, matrices, integrals, limits, and multi-line formulas remain risky.
- OCR can corrupt math and merge sparse regions. In OCR-enabled baselines, `ocr_text_sparse_or_merged` appears in `899` visual reason flags in `output/json/audit.current.json`.
- Native PDF text can flatten layout even when the visual crop is correct.
- Diagrams and answer-space furniture still interfere with both crop scope and extracted text.
- Some records are visually usable but text-untrusted; this is expected and should remain explicit.
- Some failures are mixed-category: a record can have a good crop but bad text, or good text but bad mapping/scope.
- Full reruns are expensive enough that triage baselines must be preserved and compared carefully.
- Topic trust and DeepSeek labels are useful but should not be over-relied on.
- Current output has a source-pairing mismatch affecting `11` `33autumn25` records; one false mapping pass is visible.
- There is no dedicated trusted-subset export yet.

## Current Risk Register

| Risk | Severity | Why it matters | Mitigation |
|---|---|---|---|
| Corrupted math text | High | Student apps could show or search wrong formulas if text is treated as canonical. | Keep images as source of truth; gate by `text_fidelity_status`, `question_text_trust`, and `visual_required`. |
| Scope contamination | High | A crop can include the wrong question, graph labels, answer space, or next-question material. | Preserve `scope_quality_status`, `validation_flags`, and visual review; continue targeted crop-boundary passes. |
| Mark-scheme mismatch | High | Wrong answer crops are more damaging than missing text. | Add source-pairing validation and require paper/session/component agreement before student export. |
| OCR/native text disagreement | Medium | OCR can look plausible while losing marks, subparts, or notation. | Keep OCR as candidate/fallback; audit `ocr_rejected_reasons` and compare OCR runs only against OCR baselines. |
| False confidence from topic labels | Medium | Topic search or adaptive routing can become misleading. | Treat local and DeepSeek topics as metadata; require `topic_trust_status=normal` for student routing. |
| Comparing against wrong baseline | High | Mixed OCR/no-OCR comparisons can falsely claim improvement. | Preserve baseline OCR state in docs and comparison names; compare production runs only to OCR-enabled baselines. |
| Unsafe automatic improvement loop | High | A tool could reduce counts by loosening gates or skipping visual evidence. | Keep auto-triage evidence-gated: tests must pass, OCR mode must match, `worsened_records` must be empty, and validation/trust-gate loosening is rejected without extraction evidence. |
| Stale docs | Medium | Contributors may rerun the wrong profile or trust wrong fields. | Keep README and project review tied to measured command output. |
| Visual review bottleneck | Medium | Remaining failures often need human judgment. | Improve triage notes capture and build a trusted-subset export. |
| Overfitting fixes to a few papers | Medium | Heuristics can regress unseen layouts. | Add regression fixtures from triage notes and keep worsened-record checks. |
| Source-pairing fallback mismatch | High | Current evidence shows a missing mark scheme can lead to wrong-source pairing. | Add a gate that fails or reviews any record whose question paper and mark scheme metadata disagree. |

## Student-App Readiness

### Tier 1: image-ready

Use when:

- Question image exists.
- Mark-scheme image exists.
- `notes.mapping_status=pass`.
- `notes.scope_quality_status` is acceptable.
- `visual_curation_status=ready`.
- Source question paper and mark-scheme metadata agree.

Measured with an additional local filter on current no-OCR output: `616` records meet a strict image-ready definition. The OCR frozen baseline has `208` under the same strict file/source/status check.

### Tier 2: metadata-ready

Use when Tier 1 passes and:

- `notes.text_fidelity_status=clean`.
- `notes.topic_trust_status=normal`.
- Marks are present.
- Subpart metadata is present and does not contradict mark-scheme structure.

Measured with a strict local filter: current no-OCR output has `567`; the OCR frozen baseline has `178`.

### Tier 3: fully trusted practice item

Use when Tier 2 passes and:

- `notes.validation_status=pass`.
- `text_only_status=ready`.
- `question_text_trust=high`.
- The item has been reviewed or has strong automated confidence.

Measured with a strict local filter: current no-OCR output has `150`; the OCR frozen baseline has `41`.

Exclude from student-facing practice until reviewed:

- Mapping failures.
- Validation failures.
- Scope failures.
- Missing image files.
- Source-pairing mismatches.
- Degraded or unusable text when text is needed for the workflow.
- `topic_trust_status` other than `normal` for topic-routed practice.
- DeepSeek sidecar failures or final review-required enrichment when enrichment is needed.

## Recommended Next Passes

Use auto-triage as the control loop for these passes. The next target should come from `auto-triage-plan` on an OCR-enabled candidate, and the pass should not be reported as accepted unless `auto-triage-compare` has passing test evidence, OCR-enabled current and baseline outputs, target improvement, and no unexplained regressions.

### 1. Source-Pairing Guard

- Goal: prevent wrong mark-scheme source pairing when a companion mark scheme is missing.
- Why it matters: current output has `11` `33autumn25` records pointing at `12autumn21` mark-scheme crops, with one false `mapping_status: pass`.
- Acceptance criteria: any question/mark-scheme paper-family, component, session, or year mismatch becomes `mapping_status=fail` or a clear review status; `33autumn25_q01` can no longer pass.
- Tests to add: registry/pipeline fixture for missing `33` companion mark scheme; export audit for source metadata mismatch.
- What not to change: do not loosen mapping gates to hide the mismatch.

### 2. Remaining Math-Layout Reconstruction

- Goal: reduce wrong text for fractions, powers, trig expressions, vectors, matrices, and multi-line math.
- Why it matters: triage notes repeatedly identify text quality as a remaining cluster.
- Acceptance criteria: targeted fixtures pass, `math_text_corruption_detected` does not increase, and OCR-to-OCR comparison has no worsened records.
- Tests to add: fixtures from iteration 004 notes such as vector columns, half powers, alpha/theta inequalities, and inverse trig.
- What not to change: do not promote repaired text to high trust unless fidelity checks support it.

### 3. Separate Crop Failures From Text-Fidelity Failures

- Goal: make review status explain whether an issue is visual scope, text fidelity, or both.
- Why it matters: many records are visually good but text-untrusted, and some triage notes mix crop remnants with bad text.
- Acceptance criteria: triage primary issues distinguish crop-boundary failures from text-only corruption; issue counts remain comparable.
- Tests to add: records with good images but degraded text, and records with bad crop but readable text.
- What not to change: do not reduce hard failures by suppressing flags.

### 4. Trusted-Subset Export

- Goal: export a JSON subset for downstream student use with explicit tier filters.
- Why it matters: downstream apps should not reinvent trust filtering or load all records blindly.
- Acceptance criteria: export includes only records passing source-pair, image, mapping, scope, text/topic gates appropriate for the selected tier; rejected counts are reported by reason.
- Tests to add: fixture records for each exclusion reason and tier.
- What not to change: do not mutate the canonical `question_bank.json`.

### 5. Triage UX And Notes Capture

- Goal: make visual review notes easier to turn into regression fixtures.
- Why it matters: iteration 004 notes contain clear recurring issues, but they are still free text.
- Acceptance criteria: review notes can tag crop/text/source/mapping causes, and a follow-up report groups examples by fix owner.
- Tests to add: JSONL parsing and summary grouping.
- What not to change: do not delete or rewrite historical `review.jsonl`.

## Do Not Do Yet

- Do not build a complex frontend before a trusted-subset export exists.
- Do not treat OCR text as canonical.
- Do not auto-promote topic labels from local rules or DeepSeek.
- Do not suppress validation flags just to reduce failure counts.
- Do not use no-OCR comparisons as the production score.
- Do not expand to more syllabuses or broad new features before the trust loop stabilizes.
- Do not make `question_bank.deepseek.json` canonical extraction truth.

## Final Assessment

The project is on the right track. The strongest part is the image-first extraction plus deterministic triage loop: it preserves visual evidence and makes improvement measurable. Auto-triage now makes that loop easier to repeat without treating aggregate counts as proof. The weakest part is still trust around hard mathematical text and mark-scheme/source pairing. The most meaningful reliability improvement would be a source-pairing guard plus a trusted-subset export that downstream apps can consume without guessing.

The next implementation agent should fix the source-pairing mismatch first, then rerun an OCR-enabled full export into a new output path, compare it against the OCR baseline, and only then continue math-layout repair or trusted-subset work.
