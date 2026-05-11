# Project Review — Image-First Success and Reliable Text Extraction Phase

Review date: 2026-05-11

Primary export reviewed: `output/json/question_bank.json`

Secondary exports reviewed: `output/json/question_bank.deepseek.json`, `output_ocr_candidate/json/question_bank.json`

Key thesis: Phase 1 succeeded because images are canonical; Phase 2 begins because text is not yet canonical.

## 1. Executive Summary

The image-first goal has substantially succeeded. The current bank contains 1301 records. All 1301 question image paths are present and all 1301 question PNG files exist under `output/`. Of the 1301 records, 1290 have mark-scheme image paths and existing mark-scheme PNG files. The 11 missing mark-scheme images are all `33autumn25_q01` through `33autumn25_q11`, matching the known missing exact mark-scheme cluster. A PIL smoke audit found no invalid, tiny, or blank-looking existing question or mark-scheme PNGs.

Text extraction is not ready for student-facing mathematical display. The current export has `text_only_status=fail` for 906 records, `review` for 299, and `ready` for only 96. It has `visual_required=true` for 1203 records. It marks 905 records as `question_text_role=untrusted_math_text`, and the audit reports 906 records with degraded text fidelity. Sample records show classic math corruption: flattened powers, corrupted fractions, bad inverse/trig notation, missing absolute-value bars, bad limits, merged words, table noise, and mark-scheme notation damage.

The recommended next strategic goal is Phase 2 - Reliable Text Extraction. The goal is not "turn on OCR everywhere." The goal is to measure and improve text as a separate reliability layer while keeping rendered question and mark-scheme images as canonical. Search, topic classification, teacher review, student display, and generated-content workflows need separate trust gates.

Hypothesis check:

- Image-first extraction goal substantially succeeded: verified, with the caveat of 11 missing exact mark schemes.
- Canonical question and mark-scheme PNGs are stable source of truth: verified for existing artifacts.
- JSON contract contains many right trust/review fields: verified.
- Main weakness is text reliability: verified strongly.
- Next phase should focus on reliable text without weakening image-first safety: verified and recommended.

## 2. Current Project State

The project currently scans CAIE 9709 question-paper and mark-scheme PDFs, classifies paper type, detects top-level question spans, renders canonical question PNGs, detects mark-scheme regions, renders mark-scheme PNGs, maps questions to mark schemes, and writes schema-versioned JSON metadata.

The supported runtime entry points visible from `python -m exam_bank.cli --help` are:

- `process`
- `audit`
- `asterion-export`
- `asterion-content-lab-candidates`
- `triage-sample`
- `triage-serve`
- `triage-compare`
- `auto-triage-status`
- `auto-triage-plan`
- `auto-triage-compare`
- `auto-triage-runbook`

The extraction configuration in `config.yaml` keeps OCR disabled by default:

```yaml
ocr:
  enabled: false
```

The README recommends production-style runs with OCR enabled:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output --enable-ocr
```

The current standard `output/json/question_bank.json` reviewed here is OCR-populated: `ocr_ran=true` for all 1301 records, with 29 OCR-selected records. Older handoff materials, especially `agent_handoffs/auto_triage/iteration_007/residual_failure_burndown_report.md`, describe an earlier no-OCR export state and should be read as historical evidence rather than current state.

Files requested by the review:

- `README.md`: present; useful, and should stay synchronized with the current OCR-enabled standard output.
- `pyproject.toml`: present; project requires Python >=3.10 and uses PyMuPDF, pdfplumber, Pillow, pytesseract, PyYAML, and OpenAI.
- `config.yaml`: present; operational overrides and OCR default.
- `src/exam_bank/`: present.
- `tests/`: present.
- `output/json/question_bank.json`: present.
- `output/json/question_bank.deepseek.json`: present.
- `agent_handoffs/`: present, including OCR and auto-triage iteration materials.
- Roadmap/review/trust/audit docs: `ROADMAP.md`, `docs/ROADMAP.md`, `docs/PROJECT_REVIEW.md`, `docs/TRUST_MODEL.md`, `docs/TRIAGE_WORKFLOW.md`, `docs/AUTO_TRIAGE.md`, `REVIEW_PASS_2026-04-22.md`.
- Dedicated `ARCHITECTURE.md`: not present. Architecture exists implicitly in README, source modules, and `docs/TRUST_MODEL.md`; a standalone architecture doc is recommended.

## 3. Major Positives

The strongest achievement is that the project no longer depends on text being perfect. It has a paper-first output tree with canonical PNGs, source PDF traceability, page references, crop confidence, mapping status, validation status, visual curation status, and text-only status. This is the correct architecture for exam content where OCR and native PDF text are frequently wrong.

The question and mark-scheme workflow is clean and traceable. `question_detection.py` detects question spans and records structural diagnostics. `mark_schemes.py` treats the rendered mark-scheme image as the preserved mathematical source, while text is used to locate boundaries. `pipeline.py` wires discovery, extraction, mapping, trust classification, OCR selection, and export.

The JSON contract is already safety-aware. `models.py` and `exporters.py` expose fields such as `question_text_role`, `question_text_trust`, `visual_required`, `visual_reason_flags`, `visual_curation_status`, `text_only_status`, OCR candidate fields, validation fields, mapping fields, subparts, mark totals, topic trust, and review flags.

OCR is optional and candidate-based. `ocr.py` keeps native text unless OCR is clearly better and rejects OCR for missing question numbers, missing subparts, lost mark brackets, page furniture, next-question contamination, non-pass validation, or non-pass mapping.

DeepSeek enrichment is a sidecar. `output/json/question_bank.deepseek.json` does not mutate the raw bank. The sidecar has independent reconciliation and review fields, which is the right posture.

Audit and triage tooling is substantial. The project has `exam_bank.cli audit`, `scripts/audit_ocr_candidates.py`, `scripts/audit_question_bank_readiness.py`, difficulty audit scripts, manual triage commands, and auto-triage commands.

Tests are strong for current scope. The full suite passed:

```text
347 passed, 3 skipped
```

Tests cover schema shape, paper-first paths, OCR selection/rejection, page furniture, lost mark brackets, mapping gates, readiness audits, subpart fillability, Asterion projection behavior, and sidecar behavior.

## 4. Major Negatives / Risks

Image risks:

- 11 `33autumn25` records have no exact mark-scheme image path. They are safely marked through mapping failure rather than wrongly paired.
- 895 records have `question_crop_confidence=low` in the current OCR-populated export. Low crop confidence does not mean the image is bad, but it means the image layer still needs QA sampling and should not be inflated into "fully reviewed."
- 544 existing mark-scheme crops have `mark_scheme_crop_confidence=medium`, and 11 are blank/missing because there is no exact mark scheme.
- Some older handoff documentation describes a previous no-OCR export state, so readers could misinterpret historical readiness counts as current.

Text risks:

- Text is often readable enough for topic guessing but not reliable enough for mathematical display.
- OCR confidence and OCR readability do not measure mathematical fidelity.
- Examples show broken powers (`x2` for `x^2`), broken fractional powers (`x^{-}1_{2}`), bad trig/log/exponential notation, malformed inequalities, missing absolute-value bars, and corrupted table content.
- Mark-scheme text is especially degraded for equations, integrals, fractions, and mark-event structure.
- Subparts are detected for 968 records, but all 968 currently have `subparts_solution_marks` entirely null.
- Only 96 records are currently `text_only_status=ready`; 906 fail.
- Topic trust is strongly affected by degraded text. Only 375 records have `topic_trust_status=normal`; 892 are `degraded_text` and 34 are `review_required`.
- DeepSeek can be confident while extraction gates still require review. There are 688 high-confidence DeepSeek records with `final_review_required=true`.

Process risks:

- Generated outputs are present and some Asterion-related files are untracked. These may be intentional work in progress, but they should be handled deliberately before commit.
- `README.md` current audited-state numbers no longer match the current JSON in this workspace.
- The repo lacks a standalone `ARCHITECTURE.md`, which makes the phase transition harder to communicate.

## 5. Image-First Extraction Assessment

Evidence from the reviewed export:

| Metric | Count |
| --- | ---: |
| Total records | 1301 |
| Question image paths present | 1301 |
| Question image files existing | 1301 |
| Mark-scheme image paths present | 1290 |
| Mark-scheme image files existing | 1290 |
| Missing mark-scheme image paths | 11 |
| Existing image files invalid/tiny/blankish by smoke check | 0 |
| Mapping status pass/fail | 1280 / 21 |
| Validation status pass/review/fail | 917 / 370 / 14 |
| Paper-total status matched | 1301 |
| Question crop confidence high/low | 406 / 895 |
| Mark-scheme crop confidence high/medium/blank | 746 / 544 / 11 |

The image-first layer is good enough to be the canonical foundation. The missing mark-scheme cluster is explicit rather than hidden. The paper-first path structure is stable and product-useful:

```text
output/
  p3/
    32spring21/
      questions/q04.png
      mark_scheme/q04.png
```

Remaining image QA should focus on crop confidence, mark-scheme medium-confidence crops, the 21 mapping failures, the 14 validation failures, and the 11 missing exact mark-scheme records. The image layer should continue to be treated as canonical, not "complete with no review needed."

## 6. Text Extraction Assessment

Current text distributions:

| Field | Distribution |
| --- | --- |
| `question_text_role` | `readable_text=98`, `search_hint=298`, `untrusted_math_text=905` |
| `question_text_trust` | `high=98`, `medium=297`, `low=906` |
| `text_only_status` | `ready=96`, `review=299`, `fail=906` |
| `visual_required` | `true=1203`, `false=98` |
| `text_fidelity_status` | `clean=395`, `degraded=906` |
| `text_candidate_source` | `native=1272`, `ocr=29` |
| `text_candidate_decision` | `native_retained=1272`, `ocr_selected=29` |
| `ocr_text_role` | `readable_text=343`, `search_hint=938`, `untrusted_math_text=20` |
| `ocr_text_trust` | `high=343`, `medium=938`, `low=20` |

The current selected text is useful for search hints, classification hints, and review queues. It is not broadly safe for student display, math transcription, or generation.

Concrete text failure modes observed:

- `12spring21_q11`: native question text merges words and encodes fractional powers as `x^{-}1_{2}`; OCR loses exponent structure.
- `32spring21_q04`: differential notation appears as `(1 -cos x)ddyx`; OCR turns the equation into scrambled text.
- `32spring21_q10`: `1/2 pi` appears as `1_{2}π`; OCR turns `cos^2 x` into `cos?x` and changes the interval.
- `11autumn22_q10`: selected text has `y = 2x2 + 1` where mathematical power notation is unsafe.
- `31autumn23_q02`: complex-number inequality text loses exact absolute-value and argument notation.
- `52spring21_q05`: table content is corrupted into noise, even though the topic is guessable.
- `33autumn25_q01`: native text loses absolute-value bars; OCR preserves some bars but the record has no mark-scheme mapping and remains unsafe.

The distinction matters: a record can be image-perfect and text-poor. Several high-confidence image records still have `text_only_status=review` because math notation is not reliable enough.

Mark-scheme text is also not safe as canonical text. It preserves some mark codes, but equations and mark-event dependencies are corrupted by PDF glyph extraction. It can support mark-event candidate parsing, but not student-facing solution text without review.

## 7. JSON Contract Assessment

Strengths:

- Versioned schema: `schema_name`, `schema_version`, `record_count`, and `questions`.
- Stable source identifiers: `question_id`, `paper`, `paper_family`, `question_number`, source paths, page refs.
- Canonical artifact paths: question and mark-scheme image paths.
- Trust/readiness fields: `question_text_role`, `question_text_trust`, `visual_required`, `visual_reason_flags`, `visual_curation_status`, `text_only_status`.
- Candidate OCR fields: `ocr_ran`, `ocr_engine`, `ocr_text`, `ocr_text_trust`, candidate scores, candidate source, decision, and rejection reasons.
- Diagnostics under `notes`: validation, mapping, scope, text fidelity, topic trust, crop confidence, extraction flags, review flags.

Ambiguities and gaps:

- Some trust fields live top-level and some live under `notes`; consumers need a documented precedence rule.
- `question_text_trust=high` is too broad if read as student-safe. It should not imply math fidelity.
- `ocr_text_trust=high` means readable OCR, not math-faithful OCR.
- `text_only_status=ready` is useful but currently lacks decomposed evidence fields such as readability, math fidelity, structure, marks, and contamination.
- `subparts_solution_marks` exists but is null for every subpart-bearing record.
- The raw export lacks run manifest fields: `generated_at`, `run_id`, `pipeline_version`, `git_commit`, `model_versions`, `ocr_engine_version`, `input_manifest_sha256`, `artifact_root`, and `qa_summary`.
- The raw export does not include artifact hashes; the untracked Asterion projection code appears to add them later.

Recommended additions:

```json
"text_contract": {
  "selected_text_source": "native_pdf|ocr|hybrid|manual|none",
  "readability_trust": "high|medium|low|unknown",
  "math_fidelity_trust": "high|medium|low|unknown",
  "structure_trust": "high|medium|low|unknown",
  "marks_trust": "high|medium|low|unknown",
  "search_allowed": true,
  "classification_allowed": true,
  "teacher_review_allowed": true,
  "student_display_allowed": false,
  "generation_allowed": false,
  "requires_visual_display": true,
  "review_required": true,
  "reason_codes": ["math_fidelity_unverified", "visual_required"]
}
```

## 8. OCR Candidate Selection Assessment

The OCR candidate selector is conservative and useful, but still insufficiently measured for math fidelity.

Evidence from `scripts/audit_ocr_candidates.py --input output/json/question_bank.json`:

| Metric | Count |
| --- | ---: |
| Records | 1301 |
| OCR selected | 29 |
| Native retained | 1272 |
| `ocr_not_clearly_better` rejections | 605 |
| `ocr_validation_status_not_pass` rejections | 384 |
| `ocr_missing_question_number` rejections | 289 |
| `ocr_lost_mark_brackets` rejections | 83 |
| `page_furniture_or_header_text` rejections | 82 |
| `ocr_mapping_status_not_pass` rejections | 24 |
| `ocr_missing_subpart_labels` rejections | 17 |
| Suspicious OCR-selected records | 19 |

The selector correctly rejects many unsafe OCR candidates. It also keeps OCR-selected records gated: OCR selection does not make the text student-safe. This is a major positive.

The weakness is that OCR scoring is still primarily structural/readability scoring. It can catch obvious page furniture and missing marks, but it does not prove mathematical fidelity. OCR-selected examples still contain `@` for angle symbols, broken powers, corrupted tables, and mangled equations. OCR selection should therefore remain a candidate selection event, not a readiness event.

Recommendation: add a full-bank `audit-text-reliability` command or script that measures native, OCR, selected text, math corruption, subpart/mark preservation, and mark-scheme text separately.

## 9. Asterion Readiness Assessment

Asterion can safely ingest a conservative projection of this bank today if image display remains canonical and text-only display is blocked except for records that pass strict gates.

Safe now:

- Image-only question display for records with existing question images, mark-scheme images, mapping pass, validation pass, and acceptable visual curation.
- Search and topic-routing hints with visible trust badges.
- Teacher review workflows that show canonical images beside candidate text.
- Sidecar DeepSeek/topic/difficulty metadata as review evidence.

Must remain blocked:

- Raw OCR/native text as student-facing mathematical display for most records.
- Mark-scheme text as canonical solution display.
- Content Lab generation from unreviewed text, unreviewed skill mappings, or unreviewed mark events.
- Treating DeepSeek confidence as truth.
- Treating `ocr_text_trust=high` or `question_text_trust=high` as math fidelity.
- Counting `text_only_status=review` records as text-safe.

The untracked `src/exam_bank/asterion_export.py` and related tests show the right direction: a projection layer with `quality_gate`, usage roles, artifact integrity, subparts, and mark-event candidates. Because it is currently untracked in this workspace, it should be reviewed separately before being considered part of the committed architecture.

Content Lab / learning-intelligence suitability is not the next direct step. The bank is suitable as a visual source and review substrate. It is not yet suitable as an automatic generation substrate except through a conservative projection that requires reviewed text, reviewed skills, reviewed mark events, and explicit `generation_allowed` gates.

## 10. Test and Audit Assessment

Commands run:

```bash
git status --short
git diff --stat
.venv/bin/python -m pytest -q
.venv/bin/python -m exam_bank.cli audit --input output/json/question_bank.json
.venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json
.venv/bin/python scripts/audit_ocr_candidates.py --input output_ocr_candidate/json/question_bank.json
.venv/bin/python scripts/audit_difficulty.py --input output/json/question_bank.json
.venv/bin/python -m exam_bank.cli auto-triage-status --input output/json/question_bank.json
.venv/bin/python scripts/audit_question_bank_readiness.py --input output/json/question_bank.json --artifact-root output --out-dir /tmp/exam-bank-review-audit-output
```

Test result:

```text
347 passed, 3 skipped
```

Current test strengths:

- Schema and output contract checks.
- Runtime path and paper-first artifact checks.
- Paper recognition and question detection behavior.
- OCR enable/disable behavior.
- OCR candidate selection and rejection guards.
- Page furniture and next-question contamination tests.
- Readiness/audit report schema tests.
- Subpart mark fillability audit tests.
- Conservative Asterion projection tests.

Missing regression guards:

- Math-fidelity scoring against a reviewed gold set.
- Explicit checks for broken powers, flattened fractions, bad integral limits, inverse trig, vectors/matrices, and inequalities.
- Tests proving OCR-selected text can remain `text_only_status=fail`.
- Tests separating OCR confidence from math fidelity.
- Tests for mark-scheme event parsing against degraded text.
- Tests that prevent topic/difficulty confidence from overriding degraded text.
- Tests that prevent DeepSeek high confidence from bypassing final review gates.

The existing OCR audit command exists and is useful. The missing command is a text-reliability audit that produces a dedicated report for Phase 2.

## 11. Documentation Assessment

README strengths:

- Clearly states the pipeline is image-first.
- Describes canonical question and mark-scheme PNGs.
- Explains OCR/native text as evidence rather than truth.
- Documents triage and audit commands.
- Documents DeepSeek sidecar behavior.

README weaknesses:

- The "Current Audited State" should be kept synchronized with the OCR-enabled standard output whenever the corpus is regenerated.
- It does not yet frame the project transition clearly enough: Phase 1 succeeded because images are canonical; Phase 2 begins because text is not yet canonical.
- It does not separate readability, math fidelity, structure, marks, student display, and generation gates.

Roadmap strengths:

- Evidence-gated iterations are explicit.
- Subpart marks, canonical text candidates, run manifests, Asterion exports, and topic/difficulty reruns are already listed.

Roadmap weaknesses:

- It should name the new phase directly as "Reliable Text Extraction."
- It should make full-bank text measurement the first milestone.
- It should block Content Lab generation until reviewed text/marks/skills exist.

Architecture documentation:

- There is no standalone `ARCHITECTURE.md`.
- `docs/TRUST_MODEL.md` is strong and should be the basis for a new architecture doc.

## 12. Recommended Architecture for Reliable Text Extraction

Recommended next architecture:

```text
Canonical image artifacts
  -> raw native/OCR text candidates
  -> text candidate scoring
  -> math-fidelity scoring
  -> structure and mark scoring
  -> selected text candidate
  -> text contract
  -> Asterion-safe projection
```

Text trust levels should be explicit:

- `search_hint`
- `classification_hint`
- `teacher_review_candidate`
- `readable_plain_text`
- `math_faithful_text`
- `student_safe_text`
- `generation_safe_text`

Recommended math-fidelity scoring dimensions:

| Dimension | Example failure |
| --- | --- |
| Powers | `x2` instead of `x^2`, `e2x` instead of `e^{2x}` |
| Subscripts | sequence terms and vector labels flattened |
| Fractions | `(1 3)/(9 x 2...)` or `1_{2}pi` |
| Roots | `vx + 1` for square root notation |
| Integrals | lost integral signs or bounds |
| Summations/sequences | missing indexes or recurrence syntax |
| Trig/log/exponential | `cos?x`, `ln3`, `cos-1` ambiguity |
| Equality/inequality | missing `=`, `<`, `>`, absolute bars |
| Brackets | unmatched parentheses, missing modulus bars |
| Minus signs | hyphen/en dash/minus confusion changing meaning |
| Merged variables | `ysin x`, `ddyx`, `Acar`, `Thediagram` |
| Vectors/matrices | row/column structure lost |
| Diagrams/tables | graph labels and table cells merged into prose |
| Page furniture | headers, barcode text, margin instructions |

Suggested scoring:

```json
"math_fidelity": {
  "score": 0.0,
  "trust": "low|medium|high|unknown",
  "flags": [
    "broken_power",
    "flattened_fraction",
    "missing_modulus_bar",
    "merged_variables"
  ],
  "requires_review": true
}
```

Student-safe text requires high readability, high math fidelity, clean structure, reliable marks, no contamination, valid mapping, and either human review or strong automated evidence against a gold set.

Generation-safe text requires a stricter superset: reviewed question text, reviewed mark-scheme text/mark events, reviewed topic/skill mapping, and explicit generation approval.

## 13. Recommended Roadmap

Milestone 1 - Full-Bank Text Reliability Measurement

Acceptance criteria:

- Dedicated report over `output/json/question_bank.json`.
- Counts for native, OCR, selected text, text roles/trust, visual required, text-only status, math corruption, page furniture, missing question numbers, missing subparts, missing mark brackets, subpart mark nulls, mark-scheme text risk, topic trust, and DeepSeek review mismatches.
- Lists suspicious record IDs and promising improvement IDs.
- No extraction behavior changes.

Milestone 2 - Text Contract Formalization

Acceptance criteria:

- Add or document `readability_trust`, `math_fidelity_trust`, `structure_trust`, `marks_trust`, `student_display_allowed`, `generation_allowed`, `search_allowed`, `review_required`, and reason codes.
- OCR confidence cannot imply math fidelity.
- DeepSeek confidence cannot imply truth.

Milestone 3 - Math Corruption Detector

Acceptance criteria:

- Unit tests for powers, fractions, integrals, limits, trig/log/exponential notation, vectors, matrices, inequalities, bad brackets, bad minus signs, merged variables, and diagram/table contamination.
- Detector flags known degraded examples.
- Clean plain-text/statistics questions are not unnecessarily blocked.

Milestone 4 - Subpart and Mark Recovery

Acceptance criteria:

- Reliable subpart labels and bracket marks populate subpart-level marks where safe.
- Disagreements are flagged rather than hidden.
- Multi-subpart questions have explicit structure trust.
- Tests cover missing, repeated, nested, and ambiguous subparts.

Milestone 5 - Mark-Scheme Text Structure

Acceptance criteria:

- Mark-code candidates detect B/M/A/DM/FT/SC style evidence.
- Alternative methods, dependencies, follow-through, and special cases are candidates with review status.
- Degraded mark schemes are quarantined.
- No student-facing generated explanations are produced in this phase.

Milestone 6 - Reviewed Gold Set

Acceptance criteria:

- At least 100 reviewed question-text records.
- At least 50 reviewed mark-scheme-text records.
- Coverage across P1, P3, P4, P5, algebra, calculus, vectors, statistics, mechanics, diagrams, and tables.
- Automated scoring can be evaluated against the set.

## 14. Suggested README Update

Paste-ready replacement opening:

````markdown
# CAIE 9709 Extraction Pipeline

This project extracts CAIE 9709 question-paper and mark-scheme PDFs into a paper-first exam-bank export.

Phase 1 succeeded because images are canonical; Phase 2 begins because text is not yet canonical.

The original project goal was image-first extraction: produce reliable question PNGs, mark-scheme PNGs, and versioned JSON metadata so every extracted question has a canonical visual source of truth. That image-first layer is now the foundation of the project, subject to ongoing QA.

The next primary goal is reliable text extraction. Extracted text should eventually become accurate enough to support search, review, routing, Asterion ingestion, mark-scheme parsing, and student-safe display where appropriate. Until a record passes explicit text-fidelity gates, rendered question and mark-scheme images remain the source of truth.

## Source of Truth

Question and mark-scheme PNGs are canonical.

Extracted text is metadata until proven otherwise. It may be used as:

- a search hint
- a classification hint
- a teacher-review aid
- a mark-scheme parsing candidate
- a student-facing display source only after strict trust gates pass

OCR confidence is not mathematical fidelity. DeepSeek confidence is not truth. Both must remain review-gated unless the record passes project-defined reliability checks.

## Current Strategic Focus

The project is moving from:

```text
Phase 1: Image-first extraction
```

to:

```text
Phase 2: Reliable text extraction
```

Phase 2 focuses on:

- measuring OCR/native text behavior across the full bank
- separating readability from mathematical fidelity
- preserving question numbers, subparts, and mark brackets
- improving mark-scheme text structure
- recovering subpart-level marks
- defining student-safe and generation-safe gates
- producing Asterion-safe exports that do not overtrust raw text
````

Paste-ready JSON contract addition:

````markdown
## Text Reliability Contract

The JSON export contains text fields, but not all text fields have the same purpose or trust level.

Recommended interpretation:

- `question_text`: selected text candidate for the question
- `question_text_role`: intended role of the selected text
- `question_text_trust`: coarse trust level for the selected text
- `text_only_status`: whether the question can be safely displayed without the canonical image
- `visual_required`: whether the image is required for safe use
- `visual_reason_flags`: reasons the image remains required
- `mark_scheme_text`: extracted mark-scheme text candidate
- `subparts`: detected subpart labels
- `subparts_solution_marks`: detected or inferred marks by subpart

Important rule:

`question_text_trust` and `ocr_text_trust` must not be interpreted as student-facing mathematical correctness unless the record also passes text-only and math-fidelity gates.

Recommended future field:

```json
"text_contract": {
  "selected_text_source": "native_pdf|ocr|hybrid|manual|none",
  "readability_trust": "high|medium|low|unknown",
  "math_fidelity_trust": "high|medium|low|unknown",
  "structure_trust": "high|medium|low|unknown",
  "marks_trust": "high|medium|low|unknown",
  "student_display_allowed": false,
  "generation_allowed": false,
  "search_allowed": true,
  "review_required": true,
  "reason_codes": ["math_fidelity_unverified", "visual_required"]
}
```
````

## 15. Suggested ROADMAP Update

Paste-ready roadmap section:

````markdown
## Phase 1 - Image-First Extraction Foundation

Status: substantially complete, with ongoing QA.

Goal: convert CAIE 9709 question-paper and mark-scheme PDFs into a reliable image-first question bank.

Established:

- PDF ingestion
- paper type detection
- question region extraction
- mark-scheme region extraction
- question-to-mark-scheme mapping
- paper-first output tree
- canonical question PNGs
- canonical mark-scheme PNGs
- versioned `question_bank.json`
- visual-first metadata fields
- conservative text-only gates
- optional OCR
- optional DeepSeek sidecar enrichment
- audit and triage tooling
- regression tests around core output behavior

Phase 1 principle: question and mark-scheme images are the canonical source of truth.

## Phase 2 - Reliable Text Extraction

Status: next primary goal.

Goal: bring extracted text closer to image-level reliability without weakening the image-first safety model.

### Milestone 2.1 - Full-Bank Text Measurement

Measure native text, OCR text, selected text, text-only status, visual-required status, math corruption, page furniture, question numbers, subparts, mark brackets, mark-scheme text, topic trust, and DeepSeek review mismatches.

Acceptance criteria:

- full-bank report exists
- OCR-selected and OCR-rejected samples are manually reviewed
- suspicious OCR-selected records are listed
- no readiness inflation is introduced
- no text status is improved without evidence

### Milestone 2.2 - Text Trust Model

Separate readability from mathematical fidelity.

Acceptance criteria:

- OCR confidence is not treated as math fidelity
- text-only display requires math-fidelity evidence
- downstream consumers can safely decide how text may be used

### Milestone 2.3 - Math Corruption Detection

Detect broken powers, flattened fractions, corrupted integrals, malformed trig/log/exponential notation, merged variables, missing brackets, bad minus signs, malformed vectors/matrices, and table/diagram contamination.

Acceptance criteria:

- detector has unit tests
- detector flags known degraded examples
- detector feeds the text contract

### Milestone 2.4 - Subpart and Mark Recovery

Improve subpart labels, subpart question text candidates, subpart mark allocation, mark-scheme subpart alignment, and total-mark consistency checks.

Acceptance criteria:

- reliable bracket marks populate subpart marks
- disagreements are flagged
- multi-part questions are represented at subpart level

### Milestone 2.5 - Mark-Scheme Text Structure

Parse mark schemes into reviewed candidate mark events.

Acceptance criteria:

- B/M/A/DM/FT/SC style marks are detected where text quality allows
- degraded mark schemes are quarantined
- mark events remain machine candidates until reviewed

### Milestone 2.6 - Reviewed Text Gold Set

Create a reviewed text set for evaluation across paper families and topic types.
````

## 16. Suggested ARCHITECTURE Update

Paste-ready new `ARCHITECTURE.md` opening:

````markdown
# Architecture

## Core Principle

The pipeline is image-first.

Question and mark-scheme PNGs are canonical. Text is a candidate layer until it passes explicit reliability gates.

This separation protects downstream systems from treating weak OCR, degraded native PDF text, or LLM enrichment as truth.

## Pipeline

```text
Input PDFs
  -> PDF discovery and paper classification
  -> question-paper / mark-scheme pairing
  -> question span detection
  -> question crop rendering
  -> mark-scheme region detection
  -> mark-scheme crop rendering
  -> question-to-mark-scheme mapping
  -> paper-first image artifact export
  -> native/OCR text candidate extraction
  -> trust and review metadata
  -> versioned JSON export
  -> optional sidecars:
     - DeepSeek enrichment
     - audit reports
     - triage samples
     - Asterion-safe projections
```

## Canonical Artifact Layer

The artifact layer contains question PNGs, mark-scheme PNGs, canonical paths, source PDF metadata, page references, crop diagnostics, and mapping diagnostics.

Downstream systems should be able to display a question safely from image assets even when text is degraded.

## Text Candidate Layer

The text layer contains candidate representations, not automatic truth. Sources may include native PDF text, OCR text, selected text, mark-scheme text, manually reviewed text, or future hybrid/layout-aware text.

Each selected text value must be interpreted through a text contract:

```json
"text_contract": {
  "selected_text_source": "ocr",
  "readability_trust": "medium",
  "math_fidelity_trust": "low",
  "structure_trust": "medium",
  "marks_trust": "low",
  "search_allowed": true,
  "classification_allowed": true,
  "teacher_review_allowed": true,
  "student_display_allowed": false,
  "generation_allowed": false,
  "requires_visual_display": true,
  "review_required": true,
  "reason_codes": ["math_fidelity_low", "text_only_status_fail"]
}
```

## DeepSeek Sidecar

DeepSeek enrichment is a sidecar. It may provide topic suggestions, difficulty estimates, confidence labels, rationales, and reconciliation metadata. It must not mutate the raw extraction export and must not be treated as truth.

## Asterion Projection

Asterion should ingest a conservative projection, not raw extraction records.

The projection should answer:

- Can Asterion display this question?
- Must it display the image?
- Is text-only display allowed?
- Can it be used for search?
- Can it be used for routing?
- Can it count toward readiness?
- Can it be used as a Content Lab source?
- Does it require human review?
````

## 17. Next Iteration Plan

First bounded iteration: Full-Bank Text Reliability Measurement.

Goal: measure current text reliability without changing extraction behavior.

Create one command or script:

```bash
.venv/bin/python -m exam_bank.cli audit-text-reliability --input output/json/question_bank.json
```

or:

```bash
.venv/bin/python scripts/audit_text_reliability.py --input output/json/question_bank.json
```

Required report fields:

- total records
- paper-family distribution
- canonical question image presence
- canonical mark-scheme image presence
- `question_text_role` distribution
- `question_text_trust` distribution
- `text_only_status` distribution
- `visual_required` distribution
- `visual_curation_status` distribution
- OCR-selected and OCR-rejected counts
- native/OCR/selected text score summaries
- suspected math corruption records
- page-furniture contamination records
- missing question-number records
- missing subpart-label records
- missing mark-bracket records
- subparts with null marks
- question/mark-scheme total mismatches
- low or uncertain topic records
- records that look text-promising but remain visual-required
- records that must remain image-only

Rules:

- Do not change extraction behavior.
- Do not tune OCR thresholds.
- Do not add a new OCR engine.
- Do not alter crop detection.
- Do not alter mark-scheme mapping.
- Do not alter DeepSeek behavior.
- Do not mark any record text-only safe.
- Do not weaken review/fail gates.
- Do not treat OCR confidence as math fidelity.
- Do not treat DeepSeek labels as truth.

Acceptance criteria:

- Full tests pass.
- Report runs on the current bank.
- Report gives actionable counts and suspicious IDs.
- Missing optional OCR fields are handled.
- No generated large artifacts are committed.
- Next-loop recommendations are based on report evidence.

## 18. Open Questions

- Should older no-OCR audit/handoff materials be archived or labeled more clearly so they cannot be confused with the current OCR-enabled standard output?
- Is the missing exact `33autumn25` mark scheme available, or should those 11 records remain excluded from product-ready views?
- Should `question_text_trust=high` be renamed or decomposed to avoid downstream overinterpretation?
- What is the minimum review standard for `student_safe_text`?
- Who owns the reviewed text gold set and mark-scheme gold set?
- Should Asterion ingest only a generated projection file, never raw `question_bank.json`?
- What fields are mandatory for Content Lab generation approval?
- Should `src/exam_bank/asterion_export.py` and related output sidecars be promoted, revised, or kept out of the current commit?

## Appendix A - Evidence Snapshot

Current `git status --short` showed existing modified and untracked files before this review document was added, including modified `README.md`, `cli.py`, `ocr.py`, `pipeline.py`, tests, and untracked Asterion/export artifacts. Those changes were not reverted.

`git diff --stat` before this document showed 122 insertions and 4 deletions across existing files.

DeepSeek sidecar evidence:

| Field | Distribution |
| --- | --- |
| DeepSeek confidence | `high=729`, `medium=511`, `low=50`, `missing=11` |
| DeepSeek review required | `false=938`, `true=352`, `missing=11` |
| Final review required | `true=1246`, `false=44`, `missing=11` |
| Topic reconciliation | `match=893`, `mismatch=243`, `unmapped_label=154`, `missing=11` |
| Difficulty reconciliation | `match=885`, `mismatch=405`, `missing=11` |
| Text-only enrichment risk | `high=915`, `medium=280`, `low=95`, `missing=11` |
| High DeepSeek confidence but final review required | 688 |

Representative sample categories:

| Category | Matching count | First 10 sample IDs |
| --- | ---: | --- |
| High-confidence image records | 252 | `32spring21_q04`, `32spring21_q10`, `32summer21_q01`, `32summer21_q02`, `32summer21_q07`, `51summer21_q06`, `11autumn21_q04`, `11autumn21_q05`, `31autumn21_q07`, `32autumn21_q11` |
| Text-only fail records | 906 | `12spring21_q01`, `12spring21_q02`, `12spring21_q03`, `12spring21_q04`, `12spring21_q05`, `12spring21_q06`, `12spring21_q07`, `12spring21_q08`, `12spring21_q09`, `12spring21_q10` |
| OCR-selected records | 29 | `42spring21_q02`, `52spring21_q05`, `12summer21_q02`, `12summer21_q03`, `41summer21_q03`, `41summer21_q04`, `51summer21_q05`, `51summer21_q07`, `53summer21_q06`, `43autumn21_q04` |
| OCR-rejected records | 1272 | `12spring21_q01`, `12spring21_q02`, `12spring21_q03`, `12spring21_q04`, `12spring21_q05`, `12spring21_q06`, `12spring21_q07`, `12spring21_q08`, `12spring21_q09`, `12spring21_q10` |
| P3 records | 396 | `32spring21_q01`, `32spring21_q02`, `32spring21_q03`, `32spring21_q04`, `32spring21_q05`, `32spring21_q06`, `32spring21_q07`, `32spring21_q08`, `32spring21_q09`, `32spring21_q10` |
| Diagram/table records | 575 | `12spring21_q02`, `12spring21_q04`, `12spring21_q05`, `12spring21_q06`, `12spring21_q10`, `12spring21_q11`, `32spring21_q04`, `32spring21_q08`, `32spring21_q10`, `42spring21_q03` |
| Multi-subpart records | 968 | `12spring21_q01`, `12spring21_q05`, `12spring21_q06`, `12spring21_q07`, `12spring21_q08`, `12spring21_q09`, `12spring21_q10`, `12spring21_q11`, `32spring21_q04`, `32spring21_q05` |
| Mark-scheme-heavy records | 837 | `12spring21_q05`, `12spring21_q09`, `12spring21_q10`, `12spring21_q11`, `32spring21_q05`, `32spring21_q07`, `32spring21_q08`, `32spring21_q09`, `32spring21_q10`, `42spring21_q03` |
| Low/degraded topic trust records | 958 | `12spring21_q01`, `12spring21_q02`, `12spring21_q03`, `12spring21_q04`, `12spring21_q05`, `12spring21_q06`, `12spring21_q07`, `12spring21_q08`, `12spring21_q09`, `12spring21_q10` |
| Promising text-improvement records | 696 | `12spring21_q01`, `12spring21_q09`, `12spring21_q10`, `12spring21_q11`, `32spring21_q02`, `32spring21_q05`, `32spring21_q08`, `32spring21_q09`, `42spring21_q01`, `42spring21_q02` |

Representative sampled-record classifications:

| Record | Sample tags | Image asset | Q text | MS text | Subparts | Marks | Topic | Asterion mode | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `12spring21_q01` | text fail, OCR rejected, multi-subpart | good | degraded | mark hint | partial | bad | plausible | image-only | recover subpart marks; math compare native/OCR |
| `12spring21_q11` | diagram, multi-subpart, heavy MS | good | search hint only | degraded | partial | bad | good | text-assisted | fix merged words and fractional powers |
| `32spring21_q04` | high image, P3, diagram | good | degraded | mark hint | partial | bad | good | text-assisted | math-fidelity review |
| `32spring21_q10` | high image, P3, diagram | good | degraded | degraded | partial | bad | good | text-assisted | protect trig/power/interval notation |
| `42spring21_q02` | OCR selected, mechanics | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | keep OCR gated; fix units/angle notation |
| `52spring21_q05` | OCR selected, table | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | table extraction/layout review |
| `12summer21_q03` | OCR selected, table/curve | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | detect root/table corruption |
| `11autumn22_q10` | OCR selected, diagram | questionable crop | unsafe | degraded | partial | bad | plausible | image-only | flag `x2` power corruption |
| `31autumn23_q02` | OCR selected, P3 complex | good image, text poor | unsafe | mark hint | none | partial | plausible | image-only | restore modulus/argument notation |
| `33autumn25_q01` | missing MS image | bad for product | unsafe | missing | none | bad | plausible | blocked | obtain exact mark scheme |
| `32summer21_q01` | high image, P3 | good | search hint | mark hint | none | partial | degraded topic | text-assisted | verify notation and topic |
| `32summer21_q02` | high image, P3, diagram | good | search hint | mark hint | none | partial | good | text-assisted | visual dependency remains |
| `51summer21_q06` | high image, multi-subpart | good | readable | mark hint | partial | bad | good | text-assisted | fill subpart marks before text-only |
| `11autumn21_q04` | high image, multi-subpart | good | readable/search | mark hint | partial | bad | good | text-assisted | validate math symbols |
| `31autumn21_q07` | high image, P3, heavy MS | good | search hint | degraded | partial | bad | good | text-assisted | mark-scheme parsing review |
| `32autumn21_q11` | high image, P3, diagram | good | degraded | degraded | partial | bad | plausible | text-assisted | fix diagram/math dependency |
| `41summer21_q03` | OCR selected | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | keep text as search hint only |
| `41summer21_q04` | OCR selected | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | mechanics notation review |
| `51summer21_q05` | OCR selected, table | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | table extraction review |
| `53summer21_q06` | OCR selected, multi-subpart | questionable crop | unsafe | mark hint | partial | bad | plausible | image-only | math/text contract review |

Interpretation note: image asset classifications above use artifact existence, crop confidence, mapping status, validation status, and sampled text inspection. They are not a substitute for a complete human pixel-level review of every crop.
