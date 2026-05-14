# Roadmap

This roadmap is evidence-gated. The pipeline remains image-first: rendered question PNGs and rendered mark-scheme PNGs are the student-facing source of truth. Extracted text, OCR text, native PDF text, future vision text, topic labels, difficulty labels, crop confidence, validation fields, mapping fields, and readiness fields are support metadata for search, review, QA, and future adaptive practice.

## Current Evidence

The current measured-state baseline is [Project Audit and Optimization Review](docs/PROJECT_AUDIT_AND_OPTIMIZATION_REVIEW.md). Use that report for current counts, OCR status, generated-output names, known risks, and validation status. As of that audit, dated `2026-05-14`, `output/json/question_bank.json` is OCR-enabled; older roadmap entries that describe a no-OCR canonical export are historical iteration evidence, not current state.

Refresh current evidence with:

```bash
.venv/bin/python -m exam_bank.cli audit \
  --input output/json/question_bank.json

.venv/bin/python scripts/audit_question_bank_readiness.py \
  --input output/json/question_bank.json \
  --baseline output/triage/iteration_004/baseline_question_bank.json \
  --artifact-root output \
  --out-dir output/audits/manual
```

The iteration sections below preserve useful planning rationale and prior audit evidence. Do not copy their measured counts into current-state docs without rerunning the audit commands above.

## iteration_001 - Audit/reporting layer

Status: completed by Agent 2; pending/paired with Agent 3 focused tests as the next formal gate.

Completed:

- Implemented `scripts/audit_question_bank_readiness.py`.
- Wrote implementation notes at `agent_handoffs/iteration_001/agent2_impl_notes.md`.
- Ran the full-bank audit into ignored `output/audits/iteration_001/`.
- Confirmed reliable baseline comparison: `0` added, `0` removed, `1301` shared.
- Passed validation commands including `py_compile` and the focused existing test set: `34 passed`.

Historical documented audit result from `iteration_001`:

- At that iteration, the audited canonical export was not OCR-enabled.
- OCR-selection quality cannot yet be judged because OCR text is blank.
- Hard blockers: `23`.
- Mapping: `pass: 1281`, `fail: 20`.
- Validation: `pass: 921`, `review: 371`, `fail: 9`.
- Asterion tiers: `0: 23`, `1: 360`, `2: 185`, `3: 13`, `4: 52`, `5: 668`.
- Simple-fillable subpart mark candidates: `920`.

## iteration_002 - Agent 3 audit tests

Goal: add focused tests for the new reporting script.

Status: required gate before further extraction changes. The current worktree contains `tests/test_question_bank_readiness_audit.py`; keep this iteration explicit until those tests are accepted with the branch.

Scope:

- Field discovery.
- Top-level vs `notes` field resolution.
- Missing optional fields.
- Missing OCR fields.
- No fake zeroes for missing numeric values.
- Readiness tier classification.
- Hard blocker detection.
- Mapping/validation distribution.
- Subpart fillable detection.
- Baseline comparison.
- Deterministic output shape.

Non-goals:

- No OCR engine changes.
- No extraction behavior changes.
- No crop fixes.
- No mark-scheme mapping fixes.
- No topic/difficulty changes.
- No network/API-key tests.

## iteration_003 - Resolve OCR activation/export disconnect

Goal: determine why the canonical audited export has `ocr_ran=0`, blank `ocr_text`, and `text_candidate_source=native: 1301`.

Questions to answer:

- Is OCR disabled in config?
- Is OCR running but not exported?
- Is the audit reading the wrong source field?
- Is there a difference between historical/uploaded JSONs and current canonical repo output?
- Are top-level OCR fields and `notes` OCR fields being populated differently?
- Is the export intentionally native-only?

Deliverables:

- Root-cause note.
- Minimal fix if needed.
- Rerun audit after fix.
- Compare OCR activity before and after.

Non-goals:

- No OCR threshold tuning.
- No broad OCR quality tuning.
- No new OCR engine.
- No schema-breaking changes.

## iteration_004 - Fix hard blockers

Goal: reduce hard blockers as close to zero as possible.

Use `hard_blockers.csv` and `mapping_validation_report.csv`.

Focus:

- Mapping failures.
- Validation failures.
- Missing mark-scheme text/image paths.
- Records where mapping fail coexists with validation pass.
- Local mark-total mismatches.
- Records with missing or unusable marks.
- Artifact path failures if present.

Acceptance criteria:

- Hard blocker count reduced.
- Mapping fail count reduced.
- Validation fail count reduced.
- No record with mapping fail is presented as cleanly validation-pass without a contradiction flag.
- Audit rerun documents before/after changes.

## iteration_005 - Dedicated 2024-2025 layout/crop recovery

Goal: improve newer CAIE format records if the audit confirms they are disproportionately weak.

Focus:

- `caie_2024_2025` or equivalent format profile.
- Weak question anchors.
- Side panels.
- Barcode/page furniture exclusion.
- Terminal mark totals.
- Multi-page continuation.
- Diagram/text region union.
- Avoiding next-question contamination.
- Improving question crop confidence.
- Improving visual curation status.

Deliverables:

- Format-profile before/after metrics.
- Crop-quality before/after metrics.
- Validation/mapping before/after metrics.

## iteration_006 - Crop metadata and artifact QA

Goal: make image crops auditable and reproducible.

Proposed metadata:

```json
{
  "question_crop_bbox": [],
  "mark_scheme_crop_bbox": [],
  "crop_source_pages": [],
  "crop_includes_diagram": true,
  "crop_excludes_page_furniture": true,
  "crop_validation_reason": "",
  "image_sha256": "",
  "image_width": 0,
  "image_height": 0
}
```

Also add or strengthen artifact existence checks.

Acceptance criteria:

- Audit can verify image paths.
- Crop metadata exists for generated artifacts.
- Artifact-root checks are deterministic.
- Missing image artifacts become hard blockers or explicit review flags.

## iteration_007 - Promote subpart marks

Goal: turn detected mark values into usable subpart mark fields.

Use the audit finding that `920` records look simple-fillable.

Proposed structure:

```json
"subpart_marks": [
  {
    "part_path": ["a"],
    "marks": 3,
    "source": "question_mark_bracket",
    "confidence": "high"
  },
  {
    "part_path": ["b", "i"],
    "marks": 2,
    "source": "question_mark_bracket",
    "confidence": "medium"
  }
]
```

Support nested paths such as `["a", "i"]`, `["a", "ii"]`, and `["b"]`.

Acceptance criteria:

- Simple fillable records are populated.
- Nested records are not incorrectly flattened.
- Detected marks sum to question total where possible.
- Subpart marks agree with mark-scheme totals where available.
- Ambiguous records remain review, not falsely ready.

## iteration_008 - Canonical text candidate system

Goal: separate raw text candidates from canonical text.

Do this only after OCR activation/export state is resolved.

Move toward fields like:

```json
{
  "native_pdf_text_raw": "",
  "ocr_text_raw": "",
  "vision_text_raw": "",
  "canonical_question_text": "",
  "canonical_question_latex": "",
  "question_text_trust": "",
  "text_validation_flags": [],
  "text_needs_image": true
}
```

Principles:

- Image artifacts remain canonical for visual-required records.
- Text-ready means text preserves question number, marks, subparts, mathematical content, and scope.
- OCR/native/vision text are candidates, not automatically student-facing text.
- Do not inflate readiness/trust because text looks prettier.

## iteration_009 - Run manifest and export contract

Goal: make future JSONs auditable from the file alone.

Status: substantially implemented. New exports include `run_manifest`, `artifact_root`, QA rollups, and `output_layout` profile metadata. Generated-output inventory and dry-run cleanup planning are available through `exam_bank.cli output-inventory` and `exam_bank.cli output-cleanup-plan`.

Add top-level run metadata such as:

```json
"run_metadata": {
  "generated_at": "...",
  "run_id": "...",
  "pipeline_version": "...",
  "git_commit": "...",
  "model_versions": {
    "vision_text": "...",
    "topic_classifier": "...",
    "difficulty_model": "local-difficulty-v1"
  },
  "ocr_engine_version": "...",
  "input_manifest_sha256": "...",
  "artifact_root": "...",
  "qa_summary": {}
}
```

Acceptance criteria:

- Future audits can identify when/how the JSON was produced.
- Model/pipeline versions are explicit.
- Artifact root is explicit.
- QA summary is embedded.
- Output layout profile is explicit.
- Generated output can be inventoried without deleting or moving files.

## iteration_010 - Tiered Asterion exports

Goal: generate separate exports for different uses.

Current compatibility step: conservative Asterion exports now default to `output*/asterion/exports/latest/` instead of mixing sidecars into `json/` or paper artifact folders. Tiered slices remain future work.

Suggested exports:

```text
asterion_gold.json
asterion_multimodal.json
question_bank_master.json
```

Suggested meanings:

- `asterion_gold.json`: clean pilot subset.
- `asterion_multimodal.json`: image-canonical records with clean mapping/validation and usable mark schemes.
- `question_bank_master.json`: full bank with all flags retained for continued extraction work.

Do not implement until readiness tier semantics are stable.

## iteration_011 - Topic/difficulty rerun

Goal: rerun or improve topic/difficulty only after extraction quality improves.

Do not optimize topic/difficulty before crop, mapping, marks, and canonical text are more reliable.

Current topic/difficulty should be treated as support metadata, not final truth, especially where text fidelity is degraded.
