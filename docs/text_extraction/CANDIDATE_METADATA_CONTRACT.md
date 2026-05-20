# Candidate Metadata Contract

Date: 2026-05-20

Scope: additive metadata for future advisory text gates, measurement, and candidate replay. This contract does not change OCR/native selection, selected question text, canonical image paths, mark-scheme image paths, or Asterion exports.

## Boundary

Canonical question images and mark-scheme images remain the source of truth. Candidate metadata is advisory evidence only. It may explain why text should be reviewed, replay a selector decision, or compare future candidate gates, but it must not overwrite `question_text`, `question_image_path`, `canonical_question_artifact`, or any Asterion export field.

## Sidecar Location

Use sidecar/report files outside the canonical bank:

- Coverage report: `output/reports/text_candidate_metadata_coverage.json`
- Human-readable report: `output/reports/text_candidate_metadata_coverage.md`
- Future per-record sidecar, if persisted: `output/reports/text_candidate_metadata_sidecar.json`

The canonical `output/json/question_bank.json` remains read-only for this workflow.

## Sidecar Schema

Schema name: `exam_bank.text_candidate_metadata.sidecar`

Schema version: `1`

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_name` | Must be `exam_bank.text_candidate_metadata.sidecar`. |
| `schema_version` | Contract version. |
| `record_count` | Number of candidate metadata records. |
| `records` | Candidate metadata records keyed by `record_id`. |

Per-record fields:

| Field | Meaning |
| --- | --- |
| `record_id` | Stable question id, matching `question_id` in the bank. |
| `paper_id` | Paper/session id. |
| `paper_family` | P1/P3/P4/P5 family. |
| `question_number` | Source question number. |
| `canonical_question_artifact` | Reference only; never modified by this sidecar. |
| `question_image_path` | Reference only; never modified by this sidecar. |
| `mark_scheme_image_path` | Reference only; never modified by this sidecar. |
| `crop_metadata` | Crop dimensions, confidence, page bboxes, normalized area when page dimensions are available. |
| `candidates` | Raw native/OCR/profile candidate windows, provenance, confidence, and warnings. |
| `selector` | Decision, selected source, accepted/rejected reasons, selector warnings, structural warnings. |
| `line_geometry` | Text-line bounding boxes when available. |
| `missing_metadata` | Honest list of unavailable fields and why they were not inferred. |

Candidate entry fields:

| Field | Meaning |
| --- | --- |
| `source` | `native`, `ocr`, `ocr_profile`, `normalized_advisory`, or another explicit candidate source. |
| `text_window` | Raw text window for replay. Empty only when missing metadata is documented. |
| `window_scope` | Crop/page/span scope used to produce the text window. |
| `provenance` | Engine/profile/source PDF details needed to reproduce the candidate. |
| `confidence` | Candidate confidence if available; otherwise null with a missing-metadata reason. |
| `warnings` | Candidate-specific warnings. |
| `rejected_reasons` | Reasons the selector rejected this candidate, if rejected. |
| `selected` | Whether this candidate was the selector's chosen advisory text. |

## Target Metadata Status

Available now:

- Crop confidence from `notes.question_crop_confidence`.
- Candidate provenance from `notes.text_candidate_source`, `notes.text_source_profile`, and `notes.text_candidate_decision`.
- Raw OCR candidate text from `ocr_text`.
- Rejected/decision reasons from `notes.ocr_rejected_reasons` and `notes.text_candidate_decision_reasons`.
- Selector warnings from review, validation, extraction-quality, and text-fidelity flags.
- Selector structural warnings derived from existing structural flags and decision reasons.
- Crop pixel dimensions can be resolved from referenced question PNG files when those files exist.

Partially available:

- Raw native text candidate window is only recoverable when native text was selected. OCR-selected records do not retain an independent raw native candidate window in the current canonical bank.
- Crop region bboxes exist in PDF-space diagnostics, but they are region-level diagnostics, not text-line boxes.

Missing now:

- Page-normalized crop area, because current diagnostics do not include source page dimensions.
- True text-line bounding boxes linked to candidate spans.
- Rejected candidate text windows with crop/page positions.
- Candidate-level crop confidence when multiple candidate windows exist.

## Non-Mutation Requirements

Any implementation using this contract must:

1. Read `output/json/question_bank.json` without mutating it.
2. Write candidate metadata only to sidecar/report paths.
3. Preserve OCR/native source selection exactly.
4. Preserve selected question text exactly.
5. Preserve canonical image and mark-scheme paths exactly.
6. Leave Asterion exports unchanged unless a later export contract explicitly adds advisory-only fields.
