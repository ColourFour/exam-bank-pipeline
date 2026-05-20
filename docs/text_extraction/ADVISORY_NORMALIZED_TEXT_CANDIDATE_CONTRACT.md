# Advisory Normalized Text Candidate Contract

Date: 2026-05-20

Scope: contract-first schema for storing or reporting normalized text candidates produced from selected/native/OCR text. This contract is advisory-only. It does not change extraction behavior, OCR/native selection, canonical question images, mark-scheme images, `question_text`, raw native PDF text, raw OCR text, question-bank JSON meaning, or Asterion exports.

## Boundary

Canonical question images and mark-scheme images remain the source of truth. Raw selected, native, OCR, and profile OCR text must be preserved exactly as extracted. A normalized candidate is a separate review artifact that may help classification, search triage, or future reviewer workflows, but it must never overwrite raw selected/native/OCR text and must never be treated as canonical student-display text.

The math-normalization spike showed useful report value, but only a small number of issue-level fixture failures were actually improved. Many repairs are inferred, warning-bearing, and require image review. This contract therefore requires provenance, warnings, and explicit display/export blocks.

## Approved Locations

Normalized candidates may be written only to report or sidecar outputs, for example:

- `output/reports/text_fidelity_fixture_baseline_normalized.json`
- Future report-only sidecar: `output/reports/advisory_normalized_text_candidates.json`

They must not be added to Asterion exports unless a later export contract explicitly approves advisory fields with provenance and warnings. They must not replace `question_text`, `currently_selected_text`, `native_pdf_text_raw`, `ocr_text_raw`, or profile OCR text.

## Schema

Schema name: `exam_bank.advisory_normalized_text_candidate`

Schema version: `1`

Per-candidate required fields:

| Field | Requirement |
| --- | --- |
| `candidate_id` | Deterministic id for this normalized candidate. Recommended format: `normcand_<hash-prefix>`. |
| `record_id` | Stable source record/question id. |
| `source_text_kind` | Raw source kind used to build the candidate, such as `selected_text`, `native_pdf_text_raw`, `ocr_text_raw`, `profile_ocr_text`, `native_pdf_text_candidate`, or `ocr_text_candidate`. |
| `source_text_hash` | SHA-256 hash of the exact raw source text. This proves raw text preservation without duplicating every source field in every candidate. |
| `question_text_normalized` | Normalized advisory candidate string. This is not canonical text. |
| `normalization_flags` | List of rule families that changed text, such as `fraction_notation_normalized` or `root_notation_normalized`. Required even when empty. |
| `normalization_confidence` | Number from `0` to `1`; confidence in the normalization candidate, not in canonical correctness. |
| `normalization_warnings` | List of warnings. Required even when empty; inferred math repairs should emit warning entries. |
| `normalization_is_advisory` | Must be `true`. Candidates with this field missing or false are invalid. |
| `created_by_version` | Rule or generator version, for example `math_normalization_rules_v1`. |
| `created_at` | Build timestamp. For reproducible fixture reports, use a deterministic build timestamp policy instead of wall clock time. |
| `provenance` | Non-empty object describing source report, normalizer, source text kind, rule version, and raw-text preservation evidence. |
| `display_allowed` | Must default to `false`. Later approval requires a separate display decision and image-reviewed evidence. |
| `export_allowed` | Must default to `false`. Later approval requires a separate export-contract decision. |

## Deterministic Build Timestamp Policy

Fixture and CI reports should not use wall clock time. Use a deterministic timestamp such as `1970-01-01T00:00:00Z`, or a caller-provided reproducible build timestamp, and include the rule version in `created_by_version`. If a runtime-generated sidecar later needs wall clock timestamps, that policy must be documented in the sidecar producer and must not affect candidate identity.

## Candidate Identity

`candidate_id` should be derived from stable inputs:

- schema name and version
- `record_id`
- `source_text_kind`
- `source_text_hash`
- `question_text_normalized`
- `created_by_version`

Do not include `display_allowed`, `export_allowed`, or non-deterministic timestamps in candidate identity.

## Raw Preservation Requirements

Any producer using this contract must:

1. Preserve raw selected/native/OCR/profile text fields exactly.
2. Store normalized text separately from canonical and raw text fields.
3. Link the candidate back to the raw source with `source_text_kind` and `source_text_hash`.
4. Include provenance and warning fields for every candidate.
5. Keep `normalization_is_advisory=true`.
6. Default `display_allowed=false` and `export_allowed=false`.
7. Leave Asterion exports unchanged until a later export contract explicitly approves advisory candidate fields.

## Provenance Requirements

`provenance` must be non-empty and should include:

- source report or sidecar path
- normalizer function or tool name
- normalizer rule version
- source text location or source field name
- raw-text preservation statement
- related fixture/report ids when available

Example:

```json
{
  "source_report": "output/reports/text_fidelity_fixture_baseline_normalized.json",
  "normalizer": "exam_bank.text_normalization.normalize_advisory_question_text",
  "rule_version": "math_normalization_rules_v1",
  "raw_text_preserved": true
}
```

## Non-Goals

This contract does not approve:

- replacing selected text
- changing canonical question-bank meaning
- using normalized text as student-display text
- exporting normalized text through Asterion
- claiming normalized text is accurate enough without image-reviewed evidence

## Implementation Hook

The contract validator lives in `src/exam_bank/text_normalized_candidate.py`. It is intentionally separate from Asterion export code and canonical question-bank generation.
