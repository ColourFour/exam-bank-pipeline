# Bad Text Fixture Notes

Date: 2026-05-20

## Purpose

`tests/fixtures/text_fidelity/bad_text_records.json` freezes a deterministic set of visually suspicious advisory-text records from `docs/text_extraction/TEXT_EXTRACTION_FAILURE_AUDIT.md`.

The fixture set is for improving text-fidelity review and regression tests only. Canonical question images and mark-scheme images remain the source of truth. Normalized text, OCR text, native PDF text, selected text, topic labels, difficulty labels, and advisory fields must not be promoted to canonical status from this fixture.

## Source

- Primary source: `docs/text_extraction/TEXT_EXTRACTION_FAILURE_AUDIT.md`
- Current bank used to populate record metadata and text excerpts: `output/json/question_bank.json`
- Fixture count: 36 records
- No OCR was re-run.
- No production extraction behavior was changed.
- No canonical image outputs were changed.

The current bank does not always preserve a separate raw native-PDF text field once text selection has happened. For records whose selected source profile is `native_pdf`, `native_pdf_text_raw` is populated from the currently selected question text. For `hybrid` or `ocr` records, it is left as `null` unless a separately preserved native field becomes available later. OCR text is copied from the current `ocr_text` field when present.

## Coverage

The manifest intentionally includes more than the minimum acceptance threshold:

- 36 total records.
- More than 6 failure categories represented.
- More than 10 math-notation-related records.
- More than 5 crop-boundary, reading-order, page-furniture, table, or diagram-contamination records.
- P1, P3, P4, and P5 examples.
- Native, OCR, and hybrid text source profiles.

Primary fixture lanes:

- Math layout: fractions, powers, integrals, derivatives, trig symbols, complex numbers, vectors, inequalities, and polynomial layout.
- Structural anchors: question number presence, displaced question number, mark bracket presence, and subpart order.
- Reading order and contamination: diagrams, mechanics graphs, tables, axis labels, answer-space artifacts, and page furniture.
- Clean-crop degraded text: records where a visually clean crop still has unsafe advisory text.
- OCR/native disagreement: records where OCR and selected/native text differ in ways that are useful for future selection tests.

## Expected Text

The fixture does not attempt to provide perfect full transcriptions. Most records use structural expectations because the source images are canonical and many prompts are math-heavy. Examples include:

- Contains the expected question number.
- Contains the expected mark bracket.
- Preserves subpart labels and order.
- Preserves denominator, integral-bound, vector, matrix, inequality, or Greek-symbol structure.
- Does not include unrelated page furniture or answer-space artifacts.
- Does not truncate before the end of the prompt.

## Validation

Run:

```bash
.venv/bin/python -m pytest tests/test_text_fidelity_fixture_manifest.py
```

The validation test checks that:

- The fixture has 30 to 50 records.
- Record IDs are unique and resolve in `output/json/question_bank.json`.
- Referenced question and mark-scheme image paths exist under `output/`.
- Each record has structural expectations and failure tags.
- Category coverage meets the fixture acceptance thresholds.

## Known Gaps

The audit found no confirmed active next-question contamination in current selected text. The fixture therefore covers adjacent risks such as displaced question anchors, page furniture, table/diagram reading-order contamination, and crop-boundary suspicion rather than claiming confirmed next-question contamination.

The fixture also does not guarantee that every expected structural assertion is machine-checkable today. It is designed as a stable review and regression corpus for future advisory-text improvements.
