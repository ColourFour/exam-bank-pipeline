# Math Normalization Spike

This spike adds an advisory-only normalization candidate to the frozen text-fidelity fixture report. It does not change production extraction, canonical question images, mark-scheme images, `question_text`, raw native PDF text, raw OCR text, question-bank JSON, or Asterion exports.

Canonical question and mark-scheme images remain the source of truth. Normalized text is a review candidate only.

## Scope

- Code path: `scripts/report_text_fidelity_fixtures.py --include-normalized`
- Input fixture: `tests/fixtures/text_fidelity/bad_text_records.json`
- Report outputs:
  - `output/reports/text_fidelity_fixture_baseline_normalized.json`
  - `output/reports/text_fidelity_fixture_baseline_normalized.md`
- Per-record raw fields preserved in normalized reports:
  - `native_pdf_text_raw`
  - `ocr_text_raw`
  - `selected_text_raw`
- Per-record advisory candidate fields:
  - `question_text_normalized`
  - `normalization_flags`
  - `normalization_confidence`
  - `normalization_warnings`
  - `normalization_is_advisory`

## Normalization Rules

The rules are conservative regex repairs for repeated fixture artifacts:

- Unicode math glyph cleanup: en/em/minus dashes, ligatures, prime marks.
- Fraction flattening: fixture-style `1_{2}` and stacked numeric fragments to advisory `1/2`.
- Power cleanup: `?` as a possible lost square only next to formula punctuation, and `e^{-}2x` to `e^{-2x}`.
- Root cleanup: known fixture root spans such as `vx + 1+3` to `sqrt(x + 1) + 3`, with warnings.
- Trig/log cleanup: `cos 21` to `cos(2θ)` only when `θ` appears elsewhere; `ln x^{4}` to `ln(x^{4})`.
- Vector/matrix cleanup: glyph artifacts such as `---OM¿` to `vector(OM)` and `@ 41A` to `column_vector(4,1)`, with warnings.
- Inequality cleanup: `in equal it y` to `inequality`; fixture `x20` to `x > 0`, with warning.
- CAIE subpart layout: spaces and line breaks around `(a)`, `(b)`, `(i)`, etc.
- Common spacing repairs for merged fixture prose.

Each changed rule emits a flag. Inferred repairs emit warnings and reduce confidence.

## Results

The normalized fixture baseline contains 36 records.

- Issue-level improvements: 4 records
- Clearer failure classifications: 26 records
- Unchanged by normalization: 6 records
- Total records with measurable improvement or clearer classification: 30

Issue-level improvements:

- `12summer23_q01`: `ddyx` repaired to `dy/dx`, resolving the derivative-notation expectation.
- `35summer25_q04`: `x20` repaired to advisory `x > 0`, resolving the inequality expectation.
- `12spring22_q08`: lost square glyphs in the circle equation are made explicit enough to clear a math-signal-loss check, but remain warning-bearing.
- `12autumn23_q06`: the translation-vector glyph is classified as vector/matrix ambiguity instead of generic math-symbol loss.

Clearer-but-not-repaired examples:

- `12summer21_q03`: root notation is normalized to `sqrt(...)`, but the root span is inferred and needs image review.
- `33autumn21_q05`: `cos 21` becomes advisory `cos(2θ)`, but this is inferred from nearby theta symbols.
- `31summer21_q01`, `32summer21_q01`, `33autumn21_q02`: `in equal it y` is normalized to `inequality`, improving readability without changing mathematical content.
- Several diagram/table fixtures gain spacing and subpart-line-break flags, clarifying that their failures are mostly reading-order or table-layout problems, not fully recoverable by text normalization.

Unchanged examples:

- `33summer24_q03`: missing mark bracket cannot be safely restored from selected text.
- `33autumn25_q07`: truncated ending and missing mark bracket remain unresolved.
- `32spring23_q04`: complex denominator corruption remains unresolved.
- `11summer23_q01` and `12spring21_q02`: selected text is already plausible enough that no fixture rule fires.

## False-Positive Risks

- `?` to `^{2}` is intentionally limited to formula punctuation. Plain prose such as `Why is x?` is not rewritten.
- Root spans are hard to infer from linearized OCR/PDF text; root repairs are low-authority and warning-bearing.
- `cos 21` to `cos(2θ)` only fires when theta appears elsewhere, but it still requires image review.
- Vector/matrix glyphs such as `@` and `¿` are ambiguous; these repairs are useful for classification, not canonical display.
- Spacing repairs can improve similarity scores without truly fixing reading order. The report distinguishes these as clearer classifications unless an expectation check is actually resolved.

## Recommendation

Normalization is useful as a fixture/report advisory layer and as a triage aid. It should not be promoted broadly into production exports yet.

Before any broader implementation pass:

- Keep the candidate field separate from `question_text`, `ocr_text`, and raw native text.
- Require explicit provenance fields and warning flags.
- Evaluate against image-reviewed ground truth, not only fixture structural checks.
- Separate low-risk formatting cleanup from inferred math repairs.
- Add per-rule precision checks before enabling any candidate outside fixture reports.
