# OCR Profile Experiment

This is a fixture-only experiment against `tests/fixtures/text_fidelity/bad_text_records.json`.
It does not change production OCR selection, global OCR thresholds, canonical crops, `question_bank.json`,
Asterion exports, or the image-first source-of-truth policy.

## Run

- Command: `.venv/bin/python scripts/experiment_ocr_profiles.py --fixtures tests/fixtures/text_fidelity/bad_text_records.json`
- JSON report: `output/reports/ocr_profile_experiment.json`
- Markdown report: `output/reports/ocr_profile_experiment.md`
- Fixture records: 36
- Runtime blockers: none; Pillow, pytesseract, and local Tesseract were available.

## Results

| Profile | Average fixture score | Improved | Regressed | Total runtime | Average runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| formula_heavy | 91.25 | 13 | 10 | 18.53s | 0.51s |
| dense_algebra | 90.00 | 11 | 10 | 25.17s | 0.70s |
| grayscale_threshold | 89.31 | 12 | 11 | 19.23s | 0.53s |
| padding_variant | 89.31 | 11 | 11 | 20.68s | 0.57s |
| diagram_safe | 89.17 | 10 | 11 | 18.98s | 0.53s |
| table_preserving | 83.19 | 5 | 11 | 9.88s | 0.27s |
| baseline_current | 82.78 | 0 | 0 | 0.00s | 0.00s |

Best average fixture score: `formula_heavy`.
Worst average fixture score: `baseline_current`, although it is also the zero-cost frozen selected-text baseline.

## Family And Layout Signals

The profile results are not uniform by paper family or fixture layout type.

| Group | Best profile |
| --- | --- |
| P1 fixtures | `formula_heavy` |
| P3 fixtures | `formula_heavy` |
| P4 fixtures | `grayscale_threshold` |
| P5 fixtures | `table_preserving` |
| formula-heavy layouts | `formula_heavy` |
| table-preserving layouts | `padding_variant` |
| diagram-safe layouts | `baseline_current` |

The table-preserving profile is fastest but has the weakest average score and the highest fail count.
The dense algebra profile is slower than `formula_heavy` without beating it on average.

## Notable Improvements

Large gains appeared on fixtures where OCR recovered missing anchors, mark brackets, or simple structural expectations:

- `52spring22_q06`: all OCR profiles improved by +100 over the baseline fixture score.
- `35summer25_q04`: all OCR profiles improved by +80.
- `33autumn25_q07`: all OCR profiles improved by +60.
- `33summer24_q03`, `13autumn25_q09`, `43autumn21_q06`, and `41summer23_q06`: most OCR profiles improved by +40.
- `12summer23_q01`: `formula_heavy` and `table_preserving` improved by +25.

## Regressions

Regressions are material and are the main reason not to promote any profile directly into production selection:

- Integral fixtures such as `33autumn21_q04` lost the expected integral sign under every OCR profile.
- Theta/math-symbol fixtures such as `33autumn21_q05` and `11summer23_q01` regressed under every OCR profile.
- Several P3 algebra/calculus fixtures introduced likely math-symbol loss, including `32spring23_q04`, `32summer21_q01`, `32spring23_q03`, and `31summer23_q09`.
- `table_preserving` missed question anchors on `31summer22_q09`, `42autumn21_q01`, and `33summer21_q07`.

## Conclusion

Targeted OCR profiles are justified as an advisory experiment and as possible per-layout candidates, but not as a production default.
The measured gains are real on anchor-sensitive and some P5/table-like fixtures, while symbol-heavy calculus and trigonometry fixtures show clear regressions.

Recommended next steps:

- Keep profile outputs report-only until image-reviewed ground truth confirms precision.
- Add a small routing experiment that chooses candidate profiles by fixture layout type, then compare against the same baseline.
- Add per-profile OCR text snippets for reviewed records only, so regressions can be inspected without treating OCR text as canonical.
- If any profile is later considered for production, gate it behind existing text-fidelity and visual-required checks rather than global thresholds.
