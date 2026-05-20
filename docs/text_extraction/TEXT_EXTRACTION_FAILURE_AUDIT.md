# Text Extraction Failure Audit

Date: 2026-05-20

Scope: documentation-only audit of current generated outputs. This report does not change extraction behavior, OCR thresholds, canonical image generation, trust gates, topic labels, or advisory fields.

## Image-First Contract

This project remains image-first. Canonical question images and mark-scheme images are the source of truth. Native PDF text, OCR text, normalized text, topic labels, difficulty labels, and advisory fields are secondary metadata for search, triage, and review workflows only.

Every failure below is therefore an advisory text failure unless separately confirmed by visual crop review. The canonical image remains safe when the question PNG and mark-scheme PNG still point to the visual source artifacts and downstream consumers continue to prefer those images over text.

## Corpus Inspected

- Current bank inspected: `output/json/question_bank.json`
- Records inspected by script/statistics: 1301
- Paper family coverage: P1 = 401, P3 = 396, P4 = 258, P5 = 246
- Concrete examples catalogued below: 36
- Example coverage: P1 = 9, P3 = 15, P4 = 3, P5 = 9
- Selected text source distribution in current bank: native = 1268, OCR = 33
- Text fidelity status in current bank: clean = 1259, degraded = 42

## Overall Findings

The requested risks are present in current outputs, but they are concentrated in advisory text fields rather than canonical visual artifacts.

| Risk | Present? | Evidence summary |
| --- | --- | --- |
| Flattened math changing meaning | Yes | Many P1/P3 examples flatten powers, fractions, integrals, trig symbols, derivatives, vectors, and complex-number notation. |
| OCR/native disagreement | Yes | OCR-selected records and rejected OCR snippets disagree with native text. Some OCR repairs structure while introducing symbol errors. |
| Missing question numbers | Yes | 7 selected texts do not begin with the expected question number. |
| Missing mark brackets | Yes | 5 selected texts have no `[...]` mark bracket despite mark totals in image/mark scheme metadata. |
| Lost subpart labels | Yes, but as ordering/anchor loss rather than detected internal gaps | No `missing_internal_subparts` flags were found, but examples show selected text beginning at `(b)` or reordering `(a)`, `(b)`, `(c)` context. |
| Suspiciously short selected text | Yes | 77 records have selected text score <= 40; several are short math prompts where the text field looks readable but mathematically unsafe. |
| Next-question contamination | Not confirmed in current selected text | Current structured detector reports 0 `contamination_detected` records. Keep a fixture category because page furniture and question-anchor failures are adjacent risks. |
| Answer-space/furniture dominance | Yes | Table/diagram regions are represented by gibberish, axis/furniture tokens, or answer-space artifacts in selected text. |
| Degraded text despite clean visual crop | Yes | 5 degraded records have high question crop confidence; 30 high-crop records have selected score < 50. |

## Evidence Counts

| Signal | Count |
| --- | ---: |
| Total records inspected | 1301 |
| OCR-selected records | 33 |
| OCR-selected with low question crop confidence | 21 |
| Degraded text records | 42 |
| Degraded text with high question crop confidence | 5 |
| Selected text missing expected leading question number | 7 |
| Selected text missing mark brackets | 5 |
| Selected text score <= 40 | 77 |
| High question crop confidence with selected score < 50 | 30 |
| Detected internal subpart sequence gaps | 0 |
| Detected next-question contamination | 0 |

## Concrete Examples

Notes:

- `Raw native text` is listed as "same as selected" when the selected source is native because the current bank does not preserve a second top-level native-text field.
- For OCR-selected rows, raw native text is not separately persisted in `output/json/question_bank.json`; the native score is included where available.
- Text excerpts are exact excerpts from current output fields, shortened only for readability.

| # | Record/question id | Paper id | Family/session | Q no. | Selected text source | Raw native text if available | Raw OCR text if available | Currently selected text | Observed failure type | Why canonical image remains safe | Suggested fixture category |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `12summer21_q03` | `12summer21` | P1 / June 2021, component 12 | 3 | OCR | Not separately persisted; native_score=87 | `3 The equation of a curve is y = (x — 3)vx + 1+3...` | Same as OCR | Flattened math; OCR-selected symbol corruption (`sqrt`/power lost). | PNG paths remain `p1/12summer21/questions/q03.png` and `p1/12summer21/mark_scheme/q03.png`. | math-layout/native-vs-ocr |
| 2 | `32autumn21_q04` | `32autumn21` | P3 / November 2021, component 32 | 4 | native | Same as selected | `4 Express 4x? — 13x + 13 (2x — 1)(x - 3) in partial fractions. [5]` | `4 Express (42xx^{2} - -(13x 13)/(1 x 3))( + - ) in partial fractions. [5]` | Flattened fraction changes algebraic meaning; OCR/native disagreement. | PNG paths remain `p3/32autumn21/questions/q04.png` and `p3/32autumn21/mark_scheme/q04.png`. | rational-expression-math |
| 3 | `33autumn21_q04` | `33autumn21` | P3 / November 2021, component 33 | 4 | native | Same as selected | `Te 4 Find the exact value of \| xsin 4x dx. 1 3"` | `4 Find the exact value of ∫_{1}_{3}^{π}_{π} x sin^{ 1}_{2}x dx. [5]` | Integral bounds/function flattened; OCR/native disagreement. | PNG paths remain `p3/33autumn21/questions/q04.png` and `p3/33autumn21/mark_scheme/q04.png`. | integral-layout |
| 4 | `33autumn21_q05` | `33autumn21` | P3 / November 2021, component 33 | 5 | native | Same as selected | `5 Solve the equation sin 0 = 3 cos 26 + 2, for 0° < @ < 360°. [5]` | `5 Solve the equation sin θ = 3 cos 21 + 2, for 0° ≤ θ ≤ 360°. [5]` | Trig argument corrupted (`2θ` appears as `21`). | PNG paths remain `p3/33autumn21/questions/q05.png` and `p3/33autumn21/mark_scheme/q05.png`. | trig-symbol-fidelity |
| 5 | `31summer22_q09` | `31summer22` | P3 / June 2022, component 31 | 9 | native | Same as selected | `...OABCDEFG is a cuboid... Unit vectors i, j and k are parallel to OA, OC and OG...` | `9 vectorsIn the diagram, i,j and OABCDEFG k areparallel is a cuboid...` | Diagram text order collapse; furniture/diagram dominance. | PNG paths remain `p3/31summer22/questions/q09.png` and `p3/31summer22/mark_scheme/q09.png`. | diagram-reading-order |
| 6 | `32spring23_q04` | `32spring23` | P3 / March 2023, component 32 | 4 | native | Same as selected | `4 Solve the equation 5z x a Tan + 30+ 101 = 0...` | `4 Solve the equation (5z)/(1 2i) + -zz^{*} + 30 + 10i = 0... [5]` | Complex-number fraction flattened; signs/denominator unsafe. | PNG paths remain `p3/32spring23/questions/q04.png` and `p3/32spring23/mark_scheme/q04.png`. | complex-equation-layout |
| 7 | `12summer23_q01` | `12summer23` | P1 / June 2023, component 12 | 1 | native | Same as selected | `...d 4 . The equation of a curve is such that cy = for x > 3...` | `1 (The4,5equation). ofa curve is such that ddyx = (x -43)^{3} forx > 3... [3]` | Derivative/fraction flattened; word order degraded despite high crop. | PNG paths remain `p1/12summer23/questions/q01.png` and `p1/12summer23/mark_scheme/q01.png`. | derivative-layout |
| 8 | `13summer23_q01` | `13summer23` | P1 / June 2023, component 13 | 1 | native | Same as selected | `The diagram shows the graph of y = f(x)...` | `1 lines The diagram A^{′}B^{′} and shows B^{′}C^{′} the form graph...` | Diagram reading order contamination; selected score low despite high crop. | PNG paths remain `p1/13summer23/questions/q01.png` and `p1/13summer23/mark_scheme/q01.png`. | transformation-diagram-order |
| 9 | `33summer24_q03` | `33summer24` | P3 / June 2024, component 33 | 3 | native | Same as selected | `3. The square roots of 24—7i... exact Cartesian form.` | `3 The square roots of 24 - 7i... exact Cartesian form.` | Missing mark bracket in selected text. | PNG paths remain `p3/33summer24/questions/q03.png` and `p3/33summer24/mark_scheme/q03.png`. | mark-bracket-presence |
| 10 | `35summer25_q04` | `35summer25` | P3 / June 2025, component 35 | 4 | native | Same as selected | `4 Find the exact coordinates... y = 3x7 Inx*, for x > 0. [5]` | `4 Find the exact coordinates... y = 3x^{3} ln x^{4}, for x20.` | Missing mark bracket; inequality `x > 0` collapsed to `x20`. | PNG paths remain `p3/35summer25/questions/q04.png` and `p3/35summer25/mark_scheme/q04.png`. | mark-bracket-and-inequality |
| 11 | `33autumn25_q07` | `33autumn25` | P3 / November 2025, component 33 | 7 | native | Same as selected | `7 Solve the equation 33 722 *+20+8i = 0... [6]` | `7 Solve the equation (5z)/(2 - i) - zz^{*} + 20 + 8i = 0... where x and y are` | Missing mark bracket and apparent truncation. | PNG paths remain `p3/33autumn25/questions/q07.png` and `p3/33autumn25/mark_scheme/q07.png`. | truncated-math-prompt |
| 12 | `13autumn25_q09` | `13autumn25` | P1 / November 2025, component 13 | 9 | native | Same as selected | `...curve with equation y = 5x+4 and the line y = 4.5... [8]` | `9 The diagram shows part of the curve with equation y = 12x + 4x and the line y = 4 5...` | Diagram/math flattening; missing mark bracket; likely truncation. | PNG paths remain `p1/13autumn25/questions/q09.png` and `p1/13autumn25/mark_scheme/q09.png`. | volume-diagram-text |
| 13 | `43autumn21_q06` | `43autumn21` | P4 / November 2021, component 43 | 6 | OCR | Not separately persisted; native_score=129 | `The diagram shows a particle of mass 5 kg...` | Same as OCR | Missing leading question number in selected text. | PNG paths remain `p4/43autumn21/questions/q06.png` and `p4/43autumn21/mark_scheme/q06.png`. | question-anchor-presence |
| 14 | `12spring22_q08` | `12spring22` | P1 / March 2022, component 12 | 8 | OCR | Not separately persisted; native_score=65 | `The diagram shows the circle with equation (x — 2)? + y? = 8...` | Same as OCR | Missing leading question number; powers flattened as `?`. | PNG paths remain `p1/12spring22/questions/q08.png` and `p1/12spring22/mark_scheme/q08.png`. | question-anchor-plus-powers |
| 15 | `52spring22_q06` | `52spring22` | P5 / March 2022, component 52 | 6 | OCR | Not separately persisted; native_score=157 | `(b) A factory produces chocolates... (a) Find the probability...` | Same as OCR | Lost/shifted subpart order; selected text begins at `(b)` before `(a)`. | PNG paths remain `p5/52spring22/questions/q06.png` and `p5/52spring22/mark_scheme/q06.png`. | subpart-ordering |
| 16 | `32summer23_q09` | `32summer23` | P3 / June 2023, component 32 | 9 | OCR | Not separately persisted; native_score=14 | `2x7 +17x-17 9 Letf(xye 2 ett) = Tapas ae...` | Same as OCR | Missing leading question number position; severe rational/integral corruption. | PNG paths remain `p3/32summer23/questions/q09.png` and `p3/32summer23/mark_scheme/q09.png`. | severe-math-ocr |
| 17 | `41summer23_q06` | `41summer23` | P4 / June 2023, component 41 | 6 | OCR | Not separately persisted; native_score=120 | `Two particles P and Q...` | Same as OCR | Missing leading question number in selected text. | PNG paths remain `p4/41summer23/questions/q06.png` and `p4/41summer23/mark_scheme/q06.png`. | question-anchor-presence |
| 18 | `52spring21_q05` | `52spring21` | P5 / March 2021, component 52 | 5 | OCR | Not separately persisted; native_score=75 | `...summarised in the following table. FFeweny [\| \|e fo [fs \|... Qty ee we Ve...` | Same as OCR | Table/furniture dominance; OCR-selected gibberish inside table/answer-space area. | PNG paths remain `p5/52spring21/questions/q05.png` and `p5/52spring21/mark_scheme/q05.png`. | table-ocr-furniture |
| 19 | `51summer21_q05` | `51summer21` | P5 / June 2021, component 51 | 5 | OCR | Not separately persisted; native_score=54 | `...Time (¢ seconds) 0<r<10... Qe ee we see es WU Be...` | Same as OCR | Table extraction confused variables and answer furniture. | PNG paths remain `p5/51summer21/questions/q05.png` and `p5/51summer21/mark_scheme/q05.png`. | table-ocr-furniture |
| 20 | `52summer21_q02` | `52summer21` | P5 / June 2021, component 52 | 2 | OCR | Not separately persisted; native_score=53 | `...standard deviation okg... Find the value of o. [4]` | Same as OCR | OCR/native disagreement likely around sigma; selected text uses `o`/`okg`. | PNG paths remain `p5/52summer21/questions/q02.png` and `p5/52summer21/mark_scheme/q02.png`. | greek-symbol-ocr |
| 21 | `52summer22_q02` | `52summer22` | P5 / June 2022, component 52 | 2 | native | Same as selected | `2 A fair 6-sided die has the numbers 1, 2, 2, 3, 3, 3...` | `2 A fair 6-sided diehas the numbers1,2,2,3,3,3...` | Degraded text despite high crop; missing spaces but visual crop is clean. | PNG paths remain `p5/52summer22/questions/q02.png` and `p5/52summer22/mark_scheme/q02.png`. | clean-crop-text-degradation |
| 22 | `53summer22_q07` | `53summer22` | P5 / June 2022, component 53 | 7 | OCR | Not separately persisted; native_score=118 | `...The group consists of four families. ¢ Mr and Mrs Kenny...` | Same as OCR | Bullet/list OCR degradation despite high crop. | PNG paths remain `p5/53summer22/questions/q07.png` and `p5/53summer22/mark_scheme/q07.png`. | bullet-list-ocr |
| 23 | `11autumn23_q04` | `11autumn23` | P1 / November 2023, component 11 | 4 | native | Same as selected | `(b) (c) The transformation R denotes...` | `4 The transformation R denotes a reflection in the@ A x-axis...` | Degraded clean crop; transformation vector/furniture inserted into prose. | PNG paths remain `p1/11autumn23/questions/q04.png` and `p1/11autumn23/mark_scheme/q04.png`. | transformation-vector-layout |
| 24 | `12autumn23_q06` | `12autumn23` | P1 / November 2023, component 12 | 6 | native | Same as selected | `6 The equation of a curve is y = x -— 8x45...` | `6 The equation of a curve is y = x^{2}-8x + 5... translated by@ 41A...` | Degraded clean crop; translation vector flattened into garbage. | PNG paths remain `p1/12autumn23/questions/q06.png` and `p1/12autumn23/mark_scheme/q06.png`. | transformation-vector-layout |
| 25 | `51autumn23_q05` | `51autumn23` | P5 / November 2023, component 51 | 5 | native | Same as selected | `5 A red spinner has four sides labelled...` | `5 Aredspinnerhasfoursideslabelled1,2,3,4...` | Degraded text despite high crop; table/prose spacing collapsed. | PNG paths remain `p5/51autumn23/questions/q05.png` and `p5/51autumn23/mark_scheme/q05.png`. | clean-crop-spacing |
| 26 | `42autumn21_q01` | `42autumn21` | P4 / November 2021, component 42 | 1 | native | Same as selected | `...velocity-time graph which models the motion of a car...` | `1 Thesixstraightdiagramlineshowssegments.avelocity-time... 20 m sacar._{-}The_{1}...` | Diagram/furniture dominance; prose and axis labels interleaved. | PNG paths remain `p4/42autumn21/questions/q01.png` and `p4/42autumn21/mark_scheme/q01.png`. | mechanics-graph-reading-order |
| 27 | `52autumn21_q01` | `52autumn21` | P5 / November 2021, component 52 | 1 | native | Same as selected | `(b) (c) Each of the 180 students...` | `1 Eachofthe180studentsatacollegeplaysexactlyoneofthepiano...` | Table/prose merge; raw OCR shows subpart labels before stem. | PNG paths remain `p5/52autumn21/questions/q01.png` and `p5/52autumn21/mark_scheme/q01.png`. | table-reading-order |
| 28 | `31summer21_q01` | `31summer21` | P3 / June 2021, component 31 | 1 | native | Same as selected | `1 Solve the inequality 2\|3x - 1\| <\|x+ 1]. [4]` | `1 Solve the in equal it y 2\|3x -1\| < \|x + 1\|. [4]` | Suspiciously short selected text; word split (`inequality`). | PNG paths remain `p3/31summer21/questions/q01.png` and `p3/31summer21/mark_scheme/q01.png`. | short-math-text |
| 29 | `32summer21_q01` | `32summer21` | P3 / June 2021, component 32 | 1 | native | Same as selected | `1 Solve the inequality \|2x — 1\| < 3\|x+ 1]. [4]` | `1 Solve the in equal it y \|2x -1\| < 3\|x + 1\|. [4]` | Suspiciously short selected text; word split and OCR/native disagreement. | PNG paths remain `p3/32summer21/questions/q01.png` and `p3/32summer21/mark_scheme/q01.png`. | short-math-text |
| 30 | `33autumn21_q02` | `33autumn21` | P3 / November 2021, component 33 | 2 | native | Same as selected | `2 (a) Sketch the graph of y = \|2x — 3}. (b) Solve the inequality... [3] [1]` | `2 (a) Sketch the graph of y = \|2x -3\|. [1] (b) Solve the in equal it y... [3]` | OCR/native disagreement; mark order differs; word split. | PNG paths remain `p3/33autumn21/questions/q02.png` and `p3/33autumn21/mark_scheme/q02.png`. | subpart-mark-order |
| 31 | `32spring23_q03` | `32spring23` | P3 / March 2023, component 32 | 3 | native | Same as selected | `3. The polynomial 2x* + ax* + bx — 1...` | `3 divided by Thepolynomial x^{2}-x2 +x 1 the remainder is 3^{4} + ax^{3}...` | Clean-crop low-score text; polynomial reading order changes meaning. | PNG paths remain `p3/32spring23/questions/q03.png` and `p3/32spring23/mark_scheme/q03.png`. | polynomial-reading-order |
| 32 | `11summer23_q01` | `11summer23` | P1 / June 2023, component 11 | 1 | native | Same as selected | `1 Solve the equation 4 sin 6 + tan @ = 0...` | `1 Solve the equation 4 sin θ + tan θ = 0 for 0° < θ < 180°. [3]` | OCR/native disagreement on theta; selected native looks better but remains short math. | PNG paths remain `p1/11summer23/questions/q01.png` and `p1/11summer23/mark_scheme/q01.png`. | greek-symbol-compare |
| 33 | `12spring21_q02` | `12spring21` | P1 / March 2021, component 12 | 2 | native | Same as selected | `2. By using a suitable substitution, solve the equation 4 2 (2x - 3) ox=3P 3=0. [4]` | `2 By using a suitable substitution, solve the equation (2x -3)^{2}- (2x -43)^{2} -3 = 0. [4]` | Suspiciously short math; exponent/operand flattened. | PNG paths remain `p1/12spring21/questions/q02.png` and `p1/12spring21/mark_scheme/q02.png`. | short-algebra-text |
| 34 | `33summer21_q07` | `33summer21` | P3 / June 2021, component 33 | 7 | native | Same as selected | `For the curve shown in the diagram...` | `7 For the curve shown in the diagram,the normal... to tan The curve is such that...` | Diagram/math prose interleaving; degraded selected text. | PNG paths remain `p3/33summer21/questions/q07.png` and `p3/33summer21/mark_scheme/q07.png`. | diagram-plus-calculus |
| 35 | `52summer21_q04` | `52summer21` | P5 / June 2021, component 52 | 4 | native | Same as selected | `4 A fair spinner has sides numbered 1, 2, 2...` | `4 A fair spinner has sides numbered1,2,2... The number onthe side onwhicha spinner...` | Degraded spacing; prose still readable but not high-quality text. | PNG paths remain `p5/52summer21/questions/q04.png` and `p5/52summer21/mark_scheme/q04.png`. | spacing-normalization-review |
| 36 | `31summer23_q09` | `31summer23` | P3 / June 2023, component 31 | 9 | native | Same as selected | `(b) (c) a The constant a is such that \| xe dx = : 0...` | `a 9 The constant a is such that ∫_{0} xe^{-}2x dx = 1_{8}...` | Question-number anchor displaced; integral layout flattened. | PNG paths remain `p3/31summer23/questions/q09.png` and `p3/31summer23/mark_scheme/q09.png`. | question-anchor-integral |

## Improvement Evidence

The five improvement themes below are evaluated only as fixture and review candidates. This report does not recommend changing current extraction behavior without a measured candidate run.

| Proposed improvement | Supported by evidence? | Reason |
| --- | --- | --- |
| Add math-fidelity fixtures for fractions, powers, integrals, trig, vectors, and complex numbers. | Supported | Examples 1-8, 10-12, 16, 31, 33, and 36 show meaning-changing flattening. |
| Keep OCR selection conservative and review OCR-selected cases before loosening thresholds. | Supported | 33 OCR-selected records exist; 21 have low crop confidence. Examples 1, 13-20, and 22 show OCR can repair anchors while adding symbol/table errors. |
| Add structural fixtures for expected question number, mark brackets, and subpart ordering. | Supported | 7 selected texts miss leading question numbers, 5 miss mark brackets, and examples 15, 27, 30, and 36 show subpart/order anomalies. |
| Add table/diagram/furniture fixtures instead of treating extracted table text as canonical. | Supported | Examples 5, 8, 18, 19, 26, 27, and 34 show axis/table/diagram text interleaving. |
| Add a clean-crop degraded-text fixture lane. | Supported | 5 degraded records have high crop confidence; examples 21-25 show visual crop quality does not guarantee text quality. |
| Broad OCR threshold loosening. | Not supported | OCR-selected examples include serious symbol and table corruption; this audit supports targeted fixtures and review, not looser global OCR selection. |
| Next-question contamination fixes as an immediate current-output target. | Not yet supported | Current structured metadata reports 0 `contamination_detected` records. Maintain regression fixtures, but this audit does not prove active selected-text contamination. |
| Promoting advisory text to source-of-truth status. | Not supported | The strongest failures are exactly in advisory text. Canonical images remain the safe source. |

## Recommended Next Fixture Categories

1. `math_layout_fraction_power_integral`: P1/P3 algebra, calculus, complex-number, and trig prompts where native and OCR disagree.
2. `question_anchor_and_mark_bracket_presence`: selected text must preserve expected leading question number and mark brackets without trusting the text as canonical.
3. `subpart_ordering_and_context`: prompts where selected text begins at `(b)` or raw OCR/native orders `(a)`, `(b)`, `(c)` incorrectly.
4. `table_diagram_furniture_noise`: P4/P5 graph/table prompts with axis labels, answer-space remnants, and table OCR gibberish.
5. `clean_crop_degraded_text`: high crop-confidence records with degraded or low-score selected text.
6. `greek_symbol_and_units`: theta, sigma, pi, inequality signs, and units such as `m s^{-1}`.
7. `short_math_prompt_review`: short selected text with low score where a single exponent/sign error changes the problem.

## Validation Notes

This audit intentionally created documentation only. No extraction code, generated question/mark-scheme images, generated JSON outputs, OCR thresholds, trust gates, or Asterion exports were modified by this report.

Validation commands used:

```bash
git status --short
find output -type f \( -name '*.json' -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/exam_bank_output_hashes_before.sha256
.venv/bin/pytest tests/test_output_contract.py tests/test_asterion_export.py tests/test_advisory_evidence.py
find output -type f \( -name '*.json' -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/exam_bank_output_hashes_after.sha256
diff -u /tmp/exam_bank_output_hashes_before.sha256 /tmp/exam_bank_output_hashes_after.sha256
git status --short
```

The final command results are summarized in the final response for this audit.
