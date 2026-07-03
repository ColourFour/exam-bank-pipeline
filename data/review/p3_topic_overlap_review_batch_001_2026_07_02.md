# P3 Topic Overlap Review Batch 001 - 2026-07-02

Scope: five papers selected from the P3 coverage audit: `03summer08`, `03summer09`, `03winter08`, `31summer10`, `31summer11`.

Review policy:
- `primary_topic` is the dominant assessed topic by marks and prompt intent.
- `secondary_topics` are substantial mark-bearing cross-topic skills that should count for paper coverage without erasing the primary topic.
- `coverage_topics = primary_topic + secondary_topics`; this is the proposed basis for paper-level at-least-one-topic checks.
- Routine integration inside a separable differential-equation question is not separately counted as integration unless the prompt separately assesses an integration topic outside the DE objective.

Topic abbreviations: ALG algebra, LOG log/exponential, TRIG trigonometry, DIFF differentiation, INT integration, NUM numerical solution, VEC vectors, DE differential equations, CX complex numbers.

## Summary

| Paper | Current relevant questions | Excluded current-syllabus | Missing using current single-topic counts | Missing after coverage topics | Coverage counts |
|---|---:|---|---|---|---|
| 03summer08 | 10 |  | INT |  | ALG=2 LOG=1 TRIG=2 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 03summer09 | 9 | 03summer09_q02 | INT, DE, CX |  | ALG=2 LOG=1 TRIG=1 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 03winter08 | 10 |  | TRIG, INT |  | ALG=2 LOG=1 TRIG=2 DIFF=2 INT=1 NUM=1 VEC=1 DE=1 CX=1 |
| 31summer10 | 10 |  | INT |  | ALG=2 LOG=1 TRIG=3 DIFF=1 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 31summer11 | 10 |  | DIFF, DE |  | ALG=2 LOG=3 TRIG=2 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |

## Proposed Question-Level Decisions

### 03summer08

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 03summer08_q01 | DIFF | ALG |  | ALG | relabel_primary | Modulus inequality solved by squaring/linear cases; no calculus is assessed. |
| 2 | 03summer08_q02 | LOG | LOG |  | LOG | keep | Exponential equation solved via substitution/quadratic and logarithm. |
| 3 | 03summer08_q03 | NUM | NUM | TRIG | NUM, TRIG | add_secondary_topic | Main assessment is fixed-point iteration; setup uses trig/radian geometry to form the equation. |
| 4 | 03summer08_q04 | TRIG | TRIG |  | TRIG | keep | Tangent addition formulae and trigonometric equation. |
| 5 | 03summer08_q05 | CX | CX |  | CX | keep | Argand locus and complex-number algebra. |
| 6 | 03summer08_q06 | DIFF | DIFF |  | DIFF | keep | Implicit differentiation and horizontal tangent. |
| 7 | 03summer08_q07 | ALG | ALG | INT | ALG, INT | add_secondary_topic | Partial fractions are assessed, then a definite integral is evaluated from them. |
| 8 | 03summer08_q08 | DE | DE |  | DE | keep | Separable differential equation from a geometric condition. |
| 9 | 03summer08_q09 | DIFF | INT | DIFF | INT, DIFF | relabel_primary_add_secondary | Part (i) is differentiation, but part (ii) is a 6-mark volume-of-revolution integration task, so integration is the stronger primary. |
| 10 | 03summer08_q10 | VEC | VEC |  | VEC | keep | Vector lines/intersection/angle geometry. |

### 03summer09

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 03summer09_q01 | LOG | LOG |  | LOG | keep | Logarithmic equation solved with exponentials. |
| 2 | 03summer09_q02 | DIFF | EXCL |  |  | exclude_current_syllabus | Trapezium-rule numerical integration is no longer current-syllabus P3 content; current differentiation label is wrong. |
| 3 | 03summer09_q03 | TRIG | TRIG |  | TRIG | keep | Trigonometric identity and equation. |
| 4 | 03summer09_q04 | NUM | NUM |  | NUM | keep | Root location, fixed-point formula, and iteration. |
| 5 | 03summer09_q05 | ALG | ALG |  | ALG | keep | Binomial expansion with algebraic coefficient condition. |
| 6 | 03summer09_q06 | DIFF | DIFF |  | DIFF | keep_with_split_needed | The Q6 portion is parametric differentiation and tangent geometry; crop also includes Q7 and must be split. |
| 7 | 03summer09_q07 |  | CX |  | CX | restore_split_question | Q7 is visible inside the Q6 question crop but missing as its own extracted question; it assesses complex roots, Argand diagram, modulus/argument. |
| 8 | 03summer09_q08 | ALG | DE | ALG, INT | DE, ALG, INT | relabel_primary_add_secondary | Part (i) partial fractions support part (ii), but the main task is solving a separable differential equation; integration is also explicitly assessed. |
| 9 | 03summer09_q09 | VEC | VEC |  | VEC | keep | Line/plane vector geometry and perpendicular distance. |
| 10 | 03summer09_q10 | DIFF | INT | DIFF | INT, DIFF | relabel_primary_add_secondary | Part (i) differentiates to find a maximum; parts (ii)-(iii) are area by substitution and exact integration, so integration is the stronger primary. |

### 03winter08

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 03winter08_q01 | LOG | LOG |  | LOG | keep | Logarithmic equation using log laws. |
| 2 | 03winter08_q02 | ALG | ALG |  | ALG | keep | Binomial expansion. |
| 3 | 03winter08_q03 | DIFF | DIFF |  | DIFF | keep | Differentiate and solve stationary-point condition. |
| 4 | 03winter08_q04 | DIFF | DIFF | TRIG | DIFF, TRIG | add_secondary_topic | Parametric differentiation is primary; final simplification uses trig identities. |
| 5 | 03winter08_q05 | ALG | ALG |  | ALG | keep | Polynomial divisibility/factorisation and inequality. |
| 6 | 03winter08_q06 | LOG | TRIG |  | TRIG | relabel_primary | R sin(x+alpha) form and trig equation; no log/exponential assessment. |
| 7 | 03winter08_q07 | VEC | VEC |  | VEC | keep | Plane angle and line of intersection using vectors. |
| 8 | 03winter08_q08 | DE | DE |  | DE | keep | Rate model leading to separable differential equation. |
| 9 | 03winter08_q09 | NUM | NUM | INT | NUM, INT | add_secondary_topic | Numerical root work is the main arc, but part (i) is a 5-mark integration-by-parts derivation of the equation. |
| 10 | 03winter08_q10 | CX | CX |  | CX | keep | Complex modulus/argument and Argand geometry. |

### 31summer10

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 31summer10_q01 | DIFF | ALG |  | ALG | relabel_primary | Modulus inequality with parameter; no calculus is assessed. |
| 2 | 31summer10_q02 | TRIG | TRIG |  | TRIG | keep | Trig equation using cos 2A. |
| 3 | 31summer10_q03 | LOG | LOG |  | LOG | keep | Log-linearisation of power-law relationship. |
| 4 | 31summer10_q04 | TRIG | TRIG | INT | TRIG, INT | add_secondary_topic | Trig identity is primary; part (ii) asks for a definite trigonometric integral. |
| 5 | 31summer10_q05 | DE | DE |  | DE | keep | Separable differential equation. |
| 6 | 31summer10_q06 | NUM | NUM | TRIG | NUM, TRIG | add_secondary_topic | Iteration/root finding is primary; setup uses segment geometry and sine. |
| 7 | 31summer10_q07 | CX | CX |  | CX | keep | Complex number and Argand region. |
| 8 | 31summer10_q08 | ALG | INT | ALG | INT, ALG | relabel_primary_add_secondary | Partial fractions support a 5-mark definite integral; integration is the stronger primary. |
| 9 | 31summer10_q09 | DIFF | DIFF |  | DIFF | keep | Differentiate to obtain normal gradient and optimise it. |
| 10 | 31summer10_q10 | VEC | VEC |  | VEC | keep | Vector lines, angle, and plane. |

### 31summer11

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 31summer11_q01 | ALG | ALG |  | ALG | keep | Binomial expansion. |
| 2 | 31summer11_q02 | LOG | DIFF |  | DIFF | relabel_primary | Prompt is explicitly find dy/dx; ln and tan are function inputs, but the assessed skill is differentiation. |
| 3 | 31summer11_q03 | VEC | VEC |  | VEC | keep | Plane equation and angle with axis using vectors. |
| 4 | 31summer11_q04 | ALG | ALG | LOG | ALG, LOG | add_secondary_topic | Polynomial factorisation is primary; part (ii) solves an exponential equation via 3^y and logs. |
| 5 | 31summer11_q05 | LOG | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Implicit differentiation supplies most of the method; exponential/log values are substantial context. |
| 6 | 31summer11_q06 | NUM | NUM | TRIG | NUM, TRIG | add_secondary_topic | Iteration is primary; setup and final chord length use circular/trigonometric geometry. |
| 7 | 31summer11_q07 | LOG | INT | LOG | INT, LOG | relabel_primary_add_secondary | This is an integration question using substitution and integration by parts on ln; log is secondary, not primary. |
| 8 | 31summer11_q08 | CX | CX |  | CX | keep | Complex arithmetic, argument, and loci. |
| 9 | 31summer11_q09 | TRIG | TRIG | INT | TRIG, INT | add_secondary_topic | Trig identity/equation is primary; part (ii)(b) asks for an exact integral of cos^4 theta. |
| 10 | 31summer11_q10 | INT | DE |  | DE | relabel_primary | This is a population model defined by dN/dt and solved as a separable differential equation; integration is method, DE is the topic. |

## Implementation Notes

- `topic_bank_reviewed_decisions.v1.json` can express `reviewed_topic` but cannot express secondary coverage topics or split-restoration intent.
- Add a sidecar or schema extension with `primary_topic`, `secondary_topics`, `coverage_topics`, and `count_policy` so coverage tests can count duplicate topic membership while packet manifests can still distinguish unique questions from topic placements.
- Packet summaries should report both `unique_question_count` and `topic_placement_count`; otherwise duplicate topic placement will look like inflated problem inventory.
- `03summer09_q06` requires extraction repair: the question image contains both Q6 and Q7, but there is no separate `03summer09_q07` record or mark-scheme artifact in the current output.
