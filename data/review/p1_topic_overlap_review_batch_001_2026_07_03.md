# P1 Topic Overlap Review Batch 001 - 2026-07-03

Scope: five early failing P1 papers selected from `output/topic_packets/topic_packet_summary.json`: `01summer08`, `01summer09`, `01winter08`, `11summer10`, `11summer11`.

Review policy:
- `primary_topic` is the dominant assessed topic by marks and prompt intent.
- `secondary_topics` are substantial mark-bearing cross-topic skills that should count for paper coverage without erasing the primary topic.
- `coverage_topics = primary_topic + secondary_topics`; this is the proposed basis for paper-level at-least-one-topic checks.
- Legacy P1 3D vector/scalar-product questions are not current P1 packet topics and are excluded rather than forced into nearby current topics.

Audit orientation: P1 currently has 102 papers in the coverage audit, with 88 failing `min one / max three` before P1 sidecar review. The main missing topics are quadratics (58 papers) and integration (29 papers).

Topic abbreviations: QUAD quadratics, FUNC functions, COORD coordinate geometry, CIRC circular measure, TRIG trigonometry, SERIES series, DIFF differentiation, INT integration.

## Summary

| Paper | Current relevant questions | Excluded current-syllabus | Missing using current single-topic counts | Missing after coverage topics | Coverage counts |
|---|---:|---|---|---|---|
| 01summer08 | 10 | 01summer08_q10 | CIRC |  | QUAD=2 FUNC=2 COORD=1 CIRC=1 TRIG=2 SERIES=2 DIFF=2 INT=1 |
| 01summer09 | 10 | 01summer09_q06 | INT, QUAD |  | QUAD=2 FUNC=1 COORD=1 CIRC=1 TRIG=2 SERIES=2 DIFF=2 INT=2 |
| 01winter08 | 9 | 01winter08_q04 | QUAD |  | QUAD=1 FUNC=1 COORD=1 CIRC=1 TRIG=2 SERIES=2 DIFF=3 INT=1 |
| 11summer10 | 9 | 11summer10_q10 | CIRC, QUAD | CIRC | QUAD=1 FUNC=1 COORD=1 CIRC=0 TRIG=2 SERIES=2 DIFF=2 INT=2 |
| 11summer11 | 10 | 11summer11_q04 | QUAD |  | QUAD=2 FUNC=1 COORD=1 CIRC=1 TRIG=1 SERIES=2 DIFF=3 INT=2 |

`11summer10` remains a documented genuine exception for `circular_measure`: after excluding its legacy vector question, no reviewed question has substantial circular-measure assessment.

## Proposed Question-Level Decisions

### 01summer08

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 01summer08_q01 | ALG | TRIG |  | TRIG | relabel_primary | Sine-rule triangle problem using exact trigonometric values; the source algebra label is not the assessed topic. |
| 2 | 01summer08_q02 | TRIG | TRIG |  | TRIG | keep | Trigonometric identity and equation in cos theta. |
| 3 | 01summer08_q03 | SERIES | SERIES |  | SERIES | keep | Binomial expansion and coefficient extraction. |
| 4 | 01summer08_q04 | DIFF | QUAD | DIFF | QUAD, DIFF | relabel_primary_add_secondary | Line-curve intersection is a 4-mark quadratic solve, followed by a 3-mark stationary-point differentiation check. |
| 5 | 01summer08_q05 | DIFF | CIRC |  | CIRC | relabel_primary | Arc length, sector area, and tangent-circle geometry; no calculus is assessed. |
| 6 | 01summer08_q06 | FUNC | FUNC | DIFF | FUNC, DIFF | add_secondary_topic | Inverse/domain work is dominant, with a mark-bearing derivative used to prove the function is increasing. |
| 7 | 01summer08_q07 | SERIES | SERIES |  | SERIES | keep | Geometric progression, sum to infinity, and arithmetic progression. |
| 8 | 01summer08_q08 | QUAD | QUAD | FUNC | QUAD, FUNC | add_secondary_topic | Composite-function equation is assessed through discriminant/equal-roots work; the functions context is substantial. |
| 9 | 01summer08_q09 | INT | INT |  | INT | keep | Reverse differentiation and area by integration. |
| 10 | 01summer08_q10 | FUNC | EXCL |  |  | exclude_current_syllabus | 3D vectors and scalar product are not current P1 packet topics, so this should not be counted as functions coverage. |
| 11 | 01summer08_q11 | COORD | COORD |  | COORD | keep | Straight-line gradients, perpendicularity, reflection, and coordinate distance. |

### 01summer09

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 01summer09_q01 | TRIG | TRIG |  | TRIG | keep | Trigonometric identity proof. |
| 2 | 01summer09_q02 | TRIG | QUAD |  | QUAD | relabel_primary | The assessment is forming a quadratic from a line-curve intersection and using the discriminant for two distinct roots. |
| 3 | 01summer09_q03 | SERIES | SERIES |  | SERIES | keep | Binomial expansion and coefficient condition. |
| 4 | 01summer09_q04 | TRIG | TRIG |  | TRIG | keep | Sine-graph parameters and trigonometric equation in radians. |
| 5 | 01summer09_q05 | CIRC | CIRC |  | CIRC | keep | Arc/sector perimeter and area in radians. |
| 6 | 01summer09_q06 | FUNC | EXCL |  |  | exclude_current_syllabus | 3D vectors and scalar product are not current P1 packet topics, so this should not be counted as functions coverage. |
| 7 | 01summer09_q07 | SERIES | SERIES |  | SERIES | keep | Geometric and arithmetic progression sums. |
| 8 | 01summer09_q08 | COORD | COORD |  | COORD | keep | Straight-line perpendicularity and coordinate construction. |
| 9 | 01summer09_q09 | DIFF | INT | DIFF | INT, DIFF | relabel_primary_add_secondary | The volume of revolution is the larger task; differentiating the curve for the gradient is a substantial preliminary part. |
| 10 | 01summer09_q10 | FUNC | FUNC | QUAD | FUNC, QUAD | add_secondary_topic | Function inverse/range work is primary; completing the square and quadratic range reasoning are substantial. |
| 11 | 01summer09_q11 | DIFF | DIFF | INT | DIFF, INT | add_secondary_topic | Stationary points and normal are differentiation-led, with a substantial exact area integral. |

### 01winter08

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 01winter08_q01 | SERIES | SERIES |  | SERIES | keep | Binomial coefficient extraction. |
| 2 | 01winter08_q02 | TRIG | TRIG |  | TRIG | keep | Trigonometric identity proof. |
| 3 | 01winter08_q03 | SERIES | SERIES |  | SERIES | keep | Arithmetic progression term and sum equation. |
| 4 | 01winter08_q04 | COORD | EXCL |  |  | exclude_current_syllabus | 3D vectors and scalar product are not current P1 packet topics, so this should not be counted as coordinate geometry coverage. |
| 5 | 01winter08_q05 | FUNC | TRIG |  | TRIG | relabel_primary | Trigonometric function parameters, equation solving, and graph sketching dominate; the functions notation is incidental. |
| 6 | 01winter08_q06 | CIRC | CIRC |  | CIRC | keep | Arc length, tangent length, sector area, and shaded area in radians. |
| 7 | 01winter08_q07 | COORD | DIFF |  | DIFF | relabel_primary | The main assessed task is optimizing the derived area expression by differentiation. |
| 8 | 01winter08_q08 | DIFF | DIFF | COORD | DIFF, COORD | add_secondary_topic | Normal-line equation, second intersection, and distance are substantial coordinate-geometry work after differentiation. |
| 9 | 01winter08_q09 | INT | INT | DIFF | INT, DIFF | add_secondary_topic | Area/volume by integration is primary, with a substantial tangent-gradient angle part. |
| 10 | 01winter08_q10 | FUNC | FUNC | QUAD | FUNC, QUAD | add_secondary_topic | Function composition and inverse work are primary; quadratic maximum/completing-square work is substantial. |

### 11summer10

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 11summer10_q01 | TRIG | TRIG |  | TRIG | keep | Trigonometric angle transformations and exact expression for sin x. |
| 2 | 11summer10_q02 | SERIES | SERIES |  | SERIES | keep | Binomial expansion and coefficient extraction. |
| 3 | 11summer10_q03 | SERIES | SERIES |  | SERIES | keep | Arithmetic progression term and sum work. |
| 4 | 11summer10_q04 | INT | INT |  | INT | keep | Area between curve and line by integration. |
| 5 | 11summer10_q05 | TRIG | TRIG |  | TRIG | keep | Trigonometric identity, extrema, and equation solving. |
| 6 | 11summer10_q06 | DIFF | INT | DIFF | INT, DIFF | relabel_primary_add_secondary | Reverse differentiation to recover the curve is the larger part; stationary-point differentiation is also assessed. |
| 7 | 11summer10_q07 | DIFF | DIFF |  | DIFF | keep | Differentiation to form the normal and find an intercept length. |
| 8 | 11summer10_q08 | COORD | COORD |  | COORD | keep | Straight-line gradients, intersections, and perpendicular bisector. |
| 9 | 11summer10_q09 | FUNC | QUAD | FUNC | QUAD, FUNC | relabel_primary_add_secondary | Completing the square, quadratic inequality, and discriminant dominate; function notation supplies context. |
| 10 | 11summer10_q10 | TRIG | EXCL |  |  | exclude_current_syllabus | 3D vectors and scalar product are not current P1 packet topics, so this should not be counted as trigonometry coverage. |

### 11summer11

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 11summer11_q01 | SERIES | SERIES |  | SERIES | keep | Binomial coefficient extraction. |
| 2 | 11summer11_q02 | COORD | DIFF |  | DIFF | relabel_primary | Related rates using dV/dr and dV/dt; not coordinate geometry. |
| 3 | 11summer11_q03 | INT | INT |  | INT | keep | Volume of revolution by integration. |
| 4 | 11summer11_q04 | SERIES | EXCL |  |  | exclude_current_syllabus | 3D vectors and scalar product are not current P1 packet topics, so this should not be counted as series coverage. |
| 5 | 11summer11_q05 | TRIG | TRIG |  | TRIG | keep | Trigonometric identity and equation solving. |
| 6 | 11summer11_q06 | FUNC | DIFF |  | DIFF | relabel_primary | Stationary-value optimization by differentiation; the variables context is not a functions topic. |
| 7 | 11summer11_q07 | DIFF | INT | DIFF | INT, DIFF | relabel_primary_add_secondary | Reverse differentiation to find the curve is primary; the gradient inequality is a substantial differentiation part. |
| 8 | 11summer11_q08 | SERIES | SERIES |  | SERIES | keep | Arithmetic and geometric progression models. |
| 9 | 11summer11_q09 | CIRC | CIRC |  | CIRC | keep | Arc/tangent geometry, sector area, and perimeter in radians. |
| 10 | 11summer11_q10 | COORD | COORD | QUAD | COORD, QUAD | add_secondary_topic | Coordinate line/point work is primary, with completing square and quadratic intersection also substantial. |
| 11 | 11summer11_q11 | FUNC | FUNC | QUAD | FUNC, QUAD | add_secondary_topic | Function composition/inverses are primary; solving quadratic equalities and inverse quadratic work are substantial. |

## Implementation Notes

- The merged sidecar is `data/review/p1_topic_overlap_review_merged_2026_07_03.json`.
- The sidecar only changes coverage accounting and primary packet routing; it does not duplicate PDFs for secondary topics.
- The documented exception should remain visible in coverage audit output until a later review confirms otherwise from canonical artifacts.
