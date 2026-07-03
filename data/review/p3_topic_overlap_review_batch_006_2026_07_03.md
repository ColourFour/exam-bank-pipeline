# P3 Topic Overlap Review Batch 006 - 2026-07-03

Scope: next 15 actual P3 papers selected from highlighted papers missing at least one topic after batch 005.

Review policy matches prior batches: primary topic follows dominant assessed marks/prompt intent; secondary topics count for paper coverage only.

## Summary

| Paper | Current relevant questions | Excluded current-syllabus | Missing using current single-topic counts | Missing after coverage topics | Coverage counts |
|---|---:|---|---|---|---|
| 33summer15 | 10 |  | INT |  | ALG=2 LOG=3 TRIG=2 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer16 | 10 |  | DE, TRIG |  | ALG=2 LOG=3 TRIG=3 DIFF=2 INT=1 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer17 | 11 |  | DE |  | ALG=2 LOG=3 TRIG=3 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer18 | 10 |  | DE, NUM |  | ALG=2 LOG=3 TRIG=2 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer19 | 10 |  | DE, TRIG |  | ALG=1 LOG=3 TRIG=2 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer20 | 10 |  | DE, TRIG |  | ALG=2 LOG=2 TRIG=1 DIFF=1 INT=3 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer22 | 10 |  | INT |  | ALG=2 LOG=3 TRIG=2 DIFF=2 INT=1 NUM=1 VEC=1 DE=1 CX=1 |
| 33summer24 | 11 |  | NUM | NUM | ALG=2 LOG=2 TRIG=3 DIFF=1 INT=2 NUM=0 VEC=1 DE=1 CX=2 |
| 33winter10 | 10 |  | INT, NUM |  | ALG=3 LOG=3 TRIG=2 DIFF=1 INT=3 NUM=1 VEC=1 DE=1 CX=2 |
| 33winter11 | 10 |  | DE |  | ALG=2 LOG=2 TRIG=3 DIFF=1 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33winter12 | 10 |  | INT |  | ALG=1 LOG=3 TRIG=2 DIFF=2 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33winter13 | 10 |  | DIFF, TRIG |  | ALG=2 LOG=3 TRIG=2 DIFF=1 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33winter14 | 9 |  | DE, INT |  | ALG=3 LOG=1 TRIG=2 DIFF=1 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33winter15 | 10 |  | ALG |  | ALG=1 LOG=2 TRIG=3 DIFF=3 INT=2 NUM=1 VEC=1 DE=1 CX=1 |
| 33winter16 | 10 |  | DE, TRIG |  | ALG=3 LOG=3 TRIG=3 DIFF=2 INT=1 NUM=1 VEC=1 DE=1 CX=1 |

## Changed Question-Level Decisions

### 33summer15

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 4 | 33summer15_q04 | DIFF | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Stationary-point differentiation is primary, with exponential/logarithmic curve context also assessed. |
| 6 | 33summer15_q06 | NUM | NUM | INT, TRIG | NUM, INT, TRIG | relabel_primary_add_secondary | Iteration is primary, with the fixed-point equation derived from a trigonometric integral. |
| 10 | 33summer15_q10 | ALG | INT | ALG, LOG | INT, ALG, LOG | relabel_primary_add_secondary | The exact integral is primary, with partial fractions and logarithmic exact form as substantial supporting work. |

### 33summer16

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 33summer16_q01 | DIFF | ALG |  | ALG | keep | This is a modulus inequality; no calculus is assessed. |
| 3 | 33summer16_q03 | LOG | TRIG |  | TRIG | keep | R cos form and trigonometric equation solving are the assessed topic. |
| 4 | 33summer16_q04 | DIFF | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Parametric differentiation is primary, with logarithmic functions defining the curve. |
| 5 | 33summer16_q05 | DIFF | DE | TRIG | DE, TRIG | relabel_primary_add_secondary | The prompt asks to solve a differential equation involving tangent functions; trigonometric manipulation is substantial. |
| 6 | 33summer16_q06 | NUM | NUM | DIFF, TRIG | NUM, DIFF, TRIG | relabel_primary_add_secondary | Iteration is primary, after stationary-point differentiation and trigonometric equation work form the root equation. |
| 7 | 33summer16_q07 | INT | INT | LOG | INT, LOG | relabel_primary_add_secondary | The exact integral is primary, with logarithmic form in the evaluated answer. |

### 33summer17

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 3 | 33summer17_q03 | DIFF | LOG |  | LOG | keep | This is an exponential equation solved using logarithms, not differentiation. |
| 5 | 33summer17_q05 | LOG | DIFF | LOG, TRIG | DIFF, LOG, TRIG | relabel_primary_add_secondary | Differentiating the logarithmic/trigonometric curve is the assessed task. |
| 6 | 33summer17_q06 | NUM | NUM | TRIG | NUM, TRIG | relabel_primary_add_secondary | Iteration is primary, with a cotangent equation as substantial trigonometric context. |
| 8 | 33summer17_q08 | DIFF | DE |  | DE | keep | The prompt asks to solve a differential equation; ordinary differentiation is not the dominant topic. |
| 9 | 33summer17_q09 | ALG | INT | ALG, LOG | INT, ALG, LOG | relabel_primary_add_secondary | Partial fractions support an exact integral whose result is logarithmic; integration is primary. |

### 33summer18

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 2 | 33summer18_q02 | DIFF | LOG |  | LOG | keep | This is an exponential equation solved using logarithms, not ordinary differentiation. |
| 4 | 33summer18_q04 | LOG | NUM | DIFF, LOG | NUM, DIFF, LOG | relabel_primary_add_secondary | Iteration/root finding is primary, with a derivative condition on a logarithmic/exponential curve used to form the equation. |
| 5 | 33summer18_q05 | DIFF | TRIG |  | TRIG | keep | The trigonometric identity and equation solving are the assessed topic, not differentiation. |
| 6 | 33summer18_q06 | ALG | DE | ALG, LOG | DE, ALG, LOG | relabel_primary_add_secondary | The problem is a differential equation solved using partial fractions and logarithms. |
| 7 | 33summer18_q07 | TRIG | TRIG | INT | TRIG, INT | relabel_primary_add_secondary | R-form trigonometry is primary, with a follow-on exact integral also assessed. |
| 8 | 33summer18_q08 | LOG | DIFF |  | DIFF | keep | Implicit differentiation is the assessed task; logarithms define the curve. |

### 33summer19

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 3 | 33summer19_q03 | LOG | INT | LOG, TRIG | INT, LOG, TRIG | relabel_primary_add_secondary | The exact integral is primary, with a trigonometric identity and logarithmic exact form also assessed. |
| 4 | 33summer19_q04 | LOG | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Gradient/differentiation of the logarithmic curve is primary. |
| 5 | 33summer19_q05 | DIFF | DE |  | DE | keep | The prompt asks to solve a differential equation; the current differentiation label is too narrow. |
| 7 | 33summer19_q07 | DIFF | DIFF | TRIG | DIFF, TRIG | relabel_primary_add_secondary | Parametric differentiation is primary, with trigonometric functions throughout the derivative work. |

### 33summer20

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 33summer20_q01 | DIFF | ALG |  | ALG | keep | This is a modulus inequality; no calculus is assessed. |
| 5 | 33summer20_q05 | ALG | TRIG |  | TRIG | keep | The equation and solution use trigonometric identities, not algebra as the dominant topic. |
| 7 | 33summer20_q07 | ALG | INT | ALG, LOG | INT, ALG, LOG | relabel_primary_add_secondary | Partial fractions support an exact definite integral with logarithmic terms; integration is primary. |
| 10 | 33summer20_q10 | DE | DE | INT | DE, INT | relabel_primary_add_secondary | The tank model is set up and solved as a differential equation; integration is the solution method and remains substantial coverage. |

### 33summer22

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 4 | 33summer22_q04 | DIFF | DIFF | TRIG | DIFF, TRIG | relabel_primary_add_secondary | Parametric differentiation is primary, with trigonometric simplification central to the derivative. |
| 6 | 33summer22_q06 | DIFF | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Gradient work is primary, with logarithmic/exponential functions defining the curve. |
| 10 | 33summer22_q10 | NUM | NUM | INT, LOG | NUM, INT, LOG | relabel_primary_add_secondary | Iteration is primary, with the fixed-point equation derived from an integral whose exact form is logarithmic. |

### 33summer24

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 33summer24_q01 | LOG | LOG |  | LOG | keep | This is an exponential/logarithmic equation, not differentiation. |
| 2 | 33summer24_q02 | DIFF | DIFF | TRIG | DIFF, TRIG | relabel_primary_add_secondary | Parametric differentiation is primary, with trigonometric functions defining the curve. |
| 7 | 33summer24_q07 | ALG | TRIG | ALG | TRIG, ALG | relabel_primary_add_secondary | Trigonometric equation solving is primary, with polynomial factorisation as supporting algebra. |
| 8 | 33summer24_q08 | ALG | INT | TRIG | INT, TRIG | relabel_primary_add_secondary | The exact integral is primary, with trigonometric identities used to transform the integrand. |

### 33winter10

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 2 | 33winter10_q02 | DIFF | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Differentiation is primary, with logarithmic/exponential functions defining the curve. |
| 4 | 33winter10_q04 | TRIG | TRIG | INT | TRIG, INT | relabel_primary_add_secondary | The trigonometric identity is primary, with an exact integral also assessed. |
| 5 | 33winter10_q05 | LOG | INT | ALG, LOG | INT, ALG, LOG | relabel_primary_add_secondary | Partial fractions support an exact integral whose answer is logarithmic; integration is primary. |
| 7 | 33winter10_q07 | LOG | NUM | INT, LOG | NUM, INT, LOG | relabel_primary_add_secondary | Iteration is primary, with the fixed-point equation derived from an integral involving logarithms. |
| 10 | 33winter10_q10 | ALG | CX | ALG | CX, ALG | relabel_primary_add_secondary | The polynomial root work is used to determine complex roots; complex numbers are primary with algebra support. |

### 33winter11

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 4 | 33winter11_q04 | INT | DE |  | DE | keep | The organism-growth model is set up and solved as a differential equation. |
| 5 | 33winter11_q05 | NUM | NUM | INT, LOG | NUM, INT, LOG | relabel_primary_add_secondary | Iteration is primary, with the fixed-point equation derived from an integral involving logarithms. |
| 8 | 33winter11_q08 | DIFF | DIFF | TRIG | DIFF, TRIG | relabel_primary_add_secondary | Parametric differentiation is primary, with trigonometric functions throughout. |
| 10 | 33winter11_q10 | LOG | INT | TRIG | INT, TRIG | relabel_primary_add_secondary | The exact integral is primary, with trigonometric substitution/manipulation central to the solution. |

### 33winter12

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 3 | 33winter12_q03 | DIFF | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Parametric differentiation is primary, with logarithmic functions defining the curve. |
| 5 | 33winter12_q05 | LOG | INT | DIFF, LOG | INT, DIFF, LOG | relabel_primary_add_secondary | The exact integral is the larger task, after differentiating a logarithmic function. |
| 7 | 33winter12_q07 | DIFF | INT | TRIG | INT, TRIG | relabel_primary_add_secondary | The area integral is primary, with trigonometric functions defining the curve. |

### 33winter13

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 4 | 33winter13_q04 | LOG | DIFF | LOG | DIFF, LOG | relabel_primary_add_secondary | Implicit differentiation is primary; logarithms define the curve. |
| 5 | 33winter13_q05 | NUM | NUM | INT, LOG | NUM, INT, LOG | relabel_primary_add_secondary | Iteration is primary, with the fixed-point equation derived from an integral involving logarithms. |
| 7 | 33winter13_q07 | TRIG | TRIG |  | TRIG | keep | The secant/cosecant/R-form work and equation solving are trigonometry, not logarithms. |
| 10 | 33winter13_q10 | DE | DE | TRIG | DE, TRIG | relabel_primary_add_secondary | The differential equation is primary, with cosine-squared/trigonometric integration central to the solution. |

### 33winter14

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 8 | 33winter14_q08 | DIFF | DE | INT, TRIG | DE, INT, TRIG | relabel_primary_add_secondary | The prompt explicitly asks to solve a differential equation; separation and integration by parts with trigonometric functions are substantial. |
| 10 | 33winter14_q10 | LOG | INT | ALG, LOG | INT, ALG, LOG | relabel_primary_add_secondary | Substitution and partial fractions support an exact integral whose answer is logarithmic; integration is primary. |

### 33winter15

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 1 | 33winter15_q01 | DIFF | LOG |  | LOG | keep | This is an exponential graph sketch, not a differentiation task. |
| 3 | 33winter15_q03 | DIFF | DIFF | TRIG | DIFF, TRIG | relabel_primary_add_secondary | Tangent-line differentiation is primary, with tangent/trigonometric functions defining the curve. |
| 4 | 33winter15_q04 | NUM | NUM | DIFF | NUM, DIFF | relabel_primary_add_secondary | Iteration is primary, with a gradient condition used to form the equation. |
| 5 | 33winter15_q05 | INT | INT | TRIG | INT, TRIG | relabel_primary_add_secondary | The exact integral is primary, with a trigonometric identity and substitution central to the solution. |
| 7 | 33winter15_q07 | LOG | INT | ALG, LOG | INT, ALG, LOG | relabel_primary_add_secondary | The integral is primary, with factor theorem/partial fractions as substantial algebra and logarithmic antiderivatives. |

### 33winter16

| Q | Question ID | Current | Proposed primary | Secondary | Coverage | Status | Rationale |
|---:|---|---|---|---|---|---|---|
| 2 | 33winter16_q02 | DIFF | DIFF | TRIG | DIFF, TRIG | relabel_primary_add_secondary | The gradient proof is primary, with sine/cosine identities needed to show positivity. |
| 3 | 33winter16_q03 | ALG | TRIG | ALG | TRIG, ALG | relabel_primary_add_secondary | The tan double-angle equation and trigonometric solutions are primary, with a quadratic in tan theta as algebra support. |
| 5 | 33winter16_q05 | LOG | DE | LOG | DE, LOG | relabel_primary_add_secondary | The curve is defined and solved through a differential equation, with logarithmic/exponential solution form. |
| 6 | 33winter16_q06 | LOG | INT | LOG | INT, LOG | relabel_primary_add_secondary | The substitution and exact integral are primary, with logarithmic exact form in the result. |
| 9 | 33winter16_q09 | NUM | NUM | DIFF, TRIG | NUM, DIFF, TRIG | relabel_primary_add_secondary | Iteration is primary, after differentiating a trigonometric curve to derive the tangent equation. |
