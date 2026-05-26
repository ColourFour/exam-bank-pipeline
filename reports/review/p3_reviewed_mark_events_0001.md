# P3 Reviewed Mark Events 0001

This is a mark-event population pass for already-promoted clean source-skill controls only. It does not promote source-skill records and does not use Batch 0003 retagged records.

## Summary

- Reviewed events: `11`
- Status counts: `{"approved": 11}`
- Source-skill promotions added: `0`
- Batch 0003 retagged records used: `0`

## Source Exact-Skill Records

- `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b` / `33summer23_q11_b` / `9709_p3_3_9_complex_arithmetic_polar_form`
  - Matching mark-event IDs: `["33summer23_q11_me0028", "33summer23_q11_me0029", "33summer23_q11_me0030", "33summer23_q11_me0031"]`
  - Other-part mark-event IDs existed: `True`
- `p3_exact_skill_review:batch_0002_seed:31summer24_q06:31summer24_q06_d` / `31summer24_q06_d` / `9709_p3_3_6_fixed_point_iteration`
  - Matching mark-event IDs: `["31summer24_q06_me0006", "31summer24_q06_me0007", "31summer24_q06_me0008"]`
  - Other-part mark-event IDs existed: `True`
- `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b` / `31summer24_q09_b` / `9709_p3_3_7_vector_lines`
  - Matching mark-event IDs: `["31summer24_q09_me0002", "31summer24_q09_me0003", "31summer24_q09_me0004", "31summer24_q09_me0005"]`
  - Other-part mark-event IDs existed: `True`

## Reviewed Events

### `33summer23_q11_me0028`

- Source question: `33summer23_q11`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b`
- Exact skill: `9709_p3_3_9_complex_arithmetic_polar_form`
- Question image: `output/p3/33summer23/questions/q11.png`
- Mark-scheme image: `output/p3/33summer23/mark_scheme/q11.png`
- Mark event: `B1` / `independent_statement` / confidence `high`
- Mark-scheme excerpt: 11(b) State arg(z^3) = -3π/4 B1; complete method to obtain r from z M1; r = 16 sqrt(2) A1; A1 if z = 2 - 2i obtained correctly.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: Part (b) explicitly asks for z^3 in polar/exponential form, and this B1 mark states the argument of z^3. The event is part-path matched to (b) and supports the approved complex polar/modulus exact skill.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `33summer23_q11_me0029`

- Source question: `33summer23_q11`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b`
- Exact skill: `9709_p3_3_9_complex_arithmetic_polar_form`
- Question image: `output/p3/33summer23/questions/q11.png`
- Mark-scheme image: `output/p3/33summer23/mark_scheme/q11.png`
- Mark event: `M1` / `method` / confidence `high`
- Mark-scheme excerpt: 11(b) State arg(z^3) = -3π/4 B1; complete method to obtain r from z M1; r = 16 sqrt(2) A1; A1 if z = 2 - 2i obtained correctly.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: Part (b) requires the modulus r for z^3; this M1 mark awards the complete method for obtaining r from z. It is matched to part (b) and is not an adjacent-part method.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `33summer23_q11_me0030`

- Source question: `33summer23_q11`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b`
- Exact skill: `9709_p3_3_9_complex_arithmetic_polar_form`
- Question image: `output/p3/33summer23/questions/q11.png`
- Mark-scheme image: `output/p3/33summer23/mark_scheme/q11.png`
- Mark event: `A1` / `accuracy` / confidence `high`
- Mark-scheme excerpt: 11(b) State arg(z^3) = -3π/4 B1; complete method to obtain r from z M1; r = 16 sqrt(2) A1; A1 if z = 2 - 2i obtained correctly.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This A1 mark awards the correct exact modulus r = 16 sqrt(2) for z^3 in part (b). It aligns directly with the approved complex polar/modulus source-skill record.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `33summer23_q11_me0031`

- Source question: `33summer23_q11`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0001_seed:33summer23_q11:33summer23_q11_b`
- Exact skill: `9709_p3_3_9_complex_arithmetic_polar_form`
- Question image: `output/p3/33summer23/questions/q11.png`
- Mark-scheme image: `output/p3/33summer23/mark_scheme/q11.png`
- Mark event: `A1` / `accuracy` / confidence `medium`
- Mark-scheme excerpt: 11(b) State arg(z^3) = -3π/4 B1; complete method to obtain r from z M1; r = 16 sqrt(2) A1; A1 if z = 2 - 2i obtained correctly.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This A1 mark is tied to correct use of z = 2 - 2i in part (b), supporting the same z^3 polar/exponential result. It is part-path matched and understandable as an accuracy mark within the approved target skill.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q06_me0006`

- Source question: `31summer24_q06`
- Part path: `d`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q06:31summer24_q06_d`
- Exact skill: `9709_p3_3_6_fixed_point_iteration`
- Question image: `output/p3/31summer24/questions/q06.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q06.png`
- Mark event: `M1` / `method` / confidence `high`
- Mark-scheme excerpt: 6(d) Use the iterative formula correctly at least twice M1; obtain final answer 1.50 A1; show sufficient iterations to 4 dp to justify 1.50 A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: Part (d) asks students to use the given iterative formula; this M1 mark awards using that iterative formula correctly at least twice. It is part-path matched to (d) and directly supports fixed-point iteration.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q06_me0007`

- Source question: `31summer24_q06`
- Part path: `d`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q06:31summer24_q06_d`
- Exact skill: `9709_p3_3_6_fixed_point_iteration`
- Question image: `output/p3/31summer24/questions/q06.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q06.png`
- Mark event: `A1` / `accuracy` / confidence `high`
- Mark-scheme excerpt: 6(d) Use the iterative formula correctly at least twice M1; obtain final answer 1.50 A1; show sufficient iterations to 4 dp to justify 1.50 A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This A1 mark awards the final fixed-point iteration answer 1.50 for part (d). It belongs to the reviewed part and directly supports the approved fixed-point iteration skill.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q06_me0008`

- Source question: `31summer24_q06`
- Part path: `d`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q06:31summer24_q06_d`
- Exact skill: `9709_p3_3_6_fixed_point_iteration`
- Question image: `output/p3/31summer24/questions/q06.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q06.png`
- Mark event: `A1` / `accuracy` / confidence `high`
- Mark-scheme excerpt: 6(d) Use the iterative formula correctly at least twice M1; obtain final answer 1.50 A1; show sufficient iterations to 4 dp to justify 1.50 A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This A1 mark awards sufficient iteration evidence to justify the two-decimal-place result in part (d). It is matched to the reviewed part and is safe for the fixed-point iteration gate.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q09_me0002`

- Source question: `31summer24_q09`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b`
- Exact skill: `9709_p3_3_7_vector_lines`
- Question image: `output/p3/31summer24/questions/q09.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q09.png`
- Mark event: `B1` / `independent_statement` / confidence `high`
- Mark-scheme excerpt: 9(b) Express a general point of at least one line correctly B1; equate corresponding components and solve M1; obtain lambda = -1 or mu = 0 A1; obtain -i - j - k A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: Part (b) asks for the vector point of intersection; this B1 mark awards expressing a point on one line in component form. It is matched to part (b) and supports vector-line intersection setup.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q09_me0003`

- Source question: `31summer24_q09`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b`
- Exact skill: `9709_p3_3_7_vector_lines`
- Question image: `output/p3/31summer24/questions/q09.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q09.png`
- Mark event: `M1` / `method` / confidence `high`
- Mark-scheme excerpt: 9(b) Express a general point of at least one line correctly B1; equate corresponding components and solve M1; obtain lambda = -1 or mu = 0 A1; obtain -i - j - k A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This M1 mark awards equating components and solving for a line parameter in part (b). It directly supports the approved vector-line intersection target skill.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q09_me0004`

- Source question: `31summer24_q09`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b`
- Exact skill: `9709_p3_3_7_vector_lines`
- Question image: `output/p3/31summer24/questions/q09.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q09.png`
- Mark event: `A1` / `accuracy` / confidence `high`
- Mark-scheme excerpt: 9(b) Express a general point of at least one line correctly B1; equate corresponding components and solve M1; obtain lambda = -1 or mu = 0 A1; obtain -i - j - k A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This A1 mark awards the correct parameter value in part (b), a direct intermediate for the vector-line intersection solution.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`

### `31summer24_q09_me0005`

- Source question: `31summer24_q09`
- Part path: `b`
- Source exact-skill decision: `p3_exact_skill_review:batch_0002_seed:31summer24_q09:31summer24_q09_b`
- Exact skill: `9709_p3_3_7_vector_lines`
- Question image: `output/p3/31summer24/questions/q09.png`
- Mark-scheme image: `output/p3/31summer24/mark_scheme/q09.png`
- Mark event: `A1` / `accuracy` / confidence `high`
- Mark-scheme excerpt: 9(b) Express a general point of at least one line correctly B1; equate corresponding components and solve M1; obtain lambda = -1 or mu = 0 A1; obtain -i - j - k A1.
- In matching_mark_event_ids: `true`
- Other-part mark-event IDs existed: `true`
- Decision status: `approved`
- Rationale: This A1 mark awards the final position vector of the intersection point in part (b). It is part-path matched and directly supports the approved vector-line intersections exact skill.
- Warnings: `["Approval applies only to this event ID and reviewed part path.", "Other-part mark events for the same question were not approved in this pass."]`


## Content Lab After Population

- Total candidates: `2432`
- Allow/generation-ready candidates: `0`
- Blocked-until-reviewed candidates: `2432`
- Rejected mark-event candidates: `0`
- Candidates newly passing the mark-event gate: `3`
- Remaining block reasons for those candidates: `{"mapping_or_subpart_not_reviewed_or_approved": 2, "missing_source_skill_ids": 3}`

## Exclusions

- Batch 0003 retagged records were not reviewed for mark-event approval.
- Other-part mark-event IDs were not approved in this pass.
- Records with empty or uncertain matching mark-event IDs were excluded.
