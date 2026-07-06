# Statistics Topic Coverage Repair Handoff - 2026-07-03

## Current State

Repository: `/Users/sbrooker/repos/exam-bank-pipeline`

The P3 and P1 topic packet coverage repair passes have been completed and verified through a full topic-packet regeneration.

Completed sidecars:

- P3 merged sidecar: `data/review/p3_topic_overlap_review_merged_2026_07_03.json`
- P1 merged sidecar: `data/review/p1_topic_overlap_review_merged_2026_07_03.json`
- Combined P1+P3 full-run sidecar: `data/review/topic_overlap_review_merged_p1_p3_2026_07_03.json`

Latest regenerated summary:

- `output/topic_packets/topic_packet_summary.json`
- Generated at: `2026-07-03T06:39:54.228975+00:00`
- Full packet output size: about `897M` under `output/topic_packets`
- `topic_overlap_reviews_loaded`: `1369`
- `topic_overlap_reviews_applied`: `1368`
- `tests/test_topic_packets.py`: `62 passed`, 5 SwigPy deprecation warnings

## Known Exceptions and Policy

Do not force topic labels just to satisfy coverage.

P3 accepted genuine exceptions:

- `33summer24` lacks `numerical_solution_of_equations`
- `33winter23` lacks `numerical_solution_of_equations`

P1 held-out exception pending user review:

- `11summer10` lacks `circular_measure`
- Do not fabricate a circular-measure label for this paper unless the user supplies a reviewed decision.

Use the same policy for Statistics:

- `primary_topic` is the dominant assessed, mark-bearing topic.
- `secondary_topics` count only for substantial mark-bearing overlap.
- `coverage_topics` should not duplicate the PDF in multiple packets; the sidecar model already handles coverage-only counting.
- Real exceptions should remain flagged and documented.
- Legacy/out-of-current-syllabus material should be excluded rather than forced into a current topic.

## Full Regeneration Command Used

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json \
  --reviewed-decisions data/review/topic_bank_reviewed_decisions.v1.json \
  --artifact-root output \
  --topic-overlap-review data/review/topic_overlap_review_merged_p1_p3_2026_07_03.json \
  --strict-syllabus \
  --pdf-profile print \
  --page-size a4 \
  --layout compact \
  --answer-placement end
```

Verification command:

```bash
.venv/bin/pytest tests/test_topic_packets.py
```

## Latest P1 Audit Result

P1 papers audited: `102`

- Passing: `101`
- Failing: `1`
- Remaining failure: `11summer10` missing `circular_measure`
- Over-3 topic failures: `0`

P1 packet counts:

| Topic | Questions | Coverage placements |
| --- | ---: | ---: |
| circular_measure | 101 | 105 |
| coordinate_geometry | 101 | 145 |
| differentiation | 162 | 231 |
| functions | 139 | 185 |
| integration | 126 | 178 |
| quadratics | 69 | 78 |
| series | 208 | 229 |
| trigonometry | 129 | 164 |

## Statistics Baseline

In this repository, the Asterion course `s1` maps to topic-packet paper family `p5`.

Current P5 audit from `output/topic_packets/topic_packet_summary.json`:

- P5 papers audited: `101`
- Passing: `31`
- Failing: `70`

Missing-topic counts across failing papers:

| Topic | Missing papers |
| --- | ---: |
| discrete_random_variables | 63 |
| permutations_and_combinations | 63 |
| representation_of_data | 63 |
| the_normal_distribution | 8 |
| probability | 1 |

Over-max counts:

| Topic | Papers over 3 |
| --- | ---: |
| probability | 62 |

Aggregate P5 audit coverage totals:

| Topic | Coverage count |
| --- | ---: |
| discrete_random_variables | 62 |
| permutations_and_combinations | 47 |
| probability | 382 |
| representation_of_data | 57 |
| the_normal_distribution | 141 |

This suggests the older Statistics papers, especially `06*` and `61/62/63*`, are over-collapsed into `probability` and need a sidecar pass similar to P1/P3.

First failing examples:

- `06summer08`: missing `discrete_random_variables`, `permutations_and_combinations`, `representation_of_data`; `probability` count `6`
- `06summer09`: missing `discrete_random_variables`, `permutations_and_combinations`, `representation_of_data`; `probability` count `4`
- `06winter08`: missing `discrete_random_variables`, `permutations_and_combinations`, `representation_of_data`; `probability` count `6`
- `51summer20`: missing `the_normal_distribution`
- `51summer25`: missing `probability`
- `51winter20`: missing `the_normal_distribution`
- `52summer20`: missing `the_normal_distribution`
- `52winter20`: missing `the_normal_distribution`

## Recommended Next Workflow

1. Audit P5 from `output/topic_packets/topic_packet_summary.json`, specifically `paper_topic_coverage_audit.papers` where `paper_family == "p5"` and `passes_min_one_max_three == false`.
2. Start with the high-signal older papers that are over-counted as probability and missing three topics:
   - `06summer08`
   - `06summer09`
   - `06winter08`
   - then `61summer10`, `61summer13`, `61summer14`, `61summer15`, `61summer16`, `61summer17`, `61summer18`, `61summer19`, `61winter09`, `61winter10`
3. Review papers in small batches using:
   - `output/json/question_bank.json`
   - canonical crop artifacts under `output/run_status/.../batch_artifacts`
   - generated packet manifests under `output/topic_packets/p5/<topic>/manifest.json`
4. Create Statistics sidecars analogous to P1/P3:
   - `data/review/p5_topic_overlap_review_batch_001_2026_07_03.{json,csv,md}`
   - `data/review/p5_topic_overlap_review_merged_2026_07_03.json`
5. For validation, combine P1, P3, and P5 sidecars into a single full-run sidecar, or run targeted P5 dry-runs with the P5 sidecar alone.
6. Run dry-run packet generation first, then full generation once the P5 audit is acceptable.
7. Run `.venv/bin/pytest tests/test_topic_packets.py`.

## Useful Audit Snippet

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path

summary = json.loads(Path("output/topic_packets/topic_packet_summary.json").read_text())
papers = [
    paper
    for paper in summary["paper_topic_coverage_audit"]["papers"]
    if paper.get("paper_family") == "p5"
]
failures = [paper for paper in papers if not paper.get("passes_min_one_max_three")]

missing = Counter()
over = Counter()
for paper in failures:
    missing.update(paper.get("missing_topics", []))
    over.update((paper.get("over_max_topics") or {}).keys())

print("p5 papers", len(papers))
print("p5 failures", len(failures))
print("missing", dict(sorted(missing.items())))
print("over", dict(sorted(over.items())))
for paper in failures[:20]:
    print(
        paper["paper"],
        "missing=", paper.get("missing_topics"),
        "over=", paper.get("over_max_topics"),
        "counts=", paper.get("topic_coverage_counts"),
    )
PY
```

## Current Worktree Notes

At handoff creation time, expected changed/untracked review artifacts include:

- modified `data/review/p1_topic_overlap_review_merged_2026_07_03.json`
- untracked `data/review/p1_topic_overlap_review_batch_010_2026_07_03.*` through `batch_019`
- untracked `data/review/topic_overlap_review_merged_p1_p3_2026_07_03.json`
- this handoff file

The generated topic packets under `output/topic_packets` are ignored/generated output and were regenerated successfully.
