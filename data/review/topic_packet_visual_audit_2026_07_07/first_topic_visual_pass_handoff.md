# First Topic Packet Visual Pass Handoff

Date: 2026-07-07

Start target: `p1/circular_measure`

Scope: topic-packet pipeline only. Do not regenerate Asterion, app exports, or downstream student-runtime exports during this loop.

## Current Checkpoint

- The topic packet visual-audit CLI is implemented at `src/exam_bank/topic_packet_visual_audit.py` and wired into `src/exam_bank/cli.py` as `topic-packet-visual-audit`.
- `topic-packet-visual-audit run` defaults to `--provider codex`, uses `codex exec`, attaches the rendered page PNG, and does not require `OPENAI_API_KEY`.
- `--provider openai` remains available as an explicit opt-in path and is the only path that requires `OPENAI_API_KEY`.
- The initial general packet-layout repair for very tall answer images is implemented in `src/exam_bank/topic_packets.py`.
- Regression coverage is in `tests/test_topic_packet_visual_audit.py` and `tests/test_topic_packets.py`.
- Latest verification before this handoff:

```bash
.venv/bin/python -m pytest tests/test_topic_packet_visual_audit.py tests/test_topic_packets.py tests/test_visual_topic_audit.py
# 76 passed, 5 warnings
```

## First Pass Objective

Run the first full visual QA pass on `p1/circular_measure`, starting from the known seed failures. The goal is not just to fix those exact pages; the goal is to use the pass to improve the visual-audit loop, classify root causes, decide whether fixes should be general or targeted, and then regenerate only the appropriate topic-packet artifacts.

The first pass is complete only when:

- Every rendered page in `p1/circular_measure` has a decision.
- Every seed bug page is marked `fixed` or `waived_with_reason` with a concrete reason.
- No `bug`, `needs_human`, or `not_reviewable` decision remains untriaged.
- Any generalized fix is backed by at least two independent records, or by one reusable invariant violation.
- The packet has been regenerated after fixes and the changed pages have been re-reviewed.

## Seed Bugs In First Packet

These are expected to be present in `p1/circular_measure` and must be reviewed carefully even if Codex marks nearby pages as pass:

| Type | User label | Canonical ID | Current packet ref |
| --- | --- | --- | --- |
| Problem | 2015 June P12 Q2 | `12summer15_q02` | problem 2 |
| Problem | 2020 June P13 Q5 | `13summer20_q05` | problem 3 |
| Problem | 2011 Nov P13 Q6 | `13winter11_q06` | problem 4 |
| Problem | 2013 Nov P12 Q2 | `12winter13_q02` | problem 12 |
| Problem | 2012 June P12 Q6 | `12summer12_q06` | problem 45 |
| Problem | 2024 June P13 Q3 | `13summer24_q03` | problem 48; source uses `m24` artifacts |
| Problem | 2019 June P13 Q3 | `13summer19_q03` | problem 50 |
| Problem | 2013 Nov P13 Q6 | `13winter13_q06` | problem 59 |
| Problem | 2019 Nov P12 Q4 | `12winter19_q04` | problem 77 |
| Problem | 2022 Nov P12 Q10 | `12winter22_q10` | problem 79 |
| Problem | 2019 Nov P11 Q8 | `11winter19_q08` | problem 81 |
| Problem | 2023 Nov P13 Q10 | `13winter23_q10` | problem 92; nested mark-scheme asset exists |
| Mark scheme | 2019 June P11 Q3 | `11summer19_q03` | problem 1 answer; currently very tall |
| Mark scheme | 2018 June P13 Q5 | `13summer18_q05` | problem 22 answer; currently very tall |

## First Commands

Build the page batch and rendered page PNGs for the first packet:

```bash
.venv/bin/python -m exam_bank.cli topic-packet-visual-audit build-batch \
  --paper-family p1 \
  --topic circular_measure \
  --out-dir data/review/topic_packet_visual_audit_2026_07_07 \
  --render-root output/audits/topic_packet_visual_audit_2026_07_07
```

Confirm the Codex runner path without making decisions:

```bash
.venv/bin/python -m exam_bank.cli topic-packet-visual-audit run \
  --batch data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_batch.json \
  --max-records 1 \
  --dry-run
```

Run a small first slice. This proves the Codex review loop before spending time on the full packet:

```bash
.venv/bin/python -m exam_bank.cli topic-packet-visual-audit run \
  --batch data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_batch.json \
  --out data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_decisions.jsonl \
  --max-records 5
```

Import the partial decisions with incomplete import allowed, then inspect the registry before continuing:

```bash
.venv/bin/python -m exam_bank.cli topic-packet-visual-audit import-decisions \
  --batch data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_batch.json \
  --decisions data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_decisions.jsonl \
  --out data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_bug_registry.v1.json \
  --markdown-out data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_bug_registry.v1.md \
  --allow-incomplete
```

If the first slice behaves correctly, continue without `--max-records`:

```bash
.venv/bin/python -m exam_bank.cli topic-packet-visual-audit run \
  --batch data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_batch.json \
  --out data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_decisions.jsonl
```

When the packet has full decisions, import without `--allow-incomplete`:

```bash
.venv/bin/python -m exam_bank.cli topic-packet-visual-audit import-decisions \
  --batch data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_batch.json \
  --decisions data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_decisions.jsonl \
  --out data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_bug_registry.v1.json \
  --markdown-out data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_bug_registry.v1.md
```

## Repair Decision Tree

Use this classification before editing code or regenerating artifacts:

| Symptom | Likely owner | Preferred repair |
| --- | --- | --- |
| Question text or diagram is clipped in the source PNG | `question_png_regeneration` | Fix extraction/crop regeneration for the canonical PNG, then regenerate affected packet pages. |
| Mark scheme exists only in nested metadata or source metadata path is mismatched | `mark_scheme_path_promotion` | Fix resolver/promotion behavior. This is general if a valid nested mark-scheme asset fails to resolve. |
| Mark scheme is too tall or unreadable only inside packet PDF | `topic_packet_layout` | Prefer general splitting/layout behavior over one-off exceptions. |
| Source PNG is good but packet page is downsampled or unreadable | `topic_packet_layout` | Fix packet image sizing/optimization or legibility constraints. |
| Header overlaps body, problem labels, or continuation text | `topic_packet_layout` | Fix wrapping/header height behavior. |
| Question belongs in the wrong topic packet | `topic_routing_review` | Use visual-topic-audit or topic-overlap review sidecars, not the page visual-audit sidecar. |
| Page cannot be reviewed due to missing render or unreadable metadata | `packet_visual_audit` | Fix batch construction/rendering first. |

Generalize only when the same root cause appears in at least two independent records, or when one record violates a reusable invariant. Otherwise use a targeted reviewed exception with a regression test.

## Regeneration Commands

After a targeted fix for this packet, regenerate only `p1/circular_measure` first, preserving difficulty order:

```bash
.venv/bin/python -m exam_bank.cli topic-packets \
  --paper-family p1 \
  --topic circular_measure \
  --topic-difficulty-review data/review/topic_difficulty/p1_circular_measure_b9e27a7a/topic_packet_difficulty_review.v1.json
```

After a generalized fix, regenerate all major topic packets one by one using their existing difficulty sidecars. Do not do that until the root cause is clearly generalized.

## Verification After Each Repair

Run focused tests after each code repair:

```bash
.venv/bin/python -m pytest tests/test_topic_packet_visual_audit.py tests/test_topic_packets.py tests/test_visual_topic_audit.py
```

Then rebuild the visual batch for `p1/circular_measure`, rerun only changed or previously flagged pages if possible, and re-import. The final import for the packet must be complete and must not require `--allow-incomplete`.

Before moving to the next topic packet:

- `topic_packet_visual_bug_registry.v1.json` has no unresolved blocking decisions.
- All seed-bug records are `fixed` or `waived_with_reason`.
- No unresolved missing-answer/path mismatch is introduced in the packet manifest or summary.
- `git diff --check` is clean.
- The handoff should be updated with what changed, what generalized, what stayed targeted, and the next packet to audit.

## Next Handoff Update Fields

Append a short section to this file after the first pass:

```md
## Iteration 1 Result

- Packet:
- Decisions reviewed:
- Bugs confirmed:
- Needs human:
- Fixes made:
- Generalized fixes:
- Targeted exceptions:
- Regenerated artifacts:
- Verification:
- Next packet:
```

## Iteration 1 Result

- Packet: `p1/circular_measure`
- Decisions reviewed: 117 initial rendered pages were reviewed during the first pass; after repairing sidecar-driven regeneration, 121 current rendered pages were re-reviewed/imported for the regenerated 101-problem packet.
- Bugs confirmed: 8 current packet bug pages remain, all triaged as `waived_with_reason`; the first-pass registry confirmed 25 pre-repair bug pages, including the two very tall answer pages that are now fixed.
- Needs human: 0 current records; no `bug`, `needs_human`, `not_reviewable`, or seed-level open status remains untriaged.
- Fixes made: removed unsupported Codex CLI approval flag, made the Codex decision schema strict-compatible, made exact `--topic-difficulty-review` sidecars authoritative for packet membership/order, added fail-closed handling for sidecar records that cannot be included, added mixed-seed page resolution overrides, and regenerated only `p1/circular_measure`.
- Generalized fixes: tall answer/mark-scheme blocks are split across pages instead of being scaled below legibility; verified on `13summer18_q05` and `11summer19_q03`. Exact difficulty sidecars now preserve reviewed packet membership and ordering, including records whose raw question-bank topic has drifted.
- Targeted exceptions: source/routing-owned defects were waived for `12summer15_q02`, `13winter11_q06`, `12winter13_q02`, `12summer12_q06`, `13summer24_q03`, `13winter13_q06`, `12winter19_q04`, and `13winter23_q10`. Non-seed packet defects were likewise triaged to `question_png_regeneration` or `topic_routing_review`.
- Regenerated artifacts: `output/topic_packets/p1/circular_measure/p1_circular_measure_packet.pdf`, `output/topic_packets/p1/circular_measure/manifest.json`, `data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_batch.json`, `data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_audit_decisions.jsonl`, `data/review/topic_packet_visual_audit_2026_07_07/topic_packet_visual_bug_registry.v1.json`, and rendered page PNGs under `output/audits/topic_packet_visual_audit_2026_07_07/p1_circular_measure/`.
- Verification: `topic-packet-visual-audit import-decisions` completed with 121 accepted decisions, 0 missing, 0 rejected; manifest has `problem_count: 101`, `topic_difficulty_review_applied_count: 101`, `missing_answer_count: 0`, no oversized-block warnings, and no missing mark-scheme path entries. `13summer20_q05` is fixed at seed status and appears as Problem 3 / page 2 with Difficulty `99/101`.
- Next packet: `p1/coordinate_geometry`

## Iteration 2 Result

- Packet: `p1/coordinate_geometry`
- Decisions reviewed: 98 regenerated rendered pages were reviewed/imported for the 102-problem packet after source-crop repairs.
- Bugs confirmed: 15 initial packet bug pages were confirmed before repair: 13 question-crop artifacts and 2 mark-scheme crop artifacts. After repair, the rebuilt registry has 0 bug records.
- Needs human: 0 current records; no `bug`, `needs_human`, or `not_reviewable` decision remains.
- Fixes made: generalized rendered-crop cleanup for top-edge barcode bands, isolated dense edge marks, and right-edge watermark fragments; added full-page background graphic classification so page-wide vectors are ignored for figure detection without forcing full-page crops; tightened watermark/page-edge furniture classification to avoid clipping valid prompt text.
- Generalized fixes: source PNG cleanup now covers repeated 2024/2025 side-panel barcode artifacts, legacy Papacambridge watermark fragments, and scan marks. Full-page background graphics are treated as segmentation furniture rather than prompt figures.
- Targeted exceptions: none.
- Regenerated artifacts: canonical source PNGs for the affected coordinate-geometry question/mark-scheme records, `output/topic_packets/p1/coordinate_geometry/p1_coordinate_geometry_packet.pdf`, `output/topic_packets/p1/coordinate_geometry/manifest.json`, `data/review/topic_packet_visual_audit_2026_07_07/p1_coordinate_geometry/topic_packet_visual_audit_batch.json`, `data/review/topic_packet_visual_audit_2026_07_07/p1_coordinate_geometry/topic_packet_visual_audit_decisions.jsonl`, and `data/review/topic_packet_visual_audit_2026_07_07/p1_coordinate_geometry/topic_packet_visual_bug_registry.v1.json`.
- Verification: `topic-packet-visual-audit import-decisions` completed with 98 accepted decisions, 0 missing, 0 rejected, and 0 bug records. Manifest has `problem_count: 102`, `page_count: 98`, `topic_difficulty_review_applied_count: 102`, `missing_answer_count: 0`, and no oversized-block warnings. `.venv/bin/python -m pytest tests/test_image_limits.py tests/test_image_rendering.py tests/test_question_png_segmentation.py tests/test_topic_packet_visual_audit.py tests/test_topic_packets.py tests/test_visual_topic_audit.py` passed with 185 tests and 5 warnings. `git diff --check` is clean.
- Next packet: `p1/differentiation`

## Iteration 3 Result

- Packet: `p1/differentiation`
- Decisions reviewed: 162 regenerated rendered pages were reviewed/imported for the 162-problem packet.
- Bugs confirmed: 29 question-crop source artifacts and 1 over-compressed mark-scheme page were found during the first visual pass. After source PNG regeneration and packet-layout repair, the rebuilt registry has 16 non-blocking question-crop records, all triaged as `waived_with_reason`.
- Needs human: 0 current records; no `needs_human`, `not_reviewable`, or open blocking bug decision remains.
- Fixes made: regenerated the affected differentiation source question PNGs and the `13summer19_q08` mark-scheme PNG; raised the single-image answer split threshold so dense mark schemes split before they become unreadably small.
- Generalized fixes: single-image answer/mark-scheme blocks now split across pages when they would need substantial downscaling, covering the `13summer19_q08` invariant violation and preserving readability for other dense answer images.
- Targeted exceptions: residual non-blocking source artifacts were waived on pages 1, 5, 24, 30, 37, 40, 52, 53, 54, 58, 62, 64, 65, 69, 72, and 73. These are barcode strips, edge watermark fragments, continuation notices, or permission footers that do not obscure assessable text or diagrams.
- Regenerated artifacts: canonical source PNGs for affected differentiation questions, `output/pm1/pm1_2019_s19_13_ms_q08_markscheme.png`, `output/topic_packets/p1/differentiation/p1_differentiation_packet.pdf`, `output/topic_packets/p1/differentiation/manifest.json`, `data/review/topic_packet_visual_audit_2026_07_07/p1_differentiation/topic_packet_visual_audit_batch.json`, `data/review/topic_packet_visual_audit_2026_07_07/p1_differentiation/topic_packet_visual_audit_decisions.jsonl`, and `data/review/topic_packet_visual_audit_2026_07_07/p1_differentiation/topic_packet_visual_bug_registry.v1.json`.
- Verification: `topic-packet-visual-audit import-decisions` completed with 162 accepted decisions, 0 missing, 0 rejected. Manifest has `problem_count: 162`, `page_count: 162`, `topic_difficulty_review_applied_count: 162`, `missing_answer_count: 0`, and no oversized-block warnings; dense answers split at `answer:24`, `answer:43`, and `answer:88`. `.venv/bin/python -m pytest tests/test_image_limits.py tests/test_image_rendering.py tests/test_question_png_segmentation.py tests/test_topic_packet_visual_audit.py tests/test_topic_packets.py tests/test_visual_topic_audit.py` passed with 185 tests and 5 warnings. `git diff --check` is clean.
- Next packet: `p1/functions`

## Iteration 4 Result

- Packet: `p1/functions`
- Decisions reviewed: 152 regenerated rendered pages were reviewed/imported for the 138-problem packet.
- Bugs confirmed: 2 source question-crop bugs were confirmed and fixed: `11summer15_q08` had following-question spillover, and `13summer11_q10` had previous-question spillover. The rebuilt registry has 2 bug records, both `fixed`, and 0 open records.
- Needs human: 0 current records; no `needs_human`, `not_reviewable`, or open blocking bug decision remains.
- Fixes made: regenerated the affected functions source question PNGs, rebuilt only `p1/functions` with the existing difficulty sidecar, and repaired the crop-boundary logic so previous-question heads and following-question tails can be trimmed before neighbor-question rejection. Verification also found and fixed a graph-label regression where a numeric axis/tick label outside the question-anchor column was treated as a foreign question boundary.
- Generalized fixes: foreign-question boundary trimming now runs before neighbor detection and uses the trimmed content area; crop-top trimming can remove accidental previous-question content while preserving the current question anchor; later bare question numbers are treated as real boundaries in the left anchor column but numeric graph labels outside that column remain protected when adjacent to graphics.
- Targeted exceptions: none. Nonblocking source furniture such as barcode strips, edge watermark fragments, black squares, and permission/footer text was reviewed as outside assessable content.
- Regenerated artifacts: `output/pm1/pm1_2015_s15_11_qp_q08_question.png`, `output/pm1/pm1_2011_s11_13_qp_q10_question.png`, `output/topic_packets/p1/functions/p1_functions_packet.pdf`, `output/topic_packets/p1/functions/manifest.json`, `data/review/topic_packet_visual_audit_2026_07_07/p1_functions/topic_packet_visual_audit_batch.json`, `data/review/topic_packet_visual_audit_2026_07_07/p1_functions/topic_packet_visual_audit_decisions.jsonl`, and `data/review/topic_packet_visual_audit_2026_07_07/p1_functions/topic_packet_visual_bug_registry.v1.json`.
- Verification: `topic-packet-visual-audit import-decisions` completed with 152 accepted decisions, 0 missing, 0 rejected, and 2 fixed bug records. Manifest has `problem_count: 138`, `page_count: 152`, `topic_difficulty_review_applied_count: 138`, and `missing_answer_count: 0`; the long answer warning for `answer:51` was reviewed as readable after splitting. `.venv/bin/python -m pytest tests/test_image_limits.py tests/test_image_rendering.py tests/test_question_png_segmentation.py tests/test_topic_packet_visual_audit.py tests/test_topic_packets.py tests/test_visual_topic_audit.py` passed with 191 tests and 5 warnings.
- Next packet: `p1/integration`
