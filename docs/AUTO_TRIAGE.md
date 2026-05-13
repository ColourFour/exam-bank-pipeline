# Auto-Triage

The auto-triage loop is a bounded, evidence-gated path for improving extraction quality without creating an unsafe auto-edit cycle. It measures the current corpus, selects the largest hard-failure cluster, creates an agent handoff iteration, and then requires tests plus OCR-enabled comparison evidence before accepting an improvement.

It does not edit code by itself. Agents or humans perform the targeted implementation work using the generated handoff files.

## Relationship To Manual Triage

Manual triage remains the visual review primitive. `triage-sample` freezes a baseline and creates the crop gallery; `triage-serve` records visual notes; `triage-compare` reports movement against the frozen baseline.

Auto-triage coordinates that loop. It measures the corpus, chooses the next hard-failure target, writes agent handoff files, prints exact commands, and records whether a completed pass is accepted, rejected, or needs human review.

An accepted auto-triage decision is not a trust override. Downstream consumers must still honor the record-level validation, mapping, scope, text-fidelity, topic-trust, and visual-curation gates.

## Why It Exists

Aggregate counts alone are not enough for this project. A change can reduce a validation flag while making question crops, mark-scheme mapping, or text trust worse. Auto-triage keeps the loop repeatable:

1. Measure `question_bank.json`.
2. Select one dominant hard-failure cluster.
3. Create a bounded handoff folder.
4. Generate exact triage and comparison commands.
5. Run OCR-enabled production-style output.
6. Compare against a frozen baseline.
7. Accept only when evidence improves and tests pass.

## Current Local Evidence

Current no-OCR export:

- `output/json/question_bank.json`
- OCR profile: disabled for all `1301` records.
- Hard failures: `148`.
- Dominant hard-failure cluster: `paper_total_mismatch` (`107`).
- Use this output for layout/debug comparisons only, not production OCR scoring.

Current OCR candidate:

- `output_ocr_candidate/json/question_bank.json`
- OCR profile: enabled for all `1301` records.
- Hard failures: `133`.
- Dominant hard-failure cluster: `paper_total_mismatch` (`86`).
- Latest accepted auto-triage comparison: `output_ocr_candidate/triage/iteration_002/comparison.auto-iteration-003.json`.
- That comparison moved hard failures `153 -> 133`, moved `paper_total_mismatch` `107 -> 86`, and reported `worsened_records: []`.

For new candidate runs, prefer `output/candidates/ocr/<run_id>/` or `output/candidates/ocr/latest/`. The historical `output_ocr_candidate/` root remains supported for compatibility and evidence review.

## Hard-Failure Target

Set a stopping threshold with `--target-max-hard-failures`. For example, to continue until the corpus has at most 100 hard failures:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output/candidates/ocr/latest \
  --enable-ocr
```

```bash
.venv/bin/python -m exam_bank.cli auto-triage-plan \
  --input output/candidates/ocr/latest/json/question_bank.json \
  --handoff-root agent_handoffs/auto_triage \
  --candidate-output output/candidates/ocr/latest \
  --target-max-hard-failures 100 \
  --sample-size 30
```

If the current hard-failure count is already at or below the target, the command prints `stopped: true` and does not create a new iteration folder.

You can run status against `output/json/question_bank.json` for debugging, but canonical planning and comparison should use OCR-enabled output.

## Target Selection

The next issue is selected from hard failures only. Hard failures are records where validation, mark-scheme mapping, or visual curation failed.

The primary issue comes from the existing triage logic:

1. First validation flag.
2. Mapping failure reason.
3. Visual curation failure.
4. Review-status fallback.

The largest remaining primary issue becomes the selected target. Override the implementation scope in the agent plan only when the sample proves the selected cluster is blocked or not actionable.

## Status

Use status to measure the current corpus:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status \
  --input output/json/question_bank.json \
  --output output/triage/auto-status.json
```

The report includes record count, OCR counts, text-source profile, trust/status counts, flag counts, hard-failure count, dominant failure cluster, and top issue clusters.

## Handoff Folders

Planning creates a folder such as:

```text
agent_handoffs/auto_triage/iteration_005/
  metrics_before.json
  selected_target.json
  commands.json
  agent1_request.md
```

The shared five-agent prompts live under:

```text
agent_handoffs/auto_triage/Prompt/
```

The planner also chooses an iteration number that avoids existing `agent_handoffs/auto_triage/iteration_*` folders and the configured triage root, such as `output/triage/iteration_*` or `output_ocr_candidate/triage/iteration_*`.

## Runbook

Print the next commands for a handoff created by the planner:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-runbook \
  --input output/candidates/ocr/latest/json/question_bank.json \
  --candidate-output output/candidates/ocr/latest \
  --handoff-root agent_handoffs/auto_triage
```

For older or non-default handoffs, pass `--iteration` and `--baseline-triage` explicitly so generated comparison paths match the intended frozen baseline.

The runbook includes:

- next `triage-sample`
- `triage-serve`
- full OCR rerun
- OCR verification through `auto-triage-status`
- full pytest
- auto-triage comparison

## OCR Baselines

Canonical production comparisons must use OCR-enabled current output against an OCR-enabled baseline.

Run the OCR export into a candidate folder:

```bash
.venv/bin/python -m exam_bank.cli process \
  --input input \
  --output output/candidates/ocr/latest \
  --enable-ocr
```

Then verify OCR state:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status \
  --input output/candidates/ocr/latest/json/question_bank.json
```

Do not claim production improvement from a no-OCR current output. No-OCR comparisons can help debug layout or crop behavior, but they must not be mixed into the canonical OCR score.

## Compare And Decide

After Agent 3 has run full tests and Agent 4 has produced OCR-enabled output:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-compare \
  --iteration agent_handoffs/auto_triage/iteration_005 \
  --baseline-triage output/triage/iteration_005 \
  --current output/candidates/ocr/latest/json/question_bank.json \
  --output output/triage/iteration_005/comparisons/comparison.auto-iteration-005.json \
  --test-status pass
```

This writes:

```text
agent_handoffs/auto_triage/iteration_005/
  metrics_after.json
  decision.json
```

Accepted requires:

- full tests pass
- current and baseline outputs are OCR-enabled
- hard failures decrease, or the selected target decreases materially
- `worsened_records` stays under the configured threshold
- no major unrelated status regression
- no broad validation or trust-gate loosening without evidence

The command writes `metrics_after.json` and `decision.json` into the auto-triage handoff folder. It prints the decision object. A rejected decision exits non-zero; `accepted` and `needs-human-review` exit zero so the decision file must be read before making a quality claim.

Rejected or paused conditions include:

- tests fail
- current output is no-OCR
- hard failures increase beyond the configured threshold
- `worsened_records` exceeds the configured threshold
- selected target fails to improve for two consecutive iterations
- flags appear suppressed without extraction evidence

Use `--test-status unknown` when test evidence is not available. The command will not accept the iteration in that state.

## Inspecting Worsened Records

Open the comparison file and inspect `worsened_records`. Each entry lists a shared `question_id` and the tracked status fields that moved in the wrong direction.

Treat any worsened record as a review blocker unless it is intentionally stricter and supported by crop evidence.

## Stopping Safely

Stop the loop when:

- `auto-triage-plan` reports the hard-failure target has been met.
- The dominant issue is not actionable from local code.
- The next fix would require broad validation/trust loosening.
- OCR-enabled evidence is unavailable.
- Tests cannot be made green.

The loop is evidence-gated rather than fully autonomous because the remaining failures often involve visual scope, math text fidelity, mark-scheme pairing, and source evidence. Those need reviewed image crops and regression tests, not unbounded automatic edits.
