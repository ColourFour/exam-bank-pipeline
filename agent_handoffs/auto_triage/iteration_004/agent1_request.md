# Agent 1 Request - iteration_004

Plan one narrow extraction-quality improvement pass.

Selected target:
- Issue: `paper_total_mismatch`
- Current count: `86`
- Current hard failures: `133`
- Stop threshold: `50`

Top issue counts:
- `paper_total_mismatch`: 86
- `question_mark_total_mismatch`: 28
- `weak_question_anchor`: 9
- `question_scope_contaminated`: 4
- `polluted_pass_requires_review`: 3
- `question_subparts_incomplete`: 2
- `missing_terminal_mark_total`: 1

Stop criteria:
- Stop immediately if current hard failures are at or below the configured threshold.
- Stop if the selected target is not actionable from the visual sample.
- Stop if the work would require broad validation or trust-gate loosening.

Acceptance criteria:
- Focused regression tests are added or updated for reviewed examples where practical.
- Full `.venv/bin/python -m pytest` passes.
- Canonical comparison uses an OCR-enabled current output against an OCR-enabled baseline.
- Hard failures decrease, or `paper_total_mismatch` decreases by at least one.
- `worsened_records` stays under the configured threshold.
- No broad status regression or flag suppression without extraction evidence.

What not to change:
- Do not delete or rewrite existing `output/triage` baselines.
- Do not make `question_bank.json` the source of truth over image crops.
- Do not treat extracted text, DeepSeek labels, or topic labels as canonical evidence.
- Do not do unrelated cleanup.

Commands:
- `triage_sample`:

```bash
.venv/bin/python -m exam_bank.cli triage-sample --input output_ocr_candidate/json/question_bank.json --output-root output_ocr_candidate/triage --iteration iteration_004 --issue-set hard-failures --target paper_total_mismatch --sample-size 30 --seed 1
```
- `triage_serve`:

```bash
.venv/bin/python -m exam_bank.cli triage-serve --iteration output_ocr_candidate/triage/iteration_004
```
- `full_ocr_rerun`:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output_ocr_candidate --enable-ocr
```
- `ocr_verification`:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status --input output_ocr_candidate/json/question_bank.json
```
- `full_tests`:

```bash
.venv/bin/python -m pytest
```
- `triage_comparison`:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-compare --iteration agent_handoffs/auto_triage/iteration_004 --baseline-triage output_ocr_candidate/triage/iteration_004 --current output_ocr_candidate/json/question_bank.json --output output_ocr_candidate/triage/iteration_004/comparison.auto-iteration-004.json --test-status pass
```
