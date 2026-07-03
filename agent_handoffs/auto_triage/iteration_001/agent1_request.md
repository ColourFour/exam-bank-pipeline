# Agent 1 Request - iteration_001

Plan one narrow extraction-quality improvement pass.

Selected target:
- Issue: `missing_image_detection_failure`
- Current count: `889`
- Current hard failures: `1897`
- Stop threshold: `100`

Top issue counts:
- `missing_image_detection_failure`: 889
- `mapping_failed:partial_question_block`: 661
- `mapping_failed:segmentation_failure`: 210
- `paper_total_mismatch`: 79
- `polluted_pass_requires_review`: 28
- `mapping_failed:mark_scheme_part_structure_mismatch`: 8
- `question_mark_total_mismatch`: 7
- `question_subparts_incomplete`: 6
- `question_scope_contaminated`: 4
- `visual_curation_failed`: 2

Stop criteria:
- Stop immediately if current hard failures are at or below the configured threshold.
- Stop if the selected target is not actionable from the visual sample.
- Stop if the work would require broad validation or trust-gate loosening.

Acceptance criteria:
- Focused regression tests are added or updated for reviewed examples where practical.
- Full `.venv/bin/python -m pytest` passes.
- Canonical comparison uses an OCR-enabled current output against an OCR-enabled baseline.
- Hard failures decrease, or `missing_image_detection_failure` decreases by at least one.
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
.venv/bin/python -m exam_bank.cli triage-sample --input output/json/question_bank.json --output-root /Users/sbrooker/repos/exam-bank-pipeline/output/triage --iteration iteration_001 --issue-set hard-failures --target missing_image_detection_failure --sample-size 30 --seed 1
```
- `triage_serve`:

```bash
.venv/bin/python -m exam_bank.cli triage-serve --iteration /Users/sbrooker/repos/exam-bank-pipeline/output/triage/iteration_001
```
- `full_ocr_rerun`:

```bash
.venv/bin/python -m exam_bank.cli process --input input --output output/candidates/ocr/latest --enable-ocr
```
- `ocr_verification`:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-status --input output/candidates/ocr/latest/json/question_bank.json
```
- `full_tests`:

```bash
.venv/bin/python -m pytest
```
- `triage_comparison`:

```bash
.venv/bin/python -m exam_bank.cli auto-triage-compare --iteration agent_handoffs/auto_triage/iteration_001 --baseline-triage /Users/sbrooker/repos/exam-bank-pipeline/output/triage/iteration_001 --current output/candidates/ocr/latest/json/question_bank.json --output /Users/sbrooker/repos/exam-bank-pipeline/output/triage/iteration_001/comparisons/comparison.auto-iteration-001.json --test-status pass
```
