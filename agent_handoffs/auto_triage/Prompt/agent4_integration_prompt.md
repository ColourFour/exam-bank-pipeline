# Agent 4 Integration Prompt

Verify production-style evidence after tests pass.

Required:
- Run or verify a full OCR-enabled export.
- Verify the current output is OCR-enabled with `auto-triage-status`.
- Run the auto-triage comparison against the frozen OCR-enabled baseline.
- Check `metrics_after.json`, `decision.json`, and the comparison report.
- Inspect `worsened_records` and any status regressions.

Constraints:
- OCR/no-OCR comparisons must not be mixed for production scoring.
- Image crops remain the source of truth.
- Extracted text and topic labels remain metadata.
- Do not replace canonical output until comparison evidence is understood.
