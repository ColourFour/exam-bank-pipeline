# Agent 4 Prompt — iteration_002

Read `agent_handoffs/iteration_002/agent1_plan.md`, Agent 2's notes, and Agent 3's test report.

Perform integration review for the difficulty iteration:

- Confirm no OCR or DeepSeek scope creep.
- Confirm exported difficulty fields are present and compatible.
- Inspect score distribution or representative fixture outputs if available.
- Check that difficulty evidence is structured enough to audit.
- Check that weak evidence cannot produce high-confidence difficulty.
- Confirm generated exports, PDFs, reports, or secrets are not tracked.

Run the full suite unless Agent 3 already reported a fresh full-suite pass and no later code changed.

Report:

- pass/fail verdict
- command results
- export/data-contract findings
- blocking issues
- deferrals for iteration 3
