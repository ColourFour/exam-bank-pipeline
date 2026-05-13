#!/usr/bin/env bash
set -euo pipefail

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is not set}"

.venv/bin/python -m exam_bank.cli enrich-ai \
  --input output/json/question_bank.json \
  --taxonomy exam_bank_taxonomy/canonical \
  --existing-sidecar output/json/question_bank.deepseek.json \
  --output output/json/question_bank.ai_assisted.v2.full.json \
  --status-dir output/run_status \
  --model deepseek-v4-flash \
  --include-subparts \
  --recompute-difficulty

.venv/bin/python -m exam_bank.cli ai-sidecar-audit \
  --input output/json/question_bank.ai_assisted.v2.full.json
