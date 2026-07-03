# Project Objective

Maintain and improve the CAIE 9709 exam-bank extraction pipeline as an image-first dataset builder for teacher/reviewer workflows and downstream Asterion study-site projections.

Success means the current `output/json/question_bank.json` and its sidecars point to clear, readable, correctly paired canonical question and mark-scheme PNGs; topic metadata is parsed against the official taxonomy and reviewed decisions; and each JSON question record carries accurate identity, question number, artifact paths, mark-scheme data, subparts, trust/status flags, and provenance.

Current Agent 1 priority order:

1. Produce clear output images.
2. Parse correct topic content.
3. Get the data correct from each question in the JSON file.

The loop should make small, verified improvements without weakening image-first trust boundaries. Native text, OCR text, AI labels, topic routing, mark-event evidence, difficulty labels, and Asterion exports are metadata over canonical images unless a reviewed contract explicitly promotes them.
