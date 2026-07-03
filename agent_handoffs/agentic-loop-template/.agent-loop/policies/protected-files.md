# Protected Files / Behaviors

The project is image-first. Protect evidence and trust boundaries before optimizing metrics.

## Always Protected

- Do not alter production secrets or env files.
- Do not delete or mutate source PDFs under `input/pastpapers/9709/`.
- Do not delete canonical generated image trees or question banks as part of a code fix: `output/json/question_bank.json`, `output/pm1/`, `output/pm3/`, `output/stats/`, and `output/mechanics/`.
- Do not delete frozen triage baselines such as `output*/triage/iteration_*/baseline_question_bank.json`.
- Do not silently overwrite durable topic-routing sources: `data/topic_routing/question_bank.topic_routing.v1.json` and `data/topic_routing/question_bank.topic_routing.v1.sha256`.
- Do not alter reviewed decision files under `data/review/` unless the plan is specifically about importing reviewed decisions.
- Do not alter canonical taxonomy files unless the plan is specifically about taxonomy repair and includes validation.
- Do not write real rosters, submissions, grades, or email addresses into git.

## Protected Behaviors

- Question and mark-scheme PNGs remain canonical.
- OCR/native text, normalized text, AI labels, topic routing, mark-event evidence, difficulty labels, and Asterion exports remain advisory unless a reviewed contract explicitly promotes them.
- Validation, mapping, visual-curation, topic-safety, text-trust, Asterion role gates, and student-runtime safety gates must fail closed when evidence is weak.
- Missing images, invalid topics, mapping failures, and validation failures must not be hidden by status downgrades or flag suppression.
- Student-runtime and Asterion export behavior must not be broadened unless the plan is explicitly about role-gated export behavior.
