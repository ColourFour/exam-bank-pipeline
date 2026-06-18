# Protected Files / Behaviors

The loop protects source inputs, canonical identity, strict filtering, and output trust signals. Any iteration that touches these areas must explain the risk and verification path before implementation.

## Protected files and data

- Do not delete source PDFs, canonical raw inputs, input manifests, or repository fixtures.
- Do not overwrite generated datasets, canonical image trees, asset manifests, run manifests, review baselines, or audit outputs without a manifest/diff that makes the change reviewable.
- Do not silently relabel question mappings, paper identities, question numbers, component/session/year metadata, or mark-scheme pairings.
- Do not weaken strict filters, role gates, readiness gates, validation statuses, mapping statuses, review flags, or `student_runtime_safe` rules.
- Do not treat OCR text, native PDF text, AI enrichment, topic routing, or advisory sidecars as canonical image truth unless the schema and project contract explicitly allow it.
- Do not mix question content and mark-scheme content in one asset or field unless the schema explicitly expects that combined representation.
- Do not mark ambiguous outputs as clean, reviewed, safe, valid, or passing without deterministic evidence or sampled visual review.
- Do not delete frozen triage baselines or historical comparison artifacts unless the plan is specifically an approved cleanup with a manifest.
- Do not add new dependencies, large generated reports, broad dashboards, or warehouse-like artifacts when a small validation check or review pack is enough.

## Protected behaviors

- Preserve `PaperIdentity` as the source for canonical paper IDs, question IDs, mark-scheme pairing, and asset paths.
- Keep canonical question and mark-scheme PNGs as source-of-truth evidence for consumers.
- Prefer quarantining, diffing, or writing candidate outputs over replacing canonical outputs directly.
- If a mapping is uncertain, surface it as `review` or suspicious output; do not coerce it into a clean pass.
- If cleanup is needed, distinguish raw/canonical inputs from generated output roots and require exact paths plus a deletion or quarantine manifest.
