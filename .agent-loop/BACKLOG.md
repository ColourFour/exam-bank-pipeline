# Improvement Backlog

The Planner may select from this backlog, but it may also propose a better bounded data-integrity item if repo evidence supports it.

## Candidates

- [ ] Validate PNG-to-question mapping against `question_id`, `paper`, `question_number`, canonical filename, and asset-manifest evidence.
- [ ] Validate question-to-mark-scheme pairing so each mark-scheme crop belongs to the same paper identity and question number as the question record.
- [ ] Detect question images containing unrelated question text, wrong question numbers, or obvious cross-question contamination.
- [ ] Detect crops that include neighboring questions, weak anchors, or multi-question spans using image/text evidence beyond dimensions and whitespace.
- [x] Build a corpus-level suspicious rendered-crop audit for 2008-2021 papers, prioritizing tall PNGs and question or mark-scheme images with multiple top-level anchors. Completed for current output integrity with dimension and whitespace PNG gates.
- [ ] Detect mark schemes assigned to the wrong question or containing neighboring mark-scheme entries.
- [ ] Detect missing question images, missing mark-scheme images, broken image references, and manifest entries whose files are absent.
- [ ] Detect orphan images under canonical asset roots that are not referenced by `question_bank.json` or `asset_manifest.v1.json`.
- [ ] Detect duplicate mappings where multiple records point to the same canonical question or mark-scheme image unexpectedly.
- [ ] Add deterministic checks for newly ingested datasets, especially modern/legacy consistency around `PaperIdentity`.
- [x] Generate small review packs for suspicious output samples, including the relevant question image, mark-scheme image, record metadata, and reason flags. Extended suspicious crop packs to cover rendered dimension and whitespace artifact candidates.
- [ ] Tighten manifest/schema validation for asset records, canonical paths, image-kind fields, and run-manifest provenance.
- [ ] Remove stale/generated artifacts from the repo only when policy says they belong in ignored output roots or an external warehouse, and only with a manifest/diff.
- [ ] Improve a focused test or validation check that proves an output-integrity bug cannot recur.
- [ ] Add OCR-assisted or vision-assisted review for crops whose rendered image may contain a foreign top-level question number despite passing dimension and whitespace gates.

## Do not select

- Large rewrites.
- Framework swaps.
- New dependencies unless there is a strong reason.
- Cosmetic churn.
- Broad reports where a small review pack, validation check, or deterministic gate would answer the risk.
- Feature expansion that does not first protect output correctness.
- Cleanup that deletes raw PDFs, canonical inputs, manifests, baselines, or generated datasets without an explicit manifest/diff and approval.
- Audit-only work unless it creates a reusable validation check, a focused suspicious-output review pack, or concrete repair evidence.
