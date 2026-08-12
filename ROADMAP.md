# Roadmap

This is the only active roadmap. Completed work and dated measurements belong
under `docs/history/`.

## Completed foundation

- Checksummed hybrid corpus boundary with verify, hydrate, offline, mirror, and
  repair behavior.
- Lean tracked tree: source PDFs, caches, generated output/reports, and review
  run evidence are ignored without rewriting Git history.
- Lazy namespaced `exam-bank` command surface and generated command reference.
- Provenance-validated canonical review promotion.
- Canonical mark-scheme repair and a full integrity gate with no missing answer
  paths or cross-question text contamination.
- Resume cache identity includes source, configuration, pipeline, and OCR state.
- Opt-in isolated paper-worker staging with deterministic record, image, and
  diagnostic equivalence against one-worker extraction.
- Repository-policy tests, Ruff correctness/import checks, and split CI jobs.
- Homework intake/classroom and automated grading extracted into sibling
  repositories behind versioned JSON/file contracts.

## Next priorities

1. Finish moving the remaining substantial historical script implementations
   into domain modules, then remove the residual `scripts/` compatibility layer.
2. Complete physical decomposition of the largest extraction, rendering, and
   packet modules while preserving characterization outputs and schema v2.
3. Expand compact generated-PDF fixtures for every newly discovered legacy
   segmentation mode.
4. Continue reviewed promotion for P1, P3, M1, and S1 without weakening image,
   privacy, or student-runtime gates.
5. Treat Question schema evolution as an explicit versioned release; keep
   sibling schema copies byte-identical and verify their SHA-256 values.
