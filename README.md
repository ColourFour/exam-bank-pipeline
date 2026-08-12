# CAIE 9709 Exam Bank

An image-first extraction and normalization pipeline for CAIE 9709. Canonical
question and mark-scheme crops are the product; extracted text, OCR, taxonomy,
difficulty, AI output, and downstream projections are evidence layered over
those images.

Homework intake and classroom logistics now live in the sibling
`../homework-ingest` repository. Automated grading and rubric-readiness tooling
live in `../autograder`. This repository does not import either project. It
exports versioned Question JSON through the contracts in `schemas/`; the other
repositories exchange Submission and GradeResult JSON at the same file boundary.
See [`docs/REPOSITORY_SPLIT_AUDIT_2026_08_11.md`](docs/REPOSITORY_SPLIT_AUDIT_2026_08_11.md)
for the migration map.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

exam-bank data verify \
  --manifest manifests/corpora/caie_9709.active_partial.v1.json \
  --root input

exam-bank extract run --input input/pastpapers/9709 --output output
exam-bank extract integrity --input output/json/question_bank.json --artifact-root output
```

The base install contains the deterministic extraction dependencies.
Provider-backed enrichment and review commands use the optional
AI extra:

```bash
pip install -e ".[ai]"
```

The `dev` extra also installs the AI client so the complete test suite remains
available to contributors.

### Configuration

`config.yaml` contains extraction, OCR, image, and output defaults. Core PDF
extraction needs no credentials. For optional provider-backed enrichment or
review, copy the names from `.env.example` into a private shell/environment
tool and set only the provider key you use; `.env` remains ignored.

Paper-level parallelism is opt-in. A bounded worker window writes to isolated
staging roots; the parent sorts canonical records and promotes the complete run
with rollback protection, publishing the final bank JSON last:

```bash
exam-bank extract run --input input/pastpapers/9709 --output output --workers 4
```

Parallel mode currently requires debug mode to be disabled because debug
streams are deliberately kept single-writer. Concurrent publishers to the same
output root are rejected, interrupted runs cannot publish partial results, and
an input scan with no classified question papers fails closed. Use
`--allow-empty` only when an empty canonical bank is intentional.

OCR is also opt-in (`--enable-ocr`). The default `adaptive` strategy runs it
only for low-confidence native extraction and sparse supplemental regions;
set `ocr.strategy: always` in an operational config only when the extra runtime
and OCR coverage are explicitly required.

If the corpus is absent, hydrate it from the checksummed source manifest:

```bash
exam-bank data hydrate \
  --manifest manifests/corpora/caie_9709.active_partial.v1.json \
  --root input
```

Hydration downloads only missing files. Existing corrupt or mismatched files
fail closed unless `--repair` is supplied, in which case they are quarantined
before replacement. Corpus verification also opens each checksum-matching PDF
and requires at least one usable page with text or visible rendered content, so
a byte-for-byte known file cannot pass merely because a manifest captured a
structurally blank download.

The complete `caie_9709.v1.json` ledger remains the restoration and audit
contract for all 932 acquired documents. Normal extraction and CI hydrate the
derived `active_partial` ledger so quarantined inputs cannot re-enter a run.

The local mirror historically inverted `mYY` and `sYY` payload filenames for
2021–2025. Normalize them from first-page publisher evidence before extraction:

```bash
exam-bank data normalize-corpus-sessions --root input --apply
exam-bank data quarantine-invalid \
  --manifest manifests/corpora/caie_9709.v1.json \
  --root input \
  --report manifests/validations/corpus_quarantine_validation.v1.json \
  --active-manifest manifests/corpora/caie_9709.active_partial.v1.json \
  --apply
```

Quarantine is recoverable and does not rewrite the authoritative source
contract. The derived active manifest is valid for extraction, while the
quarantine validation remains `ok:false` until exact replacements are restored.

After an identity or crop regeneration, rebind review evidence before building
student-facing projections:

```bash
exam-bank data rebind-text-gold --write
exam-bank data validate-review-assets
exam-bank data export-questions --check-assets
```

Both operations compare reviewed image bytes with the current canonical bank;
changed evidence remains blocked until it is reviewed again.

## Repository boundary

Tracked source of truth:

- `src/exam_bank/`: exam-bank extraction, normalization, review, and export logic.
- `manifests/corpora/`: checksummed source-corpus contracts.
- `manifests/releases/`: promoted release provenance.
- `data/review/canonical/`: provenance-stamped promoted decisions only.
- `data/topic_routing/`: the small runtime-critical routing sidecar and checksum.
- `exam_bank_taxonomy/`: canonical taxonomy files.
- `tests/` and `tests/fixtures/`: data-independent contracts and compact fixtures.

Ignored local/generated state:

- `input/`: hydrated source PDFs.
- `output/`: generated banks, images, packets, Asterion exports, and run status.
- `reports/`: generated audits and reports.
- `tmp/`: caches and transient crops.
- `data/review/runs/`: review batches, evidence, and intermediate merges.
- legacy private classroom/submission/email roots, retained only to protect
  pre-split local data; their active owner is `../homework-ingest`.

## Validation

```bash
.venv/bin/ruff check
.venv/bin/pytest -q -m "not integration and not rendering and not sample_pipeline"
.venv/bin/pytest -q
```

CI tests Python 3.10 and 3.13, reports branch coverage for the data-independent
suite, and checksum-hydrates the declared corpus before fixture-backed tests so
missing PDFs cannot turn those checks into silent skips.

The current command list is generated in
[`docs/COMMAND_REFERENCE.md`](docs/COMMAND_REFERENCE.md). Product and privacy
boundaries are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md), and future
work belongs only in [`ROADMAP.md`](ROADMAP.md). Dated repository evidence lives
under `docs/history/`.
