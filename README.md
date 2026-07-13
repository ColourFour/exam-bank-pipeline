# CAIE 9709 Exam Bank

An image-first pipeline and local teaching platform for CAIE 9709. Canonical
question and mark-scheme crops are the product; extracted text, OCR, taxonomy,
difficulty, AI output, and downstream projections are evidence layered over
those images.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

exam-bank data verify \
  --manifest manifests/corpora/caie_9709.v1.json \
  --root input

exam-bank extract run --input input/pastpapers/9709 --output output
exam-bank extract integrity --input output/json/question_bank.json --artifact-root output
```

Paper-level parallelism is opt-in. Workers write to isolated staging roots;
the parent promotes artifacts, sorts canonical records, and atomically writes
the final bank:

```bash
exam-bank extract run --input input/pastpapers/9709 --output output --workers 4
```

Parallel mode currently requires debug mode to be disabled because debug
streams are deliberately kept single-writer.

If the corpus is absent, hydrate it from the checksummed source manifest:

```bash
exam-bank data hydrate \
  --manifest manifests/corpora/caie_9709.v1.json \
  --root input
```

Hydration downloads only missing files. Existing corrupt or mismatched files
fail closed unless `--repair` is supplied, in which case they are quarantined
before replacement.

## Repository boundary

Tracked source of truth:

- `src/exam_bank/`: product logic and lazy command registration.
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
- private classroom, submission, and email data roots.

## Validation

```bash
.venv/bin/ruff check
.venv/bin/pytest -q -m "not integration and not rendering and not sample_pipeline"
.venv/bin/pytest -q
```

The current command list is generated in
[`docs/COMMAND_REFERENCE.md`](docs/COMMAND_REFERENCE.md). Product and privacy
boundaries are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md), and future
work belongs only in [`ROADMAP.md`](ROADMAP.md). Dated repository evidence lives
under `docs/history/`.
