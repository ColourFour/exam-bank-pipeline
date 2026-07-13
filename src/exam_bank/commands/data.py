from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.corpus import (
    DEFAULT_CORPUS_MANIFEST,
    DEFAULT_CORPUS_ROOT,
    hydrate_corpus,
    verify_corpus,
    write_corpus_manifest,
)


def run_hydrate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore checksum-verified corpus files from manifest sources.")
    _add_manifest_and_root(parser)
    parser.add_argument("--repair", action="store_true", help="Replace corrupt files after quarantining them.")
    parser.add_argument("--offline", action="store_true", help="Report required downloads without network access.")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    report = hydrate_corpus(
        args.manifest,
        root=args.root,
        repair=args.repair,
        offline=args.offline,
        timeout=args.timeout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def run_verify(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify local corpus files against the authoritative manifest.")
    _add_manifest_and_root(parser)
    args = parser.parse_args(argv)
    report = verify_corpus(args.manifest, root=args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def run_manifest(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a checksummed corpus manifest from local source PDFs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)
    manifest = write_corpus_manifest(args.output, root=args.root, generated_at=args.generated_at)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "record_count": manifest["record_count"],
                "documents_sha256": manifest["documents_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_manifest_and_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
