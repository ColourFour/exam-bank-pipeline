from __future__ import annotations

import argparse
import json
from pathlib import Path

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import DEFAULT_REVIEW_BATCH_DIR
from exam_bank.p3_exact_skill.review_batch import validate_batch_0002_artifacts
from exam_bank.p3_exact_skill.review_batch import validate_batch_0003_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate P3 exact-skill review batch artifacts.")
    parser.add_argument("--batch-id", default="batch_0002")
    parser.add_argument("--batch-dir", type=Path, default=Path(DEFAULT_REVIEW_BATCH_DIR))
    parser.add_argument("--output", type=Path, default=Path("reports/p3_exact_skill_batch_validation.v1.json"))
    args = parser.parse_args(argv)

    manifest_path = args.batch_dir / f"{args.batch_id}_manifest.v1.json"
    template_path = args.batch_dir / f"{args.batch_id}_decision_template.v1.json"
    manifest = _load_json(manifest_path)
    template = _load_json(template_path)
    if args.batch_id == "batch_0002":
        errors = validate_batch_0002_artifacts(manifest, template)
    elif args.batch_id == "batch_0003":
        errors = validate_batch_0003_artifacts(manifest, template)
    else:
        errors = []
    payload = {
        "schema": "exam_bank.p3_exact_skill.review_batch.validation",
        "schema_version": 1,
        "batch_id": args.batch_id,
        "manifest_path": str(manifest_path),
        "template_path": str(template_path),
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "selected_count": manifest.get("selected_count"),
        "record_count": template.get("record_count"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_json(payload, args.output, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
