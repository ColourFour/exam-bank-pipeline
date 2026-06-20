from __future__ import annotations

import json
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_png_ocr.py"
SPEC = importlib.util.spec_from_file_location("audit_png_ocr", AUDIT_SCRIPT)
assert SPEC and SPEC.loader
audit_png_ocr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_png_ocr)
_prune_unreferenced_pngs = audit_png_ocr._prune_unreferenced_pngs


def test_prune_unreferenced_pngs_deletes_only_unreferenced_canonical_assets(tmp_path: Path) -> None:
    output = tmp_path / "output"
    referenced = output / "pm1" / "pm1_2024_m24_12_qp_q01_question.png"
    unreferenced = output / "pm1" / "pm1_2024_m24_12_ms_q02_markscheme.png"
    referenced.parent.mkdir(parents=True)
    referenced.write_bytes(b"referenced")
    unreferenced.write_bytes(b"stale")

    manifest = tmp_path / "reports" / "pruned.json"
    report = _prune_unreferenced_pngs(
        [referenced, unreferenced],
        output,
        {
            "pm1/pm1_2024_m24_12_qp_q01_question.png": {
                "artifact_type": "question",
                "question_id": "12march24_q01",
            }
        },
        question_bank_path=tmp_path / "question_bank.json",
        manifest_path=manifest,
    )

    assert referenced.exists()
    assert not unreferenced.exists()
    assert report["deleted_unreferenced_png_count"] == 1
    assert report["deleted_unreferenced_pngs"][0]["path"] == "pm1/pm1_2024_m24_12_ms_q02_markscheme.png"
    assert json.loads(manifest.read_text(encoding="utf-8"))["deleted_unreferenced_png_count"] == 1
