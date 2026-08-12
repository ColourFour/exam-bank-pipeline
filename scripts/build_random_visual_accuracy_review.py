from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import escape
import json
import math
from pathlib import Path
import random
import re
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible random visual/text accuracy review packet."
    )
    parser.add_argument("--question-bank", default="output/json/question_bank.json")
    parser.add_argument("--artifact-root", default="output")
    parser.add_argument("--ocr-audit", default="reports/png_ocr_audit_20260620.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--reuse-sample",
        default="",
        help="Reuse the question IDs from an earlier sample JSON for an exact before/after cohort.",
    )
    args = parser.parse_args()

    bank_path = Path(args.question_bank)
    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    records = [row for row in payload.get("questions", []) if isinstance(row, dict)]
    records = sorted(records, key=lambda row: str(row.get("question_id") or ""))
    sample_size = math.ceil(len(records) * args.fraction)
    if args.reuse_sample:
        previous = json.loads(Path(args.reuse_sample).read_text(encoding="utf-8"))
        selected_ids = [
            str(row.get("question_id") or "")
            for row in previous.get("questions", [])
            if isinstance(row, dict)
        ]
        record_by_id = {str(row.get("question_id") or ""): row for row in records}
        missing_ids = [question_id for question_id in selected_ids if question_id not in record_by_id]
        if missing_ids:
            raise SystemExit(f"reused sample IDs missing from question bank: {missing_ids}")
        sampled = [record_by_id[question_id] for question_id in selected_ids]
        sample_size = len(sampled)
    else:
        sampled = random.Random(args.seed).sample(records, sample_size)
        sampled = sorted(sampled, key=lambda row: str(row.get("question_id") or ""))

    ocr_rows = load_ocr_rows(Path(args.ocr_audit))
    review_rows = [
        build_review_row(row, artifact_root=artifact_root, output_dir=output_dir, ocr_rows=ocr_rows)
        for row in sampled
    ]

    sample_payload = {
        "schema_name": "exam_bank.random_visual_accuracy_sample",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_question_bank": str(bank_path),
        "artifact_root": str(artifact_root),
        "sampling": {
            "method": "reused_seeded_sample" if args.reuse_sample else "simple_random_without_replacement",
            "seed": args.seed,
            "fraction": args.fraction,
            "population_count": len(records),
            "sample_count": len(review_rows),
            "rounding": "ceil",
            "reused_sample": args.reuse_sample or None,
        },
        "sample_profile": {
            "paper_family_counts": dict(sorted(Counter(str(row.get("paper_family") or "unknown") for row in review_rows).items())),
            "question_crop_confidence_counts": dict(sorted(Counter(str(row.get("question_crop_confidence") or "unknown") for row in review_rows).items())),
            "mark_scheme_crop_confidence_counts": dict(sorted(Counter(str(row.get("mark_scheme_crop_confidence") or "unknown") for row in review_rows).items())),
        },
        "rating_contract": rating_contract(),
        "questions": review_rows,
    }
    write_json(output_dir / "sample.json", sample_payload)
    write_json(output_dir / "review_decisions.template.json", decision_template(sample_payload))
    batch_size = 3
    batch_files: list[str] = []
    for start in range(0, len(review_rows), batch_size):
        batch_number = start // batch_size + 1
        batch_name = f"batch_{batch_number:03d}.html"
        batch_files.append(batch_name)
        batch_payload = {**sample_payload, "questions": review_rows[start : start + batch_size]}
        (output_dir / batch_name).write_text(
            render_html(batch_payload, display_start=start + 1, batch_number=batch_number, batch_count=math.ceil(len(review_rows) / batch_size)),
            encoding="utf-8",
        )
    (output_dir / "index.html").write_text(render_index(sample_payload, batch_files), encoding="utf-8")
    contact_batch_size = 9
    contact_count = math.ceil(len(review_rows) / contact_batch_size)
    for start in range(0, len(review_rows), contact_batch_size):
        contact_number = start // contact_batch_size + 1
        contact_payload = {**sample_payload, "questions": review_rows[start : start + contact_batch_size]}
        (output_dir / f"contact_{contact_number:03d}.html").write_text(
            render_contact_html(
                contact_payload,
                display_start=start + 1,
                batch_number=contact_number,
                batch_count=contact_count,
            ),
            encoding="utf-8",
        )
    print(json.dumps(sample_payload["sampling"], indent=2))
    print(json.dumps(sample_payload["sample_profile"], indent=2))
    return 0


def load_ocr_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row.get("path") or "")] = row
    return rows


def build_review_row(
    record: dict[str, Any],
    *,
    artifact_root: Path,
    output_dir: Path,
    ocr_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    q_path = str(record.get("question_image_path") or record.get("canonical_question_artifact") or "")
    ms_path = str(record.get("mark_scheme_image_path") or record.get("canonical_mark_scheme_artifact") or "")
    q_abs = artifact_root / q_path if q_path else Path("")
    ms_abs = artifact_root / ms_path if ms_path else Path("")
    q_ocr = ocr_rows.get(q_path, {})
    ms_ocr = ocr_rows.get(ms_path, {})
    q_text = str(record.get("question_text") or "")
    ms_text = str(record.get("mark_scheme_text") or "")
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    return {
        "question_id": str(record.get("question_id") or ""),
        "paper": str(record.get("paper") or ""),
        "paper_family": str(record.get("paper_family") or ""),
        "question_number": str(record.get("question_number") or ""),
        "question_image_path": q_path,
        "mark_scheme_image_path": ms_path,
        "question_image_exists": bool(q_path and q_abs.is_file()),
        "mark_scheme_image_exists": bool(ms_path and ms_abs.is_file()),
        "question_image_src": relative_asset_src(output_dir, q_abs) if q_path else "",
        "mark_scheme_image_src": relative_asset_src(output_dir, ms_abs) if ms_path else "",
        "question_text": q_text,
        "mark_scheme_text": ms_text,
        "question_text_trust": record.get("question_text_trust"),
        "text_only_status": record.get("text_only_status"),
        "visual_curation_status": record.get("visual_curation_status"),
        "question_crop_confidence": notes.get("question_crop_confidence"),
        "mark_scheme_crop_confidence": notes.get("mark_scheme_crop_confidence"),
        "mapping_status": notes.get("mapping_status"),
        "validation_status": notes.get("validation_status"),
        "review_flags": notes.get("review_flags") if isinstance(notes.get("review_flags"), list) else [],
        "question_ocr_preview": str(q_ocr.get("ocr_preview") or ""),
        "mark_scheme_ocr_preview": str(ms_ocr.get("ocr_preview") or ""),
        "question_ocr_similarity": similarity(q_text[:500], str(q_ocr.get("ocr_preview") or "")),
        "mark_scheme_ocr_similarity": similarity(ms_text[:500], str(ms_ocr.get("ocr_preview") or "")),
        "question_ocr_irregularities": q_ocr.get("irregularities") if isinstance(q_ocr.get("irregularities"), list) else [],
        "mark_scheme_ocr_irregularities": ms_ocr.get("irregularities") if isinstance(ms_ocr.get("irregularities"), list) else [],
    }


def similarity(left: str, right: str) -> float | None:
    left_tokens = WORD_RE.findall(left.lower())
    right_tokens = WORD_RE.findall(right.lower())
    if not left_tokens or not right_tokens:
        return None
    return round(SequenceMatcher(None, left_tokens, right_tokens).ratio(), 4)


def relative_asset_src(output_dir: Path, asset: Path) -> str:
    import os

    return os.path.relpath(asset.resolve(), output_dir.resolve()).replace(os.sep, "/")


def rating_contract() -> dict[str, Any]:
    return {
        "image_ratings": {
            "pass": "Correct question, complete scope, readable, with no material neighboring content.",
            "minor": "Correct and usable, but includes harmless extra furniture/whitespace or a small non-material edge issue.",
            "major": "Wrong/mismatched, materially truncated/contaminated, or unreadable.",
            "missing": "Referenced image is absent.",
        },
        "text_ratings": {
            "pass": "Semantically complete and mathematically faithful; harmless typography/spacing differences allowed.",
            "minor": "Usable but has a localized non-fatal transcription or layout defect.",
            "major": "A material omission, wrong symbol/value/part, severe flattening, or unrelated contamination changes meaning/use.",
            "missing": "JSON text is empty when source text is visible.",
        },
        "overall_usable": "Both images are pass/minor, paired correctly, and both JSON texts are pass/minor.",
        "strict_accurate": "All four component ratings are pass.",
    }


def decision_template(sample_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "exam_bank.random_visual_accuracy_decisions",
        "schema_version": 1,
        "source_sample": "sample.json",
        "reviewer": "",
        "reviewed_at": "",
        "questions": [
            {
                "question_id": row["question_id"],
                "question_image_rating": "pending",
                "mark_scheme_image_rating": "pending",
                "question_text_rating": "pending",
                "mark_scheme_text_rating": "pending",
                "pairing_correct": None,
                "notes": "",
            }
            for row in sample_payload["questions"]
        ],
    }


def render_html(
    sample_payload: dict[str, Any],
    *,
    display_start: int = 1,
    batch_number: int = 1,
    batch_count: int = 1,
) -> str:
    cards = "\n".join(
        render_card(display_start + offset, row)
        for offset, row in enumerate(sample_payload["questions"])
    )
    sampling = sample_payload["sampling"]
    previous_link = f'<a href="batch_{batch_number - 1:03d}.html">Previous</a>' if batch_number > 1 else ""
    next_link = f'<a href="batch_{batch_number + 1:03d}.html">Next</a>' if batch_number < batch_count else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Random visual accuracy review</title>
<style>
body{{margin:0;background:#eef1f5;color:#172033;font:14px/1.4 system-ui,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#172033;color:white;padding:12px 18px}}header a{{color:#dce8ff;margin-left:16px}}main{{max-width:1800px;margin:auto;padding:16px}}article{{background:white;border:1px solid #cbd3df;border-radius:10px;margin:0 0 18px;overflow:hidden;break-inside:avoid}}h2{{margin:0;padding:10px 14px;background:#f7f8fa;font-size:17px}}.meta{{padding:8px 14px;color:#526079}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 12px 12px}}.panel{{border:1px solid #d5dbe5;border-radius:8px;overflow:hidden;min-width:0}}h3{{margin:0;padding:8px 10px;background:#eef2f7;font-size:14px}}img{{display:block;width:100%;height:auto;background:#fafafa}}pre{{white-space:pre-wrap;word-break:break-word;margin:0;padding:10px;max-height:320px;overflow:auto;font:13px/1.35 ui-monospace,monospace}}.diag{{padding:7px 10px;background:#fff8df;color:#5d4b00}}code{{font-family:ui-monospace,monospace}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><strong>Random visual accuracy review</strong> &nbsp; Batch {batch_number}/{batch_count} · Population {sampling['population_count']} · Sample {sampling['sample_count']} · Seed {sampling['seed']} <a href="index.html">Index</a> {previous_link} {next_link}</header>
<main>{cards}</main></body></html>"""


def render_index(sample_payload: dict[str, Any], batch_files: list[str]) -> str:
    sampling = sample_payload["sampling"]
    links = "\n".join(
        f'<li><a href="{escape(name)}">Batch {index:03d}</a></li>'
        for index, name in enumerate(batch_files, start=1)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Random visual accuracy review</title>
<style>body{{max-width:900px;margin:40px auto;font:16px/1.5 system-ui,sans-serif;color:#172033}}ul{{columns:3}}a{{color:#154fb3}}</style></head><body>
<h1>Random visual accuracy review</h1>
<p>Population {sampling['population_count']} · Sample {sampling['sample_count']} · Seed {sampling['seed']} · Method {escape(sampling['method'])}</p>
<ul>{links}</ul></body></html>"""


def render_card(index: int, row: dict[str, Any]) -> str:
    q_img = f'<img loading="lazy" src="{escape(row["question_image_src"])}">' if row["question_image_exists"] else "<p>Missing image</p>"
    ms_img = f'<img loading="lazy" src="{escape(row["mark_scheme_image_src"])}">' if row["mark_scheme_image_exists"] else "<p>Missing image</p>"
    return f"""<article id="q-{index}">
<h2>{index:03d}. {escape(row['question_id'])}</h2>
<div class="meta">family <code>{escape(row['paper_family'])}</code> · Q crop <code>{escape(str(row['question_crop_confidence']))}</code> · MS crop <code>{escape(str(row['mark_scheme_crop_confidence']))}</code> · Q OCR similarity <code>{escape(str(row['question_ocr_similarity']))}</code> · MS OCR similarity <code>{escape(str(row['mark_scheme_ocr_similarity']))}</code></div>
<div class="grid">
<section class="panel"><h3>1) question.png</h3>{q_img}</section>
<section class="panel"><h3>2) markscheme.png</h3>{ms_img}</section>
<section class="panel"><h3>3a) JSON question_text</h3><pre>{escape(row['question_text'])}</pre><div class="diag">OCR preview: {escape(row['question_ocr_preview'])}</div></section>
<section class="panel"><h3>3b) JSON mark_scheme_text</h3><pre>{escape(row['mark_scheme_text'])}</pre><div class="diag">OCR preview: {escape(row['mark_scheme_ocr_preview'])}</div></section>
</div></article>"""


def render_contact_html(
    sample_payload: dict[str, Any],
    *,
    display_start: int,
    batch_number: int,
    batch_count: int,
) -> str:
    sampling = sample_payload["sampling"]
    cards = "\n".join(
        f'''<article><h2>{display_start + offset:03d}. {escape(row["question_id"])}</h2>
<div class="meta">Q {escape(str(row["question_ocr_similarity"]))} · MS {escape(str(row["mark_scheme_ocr_similarity"]))}</div>
<div class="pair"><figure><figcaption>question</figcaption><img loading="lazy" src="{escape(row["question_image_src"])}"></figure>
<figure><figcaption>markscheme</figcaption><img loading="lazy" src="{escape(row["mark_scheme_image_src"])}"></figure></div></article>'''
        for offset, row in enumerate(sample_payload["questions"])
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Compact visual review</title>
<style>body{{margin:0;background:#e9edf3;color:#172033;font:12px/1.2 system-ui,sans-serif}}header{{background:#172033;color:white;padding:8px 12px}}main{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:8px}}article{{background:white;border:1px solid #bfc8d6;border-radius:6px;overflow:hidden}}h2{{font-size:12px;margin:0;padding:5px 7px;background:#f5f7fa}}.meta{{padding:3px 7px;color:#56637a}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:3px;padding:3px}}figure{{margin:0;min-width:0}}figcaption{{font-weight:700;text-align:center}}img{{display:block;width:100%;height:180px;object-fit:contain;object-position:top;background:#fafafa}}</style></head>
<body><header>Compact visual review · {batch_number}/{batch_count} · Population {sampling['population_count']} · Sample {sampling['sample_count']} · Seed {sampling['seed']}</header><main>{cards}</main></body></html>"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
