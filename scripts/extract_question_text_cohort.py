from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from exam_bank.config import AppConfig
from exam_bank.extraction_structure import build_structured_question_text
from exam_bank.pdf_extract import extract_pdf_layout
from exam_bank.question_detection import detect_question_spans

PAPER_RE = re.compile(
    r"^(?P<variant>\d{2})(?P<session>spring|summer|winter)(?P<year>\d{2})$"
)
SESSION_CODE = {"spring": "m", "summer": "s", "winter": "w"}
IMAGE_SOURCE_RE = re.compile(
    r"_(?P<year>\d{4})_(?P<session>[msw]\d{2})_(?P<variant>\d{1,2})_qp_"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract native PDF question text for a fixed visual-audit cohort."
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--input-root", default="input/pastpapers/9709")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.workers != 1:
        parser.error(
            "--workers must be 1 because native PDF layout extraction is not thread-safe"
        )

    cohort_path = Path(args.cohort)
    input_root = Path(args.input_root)
    output_path = Path(args.output)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    rows = cohort.get("questions")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        parser.error("cohort must contain a questions list of objects")

    source_questions: dict[str, list[str]] = {}
    source_by_question_id: dict[str, str] = {}
    for index, row in enumerate(rows):
        paper = row.get("paper")
        question_number = row.get("question_number")
        question_id = row.get("question_id")
        if not all(isinstance(value, str) and value.strip() for value in (paper, question_number, question_id)):
            parser.error(f"cohort question {index} is missing paper, question_number, or question_id")
        source_pdf = resolve_question_pdf_for_row(input_root, row)
        source_key = str(source_pdf)
        source_questions.setdefault(source_key, []).append(question_number)
        source_by_question_id[question_id] = source_key

    jobs = [
        (
            source_key,
            source_key,
            sorted(set(question_numbers), key=_question_number_sort_key),
        )
        for source_key, question_numbers in sorted(source_questions.items())
    ]
    extracted = [_extract_paper_question_text(job) for job in jobs]
    text_by_source = {source_key: texts for source_key, texts in extracted}

    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        paper = str(row["paper"])
        question_number = str(row["question_number"])
        source_key = source_by_question_id[str(row["question_id"])]
        candidate_rows.append(
            {
                "question_id": str(row["question_id"]),
                "paper": paper,
                "paper_family": str(row.get("paper_family") or ""),
                "question_number": question_number,
                "question_text": text_by_source.get(source_key, {}).get(question_number, ""),
            }
        )

    payload = {
        "schema_name": "exam_bank.native_question_text_cohort",
        "schema_version": 1,
        "source_cohort": str(cohort_path),
        "source_root": str(input_root),
        "questions": candidate_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = [row["question_id"] for row in candidate_rows if not row["question_text"].strip()]
    print(
        json.dumps(
            {
                "question_count": len(candidate_rows),
                "paper_count": len(jobs),
                "missing_question_text_count": len(missing),
                "missing_question_ids": missing,
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 1 if missing else 0


def resolve_question_pdf(input_root: Path, paper: str) -> Path:
    match = PAPER_RE.fullmatch(paper)
    if match is None:
        raise ValueError(f"Unsupported canonical paper identifier: {paper!r}")
    year_suffix = match.group("year")
    year = str(2000 + int(year_suffix))
    session = SESSION_CODE[match.group("session")]
    variant = match.group("variant")
    folder = input_root / year / "question_papers"
    names = [
        f"9709_{session}{year_suffix}_qp_{variant}.pdf",
        f"9709_{session}{year_suffix}_qp_{int(variant)}.pdf",
    ]
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Question PDF not found for {paper!r}; tried {[str(folder / name) for name in names]}")


def resolve_question_pdf_for_row(input_root: Path, row: dict[str, Any]) -> Path:
    image_path = str(row.get("question_image_path") or "")
    image_match = IMAGE_SOURCE_RE.search(Path(image_path).name)
    if image_match is not None:
        year = image_match.group("year")
        session = image_match.group("session")
        variant = image_match.group("variant")
        folder = input_root / year / "question_papers"
        names = [
            f"9709_{session}_qp_{variant}.pdf",
            f"9709_{session}_qp_{int(variant)}.pdf",
        ]
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return resolve_question_pdf(input_root, str(row.get("paper") or ""))


def _extract_paper_question_text(
    job: tuple[str, str, list[str]],
) -> tuple[str, dict[str, str]]:
    source_key, pdf_path_text, requested_numbers = job
    pdf_path = Path(pdf_path_text)
    config = AppConfig()
    config.ocr.enabled = False
    layouts = extract_pdf_layout(pdf_path, config, use_ocr=False)
    spans = detect_question_spans(layouts, pdf_path, config)
    span_by_number = {span.question_number: span for span in spans}
    texts = {
        question_number: build_structured_question_text(
            span_by_number[question_number],
            layouts,
            config,
        ).combined_question_text
        for question_number in requested_numbers
        if question_number in span_by_number
    }
    return source_key, texts


def _question_number_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


if __name__ == "__main__":
    raise SystemExit(main())
