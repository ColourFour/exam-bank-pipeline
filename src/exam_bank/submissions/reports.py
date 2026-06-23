from __future__ import annotations

import csv
from pathlib import Path

from exam_bank.submissions.models import CompletionRow


COMPLETION_FIELDS = [
    "assignment_id",
    "assignment_title",
    "class_id",
    "student_id",
    "display_name",
    "email",
    "status",
    "submitted_at",
    "late",
    "source_filename",
    "stored_pdf_path",
    "rejection_reasons",
    "notes",
]


def write_completion_csv(rows: list[CompletionRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPLETION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in COMPLETION_FIELDS})
