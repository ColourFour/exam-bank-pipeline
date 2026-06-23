from pathlib import Path

from exam_bank.submissions.ingest import load_assignment
from exam_bank.submissions.validation import validate_pdf


FIXTURES = Path("tests/fixtures/submissions")


def test_validate_simple_pdf() -> None:
    assignment = load_assignment(FIXTURES / "assignment_p3_vectors_hw1.json")

    assert validate_pdf(FIXTURES / "inbox" / "S0001.pdf", assignment) == []


def test_reject_non_pdf_file() -> None:
    assignment = load_assignment(FIXTURES / "assignment_p3_vectors_hw1.json")

    assert "not_pdf" in validate_pdf(FIXTURES / "inbox" / "not_a_pdf.txt", assignment)


def test_reject_missing_and_empty_pdf(tmp_path: Path) -> None:
    assignment = load_assignment(FIXTURES / "assignment_p3_vectors_hw1.json")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    assert validate_pdf(tmp_path / "missing.pdf", assignment) == ["missing_file"]
    assert "empty_file" in validate_pdf(empty, assignment)
