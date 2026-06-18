from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

ADVISORY_ROOT = Path("output/advisory_evidence")
INVENTORY_DIR = ADVISORY_ROOT / "inventory"
EXTRACTED_TEXT_DIR = ADVISORY_ROOT / "extracted_text"
PARSED_DIR = ADVISORY_ROOT / "parsed"
LINKING_DIR = ADVISORY_ROOT / "linking"
PREDICTIONS_DIR = ADVISORY_ROOT / "predictions"
REPORTS_DIR = ADVISORY_ROOT / "reports"
FINAL_SIDECAR_PATH = ADVISORY_ROOT / "question_bank.advisory_evidence.v1.json"

EXAMINER_REPORT_INVENTORY = INVENTORY_DIR / "examiner_report_inventory.json"
GRADE_THRESHOLD_INVENTORY = INVENTORY_DIR / "grade_threshold_inventory.json"

EXAMINER_TEXT_DIR = EXTRACTED_TEXT_DIR / "examiner_reports"
GRADE_THRESHOLD_TEXT_DIR = EXTRACTED_TEXT_DIR / "grade_thresholds"
EXAMINER_PARSED_DIR = PARSED_DIR / "examiner_reports"
GRADE_THRESHOLD_PARSED_DIR = PARSED_DIR / "grade_thresholds"

EXAMINER_LINKS_PATH = LINKING_DIR / "examiner_report_question_links.json"
GRADE_THRESHOLD_LINKS_PATH = LINKING_DIR / "grade_threshold_component_links.json"
TOPIC_EVIDENCE_PATH = PREDICTIONS_DIR / "advisory_topic_evidence.v1.json"
EXAMINER_DIFFICULTY_PATH = PREDICTIONS_DIR / "advisory_examiner_report_difficulty.v1.json"
GRADE_THRESHOLD_CONTEXT_PATH = PREDICTIONS_DIR / "advisory_grade_threshold_context.v1.json"

INVENTORY_SCHEMA = "exam_bank.advisory_evidence.inventory.v1"
EXTRACTED_TEXT_SCHEMA = "exam_bank.advisory_evidence.extracted_text.v1"
EXAMINER_PARSED_SCHEMA = "exam_bank.advisory_evidence.examiner_report_parsed.v1"
GRADE_THRESHOLD_PARSED_SCHEMA = "exam_bank.advisory_evidence.grade_thresholds_parsed.v1"
EXAMINER_LINKS_SCHEMA = "exam_bank.advisory_evidence.examiner_report_links.v1"
GRADE_THRESHOLD_LINKS_SCHEMA = "exam_bank.advisory_evidence.grade_threshold_links.v1"
TOPIC_EVIDENCE_SCHEMA = "exam_bank.advisory_evidence.topic_evidence.v1"
EXAMINER_DIFFICULTY_SCHEMA = "exam_bank.advisory_evidence.examiner_report_difficulty.v1"
GRADE_THRESHOLD_CONTEXT_SCHEMA = "exam_bank.advisory_evidence.grade_threshold_context.v1"
FINAL_SIDECAR_SCHEMA = "exam_bank.question_bank.advisory_evidence.v1"
VALIDATION_SCHEMA = "exam_bank.advisory_evidence.validation.v1"

LINK_STATUSES = {"linked", "ambiguous", "unlinked", "not_applicable"}
CONFIDENCE_LABELS = {"high", "medium", "low", "unknown"}
EVIDENCE_LEVELS = {"normal", "low", "none"}
ITEM_SIGNALS = {"easy", "moderate", "hard", "mixed", "unknown"}
CONTEXT_LABELS = {
    "paper_context_harder_than_session_peers",
    "paper_context_typical",
    "paper_context_easier_than_session_peers",
    "paper_context_unknown",
}

PROTECTED_PATHS = [
    Path("output/json/question_bank.json"),
    Path("output/json/question_bank.topic_routing.v1.json"),
    Path("output/asterion/exports/latest"),
    Path("output/pm1"),
    Path("output/pm3"),
    Path("output/stats"),
    Path("output/mechanics"),
    Path("exam_bank_taxonomy"),
]
