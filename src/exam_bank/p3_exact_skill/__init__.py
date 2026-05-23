from __future__ import annotations

P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA = "exam_bank.p3_exact_skill.reviewed_decisions"
P3_EXACT_SKILL_REVIEWED_DECISIONS_VALIDATION_SCHEMA = (
    "exam_bank.p3_exact_skill.reviewed_decisions.validation"
)
P3_EXACT_SKILL_REVIEWED_DECISIONS_SCHEMA_VERSION = 1

DEFAULT_REVIEWED_DECISIONS_PATH = "data/review/p3_exact_skill_reviewed_decisions.v1.json"
DEFAULT_P3_SKILL_MAP_PATH = "exam_bank_taxonomy/canonical/skill_maps/skill_map_9709_p3_v1.json"
DEFAULT_P3_SKILL_MAPPINGS_PATH = (
    "exam_bank_taxonomy/canonical/question_skill_mappings/question_skill_mappings_9709_p3_v1.json"
)
DEFAULT_P3_TOPIC_ASSIGNMENTS_PATH = (
    "exam_bank_taxonomy/canonical/question_topic_assignments/question_topic_assignments_9709_p3_v1.json"
)
DEFAULT_REVIEW_QUEUE_JSON_PATH = "reports/p3_exact_skill_review_queue.v1.json"
DEFAULT_REVIEW_QUEUE_REPORT_PATH = "reports/p3_exact_skill_review_queue.md"
DEFAULT_REVIEW_BATCH_DIR = "data/review/p3_exact_skill_batches"

ROUTE_STATUSES = {
    "clean",
    "thin",
    "ambiguous",
    "blocked",
    "deferred",
    "review_needed",
    "fallback_only",
}

ALLOWED_USE_CASE_KEYS = {
    "mastery",
    "guardian",
    "export",
    "source_backed_examples",
    "candidate_generation",
}

REVIEW_QUEUE_SCHEMA = "exam_bank.p3_exact_skill.review_queue"
REVIEW_BATCH_MANIFEST_SCHEMA = "exam_bank.p3_exact_skill.review_batch_manifest"
REVIEW_BATCH_TEMPLATE_SCHEMA = "exam_bank.p3_exact_skill.review_batch_template"
