from __future__ import annotations

from exam_bank.config import AppConfig
from exam_bank.topic_confidence_rescoring import (
    build_topic_confidence_rescoring_report,
    rescore_topic_confidence_record,
)


def _record(
    *,
    question_id: str = "12summer21_q01",
    topic: str = "binomial_expansion",
    question_text: str = "Find the coefficient of x^2 in the expansion of (1 + 2x)^6. [3]",
    mark_scheme_text: str = "Use the binomial expansion and collect the coefficient of x^2.",
    flags: list[str] | None = None,
    topic_confidence: str = "low",
    text_only_status: str = "ready",
    question_text_trust: str = "high",
    visual_required: bool = False,
    question_image_path: str = "pm1/q01.png",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "12summer21",
        "paper_family": "pm1",
        "question_number": "1",
        "topic": topic,
        "question_text": question_text,
        "question_solution_marks": 3,
        "mark_scheme_text": mark_scheme_text,
        "ocr_text": question_text,
        "question_text_trust": question_text_trust,
        "text_only_status": text_only_status,
        "visual_required": visual_required,
        "question_image_path": question_image_path,
        "mark_scheme_image_path": "pm1/ms01.png",
        "notes": {
            "topic_confidence": topic_confidence,
            "review_flags": flags or [],
            "source_pdf": "input/pastpapers/9709/2021/question_papers/9709_s21_qp_12.pdf",
            "question_text_trust": question_text_trust,
            "text_only_status": text_only_status,
            "visual_required": visual_required,
        },
    }


def _rescore(record: dict[str, object]) -> dict[str, object]:
    return rescore_topic_confidence_record(record, config=AppConfig())


def test_benign_ocr_figure_region_hint_with_strong_topic_promotes_high() -> None:
    row = _rescore(_record(flags=["ocr_hint_figure_regions"]))

    assert row["topic_confidence_rescored"] == "high"
    assert row["benign_flags_ignored"] == ["ocr_hint_figure_regions"]
    assert row["hard_low_reasons"] == []


def test_markscheme_segmentation_flag_does_not_lower_question_text_topic() -> None:
    row = _rescore(_record(flags=["legacy_markscheme_segmentation"]))

    assert row["topic_confidence_rescored"] == "high"
    assert "markscheme_weak_flag_used_for_topic" not in row["confidence_blockers"]


def test_crop_uncertain_with_strong_topic_is_not_automatic_low() -> None:
    row = _rescore(_record(flags=["crop_uncertain", "low_confidence_question_crop"]))

    assert row["topic_confidence_rescored"] == "medium"
    assert "crop_uncertain_flag" in row["confidence_blockers"]


def test_visual_text_review_with_clear_topic_is_not_automatic_low() -> None:
    row = _rescore(
        _record(
            text_only_status="review",
            visual_required=True,
            flags=["contains_graph_or_diagram_prompt"],
        )
    )

    assert row["topic_confidence_rescored"] == "medium"
    assert "visual_dependency_flag" in row["confidence_blockers"]


def test_forced_no_rule_match_remains_low() -> None:
    row = _rescore(_record(flags=["topic_forced_no_rule_match"]))

    assert row["topic_confidence_rescored"] == "low"
    assert row["hard_low_reasons"] == ["topic_forced_no_rule_match"]


def test_topic_close_score_does_not_promote_to_high() -> None:
    row = _rescore(_record(flags=["topic_close_score"]))

    assert row["topic_confidence_rescored"] == "medium"
    assert "topic_close_score" in row["confidence_blockers"]


def test_structural_ocr_rejection_remains_low() -> None:
    row = _rescore(_record(flags=["ocr_large_margin_blocked_by_structural_rejection"]))

    assert row["topic_confidence_rescored"] == "low"
    assert row["hard_low_reasons"] == ["structural_failure_flag"]


def test_missing_question_text_remains_low() -> None:
    row = _rescore(_record(question_text="", mark_scheme_text="Use the binomial expansion."))

    assert row["topic_confidence_rescored"] == "low"
    assert row["hard_low_reasons"] == ["missing_or_truncated_question_text"]


def test_unusable_question_text_trust_blocks_promotion() -> None:
    row = _rescore(_record(question_text_trust="unusable"))

    assert row["topic_confidence_rescored"] == "low"
    assert row["hard_low_reasons"] == ["unusable_question_text_trust"]


def test_rescoring_report_includes_summary_and_remaining_low_review_pack() -> None:
    payload = {
        "questions": [
            _record(question_id="q1", flags=["ocr_hint_figure_regions"]),
            _record(question_id="q2", flags=["topic_forced_no_rule_match"]),
        ]
    }

    report = build_topic_confidence_rescoring_report(payload, config=AppConfig(), generated_at="2026-06-22T00:00:00+00:00")

    assert report["summary"]["total_records"] == 2
    assert report["summary"]["promoted_low_to_high"] == 1
    assert report["summary"]["remaining_low_count"] == 1
    assert report["summary"]["remaining_low_fix_category_counts"] == {"add_topic_rule": 1}
    assert report["remaining_low_review_pack"][0]["question_id"] == "q2"
    assert report["remaining_low_review_pack"][0]["suggested_next_fix_category"] == "add_topic_rule"
