from __future__ import annotations

from exam_bank.partial_question_block_recovery import recover_partial_question_blocks_payload


def _record(question_id: str = "11summer17_q04") -> dict:
    return {
        "question_id": question_id,
        "paper": "11summer17",
        "question_number": "4",
        "question_text": "4 (a) Find the sum of all the terms in the progression. [4]",
        "ocr_text": (
            "4 (a) An arithmetic progression has a first term of 32, a 5th term of 22 and a last term of -28. "
            "Find the sum of all the terms in the progression. [4] "
            "(b) Each year a school allocates a sum of money for the library. Find the total amount allocated. [3]"
        ),
        "mark_scheme_text": "4(a) d = -2.5 B1 Total: 4\n4(b) S_10 = 22400 Total: 3",
        "notes": {
            "mapping_status": "fail",
            "mapping_failure_reason": "partial_question_block",
            "question_crop_confidence": "low",
            "mark_scheme_crop_confidence": "low",
            "review_flags": ["low_confidence_question_crop", "crop_uncertain"],
            "validation_flags": [],
            "question_total_detected": 7,
            "mark_scheme_total_detected": 7,
            "text_fidelity_status": "clean",
            "question_crop_diagnostics": {
                "regions": [
                    {
                        "page_number": 6,
                        "text_bbox": {"x0": 50, "y0": 65, "x1": 545, "y1": 90},
                    }
                ]
            },
        },
    }


def test_recovers_only_partial_question_block_records() -> None:
    payload = {
        "questions": [
            _record(),
            {
                **_record("11summer17_q05"),
                "notes": {
                    **_record("11summer17_q05")["notes"],
                    "mapping_failure_reason": "missing_answer",
                },
            },
        ]
    }

    recovered, report = recover_partial_question_blocks_payload(payload, generated_at="2026-06-29T00:00:00+00:00")

    first, second = recovered["questions"]
    assert first["notes"]["mapping_status"] == "pass"
    assert first["notes"]["mapping_failure_reason"] == ""
    assert first["notes"]["partial_question_block_recovery"]["mark_scheme_mapping_modified"] is False
    assert second["notes"]["mapping_status"] == "fail"
    assert second["notes"]["mapping_failure_reason"] == "missing_answer"
    assert report["summary"]["before_partial_question_block"] == 1
    assert report["summary"]["after_partial_question_block"] == 0
    assert report["summary"]["traceable_reduction_rate"] == 1.0


def test_skips_when_no_contiguous_text_span_meets_threshold() -> None:
    record = _record()
    record["ocr_text"] = "4 short. [1]"
    record["question_text"] = "4 short. [1]"

    recovered, report = recover_partial_question_blocks_payload({"questions": [record]}, generated_at="2026-06-29T00:00:00+00:00")

    assert recovered["questions"][0]["notes"]["mapping_status"] == "fail"
    assert report["summary"]["after_partial_question_block"] == 1
    assert report["skipped"][0]["skip_reason"] == "no_contiguous_text_span"


def test_split_region_recovery_requires_spatial_adjacency_threshold() -> None:
    record = _record()
    record["notes"]["review_flags"] = ["crop_split_prompt_regions"]
    record["notes"]["question_crop_diagnostics"] = {
        "regions": [
            {"page_number": 1, "text_bbox": {"x0": 50, "y0": 80, "x1": 180, "y1": 100}},
            {"page_number": 1, "text_bbox": {"x0": 400, "y0": 500, "x1": 520, "y1": 530}},
        ]
    }

    recovered, report = recover_partial_question_blocks_payload({"questions": [record]}, generated_at="2026-06-29T00:00:00+00:00")

    assert recovered["questions"][0]["notes"]["mapping_status"] == "fail"
    assert report["skipped"][0]["skip_reason"] == "spatial_adjacency_below_threshold"
