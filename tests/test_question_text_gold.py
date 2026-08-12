from __future__ import annotations

import hashlib
from pathlib import Path

from exam_bank.question_text_gold import rebind_question_text_gold_registry


def test_gold_paths_rebind_only_when_current_image_hash_matches(tmp_path: Path) -> None:
    output = tmp_path / "output"
    image = output / "pm1" / "question.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"question")
    question_bank = {
        "questions": [
            {
                "question_id": "11summer24_q01",
                "paper_family": "pm1",
                "question_image_paths": ["pm1/question.png"],
            }
        ]
    }
    gold = {
        "schema_name": "exam_bank.question_text_exact_gold",
        "all_records_verified": True,
        "records": [
            {
                "question_id": "11summer24_q01",
                "paper_family": "legacy",
                "question_text": "Find x.",
                "source_image_path": "output/legacy/question.png",
                "source_image_sha256": hashlib.sha256(b"question").hexdigest(),
                "review_status": "verified",
            }
        ],
    }

    result = rebind_question_text_gold_registry(
        question_bank,
        gold,
        artifact_root=output,
        base_dir=tmp_path,
    )

    assert result["report"]["ok"] is True
    assert result["report"]["rebound_count"] == 1
    assert result["registry"]["all_records_verified"] is True
    assert result["registry"]["records"][0]["source_image_path"] == "output/pm1/question.png"
    assert result["registry"]["records"][0]["paper_family"] == "pm1"
    assert gold["records"][0]["source_image_path"] == "output/legacy/question.png"


def test_changed_gold_image_requires_re_review(tmp_path: Path) -> None:
    output = tmp_path / "output"
    image = output / "pm1" / "question.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"changed")
    question_bank = {
        "questions": [
            {
                "question_id": "11summer24_q01",
                "paper_family": "pm1",
                "question_image_paths": ["pm1/question.png"],
            }
        ]
    }
    gold = {
        "all_records_verified": True,
        "records": [
            {
                "question_id": "11summer24_q01",
                "question_text": "Find x.",
                "source_image_path": "output/legacy/question.png",
                "source_image_sha256": hashlib.sha256(b"reviewed").hexdigest(),
                "review_status": "verified",
            }
        ],
    }

    result = rebind_question_text_gold_registry(
        question_bank,
        gold,
        artifact_root=output,
        base_dir=tmp_path,
    )

    assert result["report"]["ok"] is False
    assert result["report"]["re_review_required_question_ids"] == ["11summer24_q01"]
    assert result["registry"]["all_records_verified"] is False
    assert result["registry"]["records"][0]["review_status"] == "re_review_required"


def test_unique_reviewed_image_hash_corrects_a_mechanical_identity_rewrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    spring = output / "pm1" / "spring.png"
    summer = output / "pm1" / "summer.png"
    spring.parent.mkdir(parents=True)
    spring.write_bytes(b"different March question")
    summer.write_bytes(b"reviewed June question")
    question_bank = {
        "questions": [
            {
                "question_id": "12spring19_q02",
                "paper_family": "pm1",
                "question_image_paths": ["pm1/spring.png"],
            },
            {
                "question_id": "12summer19_q02",
                "paper_family": "pm1",
                "question_image_paths": ["pm1/summer.png"],
            },
        ]
    }
    gold = {
        "all_records_verified": True,
        "records": [
            {
                "question_id": "12spring19_q02",
                "question_text": "The reviewed June question.",
                "source_image_path": "output/legacy.png",
                "source_image_sha256": hashlib.sha256(b"reviewed June question").hexdigest(),
                "review_status": "verified",
            }
        ],
    }

    result = rebind_question_text_gold_registry(
        question_bank,
        gold,
        artifact_root=output,
        base_dir=tmp_path,
    )

    assert result["report"]["ok"] is True
    assert result["report"]["reidentified_questions"] == [
        {
            "legacy_question_id": "12spring19_q02",
            "canonical_question_id": "12summer19_q02",
            "source_image_sha256": hashlib.sha256(b"reviewed June question").hexdigest(),
        }
    ]
    assert result["registry"]["records"][0]["question_id"] == "12summer19_q02"
    assert result["registry"]["records"][0]["review_status"] == "verified"
