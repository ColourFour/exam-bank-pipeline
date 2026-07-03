from __future__ import annotations

import json
from pathlib import Path

from exam_bank.topic_routing_refresh import refresh_topic_routing


ROOT = Path(__file__).resolve().parents[1]


def test_refresh_topic_routing_preserves_matching_existing_reviewed_route(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "61winter09_q03",
                paper="61winter09",
                family="stats",
                component="61",
                topic="normal_distribution",
                topic_confidence="high",
            )
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"
    _write_sidecar(
        routing,
        {
            "61winter09_q03": {
                "primary_topic_id": "9709_s1_topic_the_normal_distribution",
                "topic_distribution": [{"topic_id": "9709_s1_topic_the_normal_distribution", "fit_percent": 100}],
                "confidence": "low",
                "review_required": True,
                "review_reasons": ["manual_review_required"],
                "evidence_used": ["question_text"],
                "routing_source": "deepseek_topic_routing",
                "llm_model": "deepseek-v4-flash",
                "llm_prompt_version": "topic_routing_v1",
                "evidence_packet_hash": "0" * 64,
                "paper_family": "p5",
            }
        },
    )

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "report",
        write=False,
        generated_at="2026-06-27T00:00:00+00:00",
    )

    assert report["summary"]["sidecar_entries"] == 1
    assert report["summary"]["preserved_existing_entries"] == 1
    assert report["summary"]["preserved_reviewed_entries"] == 1
    assert report["summary"]["existing_hash_refreshed_count"] == 1
    assert report["summary"]["conflicts_count"] == 0


def test_refresh_topic_routing_reports_existing_topic_conflict(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "61winter09_q03",
                paper="61winter09",
                family="stats",
                component="61",
                topic="normal_distribution",
                topic_confidence="high",
            )
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"
    _write_sidecar(
        routing,
        {
            "61winter09_q03": {
                "primary_topic_id": "9709_s1_topic_probability",
                "topic_distribution": [{"topic_id": "9709_s1_topic_probability", "fit_percent": 100}],
                "confidence": "high",
                "review_required": False,
                "review_reasons": [],
                "evidence_used": ["question_text"],
                "routing_source": "deepseek_topic_routing",
                "llm_model": "deepseek-v4-flash",
                "llm_prompt_version": "topic_routing_v1",
                "evidence_packet_hash": "0" * 64,
                "paper_family": "p5",
            }
        },
    )

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "report",
        write=False,
        generated_at="2026-06-27T00:00:00+00:00",
    )

    assert report["summary"]["sidecar_entries"] == 1
    assert report["summary"]["conflicts_count"] == 1
    assert report["summary"]["new_review_required_entries"] == 1
    assert report["conflicts"][0]["existing_primary_topic_id"] == "9709_s1_topic_probability"
    assert report["conflicts"][0]["normalized_primary_topic_id"] == "9709_s1_topic_the_normal_distribution"


def test_refresh_topic_routing_preserves_autumn_to_winter_existing_route_alias(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "61winter21_q03",
                paper="61winter21",
                family="stats",
                component="61",
                topic="normal_distribution",
                topic_confidence="high",
            )
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"
    _write_sidecar(
        routing,
        {
            "61autumn21_q03": {
                "primary_topic_id": "9709_s1_topic_the_normal_distribution",
                "topic_distribution": [{"topic_id": "9709_s1_topic_the_normal_distribution", "fit_percent": 100}],
                "confidence": "high",
                "review_required": False,
                "review_reasons": [],
                "evidence_used": ["question_text"],
                "routing_source": "deepseek_topic_routing",
                "llm_model": "deepseek-v4-flash",
                "llm_prompt_version": "topic_routing_v1",
                "evidence_packet_hash": "0" * 64,
                "paper": "61autumn21",
                "paper_family": "p5",
                "question_number": "3",
            }
        },
    )

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "report",
        write=False,
        generated_at="2026-06-27T00:00:00+00:00",
    )

    assert report["summary"]["preserved_existing_entries"] == 1
    assert report["summary"]["preserved_via_alias_entries"] == 1
    assert report["summary"]["conflicts_count"] == 0


def test_refresh_topic_routing_writes_complete_sidecar_and_sha(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record("06summer08_q04", paper="06summer08", family="stats", component="06", topic="normal_distribution"),
            _record("41winter20_q06", paper="41winter20", family="stats", component="41", topic="connected_particles_energy"),
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "topic_routing_refresh",
        write=True,
        generated_at="2026-06-27T00:00:00+00:00",
    )

    payload = json.loads(routing.read_text(encoding="utf-8"))
    assert report["summary"]["coverage_complete"] is True
    assert report["summary"]["sidecar_entries"] == 2
    assert report["summary"]["explicit_exclusions"] == 0
    assert set(payload["records"]) == {"06summer08_q04", "41winter20_q06"}
    assert payload["records"]["06summer08_q04"]["primary_topic_id"] == "9709_s1_topic_the_normal_distribution"
    assert payload["records"]["41winter20_q06"]["primary_topic_id"] == "9709_m1_topic_energy_work_and_power"
    assert routing.with_suffix(".sha256").is_file()
    assert (tmp_path / "topic_routing_refresh.json").is_file()
    assert (tmp_path / "topic_routing_refresh.md").is_file()


def _write_question_bank(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "question_bank.json"
    path.write_text(
        json.dumps({"schema_name": "exam_bank.question_bank", "schema_version": 2, "questions": records}),
        encoding="utf-8",
    )
    return path


def _write_sidecar(path: Path, records: dict[str, dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.topic_routing_sidecar",
                "schema_version": 1,
                "record_count": len(records),
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _record(
    question_id: str,
    *,
    paper: str,
    family: str,
    component: str,
    topic: str,
    topic_confidence: str = "high",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": paper,
        "paper_family": family,
        "question_number": question_id.rsplit("_q", 1)[-1],
        "topic": topic,
        "question_text": "The random variable X has a normal distribution.",
        "question_text_trust": "high",
        "question_text_role": "readable_text",
        "text_only_status": "ready",
        "mark_scheme_text": "Use standardisation and the normal distribution table.",
        "visual_required": False,
        "notes": {
            "source_paper_code": component,
            "topic_confidence": topic_confidence,
            "mapping_status": "pass",
            "validation_status": "pass",
            "scope_quality_status": "clean",
            "question_crop_confidence": "high",
            "visual_curation_status": "ready",
            "text_only_status": "ready",
        },
    }
