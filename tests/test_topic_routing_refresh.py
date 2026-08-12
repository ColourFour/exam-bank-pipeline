from __future__ import annotations

import json
from pathlib import Path

from exam_bank.topic_packets import load_packet_taxonomy
from exam_bank.topic_routing_refresh import (
    build_deterministic_route_record,
    refresh_topic_routing,
    route_context_for_record,
)


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
                "51winter21_q03",
                paper="51winter21",
                family="stats",
                component="51",
                topic="normal_distribution",
                topic_confidence="high",
            )
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"
    _write_sidecar(
        routing,
        {
            "51autumn21_q03": {
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
                "paper": "51autumn21",
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


def test_refresh_claims_legacy_march_route_before_new_june_record(tmp_path: Path) -> None:
    june = _record(
        "12summer21_q01",
        paper="12summer21",
        family="pm1",
        component="12",
        topic="binomial_expansion",
    )
    june["notes"]["source_pdf"] = "input/pastpapers/9709/2021/question_papers/9709_s21_qp_12.pdf"
    march = _record(
        "12spring21_q01",
        paper="12spring21",
        family="pm1",
        component="12",
        topic="binomial_expansion",
    )
    march["notes"]["source_pdf"] = "input/pastpapers/9709/2021/question_papers/9709_m21_qp_12.pdf"
    question_bank = _write_question_bank(tmp_path, [june, march])
    routing = tmp_path / "question_bank.topic_routing.v1.json"
    _write_sidecar(
        routing,
        {
            "12summer21_q01": {
                "primary_topic_id": "9709_p1_topic_series",
                "topic_distribution": [{"topic_id": "9709_p1_topic_series", "fit_percent": 100}],
                "confidence": "low",
                "review_required": True,
                "review_reasons": ["manual_review_required"],
                "evidence_used": ["question_text"],
                "routing_source": "legacy_march_review",
                "evidence_packet_hash": "0" * 64,
                "source_record_hash": "1" * 64,
                "paper": "12summer21",
                "paper_family": "p1",
                "question_number": "1",
            }
        },
    )

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "report",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    records = json.loads(routing.read_text(encoding="utf-8"))["records"]
    assert records["12spring21_q01"]["routing_source"] == "legacy_march_review"
    assert records["12spring21_q01"]["previous_question_id"] == "12summer21_q01"
    assert records["12spring21_q01"]["source_session_code"] == "m21"
    assert records["12summer21_q01"]["routing_source"] == "deterministic_topic_packet_normalization"
    assert records["12summer21_q01"]["source_session_code"] == "s21"
    assert "previous_question_id" not in records["12summer21_q01"]
    assert report["summary"]["preserved_via_alias_entries"] == 1


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
    assert (tmp_path / "question_bank_release_manifest.v1.json").is_file()
    assert (tmp_path / "topic_routing_refresh.json").is_file()
    assert (tmp_path / "topic_routing_refresh.md").is_file()


def test_refresh_keeps_p6_as_review_only_s2_without_collapsing_to_s1(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "62winter24_q01",
                paper="62winter24",
                family="stats",
                component="62",
                topic="normal_distribution",
            )
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "p6_refresh",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    route = json.loads(routing.read_text(encoding="utf-8"))["records"]["62winter24_q01"]
    assert report["summary"]["sidecar_entries"] == 1
    assert report["summary"]["unresolved_count"] == 1
    assert route["paper_family"] == "p6"
    assert route["course_id"] == "s2"
    assert route["primary_topic_id"] == ""
    assert route["review_required"] is True
    assert "topic_normalization_unsupported_component_family" in route["review_reasons"]


def test_refresh_maps_pre_2020_p6_through_s1_p5_taxonomy(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "62winter19_q01",
                paper="62winter19",
                family="stats",
                component="62",
                topic="normal_distribution",
            )
        ],
    )
    routing = tmp_path / "legacy_p6.topic_routing.v1.json"

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "legacy_p6_refresh",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    route = json.loads(routing.read_text(encoding="utf-8"))["records"]["62winter19_q01"]
    assert report["summary"]["unresolved_count"] == 0
    assert route["paper_family"] == "p5"
    assert route["course_id"] == "s1"
    assert route["primary_topic_id"] == "9709_s1_topic_the_normal_distribution"
    assert route["normalization_status"] == "resolved"


def test_refresh_fails_closed_for_yearless_component_six(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "component6_unknown_q01",
                paper="unknown",
                family="stats",
                component="62",
                topic="normal_distribution",
            )
        ],
    )
    routing = tmp_path / "yearless_p6.topic_routing.v1.json"

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "yearless_p6_refresh",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    route = json.loads(routing.read_text(encoding="utf-8"))["records"]["component6_unknown_q01"]
    assert report["summary"]["unresolved_count"] == 1
    assert route["paper_family"] == ""
    assert route["course_id"] is None
    assert route["review_required"] is True
    assert route["normalization_status"] == "ambiguous_component_era"
    assert "topic_normalization_ambiguous_component_era" in route["review_reasons"]


def test_refresh_recovers_legacy_p6_topic_from_stale_blank_s2_route(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "61winter19_q03",
                paper="61winter19",
                family="stats",
                component="61",
                topic="probability",
            )
        ],
    )
    routing = tmp_path / "recover_legacy_p6.topic_routing.v1.json"
    _write_sidecar(
        routing,
        {
            "61winter19_q03": {
                "primary_topic_id": "",
                "topic_distribution": [],
                "confidence": "high",
                "review_required": True,
                "review_reasons": ["topic_normalization_unsupported_component_family"],
                "paper": "61winter19",
                "paper_family": "p6",
                "course_id": "s2",
                "normalization_status": "unsupported_component_family",
            }
        },
    )

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "recover_legacy_p6_refresh",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    route = json.loads(routing.read_text(encoding="utf-8"))["records"]["61winter19_q03"]
    assert report["summary"]["conflicts_count"] == 1
    assert route["paper_family"] == "p5"
    assert route["course_id"] == "s1"
    assert route["primary_topic_id"] == "9709_s1_topic_probability"
    assert route["topic_distribution"] == [{"topic_id": "9709_s1_topic_probability", "fit_percent": 100}]
    assert route["previous_route_conflict"]["type"] == "missing_primary_topic"
    assert "existing_route_conflicts_with_normalized_topic" not in route["review_reasons"]


def test_refresh_blocks_pre_2020_p5_mechanics_2_as_unsupported(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "51winter19_q01",
                paper="51winter19",
                family="stats",
                component="51",
                topic="normal_distribution",
            )
        ],
    )
    routing = tmp_path / "legacy_p5.topic_routing.v1.json"

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "legacy_p5_refresh",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    route = json.loads(routing.read_text(encoding="utf-8"))["records"]["51winter19_q01"]
    assert report["summary"]["unresolved_count"] == 1
    assert route["paper_family"] == ""
    assert route["course_id"] is None
    assert route["primary_topic_id"] == ""
    assert route["normalization_status"] == "unsupported_component_era"
    assert "topic_normalization_unsupported_component_era" in route["review_reasons"]


def test_refresh_does_not_move_p1_algebra_label_into_p3_course(tmp_path: Path) -> None:
    question_bank = _write_question_bank(
        tmp_path,
        [
            _record(
                "11summer24_q01",
                paper="11summer24",
                family="pm1",
                component="11",
                topic="algebra",
            )
        ],
    )
    routing = tmp_path / "question_bank.topic_routing.v1.json"

    report = refresh_topic_routing(
        question_bank_path=question_bank,
        taxonomy_path=ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json",
        canonical_taxonomy_root=ROOT / "exam_bank_taxonomy/canonical",
        routing_path=routing,
        report_prefix=tmp_path / "p1_cross_family_refresh",
        write=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    route = json.loads(routing.read_text(encoding="utf-8"))["records"]["11summer24_q01"]
    assert report["summary"]["unresolved_count"] == 1
    assert route["paper_family"] == "p1"
    assert route["course_id"] == "p1"
    assert route["primary_topic_id"] == ""
    assert route["review_required"] is True
    assert "topic_normalization_component_family_topic_mismatch" in route["review_reasons"]


def test_refresh_uses_component_authority_for_p4_p5_p6_and_keeps_s1_s2_distinct(tmp_path: Path) -> None:
    source_records = [
        _record(
            "42winter24_q01",
            paper="42winter24",
            family="stats",
            component="42",
            topic="connected_particles_energy",
        ),
        _record(
            "52winter24_q01",
            paper="52winter24",
            family="mechanics",
            component="52",
            topic="normal_distribution",
        ),
        _record(
            "62winter24_q01",
            paper="62winter24",
            family="mechanics",
            component="62",
            topic="normal_distribution",
        ),
    ]
    taxonomy = load_packet_taxonomy(ROOT / "exam_bank_taxonomy/caie_9709_syllabus_topics.v1.json")
    records = {
        str(record["question_id"]): build_deterministic_route_record(
            record,
            route_context_for_record(record, taxonomy, ROOT / "exam_bank_taxonomy/canonical"),
            generated_at="2026-08-09T00:00:00+00:00",
        )
        for record in source_records
    }

    assert (records["42winter24_q01"]["paper_family"], records["42winter24_q01"]["course_id"]) == ("p4", "m1")
    assert (records["52winter24_q01"]["paper_family"], records["52winter24_q01"]["course_id"]) == ("p5", "s1")
    assert records["52winter24_q01"]["review_required"] is False
    assert (records["62winter24_q01"]["paper_family"], records["62winter24_q01"]["course_id"]) == ("p6", "s2")
    assert records["62winter24_q01"]["review_required"] is True
    assert records["62winter24_q01"]["primary_topic_id"] == ""


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
