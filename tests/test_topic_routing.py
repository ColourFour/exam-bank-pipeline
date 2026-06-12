from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exam_bank import topic_routing


P1_SERIES = "9709_p1_topic_series"
P1_DIFFERENTIATION = "9709_p1_topic_differentiation"
P3_COMPLEX = "9709_p3_topic_complex_numbers"


def _record(question_id: str = "12spring24_q01", *, paper_family: str = "p1", **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": paper_family,
        "question_number": question_id.rsplit("_q", 1)[-1],
        "question_text": "Find the first three terms in the binomial expansion.",
        "question_text_trust": "high",
        "question_text_role": "readable_text",
        "text_only_status": "ready",
        "ocr_text": "",
        "ocr_text_trust": "",
        "ocr_text_role": "",
        "mark_scheme_text": "Use the binomial expansion formula and collect terms.",
        "visual_required": False,
        "notes": {},
        "question_image_path": "should/not/be/sent.png",
    }
    record.update(overrides)
    return record


def _route_record(
    primary_topic_id: str | None = P1_SERIES,
    *,
    distribution: list[dict[str, object]] | None = None,
    confidence: str = "high",
    review_required: bool = False,
    evidence_used: list[str] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "primary_topic_id": primary_topic_id,
        "topic_distribution": distribution
        if distribution is not None
        else ([{"topic_id": primary_topic_id, "fit_percent": 100}] if primary_topic_id else []),
        "confidence": confidence,
        "review_required": review_required,
        "review_reasons": ["needs_human_audit"] if review_required else [],
        "evidence_used": evidence_used if evidence_used is not None else ["question_text", "mark_scheme_text"],
    }
    if extra:
        item.update(extra)
    return item


def _route_payload(question_id: str, item: dict[str, object] | None = None) -> str:
    return json.dumps({"records": {question_id: item or _route_record()}})


def _route_records_payload(records: dict[str, dict[str, object]]) -> str:
    return json.dumps({"records": records})


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_client(*responses: object) -> object:
    calls: list[dict[str, object]] = []
    iterator = iter(responses)

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**kwargs: object) -> object:
                    calls.append(kwargs)
                    result = next(iterator)
                    if isinstance(result, Exception):
                        raise result
                    return result

            completions = _Completions()

        chat = _Chat()

    client = FakeClient()
    setattr(client, "calls", calls)
    return client


def _parse(raw: str, question_id: str = "12spring24_q01") -> dict[str, object]:
    return _parse_result(raw, question_id)["records"][question_id]


def _parse_result(raw: str, question_id: str = "12spring24_q01") -> dict[str, object]:
    packet = topic_routing.build_topic_routing_question_packet(
        _record(question_id),
        taxonomy_root="exam_bank_taxonomy/canonical",
    ).packet
    all_ids = topic_routing.load_all_topic_ids("exam_bank_taxonomy/canonical")
    return topic_routing.parse_topic_routing_model_json(
        raw,
        expected_packets=[packet],
        allowed_topic_ids={topic["topic_id"] for topic in packet["allowed_topics"]},
        all_topic_ids=all_ids,
    )


def _evidence_hash(record: dict[str, object] | None = None) -> str:
    return topic_routing.build_topic_routing_question_packet(
        record or _record(),
        taxonomy_root="exam_bank_taxonomy/canonical",
    ).evidence_packet_hash


def test_topic_route_single_topic_100_percent_classification() -> None:
    parsed = _parse(_route_payload("12spring24_q01"))

    assert parsed["primary_topic_id"] == P1_SERIES
    assert parsed["topic_distribution"] == [{"topic_id": P1_SERIES, "fit_percent": 100}]
    assert parsed["confidence"] == "high"


def test_topic_route_multi_topic_classification_sums_to_100() -> None:
    parsed = _parse(
        _route_payload(
            "12spring24_q01",
            _route_record(
                P1_SERIES,
                distribution=[
                    {"topic_id": P1_SERIES, "fit_percent": 70},
                    {"topic_id": P1_DIFFERENTIATION, "fit_percent": 30},
                ],
            ),
        )
    )

    assert sum(item["fit_percent"] for item in parsed["topic_distribution"]) == 100


def test_topic_route_rejects_percentages_over_100() -> None:
    parsed = _parse_result(
        _route_payload(
            "12spring24_q01",
            _route_record(
                P1_SERIES,
                distribution=[
                    {"topic_id": P1_SERIES, "fit_percent": 80},
                    {"topic_id": P1_DIFFERENTIATION, "fit_percent": 30},
                ],
            ),
        )
    )

    assert parsed["records"] == {}
    assert "total exactly 100" in parsed["record_errors"]["12spring24_q01"]["message"]


def test_topic_route_rejects_invented_topic_ids() -> None:
    parsed = _parse_result(_route_payload("12spring24_q01", _route_record("invented_topic")))

    assert parsed["records"] == {}
    assert parsed["record_errors"]["12spring24_q01"]["type"] == topic_routing.AI_FAILURE_TAXONOMY_VALIDATION_ERROR


def test_topic_route_rejects_topic_ids_outside_paper_family() -> None:
    parsed = _parse_result(_route_payload("12spring24_q01", _route_record(P3_COMPLEX)))

    assert parsed["records"] == {}
    assert parsed["record_errors"]["12spring24_q01"]["type"] == "topic_outside_allowed_paper_family"


def test_topic_route_accepts_low_confidence_review_required_output() -> None:
    parsed = _parse(
        _route_payload(
            "12spring24_q01",
            _route_record(P1_SERIES, confidence="low", review_required=True),
        )
    )

    assert parsed["confidence"] == "low"
    assert parsed["review_required"] is True


def test_topic_route_weak_evidence_record_is_marked_review_required(tmp_path: Path) -> None:
    client = _fake_client()
    records, manifest = topic_routing.route_topic_records(
        [_record(question_text="", mark_scheme_text="", text_only_status="fail")],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    routed = records["12spring24_q01"]
    assert routed["review_required"] is True
    assert routed["course_id"] == "p1"
    assert routed["component_name"] == "Pure Mathematics 1"
    assert routed["review_reasons"] == [topic_routing.REVIEW_WEAK_EVIDENCE]
    assert routed["primary_topic_id"] is None
    assert routed["evidence_packet_hash"] == _evidence_hash(_record(question_text="", mark_scheme_text="", text_only_status="fail"))
    assert getattr(client, "calls") == []
    assert manifest["provider_batch_count"] == 0


def test_topic_route_visual_required_with_insufficient_text_is_marked_review_required(tmp_path: Path) -> None:
    client = _fake_client()
    records, _manifest = topic_routing.route_topic_records(
        [
            _record(
                visual_required=True,
                question_text="The diagram shows a shaded region.",
                mark_scheme_text="",
            )
        ],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    routed = records["12spring24_q01"]
    assert routed["review_required"] is True
    assert routed["review_reasons"] == [topic_routing.REVIEW_VISUAL_INSUFFICIENT]
    assert getattr(client, "calls") == []


def test_topic_route_evidence_packet_hash_is_deterministic() -> None:
    first = topic_routing.build_topic_routing_question_packet(
        _record(),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )
    second = topic_routing.build_topic_routing_question_packet(
        _record(),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    assert first.evidence_packet_hash == second.evidence_packet_hash
    assert first.evidence_packet_hash == topic_routing.hash_topic_routing_evidence_packet(first.packet)


def test_topic_route_evidence_packet_hash_changes_with_evidence_text() -> None:
    first = topic_routing.build_topic_routing_question_packet(
        _record(question_text="Differentiate x^2."),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )
    second = topic_routing.build_topic_routing_question_packet(
        _record(question_text="Integrate x^2."),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    assert first.evidence_packet_hash != second.evidence_packet_hash


def test_topic_route_evidence_packet_hash_changes_with_available_evidence_fields() -> None:
    with_ocr = topic_routing.build_topic_routing_question_packet(
        _record(ocr_text="OCR evidence", ocr_text_trust="high", ocr_text_role="readable_text"),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )
    without_ocr = topic_routing.build_topic_routing_question_packet(
        _record(ocr_text="", ocr_text_trust="", ocr_text_role=""),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    assert with_ocr.evidence_packet_hash != without_ocr.evidence_packet_hash


def test_topic_route_visual_required_supplies_ocr_fallback_when_question_text_untrusted() -> None:
    result = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="",
            question_text_trust="",
            question_text_role="",
            text_only_status="review",
            ocr_text="The diagram shows a curve y = x^2.",
            ocr_text_trust="low",
            ocr_text_role="untrusted_math_text",
            mark_scheme_text="Differentiate the curve.",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    packet = result.packet
    assert packet["evidence"]["ocr_text"] == "The diagram shows a curve y = x^2."
    assert packet["available_evidence_fields"] == ["mark_scheme_text", "ocr_text"]
    assert packet["evidence_sources"]["ocr_text"] == "ocr_fallback"
    assert packet["ocr_text_source"] == "ocr_fallback"
    assert result.evidence_packet_hash


def test_topic_route_visual_required_supplies_search_hint_fallback_when_question_text_untrusted() -> None:
    result = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="Search hint: trigonometric equation from graph.",
            question_text_trust="medium",
            question_text_role="search_hint",
            text_only_status="ready",
            ocr_text="",
            ocr_text_trust="",
            mark_scheme_text="Solve using graph intersections.",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    packet = result.packet
    assert packet["evidence"]["question_text"] == "Search hint: trigonometric equation from graph."
    assert packet["available_evidence_fields"] == ["mark_scheme_text", "question_text"]
    assert packet["evidence_sources"]["question_text"] == "search_hint_fallback"
    assert packet["question_text_source"] == "search_hint_fallback"


def test_topic_route_trusted_question_text_is_preferred_over_fallback_text() -> None:
    result = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="Trusted text: differentiate x^3.",
            question_text_trust="high",
            question_text_role="readable_text",
            text_only_status="ready",
            ocr_text="OCR fallback should not replace this.",
            ocr_text_trust="low",
            ocr_text_role="untrusted_math_text",
            mark_scheme_text="3x^2",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    packet = result.packet
    assert packet["evidence"]["question_text"] == "Trusted text: differentiate x^3."
    assert "ocr_text" not in packet["evidence"]
    assert packet["evidence_sources"]["question_text"] == "trusted"
    assert packet["question_text_source"] == "trusted"


def test_topic_route_visual_required_without_fallback_does_not_invent_evidence() -> None:
    result = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="",
            question_text_trust="",
            question_text_role="",
            text_only_status="fail",
            ocr_text="",
            ocr_text_trust="",
            mark_scheme_text="Use integration.",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    packet = result.packet
    assert packet["evidence"] == {"mark_scheme_text": "Use integration."}
    assert packet["available_evidence_fields"] == ["mark_scheme_text"]
    assert packet["question_text_source"] == "none"
    assert packet["ocr_text_source"] == "none"


def test_topic_route_visual_fallback_hash_changes_with_fallback_text() -> None:
    with_ocr = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="",
            question_text_trust="",
            text_only_status="review",
            ocr_text="Integrate x squared.",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )
    without_ocr = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="",
            question_text_trust="",
            text_only_status="review",
            ocr_text="",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )
    with_search_hint = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="Search hint: integration",
            question_text_trust="medium",
            question_text_role="search_hint",
            text_only_status="ready",
            ocr_text="",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )
    changed_search_hint = topic_routing.build_topic_routing_question_packet(
        _record(
            visual_required=True,
            question_text="Search hint: differentiation",
            question_text_trust="medium",
            question_text_role="search_hint",
            text_only_status="ready",
            ocr_text="",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    )

    assert with_ocr.evidence_packet_hash != without_ocr.evidence_packet_hash
    assert with_search_hint.evidence_packet_hash != changed_search_hint.evidence_packet_hash


def test_topic_route_resume_preserves_current_hashed_row() -> None:
    evidence_hash = _evidence_hash()
    record = {
        **_route_record(),
        "llm_model": "deepseek-v4-flash",
        "llm_prompt_version": topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        "routing_source": "deepseek_topic_routing",
        "evidence_packet_hash": evidence_hash,
    }

    assert topic_routing._resume_record_is_current(
        record,
        model="deepseek-v4-flash",
        prompt_version=topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        evidence_packet_hash=evidence_hash,
    )


def test_topic_route_resume_rejects_legacy_row_without_evidence_packet_hash() -> None:
    record = {
        **_route_record(),
        "llm_model": "deepseek-v4-flash",
        "llm_prompt_version": topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        "routing_source": "deepseek_topic_routing",
    }

    assert not topic_routing._resume_record_is_current(
        record,
        model="deepseek-v4-flash",
        prompt_version=topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        evidence_packet_hash=_evidence_hash(),
    )


def test_topic_route_resume_rejects_stale_hash_row() -> None:
    record = {
        **_route_record(),
        "llm_model": "deepseek-v4-flash",
        "llm_prompt_version": topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        "routing_source": "deepseek_topic_routing",
        "evidence_packet_hash": "0" * 64,
    }

    assert not topic_routing._resume_record_is_current(
        record,
        model="deepseek-v4-flash",
        prompt_version=topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        evidence_packet_hash=_evidence_hash(),
    )


def test_topic_route_provider_error_record_includes_evidence_packet_hash(tmp_path: Path) -> None:
    client = _fake_client(RuntimeError("provider unavailable"))

    records, _manifest = topic_routing.route_topic_records(
        [_record()],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    routed = records["12spring24_q01"]
    assert routed["routing_source"] == "deepseek_topic_routing_error"
    assert routed["evidence_packet_hash"] == _evidence_hash()


def test_topic_route_repairs_unsupported_evidence_used_without_failing_route(tmp_path: Path) -> None:
    record = _record(question_text="", question_text_trust="", question_text_role="", mark_scheme_text="Use the binomial expansion.")
    client = _fake_client(
        _chat_response(
            _route_records_payload(
                {
                    "12spring24_q01": _route_record(evidence_used=["ocr_text", "mark_scheme_text"]),
                }
            )
        )
    )

    routed, _manifest = topic_routing.route_topic_records(
        [record],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    route = routed["12spring24_q01"]
    assert route["routing_source"] == "deepseek_topic_routing"
    assert route["evidence_used"] == ["mark_scheme_text"]
    assert route["evidence_used_repaired"] is True
    assert route["evidence_used_original"] == ["ocr_text", "mark_scheme_text"]
    assert route["evidence_used_dropped"] == ["ocr_text"]
    assert "error" not in route


def test_topic_route_evidence_repair_recognizes_visual_ocr_fallback(tmp_path: Path) -> None:
    record = _record(
        visual_required=True,
        question_text="",
        question_text_trust="",
        question_text_role="",
        text_only_status="review",
        ocr_text="Integrate x^2 from 0 to 1.",
        ocr_text_trust="low",
        ocr_text_role="untrusted_math_text",
        mark_scheme_text="Use integration.",
    )
    client = _fake_client(
        _chat_response(
            _route_records_payload(
                {
                    "12spring24_q01": _route_record(evidence_used=["ocr_text"]),
                }
            )
        )
    )

    routed, _manifest = topic_routing.route_topic_records(
        [record],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    route = routed["12spring24_q01"]
    assert route["routing_source"] == "deepseek_topic_routing"
    assert route["evidence_used"] == ["ocr_text"]
    assert "evidence_used_repaired" not in route


def test_topic_route_repairs_all_unsupported_evidence_used_with_available_fallback(tmp_path: Path) -> None:
    record = _record(mark_scheme_text="")
    client = _fake_client(
        _chat_response(
            _route_records_payload(
                {
                    "12spring24_q01": _route_record(evidence_used=["ocr_text"]),
                }
            )
        )
    )

    routed, _manifest = topic_routing.route_topic_records(
        [record],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    route = routed["12spring24_q01"]
    assert route["routing_source"] == "deepseek_topic_routing"
    assert route["evidence_used"] == ["question_text"]
    assert route["evidence_used_repaired"] is True
    assert route["evidence_used_original"] == ["ocr_text"]
    assert route["evidence_used_dropped"] == ["ocr_text"]


def test_topic_route_no_available_evidence_does_not_silently_pass(tmp_path: Path) -> None:
    record = _record(
        question_text="",
        question_text_trust="",
        question_text_role="",
        ocr_text="",
        ocr_text_trust="",
        ocr_text_role="",
        mark_scheme_text="",
        text_only_status="ready",
    )
    client = _fake_client()

    routed, _manifest = topic_routing.route_topic_records(
        [record],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    route = routed["12spring24_q01"]
    assert route["routing_source"] == "deterministic_review_gate"
    assert route["review_required"] is True
    assert route["review_reasons"] == [topic_routing.REVIEW_WEAK_EVIDENCE]
    assert getattr(client, "calls") == []


def test_topic_route_evidence_repair_without_available_fallback_is_record_error() -> None:
    packet = topic_routing.build_topic_routing_question_packet(
        _record(
            question_text="",
            question_text_trust="",
            question_text_role="",
            ocr_text="",
            ocr_text_trust="",
            ocr_text_role="",
            mark_scheme_text="",
            text_only_status="ready",
        ),
        taxonomy_root="exam_bank_taxonomy/canonical",
    ).packet
    parsed = topic_routing.parse_topic_routing_model_json(
        _route_payload("12spring24_q01", _route_record(evidence_used=["ocr_text"])),
        expected_packets=[packet],
        allowed_topic_ids={topic["topic_id"] for topic in packet["allowed_topics"]},
        all_topic_ids=topic_routing.load_all_topic_ids("exam_bank_taxonomy/canonical"),
    )

    assert parsed["records"] == {}
    assert "no supplied evidence fallback exists" in parsed["record_errors"]["12spring24_q01"]["message"]


def test_topic_route_invalid_topic_id_still_fails_with_repairable_evidence_used(tmp_path: Path) -> None:
    client = _fake_client(
        _chat_response(
            _route_records_payload(
                {
                    "12spring24_q01": _route_record("invented_topic", evidence_used=["ocr_text", "question_text"]),
                }
            )
        )
    )

    routed, _manifest = topic_routing.route_topic_records(
        [_record()],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    route = routed["12spring24_q01"]
    assert route["routing_source"] == "deepseek_topic_routing_error"
    assert route["error"]["type"] == topic_routing.AI_FAILURE_TAXONOMY_VALIDATION_ERROR


def test_topic_route_invalid_distribution_still_fails_with_repairable_evidence_used(tmp_path: Path) -> None:
    client = _fake_client(
        _chat_response(
            _route_records_payload(
                {
                    "12spring24_q01": _route_record(
                        P1_SERIES,
                        distribution=[
                            {"topic_id": P1_SERIES, "fit_percent": 80},
                            {"topic_id": P1_DIFFERENTIATION, "fit_percent": 30},
                        ],
                        evidence_used=["ocr_text", "question_text"],
                    ),
                }
            )
        )
    )

    routed, _manifest = topic_routing.route_topic_records(
        [_record()],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    route = routed["12spring24_q01"]
    assert route["routing_source"] == "deepseek_topic_routing_error"
    assert "total exactly 100" in route["error"]["message"]


def test_topic_route_salvages_valid_and_repaired_siblings_when_one_returned_record_is_invalid(tmp_path: Path) -> None:
    question_ids = [f"12spring24_q{index:02d}" for index in range(1, 8)]
    records = [_record(question_id) for question_id in question_ids]
    response_records = {question_id: _route_record() for question_id in question_ids}
    response_records["12spring24_q04"] = _route_record(evidence_used=["ocr_text"])
    response_records["12spring24_q05"] = _route_record(
        P1_SERIES,
        distribution=[
            {"topic_id": P1_SERIES, "fit_percent": 80},
            {"topic_id": P1_DIFFERENTIATION, "fit_percent": 30},
        ],
    )
    client = _fake_client(_chat_response(_route_records_payload(response_records)))

    routed, manifest = topic_routing.route_topic_records(
        records,
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    assert len(routed) == 7
    assert sum(record["routing_source"] == "deepseek_topic_routing" for record in routed.values()) == 6
    assert routed["12spring24_q04"]["routing_source"] == "deepseek_topic_routing"
    assert routed["12spring24_q04"]["evidence_used_repaired"] is True
    assert routed["12spring24_q05"]["routing_source"] == "deepseek_topic_routing_error"
    assert routed["12spring24_q05"]["review_required"] is True
    assert "total exactly 100" in routed["12spring24_q05"]["error"]["message"]
    assert all(record.get("evidence_packet_hash") for record in routed.values())
    assert manifest["batches"][0]["batch_salvaged"] is True
    assert manifest["batches"][0]["valid_records"] == 6
    assert manifest["batches"][0]["invalid_records"] == 1


def test_topic_route_malformed_top_level_json_still_fails_whole_batch(tmp_path: Path) -> None:
    records = [_record("12spring24_q01"), _record("12spring24_q02")]
    client = _fake_client(_chat_response("{not valid json"))

    routed, manifest = topic_routing.route_topic_records(
        records,
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    assert set(routed) == {"12spring24_q01", "12spring24_q02"}
    assert all(record["routing_source"] == "deepseek_topic_routing_error" for record in routed.values())
    assert all(record["review_required"] is True for record in routed.values())
    assert manifest["batches"][0]["status"] == topic_routing.AI_FAILURE_INVALID_JSON


def test_topic_route_missing_returned_record_only_fails_missing_question(tmp_path: Path) -> None:
    records = [_record("12spring24_q01"), _record("12spring24_q02")]
    client = _fake_client(_chat_response(_route_records_payload({"12spring24_q01": _route_record()})))

    routed, manifest = topic_routing.route_topic_records(
        records,
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    assert routed["12spring24_q01"]["routing_source"] == "deepseek_topic_routing"
    assert routed["12spring24_q02"]["routing_source"] == "deepseek_topic_routing_error"
    assert "missing route record" in routed["12spring24_q02"]["error"]["message"]
    assert manifest["batches"][0]["missing_records"] == 1
    assert manifest["batches"][0]["batch_salvaged"] is True


def test_topic_route_duplicate_returned_question_id_only_fails_duplicate_question(tmp_path: Path) -> None:
    duplicate_payload = (
        '{"records":{'
        f'"12spring24_q01":{json.dumps(_route_record())},'
        f'"12spring24_q01":{json.dumps(_route_record(P1_DIFFERENTIATION))},'
        f'"12spring24_q02":{json.dumps(_route_record())}'
        "}}"
    )
    records = [_record("12spring24_q01"), _record("12spring24_q02")]
    client = _fake_client(_chat_response(duplicate_payload))

    routed, manifest = topic_routing.route_topic_records(
        records,
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    assert routed["12spring24_q01"]["routing_source"] == "deepseek_topic_routing_error"
    assert "duplicate route records returned" in routed["12spring24_q01"]["error"]["message"]
    assert routed["12spring24_q02"]["routing_source"] == "deepseek_topic_routing"
    assert manifest["batches"][0]["duplicate_records"] == 1
    assert manifest["batches"][0]["batch_salvaged"] is True


def test_topic_route_unknown_extra_returned_question_id_does_not_fail_requested_records(tmp_path: Path) -> None:
    records = [_record("12spring24_q01"), _record("12spring24_q02")]
    response = {
        "12spring24_q01": _route_record(),
        "12spring24_q02": _route_record(),
        "unknown_q99": _route_record(evidence_used=["not_requested_evidence"]),
    }
    client = _fake_client(_chat_response(_route_records_payload(response)))

    routed, manifest = topic_routing.route_topic_records(
        records,
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )

    assert set(routed) == {"12spring24_q01", "12spring24_q02"}
    assert all(record["routing_source"] == "deepseek_topic_routing" for record in routed.values())
    assert manifest["batches"][0]["unknown_returned_records"] == 1
    assert manifest["batches"][0]["batch_salvaged"] is False


def test_topic_route_provider_payload_does_not_send_full_question_bank_record(tmp_path: Path) -> None:
    client = _fake_client(_chat_response(_route_payload("12spring24_q01")))

    topic_routing.route_topic_records(
        [_record(notes={"topic_trust_status": "review_required"}, old_ai_field="do not send")],
        client=client,
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "sidecar.json",
        model="deepseek-v4-flash",
    )
    request_payload = json.loads(getattr(client, "calls")[0]["messages"][1]["content"])
    question = request_payload["questions"][0]

    assert set(question) == {
        "question_id",
        "paper_family",
        "paper",
        "question_number",
        "visual_required",
        "evidence",
        "available_evidence_fields",
        "evidence_sources",
        "question_text_source",
        "ocr_text_source",
        "allowed_topics",
    }
    assert "notes" not in json.dumps(request_payload)
    assert "question_image_path" not in json.dumps(request_payload)
    assert "old_ai_field" not in json.dumps(request_payload)
    assert "evidence_packet_hash" not in json.dumps(request_payload)


def test_topic_route_rejects_image_reference_evidence_when_no_image_was_sent() -> None:
    parsed = _parse_result(_route_payload("12spring24_q01", _route_record(evidence_used=["question_text", "image_reference"])))

    route = parsed["records"]["12spring24_q01"]
    assert route["evidence_used"] == ["question_text"]
    assert route["evidence_used_repaired"] is True
    assert route["evidence_used_dropped"] == ["image_reference"]


def test_topic_route_limit_10_writes_exactly_10_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "topic_sidecar.json"
    records = [_record(f"12spring24_q{index:02d}") for index in range(1, 13)]
    input_path.write_text(json.dumps(records), encoding="utf-8")
    response = {
        "records": {
            f"12spring24_q{index:02d}": _route_record()
            for index in range(1, 11)
        }
    }
    client = _fake_client(_chat_response(json.dumps(response)))
    monkeypatch.setattr(topic_routing, "create_client", lambda **_: client)

    exit_code = topic_routing.run_topic_routing(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--limit",
            "10",
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "topic-limit",
            "--no-progress",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["record_count"] == 10
    assert len(payload["records"]) == 10
    assert all(record.get("evidence_packet_hash") for record in payload["records"].values())


def test_topic_route_cli_defaults_to_progress_enabled() -> None:
    parser = argparse.ArgumentParser()
    topic_routing.add_topic_routing_cli_arguments(parser)

    args = parser.parse_args([])

    assert args.progress is True


def test_topic_route_no_progress_disables_terminal_updates_but_writes_status_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "topic_sidecar.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")
    client = _fake_client(_chat_response(_route_payload("12spring24_q01")))
    monkeypatch.setattr(topic_routing, "create_client", lambda **_: client)

    exit_code = topic_routing.run_topic_routing(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "topic-quiet",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    status_dir = tmp_path / "run_status" / "topic-quiet"
    assert exit_code == 0
    assert "[--------------------]" not in captured.err
    assert (status_dir / "run_status.json").exists()
    assert (status_dir / "batch_status.jsonl").exists()
    assert (status_dir / "run_manifest.json").exists()


def test_topic_route_writes_status_files_and_final_summary_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "topic_sidecar.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")
    client = _fake_client(_chat_response(_route_payload("12spring24_q01")))
    monkeypatch.setattr(topic_routing, "create_client", lambda **_: client)

    exit_code = topic_routing.run_topic_routing(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "topic-visible",
        ]
    )

    captured = capsys.readouterr()
    status_dir = tmp_path / "run_status" / "topic-visible"
    status_path = status_dir / "run_status.json"
    batch_path = status_dir / "batch_status.jsonl"
    manifest_path = status_dir / "run_manifest.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    batches = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "[--------------------]" in captured.err
    assert "topic routing" in captured.err
    assert "review 0" in captured.err
    assert "provider_fail 0" in captured.err
    assert status["run_type"] == "topic_routing"
    assert status["status"] == "completed"
    assert status["provider"] == "deepseek"
    assert status["model"] == "deepseek-v4-flash"
    assert status["prompt_version"] == "topic_routing_v1"
    assert status["status_file_path"] == str(status_path)
    assert status["batch_status_path"] == str(batch_path)
    assert status["manifest_path"] == str(manifest_path)
    assert status["checkpoint_path"] == str(status_dir)
    assert status["completed_records"] == 1
    assert status["review_required_records"] == 0
    assert status["provider_failure_records"] == 0
    assert batches[0]["status"] == "completed"
    assert manifest["final_status"] == "completed"
    assert str(output_path) in captured.out
    assert str(status_path) in captured.out
    assert str(batch_path) in captured.out
    assert str(manifest_path) in captured.out


def test_topic_route_full_run_selection_has_no_implicit_25_record_default() -> None:
    records = [_record(f"12spring24_q{index:04d}") for index in range(1, 1302)]

    selected = topic_routing.select_topic_routing_records(records, limit=None)

    assert len(selected) == 1301


def test_topic_route_resume_does_not_preserve_stale_old_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "topic_sidecar.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")
    old_payload = topic_routing.build_topic_routing_sidecar(
        {
            "12spring24_q01": {
                **_route_record(),
                "llm_model": "deepseek-v4-flash",
                "llm_prompt_version": "old_prompt",
            }
        },
        taxonomy_path="exam_bank_taxonomy/canonical",
        taxonomy_version_value=None,
        model="deepseek-v4-flash",
        prompt_version="old_prompt",
    )
    topic_routing.write_topic_routing_sidecar_payload(old_payload, output_path)
    client = _fake_client(_chat_response(_route_payload("12spring24_q01", _route_record(P1_DIFFERENTIATION))))
    monkeypatch.setattr(topic_routing, "create_client", lambda **_: client)

    topic_routing.run_topic_routing(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--resume",
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "topic-resume",
            "--no-progress",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(getattr(client, "calls")) == 1
    assert payload["records"]["12spring24_q01"]["primary_topic_id"] == P1_DIFFERENTIATION
    assert payload["records"]["12spring24_q01"]["llm_prompt_version"] == topic_routing.TOPIC_ROUTING_PROMPT_VERSION


def test_topic_route_resume_preserves_progress_and_status_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "topic_sidecar.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")
    current_payload = topic_routing.build_topic_routing_sidecar(
        {
            "12spring24_q01": {
                **_route_record(),
                "llm_model": "deepseek-v4-flash",
                "llm_prompt_version": topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
                "routing_source": "deepseek_topic_routing",
                "evidence_packet_hash": _evidence_hash(),
            }
        },
        taxonomy_path="exam_bank_taxonomy/canonical",
        taxonomy_version_value=None,
        model="deepseek-v4-flash",
        prompt_version=topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
    )
    topic_routing.write_topic_routing_sidecar_payload(current_payload, output_path)

    def fail_create_client(**_: object) -> object:
        raise AssertionError("fully resumed topic routing should not create a provider client")

    monkeypatch.setattr(topic_routing, "create_client", fail_create_client)

    exit_code = topic_routing.run_topic_routing(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--resume",
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "topic-resume-progress",
        ]
    )

    captured = capsys.readouterr()
    status_dir = tmp_path / "run_status" / "topic-resume-progress"
    status = json.loads((status_dir / "run_status.json").read_text(encoding="utf-8"))
    batches = [json.loads(line) for line in (status_dir / "batch_status.jsonl").read_text(encoding="utf-8").splitlines()]

    assert exit_code == 0
    assert "preserving_resume_records" in captured.err
    assert status["status"] == "completed"
    assert status["total_records"] == 1
    assert status["skipped_records"] == 1
    assert status["completed_records"] == 1
    assert batches[0]["status"] == "skipped"


def test_topic_route_sidecar_contains_generated_at_and_schema_metadata() -> None:
    payload = topic_routing.build_topic_routing_sidecar(
        {"12spring24_q01": _route_record()},
        taxonomy_path="exam_bank_taxonomy/canonical",
        taxonomy_version_value="test-version",
        model="deepseek-v4-flash",
        prompt_version=topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
        generated_at="2026-05-13T00:00:00+00:00",
    )

    assert payload["schema_name"] == topic_routing.TOPIC_ROUTING_SCHEMA_NAME
    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-05-13T00:00:00+00:00"
    assert payload["taxonomy_path"] == "exam_bank_taxonomy/canonical"
    assert payload["taxonomy_version"] == "test-version"
    assert payload["course_contract"]["course_ids"] == ["p1", "p3", "m1", "s1"]
    assert payload["course_contract"]["routing_labels_are_advisory"] is True


def test_topic_route_review_required_outputs_do_not_enter_strict_filters() -> None:
    payload = topic_routing.build_topic_routing_sidecar(
        {
            "strict": _route_record(P1_SERIES),
            "review": _route_record(P1_SERIES, confidence="low", review_required=True),
        },
        taxonomy_path="exam_bank_taxonomy/canonical",
        taxonomy_version_value=None,
        model="deepseek-v4-flash",
        prompt_version=topic_routing.TOPIC_ROUTING_PROMPT_VERSION,
    )

    strict = topic_routing.load_topic_routing_strict_filter_mappings(payload)

    assert set(strict) == {"strict"}
