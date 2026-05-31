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
    )["records"][question_id]


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
    with pytest.raises(topic_routing.TopicRouteValidationError, match="total exactly 100"):
        _parse(
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


def test_topic_route_rejects_invented_topic_ids() -> None:
    with pytest.raises(topic_routing.TopicRouteValidationError) as excinfo:
        _parse(_route_payload("12spring24_q01", _route_record("invented_topic")))

    assert getattr(excinfo.value, "error_type") == topic_routing.AI_FAILURE_TAXONOMY_VALIDATION_ERROR


def test_topic_route_rejects_topic_ids_outside_paper_family() -> None:
    with pytest.raises(topic_routing.TopicRouteValidationError) as excinfo:
        _parse(_route_payload("12spring24_q01", _route_record(P3_COMPLEX)))

    assert getattr(excinfo.value, "error_type") == "topic_outside_allowed_paper_family"


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
        "allowed_topics",
    }
    assert "notes" not in json.dumps(request_payload)
    assert "question_image_path" not in json.dumps(request_payload)
    assert "old_ai_field" not in json.dumps(request_payload)


def test_topic_route_rejects_image_reference_evidence_when_no_image_was_sent() -> None:
    with pytest.raises(topic_routing.TopicRouteValidationError, match="not supplied"):
        _parse(_route_payload("12spring24_q01", _route_record(evidence_used=["question_text", "image_reference"])))


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
