from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exam_bank import deepseek_enrich


def _record(
    question_id: str = "12spring24_q01",
    paper_family: str = "p1",
    *,
    local_topic: str = "binomial_expansion",
    local_difficulty: str | None = None,
    scope_quality_status: str = "clean",
    text_fidelity_status: str = "clean",
    topic_trust_status: str = "normal",
    validation_status: str = "pass",
) -> dict:
    notes = {
        "scope_quality_status": scope_quality_status,
        "text_fidelity_status": text_fidelity_status,
        "topic_trust_status": topic_trust_status,
        "validation_status": validation_status,
    }
    if local_difficulty is not None:
        notes["difficulty"] = local_difficulty
    return {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": paper_family,
        "question_number": "1",
        "question_text": "Find x.",
        "question_text_role": "readable_text",
        "question_text_trust": "high",
        "visual_required": False,
        "visual_reason_flags": [],
        "question_image_path": "p1/12spring24/questions/q01.png",
        "mark_scheme_image_path": "p1/12spring24/mark_scheme/q01.png",
        "mark_scheme_text": "x = 2",
        "question_solution_marks": 3,
        "topic": local_topic,
        "notes": notes,
    }


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                )
            )
        ]
    )


def test_missing_api_key_fails_clearly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps([_record()]), encoding="utf-8")

    with pytest.raises(deepseek_enrich.StartupConfigurationError, match="DEEPSEEK_API_KEY is required"):
        deepseek_enrich.run(["--input", str(input_path), "--output", str(output_path)])

    assert not output_path.exists()


def test_client_creation_uses_configured_base_url_and_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setattr(deepseek_enrich, "OpenAI", FakeOpenAI)

    deepseek_enrich.create_client(base_url="https://example.deepseek.local")

    assert captured == {
        "api_key": "secret-key",
        "base_url": "https://example.deepseek.local",
    }


def test_valid_json_response_parses_into_sidecar_schema() -> None:
    raw = json.dumps(
        {
            "topic": "trigonometry",
            "subtopic": "identities",
            "difficulty": "medium",
            "confidence": "high",
            "rationale": "Uses a standard identity and one algebraic simplification.",
            "review_required": False,
        }
    )

    parsed = deepseek_enrich.parse_model_json(raw)
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="trigonometry", local_difficulty="average"),
        parsed,
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    assert sidecar["deepseek_topic_raw"] == "trigonometry"
    assert sidecar["deepseek_subtopic_raw"] == "identities"
    assert sidecar["deepseek_difficulty_raw"] == "medium"
    assert sidecar["deepseek_confidence_raw"] == "high"
    assert sidecar["deepseek_confidence_normalized"] == "high"
    assert sidecar["deepseek_rationale_raw"] == "Uses a standard identity and one algebraic simplification."
    assert sidecar["deepseek_review_required_raw"] is False
    assert sidecar["deepseek_topic_normalized"] == "trigonometry"
    assert sidecar["deepseek_difficulty_normalized"] == "average"
    assert sidecar["local_topic"] == "trigonometry"
    assert sidecar["local_difficulty"] == "average"
    assert sidecar["topic_reconciliation_status"] == "match"
    assert sidecar["difficulty_reconciliation_status"] == "match"
    assert sidecar["final_review_required"] is False
    assert sidecar["final_review_reasons"] == []
    assert sidecar["enrichment_mode"] == "text_with_image_reference"
    assert sidecar["image_was_sent_to_model"] is False    
    assert sidecar["image_available"] is True
    assert sidecar["vision_model_required"] is False
    assert sidecar["text_only_enrichment_risk"] == "low"
    assert sidecar["llm_provider"] == "deepseek"
    assert sidecar["llm_model"] == "deepseek-chat"
    assert sidecar["llm_prompt_version"] == "v3"
    assert sidecar["llm_run_timestamp"] == "2026-04-23T00:00:00+00:00"


def test_known_raw_topic_label_normalizes_to_internal_canonical_label() -> None:
    assert deepseek_enrich.normalize_topic_label("Trigonometry", paper_family="p3") == "trigonometry"
    assert (
        deepseek_enrich.normalize_topic_label(
            "Mechanics",
            paper_family="p4",
            raw_subtopic="Connected Particles",
        )
        == "connected_particles"
    )


def test_known_raw_difficulty_label_normalizes_to_internal_canonical_label() -> None:
    assert deepseek_enrich.normalize_difficulty_label("Medium") == "average"
    assert deepseek_enrich.normalize_difficulty_label("Hard") == "difficult"


def test_known_numeric_and_string_confidence_values_normalize_to_internal_buckets() -> None:
    assert deepseek_enrich.normalize_confidence_value(0.91) == "high"
    assert deepseek_enrich.normalize_confidence_value(0.60) == "medium"
    assert deepseek_enrich.normalize_confidence_value(0.20) == "low"
    assert deepseek_enrich.normalize_confidence_value("82%") == "high"
    assert deepseek_enrich.normalize_confidence_value("0.55") == "medium"


def test_unmapped_labels_are_preserved_and_marked_unmapped() -> None:
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="trigonometry", local_difficulty="average", paper_family="p1"),
        {
            "topic": "Mechanics",
            "subtopic": "",
            "difficulty": "Medium",
            "confidence": "high",
            "rationale": "Raw external label does not belong to P1 taxonomy.",
            "review_required": False,
        },
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    assert sidecar["deepseek_topic_raw"] == "Mechanics"
    assert sidecar["deepseek_topic_normalized"] is None
    assert sidecar["topic_reconciliation_status"] == "unmapped_label"
    assert sidecar["final_review_required"] is True
    assert "topic_reconciliation_status:unmapped_label" in sidecar["final_review_reasons"]


def test_degraded_text_forces_final_review_even_when_deepseek_matches() -> None:
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="trigonometry", text_fidelity_status="degraded"),
        {
            "topic": "Trigonometry",
            "subtopic": "general",
            "difficulty": "easy",
            "confidence": "high",
            "rationale": "Clean conceptual match.",
            "review_required": False,
        },
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    assert sidecar["topic_reconciliation_status"] == "match"
    assert sidecar["final_review_required"] is True
    assert "text_fidelity_status:degraded" in sidecar["final_review_reasons"]


def test_enrichment_payload_marks_high_risk_when_vision_required_and_text_untrusted() -> None:
    record = _record(text_fidelity_status="degraded")
    record["question_text"] = "Sketch the graph of y x 3 6 = -."
    record["question_text_role"] = "untrusted_math_text"
    record["question_text_trust"] = "low"
    record["visual_required"] = True
    record["visual_reason_flags"] = ["contains_graph_or_diagram_prompt", "contains_math_text_corruption"]

    payload = deepseek_enrich.build_enrichment_payload(record)

    assert payload["image_available"] is True
    assert payload["vision_model_required"] is True
    assert payload["text_only_enrichment_risk"] == "high"
    assert payload["question_image_path"] == "p1/12spring24/questions/q01.png"
    assert payload["mark_scheme_image_path"] == "p1/12spring24/mark_scheme/q01.png"
    assert payload["question_text_trust"] == "low"


def test_scope_fail_forces_final_review_semantics() -> None:
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="connected_particles", paper_family="p4", scope_quality_status="fail"),
        {
            "topic": "Mechanics",
            "subtopic": "Connected Particles",
            "difficulty": "Medium",
            "confidence": "high",
            "rationale": "Broad label but clear subtopic.",
            "review_required": False,
        },
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    assert sidecar["deepseek_topic_normalized"] == "connected_particles"
    assert sidecar["topic_reconciliation_status"] == "match"
    assert sidecar["final_review_required"] is True
    assert "scope_quality_status:fail" in sidecar["final_review_reasons"]


def test_local_vs_deepseek_match_is_recorded_explicitly() -> None:
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="connected_particles", paper_family="p4"),
        {
            "topic": "Mechanics",
            "subtopic": "Connected Particles",
            "difficulty": "Medium",
            "confidence": "high",
            "rationale": "Topic and subtopic align.",
            "review_required": False,
        },
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    assert sidecar["topic_reconciliation_status"] == "match"
    assert sidecar["final_review_required"] is False


def test_local_vs_deepseek_mismatch_is_recorded_explicitly() -> None:
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="trigonometry", paper_family="p1"),
        {
            "topic": "Integration",
            "subtopic": "Definite Integrals",
            "difficulty": "easy",
            "confidence": "high",
            "rationale": "Different topic from the local label.",
            "review_required": False,
        },
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    assert sidecar["deepseek_topic_normalized"] == "integration"
    assert sidecar["topic_reconciliation_status"] == "mismatch"
    assert sidecar["final_review_required"] is True
    assert "topic_reconciliation_status:mismatch" in sidecar["final_review_reasons"]


def test_malformed_model_output_becomes_per_record_error() -> None:
    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return _chat_response("not valid json")

            completions = _Completions()

        chat = _Chat()

    sidecar = deepseek_enrich.enrich_records([_record()], client=FakeClient(), model="deepseek-chat")
    record = sidecar["12spring24_q01"]

    assert record["error"]["type"] == "parse_error"
    assert "valid JSON" in record["error"]["message"]
    assert record["error"]["raw_provider_output"] == "not valid json"
    assert record["llm_provider"] == "deepseek"
    assert record["llm_model"] == "deepseek-chat"


def test_numeric_confidence_output_is_accepted_and_bucketed() -> None:
    raw = json.dumps(
        {
            "topic": "trigonometry",
            "subtopic": "general",
            "difficulty": "easy",
            "confidence": 0.91,
            "rationale": "Direct trig identity.",
            "review_required": False,
        }
    )

    parsed = deepseek_enrich.parse_model_json(raw)
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="trigonometry"),
        parsed,
        model="deepseek-chat",
        run_timestamp="2026-04-24T00:00:00+00:00",
    )

    assert parsed["confidence"] == 0.91
    assert sidecar["deepseek_confidence_raw"] == 0.91
    assert sidecar["deepseek_confidence_normalized"] == "high"


def test_provider_exception_becomes_per_record_error_without_aborting_batch() -> None:
    responses = iter(
        [
            RuntimeError("temporary provider issue"),
            _chat_response(
                json.dumps(
                    {
                        "topic": "series_and_sequences",
                        "subtopic": "general",
                        "difficulty": "hard",
                        "confidence": "medium",
                        "rationale": "Requires linking AP and GP conditions before solving.",
                        "review_required": True,
                    }
                )
            ),
        ]
    )

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    result = next(responses)
                    if isinstance(result, Exception):
                        raise result
                    return result

            completions = _Completions()

        chat = _Chat()

    sidecar = deepseek_enrich.enrich_records(
        [_record("12spring24_q01"), _record("12spring24_q02")],
        client=FakeClient(),
        model="deepseek-chat",
    )

    assert sidecar["12spring24_q01"]["error"]["type"] == "provider_error"
    assert "temporary provider issue" in sidecar["12spring24_q01"]["error"]["message"]
    assert sidecar["12spring24_q02"]["deepseek_topic"] == "series_and_sequences"
    assert sidecar["12spring24_q02"]["llm_provider"] == "deepseek"


def test_quoted_review_required_is_rejected_as_parse_error_and_logged(tmp_path: Path) -> None:
    failure_log_path = tmp_path / "deepseek.failures.jsonl"

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return _chat_response(
                        json.dumps(
                            {
                                "topic": "binomial_expansion",
                                "subtopic": "general",
                                "difficulty": "easy",
                                "confidence": "medium",
                                "rationale": "Looks straightforward.",
                                "review_required": "false",
                            }
                        )
                    )

            completions = _Completions()

        chat = _Chat()

    sidecar = deepseek_enrich.enrich_records(
        [_record()],
        client=FakeClient(),
        model="deepseek-chat",
        failure_log_path=failure_log_path,
    )

    record = sidecar["12spring24_q01"]
    assert record["error"]["type"] == "parse_error"
    assert "review_required must be a boolean" in record["error"]["message"]
    assert '"review_required": "false"' in record["error"]["raw_provider_output"]

    lines = failure_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    failure_entry = json.loads(lines[0])
    assert failure_entry["question_id"] == "12spring24_q01"
    assert failure_entry["error_type"] == "parse_error"
    assert '"review_required": "false"' in failure_entry["raw_provider_output"]


def test_duplicate_json_keys_are_rejected() -> None:
    raw = (
        '{"topic":"binomial_expansion","topic":"trigonometry","subtopic":"general",'
        '"difficulty":"easy","confidence":"high","rationale":"dup","review_required":false}'
    )

    with pytest.raises(ValueError, match="duplicate key"):
        deepseek_enrich.parse_model_json(raw)


def test_sidecar_file_is_keyed_by_question_id_and_original_input_is_untouched(tmp_path: Path) -> None:
    input_payload = [_record("12spring24_q01"), _record("12spring24_q02")]
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return _chat_response(
                        json.dumps(
                            {
                                "topic": "binomial_expansion",
                                "subtopic": "general",
                                "difficulty": "easy",
                                "confidence": "high",
                                "rationale": "Direct expansion with a standard coefficient read-off.",
                                "review_required": False,
                            }
                        )
                    )

            completions = _Completions()

        chat = _Chat()

    records = deepseek_enrich.load_question_bank(input_path)
    selected = deepseek_enrich.select_records(records, limit=1)
    sidecar = deepseek_enrich.enrich_records(selected, client=FakeClient(), model="deepseek-chat")
    deepseek_enrich.write_sidecar(sidecar, output_path)

    written_sidecar = json.loads(output_path.read_text(encoding="utf-8"))
    original_after = json.loads(input_path.read_text(encoding="utf-8"))

    assert written_sidecar["schema_name"] == deepseek_enrich.DEEPSEEK_SIDECAR_SCHEMA_NAME
    assert written_sidecar["schema_version"] == deepseek_enrich.DEEPSEEK_SIDECAR_SCHEMA_VERSION
    assert written_sidecar["record_count"] == 1
    assert list(written_sidecar["enrichments"]) == ["12spring24_q01"]
    enrichment = written_sidecar["enrichments"]["12spring24_q01"]
    assert enrichment["llm_provider"] == "deepseek"
    assert enrichment["llm_model"] == "deepseek-chat"
    assert enrichment["llm_prompt_version"] == "v3"
    assert "llm_run_timestamp" in enrichment
    assert enrichment["deepseek_topic_raw"] == "binomial_expansion"
    assert enrichment["deepseek_topic_normalized"] == "binomial_expansion"
    assert enrichment["deepseek_difficulty_raw"] == "easy"
    assert enrichment["deepseek_difficulty_normalized"] == "easy"
    assert enrichment["deepseek_confidence_raw"] == "high"
    assert enrichment["deepseek_confidence_normalized"] == "high"
    assert enrichment["topic_reconciliation_status"] == "match"
    assert enrichment["difficulty_reconciliation_status"] == "no_local_label"
    assert "final_review_required" in enrichment
    assert "final_review_reasons" in enrichment
    assert original_after == input_payload


def test_load_question_bank_accepts_versioned_question_bank_document(tmp_path: Path) -> None:
    input_path = tmp_path / "question_bank.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.question_bank",
                "schema_version": 1,
                "record_count": 1,
                "questions": [_record("12spring24_q01")],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert deepseek_enrich.load_question_bank(input_path) == [_record("12spring24_q01")]


def test_deepseek_sidecar_export_contract_includes_success_and_error_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "question_bank.deepseek.json"
    success = deepseek_enrich.build_sidecar_success(
        _record(local_topic="trigonometry", local_difficulty="average"),
        {
            "topic": "trigonometry",
            "subtopic": "identities",
            "difficulty": "medium",
            "confidence": "high",
            "rationale": "Uses a standard identity.",
            "review_required": False,
        },
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )
    error = deepseek_enrich.build_sidecar_error(
        error_type="provider_error",
        message="temporary provider issue",
        model="deepseek-chat",
        run_timestamp="2026-04-23T00:00:00+00:00",
    )

    deepseek_enrich.write_sidecar({"12spring24_q01": success, "12spring24_q02": error}, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(payload) == {"schema_name", "schema_version", "record_count", "enrichments"}
    assert payload["schema_name"] == deepseek_enrich.DEEPSEEK_SIDECAR_SCHEMA_NAME
    assert payload["schema_version"] == deepseek_enrich.DEEPSEEK_SIDECAR_SCHEMA_VERSION
    assert payload["record_count"] == 2

    success_record = payload["enrichments"]["12spring24_q01"]
    assert {
        "deepseek_topic",
        "deepseek_subtopic",
        "deepseek_difficulty",
        "deepseek_confidence",
        "deepseek_confidence_normalized",
        "deepseek_rationale",
        "deepseek_review_required",
        "deepseek_topic_raw",
        "deepseek_subtopic_raw",
        "deepseek_difficulty_raw",
        "deepseek_confidence_raw",
        "deepseek_rationale_raw",
        "deepseek_review_required_raw",
        "deepseek_topic_normalized",
        "deepseek_difficulty_normalized",
        "local_topic",
        "local_difficulty",
        "topic_reconciliation_status",
        "difficulty_reconciliation_status",
        "final_review_required",
        "final_review_reasons",
        "enrichment_mode",
        "image_available",
        "question_image_path",
        "mark_scheme_image_path",
        "vision_model_required",
        "question_text_role",
        "question_text_trust",
        "text_only_enrichment_risk",
        "llm_provider",
        "llm_model",
        "llm_prompt_version",
        "llm_run_timestamp",
    }.issubset(success_record)

    error_record = payload["enrichments"]["12spring24_q02"]
    assert {"error", "llm_provider", "llm_model", "llm_prompt_version", "llm_run_timestamp"}.issubset(error_record)
    assert {"type", "message"}.issubset(error_record["error"])


def test_dry_run_skips_external_calls_and_output_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps([_record("12spring24_q01"), _record("12spring24_q02")]), encoding="utf-8")

    exit_code = deepseek_enrich.run(
        ["--input", str(input_path), "--output", str(output_path), "--limit", "1", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "12spring24_q01" in captured.out
    assert not output_path.exists()


def test_cli_success_exits_zero_and_writes_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return _chat_response(
                        json.dumps(
                            {
                                "topic": "binomial_expansion",
                                "subtopic": "general",
                                "difficulty": "easy",
                                "confidence": "high",
                                "rationale": "Direct expansion.",
                                "review_required": False,
                            }
                        )
                    )

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: FakeClient())

    exit_code = deepseek_enrich.run(["--input", str(input_path), "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["record_count"] == 1
    assert "error" not in payload["enrichments"]["12spring24_q01"]


def test_cli_partial_provider_failure_preserves_sidecar_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps([_record("12spring24_q01"), _record("12spring24_q02")]), encoding="utf-8")
    responses = iter(
        [
            RuntimeError("temporary provider issue"),
            _chat_response(
                json.dumps(
                    {
                        "topic": "series_and_sequences",
                        "subtopic": "general",
                        "difficulty": "hard",
                        "confidence": "medium",
                        "rationale": "Requires linking AP and GP facts.",
                        "review_required": True,
                    }
                )
            ),
        ]
    )

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    result = next(responses)
                    if isinstance(result, Exception):
                        raise result
                    return result

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: FakeClient())

    exit_code = deepseek_enrich.run(["--input", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["record_count"] == 2
    assert payload["enrichments"]["12spring24_q01"]["error"]["type"] == "provider_error"
    assert payload["enrichments"]["12spring24_q02"]["deepseek_topic"] == "series_and_sequences"
    assert "1 successes and 1 failures (1 provider/API failures)" in captured.out


def test_cli_total_provider_failure_preserves_sidecar_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps([_record("12spring24_q01"), _record("12spring24_q02")]), encoding="utf-8")

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    raise RuntimeError("provider unavailable")

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: FakeClient())

    exit_code = deepseek_enrich.run(["--input", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["record_count"] == 2
    assert all(record["error"]["type"] == "provider_error" for record in payload["enrichments"].values())
    assert "All attempted DeepSeek enrichments failed with provider/API errors" in captured.out

def test_numeric_difficulty_output_is_accepted_and_bucketed() -> None:
    raw = json.dumps(
        {
            "topic": "momentum_impulse",
            "subtopic": "conservation of momentum",
            "difficulty": 1,
            "confidence": 0.95,
            "rationale": "Direct conservation of momentum.",
            "review_required": False,
        }
    )

    parsed = deepseek_enrich.parse_model_json(raw)
    sidecar = deepseek_enrich.build_sidecar_success(
        _record(local_topic="momentum_impulse", paper_family="p4"),
        parsed,
        model="deepseek-v4-flash",
        run_timestamp="2026-04-24T00:00:00+00:00",
    )

    assert parsed["difficulty"] == 1
    assert sidecar["deepseek_difficulty_raw"] == 1
    assert sidecar["deepseek_difficulty_normalized"] == "easy"

def test_mechanics_topic_aliases_normalize_when_valid_for_family() -> None:
    assert (
        deepseek_enrich.normalize_topic_label("Work, Energy and Power", paper_family="p4")
        == "power_and_resistance"
    )
    assert (
        deepseek_enrich.normalize_topic_label("forces and equilibrium", paper_family="p4")
        == "equilibrium_particle"
    )

def test_cli_allow_provider_failure_keeps_total_failure_soft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.deepseek.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    raise RuntimeError("provider unavailable")

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: FakeClient())

    exit_code = deepseek_enrich.run(
        ["--input", str(input_path), "--output", str(output_path), "--allow-provider-failure"]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["enrichments"]["12spring24_q01"]["error"]["type"] == "provider_error"
    assert "All attempted DeepSeek enrichments failed" not in captured.out
