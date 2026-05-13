from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exam_bank import deepseek_enrich
from exam_bank.run_status import RunStatusTracker


REQUIRED_RUN_STATUS_KEYS = {
    "run_id",
    "run_type",
    "command",
    "started_at",
    "updated_at",
    "finished_at",
    "status",
    "current_phase",
    "current_batch_id",
    "current_paper",
    "current_paper_family",
    "completed_batches",
    "total_batches",
    "completed_records",
    "total_records",
    "percent_complete",
    "elapsed_seconds",
    "estimated_remaining_seconds",
    "successful_records",
    "failed_records",
    "skipped_records",
    "retry_count",
    "output_path",
    "error_summary",
}


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


def _reasoning_response(reasoning_content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    reasoning_content=reasoning_content,
                )
            )
        ]
    )


def _taxonomy() -> deepseek_enrich.CanonicalTaxonomy:
    return deepseek_enrich.load_canonical_taxonomy("exam_bank_taxonomy/canonical", "p1")


def _ai_item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "question_id": "12spring24_q01",
        "subpart_id": None,
        "paper_family": "p1",
        "primary_topic_id": "9709_p1_topic_series",
        "secondary_topic_ids": [],
        "subtopic_ids": ["9709_p1_subtopic_series_binomial_positive_integer"],
        "skill_ids": ["9709_p1_series_binomial_positive_integer"],
        "method_families": ["binomial expansion"],
        "prerequisite_skill_ids": [],
        "exam_techniques": ["use ascending powers"],
        "common_mistakes": ["dropping powers of x"],
        "worked_example_seed": "Use binomial expansion and collect the requested coefficient.",
        "warmup_seed": "Expand (1 + x)^4 up to the x^2 term.",
        "strict_filter_candidate": True,
        "strict_filter_reason": "Direct assessed binomial expansion with clear evidence.",
        "evidence_used": ["question_text", "mark_scheme_text"],
        "evidence_missing": [],
        "confidence": 0.9,
        "review_required": False,
        "review_reasons": [],
        "ai_difficulty_estimate": "easy",
        "ai_difficulty_score": 0.2,
        "ai_difficulty_factors": ["single method family"],
        "needs_new_subtopic_candidate": False,
        "suggested_new_subtopic": None,
        "needs_new_skill_candidate": False,
        "suggested_new_skill": None,
        "mapping_source": "deepseek_ai_assisted",
        "reviewed_status": "machine_candidate",
    }
    item.update(overrides)
    return item


def _ai_payload(*items: dict[str, object]) -> str:
    return json.dumps({"items": list(items)})


def _fake_ai_client(*responses: object) -> object:
    calls: list[dict[str, object]] = []
    response_iter = iter(responses)

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**kwargs: object) -> SimpleNamespace:
                    calls.append(kwargs)
                    result = next(response_iter)
                    if isinstance(result, Exception):
                        raise result
                    return result

            completions = _Completions()

        chat = _Chat()

    client = FakeClient()
    setattr(client, "calls", calls)
    return client


def _ai_sidecar_record(
    question_id: str,
    *,
    prompt_version: str = "ai_assisted_v2",
    run_timestamp: str = "2026-05-12T00:00:00+00:00",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "paper": "12spring24",
        "paper_family": "p1",
        "ai_assisted_schema_version": 2,
        "ai_assisted_items": [_ai_item(question_id=question_id)],
        "strict_filter_candidates": [{"subpart_id": None, "skill_ids": ["9709_p1_series_binomial_positive_integer"]}],
        "mapping_source": "deepseek_ai_assisted",
        "reviewed_status": "machine_candidate",
        "llm_provider": "deepseek",
        "llm_model": "deepseek-v4-flash",
        "llm_prompt_version": prompt_version,
        "llm_run_timestamp": run_timestamp,
    }


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


def test_prompt_requests_numeric_difficulty_score_with_cohort_scale() -> None:
    payload = deepseek_enrich.build_enrichment_payload(_record())
    system_message = deepseek_enrich.build_messages(payload)[0]

    assert "difficulty must be a JSON number from 0 to 100" in system_message["content"]
    assert "100 representative CAIE 9709 secondary students attempted this item" in system_message["content"]
    assert "available marks not received" in system_message["content"]
    assert "not a string label" in system_message["content"]


def test_valid_json_response_parses_into_sidecar_schema() -> None:
    raw = json.dumps(
        {
            "topic": "trigonometry",
            "subtopic": "identities",
            "difficulty": 50,
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
    assert sidecar["deepseek_difficulty_raw"] == 50
    assert sidecar["deepseek_difficulty_score"] == 50
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
    assert sidecar["llm_prompt_version"] == "v4"
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
    assert deepseek_enrich.normalize_difficulty_label(20) == "easy"
    assert deepseek_enrich.normalize_difficulty_label(50) == "average"
    assert deepseek_enrich.normalize_difficulty_label(80) == "difficult"
    assert deepseek_enrich.normalize_difficulty_label("2") == "average"
    assert deepseek_enrich.normalize_difficulty_label("50%") == "average"


def test_model_difficulty_must_be_numeric_score() -> None:
    raw = json.dumps(
        {
            "topic": "trigonometry",
            "subtopic": "identities",
            "difficulty": "average",
            "confidence": "high",
            "rationale": "String labels are no longer accepted for model difficulty output.",
            "review_required": False,
        }
    )

    with pytest.raises(ValueError, match="difficulty must be a numeric score"):
        deepseek_enrich.parse_model_json(raw)


def test_model_difficulty_score_must_be_in_range() -> None:
    raw = json.dumps(
        {
            "topic": "trigonometry",
            "subtopic": "identities",
            "difficulty": 101,
            "confidence": "high",
            "rationale": "Out of range scores should not be accepted.",
            "review_required": False,
        }
    )

    with pytest.raises(ValueError, match="difficulty must be a numeric score"):
        deepseek_enrich.parse_model_json(raw)


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
            "difficulty": 50,
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
            "difficulty": 20,
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
            "difficulty": 50,
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
            "difficulty": 50,
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
            "difficulty": 20,
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
            "difficulty": 20,
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
                        "difficulty": 80,
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
                                "difficulty": 20,
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
        '"difficulty":20,"confidence":"high","rationale":"dup","review_required":false}'
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
                                "difficulty": 20,
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
    assert enrichment["llm_prompt_version"] == "v4"
    assert "llm_run_timestamp" in enrichment
    assert enrichment["deepseek_topic_raw"] == "binomial_expansion"
    assert enrichment["deepseek_topic_normalized"] == "binomial_expansion"
    assert enrichment["deepseek_difficulty_raw"] == 20
    assert enrichment["deepseek_difficulty_score"] == 20
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
            "difficulty": 50,
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
        "deepseek_difficulty_score",
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
        "local_difficulty_score",
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
                                "difficulty": 20,
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
                        "difficulty": 80,
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
    assert sidecar["deepseek_difficulty_score"] == 1
    assert sidecar["deepseek_difficulty_normalized"] == "easy"


def test_local_difficulty_score_zero_is_preserved_for_reconciliation() -> None:
    record = _record(local_topic="trigonometry", local_difficulty=None)
    record["difficulty_score"] = 0

    sidecar = deepseek_enrich.build_sidecar_success(
        record,
        {
            "topic": "trigonometry",
            "subtopic": "general",
            "difficulty": 0,
            "confidence": "high",
            "rationale": "Very routine.",
            "review_required": False,
        },
        model="deepseek-v4-flash",
        run_timestamp="2026-04-24T00:00:00+00:00",
    )

    assert sidecar["local_difficulty_score"] == 0
    assert sidecar["local_difficulty"] == "easy"
    assert sidecar["difficulty_reconciliation_status"] == "match"


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


def test_existing_sidecar_v1_is_still_readable(tmp_path: Path) -> None:
    path = tmp_path / "question_bank.deepseek.full.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.deepseek_sidecar",
                "schema_version": 1,
                "record_count": 1,
                "enrichments": {"12spring24_q01": {"deepseek_topic": "binomial expansion"}},
            }
        ),
        encoding="utf-8",
    )

    loaded = deepseek_enrich.load_existing_sidecar(path)

    assert loaded["12spring24_q01"]["deepseek_topic"] == "binomial expansion"


def test_ai_assisted_record_preserves_v1_fields_where_appropriate() -> None:
    taxonomy = _taxonomy()
    existing = {
        "deepseek_topic": "binomial expansion",
        "deepseek_difficulty_score": 25,
        "topic_reconciliation_status": "match",
    }
    item = deepseek_enrich.validate_ai_assisted_item(_ai_item(), taxonomy=taxonomy)

    merged = deepseek_enrich.build_ai_assisted_record(
        _record(),
        items=[item],
        existing_enrichment=existing,
        model="deepseek-v4-flash",
        prompt_version="ai_assisted_v2",
        run_timestamp="2026-05-11T00:00:00+00:00",
        batch_id="p1_12spring24",
        batch_input_hash="abc",
        taxonomy=taxonomy,
    )

    assert merged["deepseek_topic"] == "binomial expansion"
    assert merged["deepseek_difficulty_score"] == 25
    assert merged["ai_assisted_items"][0]["primary_topic_id"] == "9709_p1_topic_series"
    assert merged["strict_filter_candidates"][0]["skill_ids"] == ["9709_p1_series_binomial_positive_integer"]


def test_ai_assisted_validation_enforces_unknown_topic_subtopic_and_skill_ids() -> None:
    taxonomy = _taxonomy()

    with pytest.raises(ValueError, match="primary_topic_id"):
        deepseek_enrich.validate_ai_assisted_item(_ai_item(primary_topic_id="invented_topic"), taxonomy=taxonomy)

    with pytest.raises(ValueError, match="subtopic_ids"):
        deepseek_enrich.validate_ai_assisted_item(_ai_item(subtopic_ids=["invented_subtopic"]), taxonomy=taxonomy)

    with pytest.raises(ValueError, match="skill_ids"):
        deepseek_enrich.validate_ai_assisted_item(_ai_item(skill_ids=["invented_skill"]), taxonomy=taxonomy)


def test_ai_assisted_subpart_id_validation_handles_parent_empty_valid_and_invalid_subparts() -> None:
    taxonomy = _taxonomy()
    parent = deepseek_enrich.validate_ai_assisted_item(
        _ai_item(subpart_id=None),
        taxonomy=taxonomy,
        expected_question_ids={"12spring24_q01"},
        expected_subpart_ids=set(),
        expected_subpart_ids_by_question={"12spring24_q01": set()},
    )
    assert parent["subpart_id"] is None

    normalized_empty_parent = deepseek_enrich.validate_ai_assisted_item(
        _ai_item(subpart_id=""),
        taxonomy=taxonomy,
        expected_question_ids={"12spring24_q01"},
        expected_subpart_ids=set(),
        expected_subpart_ids_by_question={"12spring24_q01": set()},
    )
    assert normalized_empty_parent["subpart_id"] is None

    subpart = deepseek_enrich.validate_ai_assisted_item(
        _ai_item(subpart_id="12spring24_q01_a"),
        taxonomy=taxonomy,
        expected_question_ids={"12spring24_q01"},
        expected_subpart_ids={"12spring24_q01_a"},
        expected_subpart_ids_by_question={"12spring24_q01": {"12spring24_q01_a"}},
    )
    assert subpart["subpart_id"] == "12spring24_q01_a"

    with pytest.raises(ValueError) as empty_exc:
        deepseek_enrich.validate_ai_assisted_item(
            _ai_item(subpart_id=""),
            taxonomy=taxonomy,
            expected_question_ids={"12spring24_q01"},
            expected_subpart_ids={"12spring24_q01_a"},
            expected_subpart_ids_by_question={"12spring24_q01": {"12spring24_q01_a"}},
        )
    assert getattr(empty_exc.value, "error_type") == deepseek_enrich.AI_FAILURE_INVALID_SUBPART_ID_EMPTY_STRING

    with pytest.raises(ValueError) as invalid_exc:
        deepseek_enrich.validate_ai_assisted_item(
            _ai_item(subpart_id="a"),
            taxonomy=taxonomy,
            expected_question_ids={"12spring24_q01"},
            expected_subpart_ids={"12spring24_q01_a"},
            expected_subpart_ids_by_question={"12spring24_q01": {"12spring24_q01_a"}},
        )
    assert getattr(invalid_exc.value, "error_type") == deepseek_enrich.AI_FAILURE_UNEXPECTED_SUBPART_ID


def test_suggested_new_subtopics_and_skills_are_review_only() -> None:
    taxonomy = _taxonomy()
    item = deepseek_enrich.validate_ai_assisted_item(
        _ai_item(
            strict_filter_candidate=False,
            needs_new_subtopic_candidate=True,
            suggested_new_subtopic="binomial product coefficient extraction",
            needs_new_skill_candidate=True,
            suggested_new_skill="multiply two binomial expansions and collect x^2",
            review_required=True,
            review_reasons=["taxonomy_gap_suggestion"],
        ),
        taxonomy=taxonomy,
    )

    assert item["suggested_new_subtopic"] == "binomial product coefficient extraction"
    assert item["suggested_new_skill"] == "multiply two binomial expansions and collect x^2"
    assert item["subtopic_ids"] == ["9709_p1_subtopic_series_binomial_positive_integer"]
    assert item["skill_ids"] == ["9709_p1_series_binomial_positive_integer"]
    assert item["strict_filter_candidate"] is False


def test_strict_filter_candidate_fails_closed_for_low_confidence_prerequisite_and_review_required() -> None:
    taxonomy = _taxonomy()

    low_confidence = deepseek_enrich.validate_ai_assisted_item(_ai_item(confidence=0.3), taxonomy=taxonomy)
    assert low_confidence["strict_filter_candidate"] is False
    assert "confidence_below_strict_filter_threshold" in low_confidence["review_reasons"]

    prerequisite_only = deepseek_enrich.validate_ai_assisted_item(
        _ai_item(
            skill_ids=[],
            prerequisite_skill_ids=["9709_p1_series_binomial_positive_integer"],
            strict_filter_reason="Prerequisite only algebra support.",
        ),
        taxonomy=taxonomy,
    )
    assert prerequisite_only["strict_filter_candidate"] is False
    assert "prerequisite_only_mapping" in prerequisite_only["review_reasons"]

    review_required = deepseek_enrich.validate_ai_assisted_item(
        _ai_item(review_required=True, review_reasons=["visual_confirmation_needed"]),
        taxonomy=taxonomy,
    )
    assert review_required["strict_filter_candidate"] is False
    assert "review_required" in review_required["review_reasons"]


def test_human_reviewed_records_are_never_overwritten() -> None:
    taxonomy = _taxonomy()
    existing = {"reviewed_status": "reviewed", "primary_topic_id": "human_choice"}
    item = deepseek_enrich.validate_ai_assisted_item(_ai_item(), taxonomy=taxonomy)

    merged = deepseek_enrich.build_ai_assisted_record(
        _record(),
        items=[item],
        existing_enrichment=existing,
        model="deepseek-v4-flash",
        prompt_version="ai_assisted_v2",
        run_timestamp="2026-05-11T00:00:00+00:00",
        batch_id="p1_12spring24",
        batch_input_hash="abc",
        taxonomy=taxonomy,
    )

    assert merged["primary_topic_id"] == "human_choice"
    assert merged["ai_assisted_preserved_human_review"] is True
    assert "ai_assisted_items" not in merged


def test_ai_assisted_error_record_clears_old_strict_filter_candidates() -> None:
    record = deepseek_enrich.build_ai_assisted_error_record(
        _record(),
        existing_enrichment={
            "strict_filter_candidates": [{"subpart_id": None, "skill_ids": ["old_skill"]}],
            "ai_assisted_items": [_ai_item()],
        },
        error_type=deepseek_enrich.AI_FAILURE_INVALID_JSON,
        message="bad json",
        model="deepseek-v4-flash",
        prompt_version="ai_assisted_v2",
        run_timestamp="2026-05-13T00:00:00+00:00",
        batch_id="p1_12spring24",
        batch_input_hash="abc",
    )

    assert record["ai_assisted_items"] == []
    assert record["strict_filter_candidates"] == []


def test_parse_recovery_extracts_json_wrapped_in_final_content_but_rejects_reasoning_only() -> None:
    taxonomy = _taxonomy()
    parsed, metadata = deepseek_enrich.parse_response_with_recovery(
        _chat_response("provider prefix " + _ai_payload(_ai_item()) + " provider suffix"),
        lambda raw: deepseek_enrich.parse_ai_assisted_model_json(raw, taxonomy=taxonomy),
    )

    assert parsed["items"][0]["skill_ids"] == ["9709_p1_series_binomial_positive_integer"]
    assert metadata["parse_recovered"] is True
    assert metadata["parse_recovery_source"] == "message.content"

    with pytest.raises(ValueError) as excinfo:
        deepseek_enrich.parse_response_with_recovery(
            _reasoning_response(_ai_payload(_ai_item())),
            lambda raw: deepseek_enrich.parse_ai_assisted_model_json(raw, taxonomy=taxonomy),
        )
    assert getattr(excinfo.value, "error_type") == deepseek_enrich.AI_FAILURE_REASONING_CONTENT_ONLY

    with pytest.raises(ValueError) as empty_exc:
        deepseek_enrich.parse_response_with_recovery(
            _chat_response(""),
            lambda raw: deepseek_enrich.parse_ai_assisted_model_json(raw, taxonomy=taxonomy),
        )
    assert getattr(empty_exc.value, "error_type") == deepseek_enrich.AI_FAILURE_EMPTY_CONTENT


def test_ai_assisted_invalid_json_is_classified() -> None:
    taxonomy = _taxonomy()

    with pytest.raises(ValueError) as excinfo:
        deepseek_enrich.parse_ai_assisted_model_json("{not valid json", taxonomy=taxonomy)

    assert getattr(excinfo.value, "error_type") == deepseek_enrich.AI_FAILURE_INVALID_JSON


def test_ai_assisted_accepts_deepseek_smoke_shape_with_null_strict_filter_reason() -> None:
    taxonomy = _taxonomy()
    raw = _ai_payload(
        _ai_item(
            question_id="11autumn21_q01",
            ai_difficulty_factors="Binomial expansion with fractional term",
            common_mistakes="Sign errors",
            exam_techniques="Apply binomial expansion formula carefully",
            evidence_missing=None,
            strict_filter_candidate=True,
            strict_filter_reason=None,
            review_required=False,
            review_reasons=[],
            confidence=0.85,
        )
    )

    parsed = deepseek_enrich.parse_ai_assisted_model_json(
        raw,
        taxonomy=taxonomy,
        expected_records=[_record("11autumn21_q01")],
    )

    item = parsed["items"][0]
    assert item["question_id"] == "11autumn21_q01"
    assert item["subpart_id"] is None
    assert item["ai_difficulty_factors"] == ["Binomial expansion with fractional term"]
    assert item["common_mistakes"] == ["Sign errors"]
    assert item["exam_techniques"] == ["Apply binomial expansion formula carefully"]
    assert item["evidence_missing"] == []
    assert item["strict_filter_candidate"] is True
    assert item["strict_filter_reason"] == "Direct assessed evidence maps to validated canonical topic, subtopic, and skill IDs."


def test_ai_assisted_schema_error_includes_item_index_and_field_path() -> None:
    taxonomy = _taxonomy()

    with pytest.raises(ValueError) as excinfo:
        deepseek_enrich.parse_ai_assisted_model_json(
            _ai_payload(_ai_item(confidence="high")),
            taxonomy=taxonomy,
            expected_records=[_record()],
        )

    assert getattr(excinfo.value, "error_type") == deepseek_enrich.AI_FAILURE_SCHEMA_VALIDATION_ERROR
    assert str(excinfo.value) == "items[0].confidence must be a JSON number from 0 to 1."


def test_provider_errors_and_old_prompt_versions_are_resumable() -> None:
    records = [_record("12spring24_q01"), _record("12spring24_q02"), _record("12spring24_q03")]
    existing = {
        "12spring24_q01": {"error": {"type": "provider_error"}, "llm_prompt_version": "ai_assisted_v2"},
        "12spring24_q02": {"ai_assisted_items": [], "llm_prompt_version": "old_prompt"},
        "12spring24_q03": {"ai_assisted_items": [], "llm_prompt_version": "ai_assisted_v2"},
    }

    selected = deepseek_enrich.select_ai_assisted_records(
        records,
        existing_sidecar=existing,
        resume=True,
        prompt_version="ai_assisted_v2",
    )

    assert [record["question_id"] for record in selected] == ["12spring24_q01", "12spring24_q02"]


def test_ai_assisted_dry_run_does_not_call_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")

    def fail_create_client(**_: object) -> object:
        raise AssertionError("dry-run should not create a network client")

    monkeypatch.setattr(deepseek_enrich, "create_client", fail_create_client)

    exit_code = deepseek_enrich.run_ai_assisted(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--dry-run",
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "ai-dry-run",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    assert not output_path.exists()
    status = json.loads((tmp_path / "run_status" / "ai-dry-run" / "run_status.json").read_text(encoding="utf-8"))
    assert REQUIRED_RUN_STATUS_KEYS <= set(status)
    assert status["run_type"] == "ai_enrichment"
    assert status["status"] == "completed"
    assert status["skipped_records"] == 1


def test_ai_assisted_prompt_includes_allowed_ids_and_no_invention_instruction() -> None:
    taxonomy = _taxonomy()
    record = {**_record(), "subparts": ["a", "b"]}
    payload = deepseek_enrich.build_ai_assisted_batch_payload([record], taxonomy=taxonomy, include_subparts=True)
    messages = deepseek_enrich.build_ai_assisted_messages(payload)

    assert "Do not invent canonical topic IDs" in messages[0]["content"]
    assert "Do not invent subpart IDs" in messages[0]["content"]
    assert "Return exactly one valid JSON object" in messages[0]["content"]
    assert "Do not include markdown" in messages[0]["content"]
    assert "exactly one parent/top-level object per input question" in messages[0]["content"]
    assert "Do not output subpart-level items in this pass" in messages[0]["content"]
    assert "allowed_subpart_ids" in messages[1]["content"]
    assert "12spring24_q01_a" in messages[1]["content"]
    assert "9709_p1_topic_series" in messages[1]["content"]
    assert "9709_p1_subtopic_series_binomial_positive_integer" in messages[1]["content"]
    assert "9709_p1_series_binomial_positive_integer" in messages[1]["content"]


def test_paper_batched_outputs_merge_deterministically_and_failed_batches_do_not_corrupt_successes(tmp_path: Path) -> None:
    responses = iter(
        [
            _chat_response(_ai_payload(_ai_item(question_id="12spring24_q01"))),
            RuntimeError("temporary provider issue"),
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

    records = [
        _record("12spring24_q01"),
        {**_record("13spring24_q01"), "paper": "13spring24"},
    ]
    tracker = RunStatusTracker(
        run_id="ai-partial",
        run_type="ai_enrichment",
        status_root=tmp_path / "run_status",
        command="enrich-ai",
        progress=False,
    )
    tracker.start(phase="preparing_batches")

    enrichments, manifest = deepseek_enrich.enrich_ai_assisted_records(
        records,
        client=FakeClient(),
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "question_bank.ai_assisted.v2.json",
        model="deepseek-v4-flash",
        include_subparts=True,
        batch_by_paper=True,
        progress=tracker,
    )
    tracker.finish("completed")

    assert manifest["batch_count"] == 2
    assert enrichments["12spring24_q01"]["ai_assisted_items"][0]["question_id"] == "12spring24_q01"
    assert enrichments["13spring24_q01"]["error"]["type"] == deepseek_enrich.AI_FAILURE_PROVIDER_API_ERROR
    assert enrichments["12spring24_q01"]["batch_id"] != enrichments["13spring24_q01"]["batch_id"]
    batch_statuses = [
        json.loads(line)["status"]
        for line in tracker.batch_status_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert batch_statuses == ["completed", "failed"]


def test_invalid_multi_record_ai_batch_retries_individual_records(tmp_path: Path) -> None:
    responses = iter(
        [
            _chat_response('{"items": ['),
            _chat_response(_ai_payload(_ai_item(question_id="12spring24_q01", strict_filter_reason=None))),
            _chat_response(_ai_payload(_ai_item(question_id="12spring24_q02", strict_filter_reason=None))),
        ]
    )

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return next(responses)

            completions = _Completions()

        chat = _Chat()

    tracker = RunStatusTracker(
        run_id="ai-individual-retry",
        run_type="ai_enrichment",
        status_root=tmp_path / "run_status",
        command="enrich-ai",
        progress=False,
    )
    tracker.start(phase="preparing_batches")
    records = [_record("12spring24_q01"), _record("12spring24_q02")]

    enrichments, manifest = deepseek_enrich.enrich_ai_assisted_records(
        records,
        client=FakeClient(),
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=tmp_path / "question_bank.ai_assisted.v2.json",
        model="deepseek-v4-flash",
        batch_by_paper=True,
        progress=tracker,
    )
    tracker.finish("completed")

    assert manifest["batches"][0]["status"] == "success_individual_retry"
    assert enrichments["12spring24_q01"]["ai_assisted_items"]
    assert enrichments["12spring24_q02"]["ai_assisted_items"]
    status = json.loads(tracker.run_status_path.read_text(encoding="utf-8"))
    assert status["successful_records"] == 2
    assert status["failed_records"] == 0


def test_final_output_includes_metadata_batch_ids_hashes_model_prompt_and_calibrated_difficulty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    records = [
        {**_record("12spring24_q01"), "question_solution_marks": 2},
        {**_record("12spring24_q02"), "question_solution_marks": 10},
    ]
    input_path.write_text(json.dumps(records), encoding="utf-8")
    responses = iter(
        [
            _chat_response(
                _ai_payload(
                    _ai_item(question_id="12spring24_q01", ai_difficulty_score=0.1),
                    _ai_item(question_id="12spring24_q02", ai_difficulty_score=0.95, ai_difficulty_estimate="difficult"),
                )
            )
        ]
    )

    class FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return next(responses)

            completions = _Completions()

        chat = _Chat()

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: FakeClient())

    exit_code = deepseek_enrich.run_ai_assisted(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--limit",
            "2",
            "--include-subparts",
            "--recompute-difficulty",
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "ai-status",
            "--no-progress",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "run_status" / "ai-status" / "run_status.json").read_text(encoding="utf-8"))
    batches = [
        json.loads(line)
        for line in (tmp_path / "run_status" / "ai-status" / "batch_status.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    enrichments = payload["enrichments"]
    assert exit_code == 0
    assert REQUIRED_RUN_STATUS_KEYS <= set(status)
    assert status["run_type"] == "ai_enrichment"
    assert status["status"] == "completed"
    assert status["successful_records"] == 2
    assert batches[0]["status"] == "completed"
    assert payload["schema_name"] == "exam_bank.ai_assisted_sidecar"
    assert payload["metadata"]["prompt_version"] == "ai_assisted_v2"
    assert payload["metadata"]["run_summary"]["total_records_written"] == 2
    assert payload["metadata"]["run_summary"]["attempted_records"] == 2
    assert payload["metadata"]["run_summary"]["successful_new_records"] == 2
    assert payload["metadata"]["run_summary"]["failed_new_records"] == 0
    assert payload["metadata"]["run_summary"]["preserved_records"] == 0
    assert payload["metadata"]["run_summary"]["current_run_usable_for_asterion"] is True
    assert payload["metadata"]["run_manifest"]["batches"][0]["batch_id"] == "p1_12spring24"
    assert enrichments["12spring24_q01"]["llm_model"] == "deepseek-v4-flash"
    assert enrichments["12spring24_q01"]["llm_prompt_version"] == "ai_assisted_v2"
    assert "input_hash" in enrichments["12spring24_q01"]
    assert "batch_id" in enrichments["12spring24_q01"]
    assert enrichments["12spring24_q02"]["difficulty_rank_within_paper_family"] == 1
    assert enrichments["12spring24_q02"]["deterministic_difficulty_band"] == "advanced"
    assert enrichments["12spring24_q01"]["deterministic_difficulty_band"] == "foundation"


def test_ai_sidecar_audit_reports_mixed_preserved_attempted_and_failure_reasons(tmp_path: Path) -> None:
    path = tmp_path / "question_bank.ai_assisted.v2.json"
    payload = {
        "schema_name": "exam_bank.ai_assisted_sidecar",
        "schema_version": 2,
        "metadata": {
            "prompt_version": "ai_assisted_v2",
            "run_manifest": {
                "run_timestamp": "2026-05-13T00:00:00+00:00",
                "prompt_version": "ai_assisted_v2",
                "batches": [
                    {
                        "status": deepseek_enrich.AI_FAILURE_INVALID_JSON,
                        "question_ids": ["new_fail"],
                    }
                ],
            },
        },
        "enrichments": {
            "old_success": {
                "llm_prompt_version": "v4",
                "llm_run_timestamp": "2026-05-07T00:00:00+00:00",
                "ai_assisted_items": [_ai_item(question_id="old_success")],
                "strict_filter_candidates": [{"subpart_id": None, "skill_ids": ["old_skill"]}],
            },
            "new_fail": {
                "llm_prompt_version": "ai_assisted_v2",
                "llm_run_timestamp": "2026-05-13T00:00:00+00:00",
                "error": {"type": deepseek_enrich.AI_FAILURE_INVALID_JSON, "message": "bad json"},
                "ai_assisted_items": [],
                "strict_filter_candidates": [],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = deepseek_enrich.audit_ai_assisted_sidecar(path)

    assert report["total_records_written"] == 2
    assert report["preserved_records"] == 1
    assert report["attempted_records"] == 1
    assert report["successful_new_records"] == 0
    assert report["failed_new_records"] == 1
    assert report["parse_failures"] == 1
    assert report["new_failures_by_reason"] == {deepseek_enrich.AI_FAILURE_INVALID_JSON: 1}
    assert report["mixed_prompt_versions"] is True
    assert report["current_run_usable_for_asterion"] is False


def test_ai_assisted_fresh_run_attempts_all_filtered_records_without_existing_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    input_path.write_text(
        json.dumps([_record("12spring24_q01"), _record("12spring24_q02"), _record("12spring24_q03")]),
        encoding="utf-8",
    )
    client = _fake_ai_client(
        _chat_response(
            _ai_payload(
                _ai_item(question_id="12spring24_q01"),
                _ai_item(question_id="12spring24_q02"),
                _ai_item(question_id="12spring24_q03"),
            )
        )
    )

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: client)
    exit_code = deepseek_enrich.run_ai_assisted(
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
            "ai-fresh",
            "--no-progress",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report = deepseek_enrich.audit_ai_assisted_sidecar(output_path)

    assert exit_code == 0
    assert len(getattr(client, "calls")) == 1
    assert payload["metadata"]["fresh_run"] is True
    assert payload["metadata"]["resume"] is False
    assert payload["metadata"]["selected_count"] == 3
    assert payload["metadata"]["records_to_attempt_count"] == 3
    assert payload["metadata"]["preserving_existing_records"] is False
    assert payload["metadata"]["preservation_source"] is None
    assert report["total_records"] == 3
    assert report["attempted_records"] == 3
    assert report["successful_new_records"] == 3
    assert report["preserved_records"] == 0
    assert report["records_by_llm_prompt_version"] == {"ai_assisted_v2": 3}
    assert report["mixed_prompt_versions"] is False
    assert report["safe_to_use_for_asterion_export"] is True


def test_ai_assisted_fresh_run_fails_if_output_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    input_path.write_text(json.dumps([_record("12spring24_q01")]), encoding="utf-8")
    original_payload = {
        "schema_name": "exam_bank.ai_assisted_sidecar",
        "schema_version": 2,
        "enrichments": {
            "12spring24_q01": _ai_sidecar_record("12spring24_q01", prompt_version="v4"),
        },
    }
    output_path.write_text(json.dumps(original_payload), encoding="utf-8")

    def fail_create_client(**_: object) -> object:
        raise AssertionError("fresh output-exists failure should happen before provider setup")

    monkeypatch.setattr(deepseek_enrich, "create_client", fail_create_client)

    with pytest.raises(deepseek_enrich.StartupConfigurationError, match="Output path already exists"):
        deepseek_enrich.run_ai_assisted(
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
                "ai-output-exists",
                "--no-progress",
            ]
        )

    assert json.loads(output_path.read_text(encoding="utf-8")) == original_payload


def test_ai_assisted_resume_preserves_completed_output_records_and_attempts_stale_or_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    input_path.write_text(
        json.dumps([_record("12spring24_q01"), _record("12spring24_q02"), _record("12spring24_q03")]),
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.ai_assisted_sidecar",
                "schema_version": 2,
                "enrichments": {
                    "12spring24_q01": _ai_sidecar_record("12spring24_q01"),
                    "12spring24_q02": _ai_sidecar_record("12spring24_q02", prompt_version="v4"),
                },
            }
        ),
        encoding="utf-8",
    )
    client = _fake_ai_client(
        _chat_response(
            _ai_payload(
                _ai_item(question_id="12spring24_q02"),
                _ai_item(question_id="12spring24_q03"),
            )
        )
    )

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: client)
    exit_code = deepseek_enrich.run_ai_assisted(
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
            "ai-resume",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report = deepseek_enrich.audit_ai_assisted_sidecar(output_path)
    assert len(getattr(client, "calls")) == 1
    assert payload["metadata"]["resume"] is True
    assert payload["metadata"]["preservation_source"] == "output_resume"
    assert payload["metadata"]["preserving_existing_records"] is True
    assert report["total_records"] == 3
    assert report["attempted_records"] == 2
    assert report["successful_new_records"] == 2
    assert report["preserved_records"] == 1
    assert report["records_by_llm_prompt_version"] == {"ai_assisted_v2": 3}
    assert report["safe_to_use_for_asterion_export"] is True


def test_ai_assisted_resume_force_rerun_replaces_stale_output_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    input_path.write_text(
        json.dumps([_record("12spring24_q01"), _record("12spring24_q02"), _record("12spring24_q03")]),
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.ai_assisted_sidecar",
                "schema_version": 2,
                "enrichments": {
                    question_id: _ai_sidecar_record(question_id, prompt_version="v4")
                    for question_id in ["12spring24_q01", "12spring24_q02", "12spring24_q03"]
                },
            }
        ),
        encoding="utf-8",
    )
    client = _fake_ai_client(
        _chat_response(
            _ai_payload(
                _ai_item(question_id="12spring24_q01"),
                _ai_item(question_id="12spring24_q02"),
                _ai_item(question_id="12spring24_q03"),
            )
        )
    )

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: client)
    exit_code = deepseek_enrich.run_ai_assisted(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--output",
            str(output_path),
            "--resume",
            "--force-rerun",
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "ai-force-rerun",
            "--no-progress",
        ]
    )

    report = deepseek_enrich.audit_ai_assisted_sidecar(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(getattr(client, "calls")) == 1
    assert payload["metadata"]["force_rerun"] is True
    assert report["total_records"] == 3
    assert report["attempted_records"] == 3
    assert report["preserved_records"] == 0
    assert report["records_by_llm_prompt_version"] == {"ai_assisted_v2": 3}
    assert report["stale_records_from_existing_sidecar"] == 0
    assert report["safe_to_use_for_asterion_export"] is True


def test_ai_assisted_existing_sidecar_is_evidence_not_final_preservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    existing_path = tmp_path / "existing.json"
    input_path.write_text(json.dumps([_record("12spring24_q01"), _record("12spring24_q02")]), encoding="utf-8")
    existing_path.write_text(
        json.dumps(
            {
                "schema_name": "exam_bank.ai_assisted_sidecar",
                "schema_version": 2,
                "enrichments": {
                    "12spring24_q02": {
                        **_ai_sidecar_record("12spring24_q02", prompt_version="v4"),
                        "deepseek_topic": "binomial expansion",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    client = _fake_ai_client(
        _chat_response(
            _ai_payload(
                _ai_item(question_id="12spring24_q01"),
                _ai_item(question_id="12spring24_q02"),
            )
        )
    )

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: client)
    exit_code = deepseek_enrich.run_ai_assisted(
        [
            "--input",
            str(input_path),
            "--taxonomy",
            "exam_bank_taxonomy/canonical",
            "--existing-sidecar",
            str(existing_path),
            "--output",
            str(output_path),
            "--status-dir",
            str(tmp_path / "run_status"),
            "--run-id",
            "ai-existing-evidence",
            "--no-progress",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report = deepseek_enrich.audit_ai_assisted_sidecar(output_path)
    request_payload = json.loads(getattr(client, "calls")[0]["messages"][1]["content"])

    assert exit_code == 0
    assert "existing_deepseek_v1" in request_payload["questions"][1]
    assert payload["metadata"]["existing_sidecar_path"] == str(existing_path)
    assert payload["metadata"]["preservation_source"] is None
    assert report["total_records"] == 2
    assert report["attempted_records"] == 2
    assert report["preserved_records"] == 0
    assert report["records_by_llm_prompt_version"] == {"ai_assisted_v2": 2}
    assert report["safe_to_use_for_asterion_export"] is True


def test_ai_assisted_limit_run_writes_only_limited_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "question_bank.json"
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    records = [_record(f"12spring24_q{index:02d}") for index in range(1, 13)]
    input_path.write_text(json.dumps(records), encoding="utf-8")
    client = _fake_ai_client(
        _chat_response(
            _ai_payload(*[_ai_item(question_id=f"12spring24_q{index:02d}") for index in range(1, 11)])
        )
    )

    monkeypatch.setattr(deepseek_enrich, "create_client", lambda **_: client)
    exit_code = deepseek_enrich.run_ai_assisted(
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
            "ai-limit",
            "--no-progress",
        ]
    )

    report = deepseek_enrich.audit_ai_assisted_sidecar(output_path)

    assert exit_code == 0
    assert report["total_records"] == 10
    assert report["attempted_records"] == 10
    assert report["preserved_records"] == 0
    assert report["records_by_llm_prompt_version"] == {"ai_assisted_v2": 10}


def test_ai_resume_uses_completed_batch_cache_without_calling_provider(tmp_path: Path) -> None:
    output_path = tmp_path / "question_bank.ai_assisted.v2.json"
    records = [_record("12spring24_q01")]

    class FirstClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    return _chat_response(_ai_payload(_ai_item(question_id="12spring24_q01")))

            completions = _Completions()

        chat = _Chat()

    first_tracker = RunStatusTracker(
        run_id="first",
        run_type="ai_enrichment",
        status_root=tmp_path / "run_status",
        command="enrich-ai",
        progress=False,
    )
    first_tracker.start(phase="preparing_batches")
    deepseek_enrich.enrich_ai_assisted_records(
        records,
        client=FirstClient(),
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=output_path,
        model="deepseek-v4-flash",
        batch_by_paper=True,
        progress=first_tracker,
    )
    first_tracker.finish("completed")

    class FailingClient:
        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_: object) -> SimpleNamespace:
                    raise AssertionError("resume should read the completed batch cache")

            completions = _Completions()

        chat = _Chat()

    second_tracker = RunStatusTracker(
        run_id="second",
        run_type="ai_enrichment",
        status_root=tmp_path / "run_status",
        command="enrich-ai --resume",
        progress=False,
    )
    second_tracker.start(phase="preparing_batches")
    enrichments, _manifest = deepseek_enrich.enrich_ai_assisted_records(
        records,
        client=FailingClient(),
        taxonomy_root="exam_bank_taxonomy/canonical",
        output_path=output_path,
        model="deepseek-v4-flash",
        batch_by_paper=True,
        resume=True,
        progress=second_tracker,
        resume_completed_batch_ids={"p1_12spring24"},
    )
    second_tracker.finish("completed")

    status = json.loads(second_tracker.run_status_path.read_text(encoding="utf-8"))
    batches = [
        json.loads(line)
        for line in second_tracker.batch_status_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert enrichments["12spring24_q01"]["ai_assisted_items"][0]["question_id"] == "12spring24_q01"
    assert len(batches) == 1
    assert batches[0]["status"] == "skipped"
    assert batches[0]["record_count"] == 1
    assert batches[0]["skipped_records"] == 1
    assert status["skipped_records"] == 1


def test_difficulty_percentile_is_computed_within_paper_family_not_globally() -> None:
    enrichments = {
        "p1_easy": {"paper_family": "p1", "ai_assisted_items": [_ai_item(question_id="p1_easy", ai_difficulty_score=0.1)]},
        "p1_hard": {"paper_family": "p1", "ai_assisted_items": [_ai_item(question_id="p1_hard", ai_difficulty_score=0.9)]},
        "p3_easy": {
            "paper_family": "p3",
            "ai_assisted_items": [
                {
                    **_ai_item(question_id="p3_easy", paper_family="p3", ai_difficulty_score=0.1),
                    "primary_topic_id": "9709_p3_topic_algebra",
                    "subtopic_ids": ["9709_p3_subtopic_polynomial_division_factor_remainder"],
                    "skill_ids": ["9709_p3_3_1_polynomial_division_factor_remainder"],
                }
            ],
        },
        "p3_hard": {
            "paper_family": "p3",
            "ai_assisted_items": [
                {
                    **_ai_item(question_id="p3_hard", paper_family="p3", ai_difficulty_score=0.9),
                    "primary_topic_id": "9709_p3_topic_algebra",
                    "subtopic_ids": ["9709_p3_subtopic_polynomial_division_factor_remainder"],
                    "skill_ids": ["9709_p3_3_1_polynomial_division_factor_remainder"],
                }
            ],
        },
    }
    records = [
        {**_record("p1_easy"), "paper_family": "p1"},
        {**_record("p1_hard"), "paper_family": "p1"},
        {**_record("p3_easy"), "paper_family": "p3"},
        {**_record("p3_hard"), "paper_family": "p3"},
    ]

    calibrated = deepseek_enrich.calibrate_difficulty_by_paper_family(enrichments, records)

    assert calibrated["p1_hard"]["deterministic_difficulty_percentile"] == 100
    assert calibrated["p1_easy"]["deterministic_difficulty_percentile"] == 0
    assert calibrated["p3_hard"]["deterministic_difficulty_percentile"] == 100
    assert calibrated["p3_easy"]["deterministic_difficulty_percentile"] == 0
    assert {calibrated["p1_easy"]["deterministic_difficulty_band"], calibrated["p1_hard"]["deterministic_difficulty_band"]} == {
        "foundation",
        "advanced",
    }
