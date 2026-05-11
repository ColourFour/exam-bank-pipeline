from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI
from .runtime_profile import DIFFICULTY_LABELS, PAPER_FAMILY_TAXONOMY
from .trust import Confidence, DeepSeekErrorType, ReconciliationStatus, final_review_reasons as _final_review_reasons

DEFAULT_INPUT_PATH = Path("output/json/question_bank.json")
DEFAULT_OUTPUT_PATH = Path("output/json/question_bank.deepseek.json")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
PROMPT_VERSION = "v4"
AI_ASSISTED_PROMPT_VERSION = "ai_assisted_v2"
LLM_PROVIDER = "deepseek"
DEEPSEEK_SIDECAR_SCHEMA_NAME = "exam_bank.deepseek_sidecar"
DEEPSEEK_SIDECAR_SCHEMA_VERSION = 2
AI_ASSISTED_SIDECAR_SCHEMA_NAME = "exam_bank.ai_assisted_sidecar"
AI_ASSISTED_SIDECAR_SCHEMA_VERSION = 2

PAPER_FAMILY_TO_CANONICAL_COMPONENT = {
    "p1": "p1",
    "p3": "p3",
    "p4": "m1",
    "p5": "s1",
    "m1": "m1",
    "s1": "s1",
}

CANONICAL_COMPONENT_TO_PAPER_FAMILY = {
    "p1": "p1",
    "p3": "p3",
    "m1": "p4",
    "s1": "p5",
}

AI_ASSISTED_REQUIRED_ITEM_KEYS = {
    "question_id",
    "subpart_id",
    "paper_family",
    "primary_topic_id",
    "secondary_topic_ids",
    "subtopic_ids",
    "skill_ids",
    "method_families",
    "prerequisite_skill_ids",
    "exam_techniques",
    "common_mistakes",
    "worked_example_seed",
    "warmup_seed",
    "strict_filter_candidate",
    "strict_filter_reason",
    "evidence_used",
    "evidence_missing",
    "confidence",
    "review_required",
    "review_reasons",
    "ai_difficulty_estimate",
    "ai_difficulty_score",
    "ai_difficulty_factors",
    "needs_new_subtopic_candidate",
    "suggested_new_subtopic",
    "needs_new_skill_candidate",
    "suggested_new_skill",
    "mapping_source",
    "reviewed_status",
}

STRICT_FILTER_MIN_CONFIDENCE = 0.75


@dataclass(frozen=True)
class CanonicalTaxonomy:
    """Read-only view of the active canonical taxonomy for one component."""

    paper_family: str
    component_key: str
    topic_map_path: Path
    skill_map_path: Path
    topics: dict[str, dict[str, Any]]
    subtopics: dict[str, dict[str, Any]]
    skills: dict[str, dict[str, Any]]
    subtopic_to_topic: dict[str, str]
    skill_to_subtopic: dict[str, str]
    skill_to_topic: dict[str, str]
    skill_to_asterion_region: dict[str, str]


class StartupConfigurationError(RuntimeError):
    """Raised when the enrichment run is misconfigured before processing starts."""


class ModelResponseError(ValueError):
    """Raised when provider output is present but cannot be accepted."""

    def __init__(self, message: str, *, raw_provider_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_provider_output = raw_provider_output


_UNIQUE_TOPIC_ALIASES: dict[str, str] = {}
_TOPIC_ALIASES_BY_FAMILY: dict[str, dict[str, str]] = {}
for family_name, topics in PAPER_FAMILY_TAXONOMY.items():
    if family_name == "unknown":
        continue
    family_aliases: dict[str, str] = {}
    for canonical_topic in topics:
        alias = canonical_topic.replace("_", " ").strip().lower()
        family_aliases[alias] = canonical_topic
        _UNIQUE_TOPIC_ALIASES.setdefault(alias, canonical_topic)
        if _UNIQUE_TOPIC_ALIASES.get(alias) != canonical_topic:
            _UNIQUE_TOPIC_ALIASES.pop(alias, None)
    _TOPIC_ALIASES_BY_FAMILY[family_name] = family_aliases

_DIFFICULTY_ALIASES = {
    "easy": "easy",
    "simple": "easy",
    "basic": "easy",
    "routine": "easy",
    "average": "average",
    "medium": "average",
    "moderate": "average",
    "intermediate": "average",
    "difficult": "difficult",
    "hard": "difficult",
    "challenging": "difficult",
}

_CONFIDENCE_ALIASES = {
    "high": "high",
    "medium": "medium",
    "med": "medium",
    "moderate": "medium",
    "low": "low",
}

_EXTRA_TOPIC_ALIASES = {
    "sequence": "series_and_sequences",
    "sequences": "series_and_sequences",
    "series": "series_and_sequences",
    "sequences and series": "series_and_sequences",
    "series and sequences": "series_and_sequences",
    "binomial": "binomial_expansion",
    "binomial theorem": "binomial_expansion",
    "binomial expansion": "binomial_expansion",
    "coordinate geometry": "coordinate_geometry",
    "circle geometry": "coordinate_geometry",
    "circles": "coordinate_geometry",
    "circle": "coordinate_geometry",
    "derivative": "differentiation",
    "derivatives": "differentiation",
    "differentiation": "differentiation",
    "integration": "integration",
    "integrals": "integration",
    "vectors": "vectors",
    "vector": "vectors",
    "vector geometry": "vectors",
    "complex number": "complex_numbers",
    "complex numbers": "complex_numbers",
    "argand": "complex_numbers",
    "argand diagram": "complex_numbers",
    "argand diagrams": "complex_numbers",
    "differential equation": "differential_equations",
    "differential equations": "differential_equations",
    "trig": "trigonometry",
    "trigonometry": "trigonometry",
    "trig equations": "trigonometry",
    "trigonometric equations": "trigonometry",
    "functions": "functions",
    "function": "functions",
    "transformations": "functions",
    "graph transformations": "functions",
    "kinematics": "kinematics",
    "motion": "kinematics",
    "velocity time graphs": "kinematics_graphs",
    "velocity time graph": "kinematics_graphs",
    "work energy and power": "power_and_resistance",
    "power": "power_and_resistance",
    "power and resistance": "power_and_resistance",
    "forces and equilibrium": "equilibrium_particle",
    "force equilibrium": "equilibrium_particle",
    "equilibrium": "equilibrium_particle",
    "equilibrium of a particle": "equilibrium_particle",
    "momentum impulse": "momentum_impulse",
    "momentum and impulse": "momentum_impulse",
    "conservation of momentum": "momentum_impulse",
}

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a narrow DeepSeek enrichment pass over exported question bank JSON.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to question_bank.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the DeepSeek sidecar JSON.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of selected records to enrich.")
    parser.add_argument(
        "--question-ids",
        nargs="*",
        default=None,
        help="Optional question IDs to enrich. Accepts space- or comma-separated values.",
    )
    parser.add_argument("--paper-family", default=None, help="Optional paper family filter, for example P1 or p1.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name.")
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Optional JSONL path for raw provider failure logging. Defaults next to the output sidecar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which records would be sent without making external API calls or writing an output file.",
    )
    parser.add_argument(
        "--allow-provider-failure",
        action="store_true",
        help=(
            "Exit 0 even if every attempted enrichment fails with a provider/API error. "
            "Sidecar error records are still written."
        ),
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be zero or greater.")
    args.question_ids = _parse_question_ids(args.question_ids)
    return args


def _parse_question_ids(raw_values: Sequence[str] | None) -> list[str] | None:
    if not raw_values:
        return None
    question_ids: list[str] = []
    for value in raw_values:
        for part in value.split(","):
            cleaned = part.strip()
            if cleaned:
                question_ids.append(cleaned)
    return question_ids or None


def _normalize_free_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("_", " ")
    normalized = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    return " ".join(normalized.split())


def _canonical_paper_family(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in PAPER_FAMILY_TAXONOMY:
        return text
    return "unknown"


def normalize_topic_label(raw_topic: Any, *, paper_family: Any, raw_subtopic: Any = None) -> str | None:
    family = _canonical_paper_family(paper_family)
    family_aliases = _TOPIC_ALIASES_BY_FAMILY.get(family, {})
    valid_family_topics = set(PAPER_FAMILY_TAXONOMY.get(family, ()))

    for candidate in (raw_topic, raw_subtopic):
        normalized_candidate = _normalize_free_text(candidate)
        if not normalized_candidate:
            continue

        if normalized_candidate in family_aliases:
            return family_aliases[normalized_candidate]

        extra_alias = _EXTRA_TOPIC_ALIASES.get(normalized_candidate)
        if extra_alias and extra_alias in valid_family_topics:
            return extra_alias

        if normalized_candidate in _UNIQUE_TOPIC_ALIASES:
            unique_topic = _UNIQUE_TOPIC_ALIASES[normalized_candidate]
            if not valid_family_topics or unique_topic in valid_family_topics:
                return unique_topic

    return None


def normalize_difficulty_label(raw_difficulty: Any) -> str | None:
    if isinstance(raw_difficulty, bool) or raw_difficulty is None:
        return None
    if isinstance(raw_difficulty, (int, float)):
        score = normalize_difficulty_score(raw_difficulty)
        return _difficulty_label_for_score(score) if score is not None else None

    normalized = _normalize_free_text(raw_difficulty)
    if not normalized:
        return None
    if normalized in _DIFFICULTY_ALIASES:
        return _DIFFICULTY_ALIASES[normalized]
    if normalized in DIFFICULTY_LABELS:
        return normalized
    if normalized in {"1", "one"}:
        return "easy"
    if normalized in {"2", "two"}:
        return "average"
    if normalized in {"3", "three"}:
        return "difficult"
    if normalized in {"medium hard", "mediumhard", "average hard"}:
        return "difficult"
    score = normalize_difficulty_score(raw_difficulty)
    if score is not None:
        return _difficulty_label_for_score(score)
    return None


def _difficulty_label_for_score(score: int) -> str:
    if score <= 34:
        return "easy"
    if score <= 69:
        return "average"
    return "difficult"


def normalize_difficulty_score(raw_difficulty: Any) -> int | None:
    if isinstance(raw_difficulty, bool) or raw_difficulty is None:
        return None
    if isinstance(raw_difficulty, (int, float)):
        if 0 <= float(raw_difficulty) <= 100:
            return int(round(float(raw_difficulty)))
        return None
    raw_text = str(raw_difficulty).strip()
    if not raw_text:
        return None
    try:
        score = float(raw_text.rstrip("%"))
    except ValueError:
        return None
    if 0 <= score <= 100:
        return int(round(score))
    return None


def normalize_confidence_value(raw_confidence: Any) -> str | None:
    if isinstance(raw_confidence, bool) or raw_confidence is None:
        return None
    if isinstance(raw_confidence, (int, float)):
        return _numeric_confidence_bucket(float(raw_confidence))

    raw_text = str(raw_confidence).strip()
    if not raw_text:
        return None
    normalized = _normalize_free_text(raw_text)
    if normalized in _CONFIDENCE_ALIASES:
        return _CONFIDENCE_ALIASES[normalized]
    try:
        numeric_text = raw_text.rstrip("%").strip()
        numeric = float(numeric_text)
    except ValueError:
        return None
    return _numeric_confidence_bucket(numeric, had_percent=raw_text.endswith("%"))


def _numeric_confidence_bucket(value: float, *, had_percent: bool = False) -> str | None:
    if value < 0:
        return None
    scaled = value
    if had_percent or value > 1:
        if value > 100:
            return None
        scaled = value / 100.0
    if scaled >= 0.75:
        return Confidence.HIGH
    if scaled >= 0.45:
        return Confidence.MEDIUM
    return Confidence.LOW


def require_api_key(env: dict[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    api_key = source.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise StartupConfigurationError(
            "DEEPSEEK_API_KEY is required in the environment before running DeepSeek enrichment."
        )
    return api_key


def create_client(*, base_url: str = DEFAULT_BASE_URL, api_key: str | None = None) -> OpenAI:
    resolved_api_key = api_key if api_key is not None else require_api_key()
    return OpenAI(api_key=resolved_api_key, base_url=base_url)


def load_question_bank(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StartupConfigurationError(f"Input question bank not found: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise StartupConfigurationError(f"Input question bank is not valid JSON: {input_path}") from exc

    if isinstance(payload, dict):
        raw_records = payload.get("questions")
        if raw_records is None:
            raise StartupConfigurationError("Input question bank object must contain a questions array.")
    else:
        raw_records = payload

    if not isinstance(raw_records, list):
        raise StartupConfigurationError("Input question bank must contain a JSON array of records.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            raise StartupConfigurationError(f"Question bank record at index {index} is not a JSON object.")
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            raise StartupConfigurationError(f"Question bank record at index {index} is missing question_id.")
        records.append(item)
    return records


def select_records(
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
    question_ids: Sequence[str] | None = None,
    paper_family: str | None = None,
) -> list[dict[str, Any]]:
    selected = list(records)
    if question_ids:
        wanted = {value.strip() for value in question_ids if value.strip()}
        selected = [record for record in selected if str(record.get("question_id", "")).strip() in wanted]
    if paper_family:
        wanted_family = paper_family.strip().lower()
        selected = [record for record in selected if str(record.get("paper_family", "")).strip().lower() == wanted_family]
    if limit >= 0:
        selected = selected[:limit]
    return selected


def build_enrichment_payload(record: dict[str, Any]) -> dict[str, Any]:
    notes = record.get("notes")
    note_map = notes if isinstance(notes, dict) else {}
    question_image_path = record.get("question_image_path") or _first_path(record.get("question_image_paths"))
    mark_scheme_image_path = record.get("mark_scheme_image_path") or _first_path(record.get("mark_scheme_image_paths"))
    visual_required = bool(record.get("visual_required") or note_map.get("visual_required"))

    question_text = record.get("question_text")
    question_text_trust = _raw_or_none(record.get("question_text_trust")) or _raw_or_none(note_map.get("question_text_trust"))
    question_text_role = _raw_or_none(record.get("question_text_role")) or _raw_or_none(note_map.get("question_text_role"))

    ocr_text = record.get("ocr_text")
    ocr_text_trust = _raw_or_none(record.get("ocr_text_trust")) or _raw_or_none(note_map.get("ocr_text_trust"))
    ocr_text_role = _raw_or_none(record.get("ocr_text_role")) or _raw_or_none(note_map.get("ocr_text_role"))
    text_only_status = _raw_or_none(record.get("text_only_status")) or _raw_or_none(note_map.get("text_only_status"))
    visual_curation_status = _raw_or_none(record.get("visual_curation_status")) or _raw_or_none(note_map.get("visual_curation_status"))
    payload: dict[str, Any] = {
        "question_id": str(record["question_id"]),
        "paper_family": record.get("paper_family"),
        "question_number": record.get("question_number"),

        "question_text": question_text,
        "question_text_role": question_text_role,
        "question_text_trust": question_text_trust,

        "ocr_text": ocr_text,
        "ocr_text_role": ocr_text_role,
        "ocr_text_trust": ocr_text_trust,

        "combined_text_hint": combined_text_hint(
            question_text=question_text,
            question_text_trust=question_text_trust,
            ocr_text=ocr_text,
            ocr_text_trust=ocr_text_trust,
        ),

        "image_available": bool(question_image_path),
        "image_was_sent_to_model": False,
        "vision_model_required": visual_required,
        "text_only_status": text_only_status,
        "visual_curation_status": visual_curation_status,
        "text_only_enrichment_risk": text_only_enrichment_risk(
            visual_required=visual_required,
            question_text_trust=question_text_trust,
            question_text_role=question_text_role,
        ),
    }

    optional_fields = {
        "mark_scheme_text": record.get("mark_scheme_text"),
        "question_solution_marks": record.get("question_solution_marks"),
        "question_image_path": question_image_path,
        "mark_scheme_image_path": mark_scheme_image_path,
        "visual_reason_flags": record.get("visual_reason_flags") or note_map.get("visual_reason_flags"),
        "scope_quality_status": note_map.get("scope_quality_status"),
        "text_fidelity_status": note_map.get("text_fidelity_status"),
        "topic_trust_status": note_map.get("topic_trust_status"),
        "current_local_topic": record.get("topic"),
    }
    for key, value in optional_fields.items():
        if value not in (None, "", []):
            payload[key] = value
    return payload


def _first_path(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""

def combined_text_hint(
    *,
    question_text: Any,
    question_text_trust: Any,
    ocr_text: Any,
    ocr_text_trust: Any,
) -> str:
    native = str(question_text or "").strip()
    ocr = str(ocr_text or "").strip()
    native_trust = str(question_text_trust or "").strip().lower() or "unknown"
    ocr_trust = str(ocr_text_trust or "").strip().lower() or "unknown"

    parts: list[str] = []

    if native:
        parts.append(f"NATIVE_PDF_TEXT_TRUST={native_trust}\n{native}")

    if ocr:
        parts.append(f"OCR_TEXT_TRUST={ocr_trust}\n{ocr}")

    return "\n\n---\n\n".join(parts)


def text_only_enrichment_risk(
    *,
    visual_required: bool,
    question_text_trust: Any,
    question_text_role: Any,
) -> str:
    trust = str(question_text_trust or "").strip().lower()
    role = str(question_text_role or "").strip().lower()
    if visual_required and (trust in {"low", "unusable", ""} or role in {"untrusted_math_text", "missing", ""}):
        return "high"
    if visual_required or trust == "medium" or role == "search_hint":
        return "medium"
    return "low"


def _raw_or_none(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _reconciliation_status(*, raw_value: Any, normalized_value: str | None, local_value: str | None) -> str:
    if _raw_or_none(raw_value) in (None, ""):
        return ReconciliationStatus.NO_DEEPSEEK_LABEL
    if normalized_value is None:
        return ReconciliationStatus.UNMAPPED_LABEL
    if local_value is None:
        return ReconciliationStatus.NO_LOCAL_LABEL
    if normalized_value == local_value:
        return ReconciliationStatus.MATCH
    return ReconciliationStatus.MISMATCH


def build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    instructions = (
        "You are reviewing a CAIE 9709 maths question record for suggestion-only metadata enrichment. "
        "The rendered PNG crop is the canonical question artifact, but this request does not send the image itself. "
        "Native PDF text and OCR text are lossy hints only and may corrupt equations, powers, fractions, vectors, "
        "inequalities, integrals, trigonometric notation, and graph or diagram content. "
        "Use the text hints primarily for topic and difficulty clues, not for exact reconstruction of the mathematical expression. "
        "If visual_required is true, text_only_enrichment_risk is high, text_only_status is fail, or the text appears corrupted, "
        "set review_required to true unless the topic and difficulty are still obvious from stable wording. "
        "Do not overstate confidence from corrupted math text. "
        "Return one strict JSON object only, with no markdown, no prose, no code fences, and no extra keys. "
        "The object must contain exactly these keys: "
        "topic, subtopic, difficulty, confidence, rationale, review_required. "
        "difficulty must be a JSON number from 0 to 100, not a string label. "
        "If 100 representative CAIE 9709 secondary students attempted this item, estimate the percentage of "
        "available marks not received across the cohort. "
        "Use 0 when nearly all students would receive full marks, 50 when the cohort would receive about half "
        "of the available marks, and 100 when almost no marks would be awarded. Higher means harder. "
        "confidence must be high, medium, low, or a number from 0 to 1. "
        "Use an empty string for subtopic when unavailable. "
        "Keep rationale short and concrete. "
        "review_required must be a literal JSON boolean: true or false. "
        "Do not quote the boolean."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def request_suggestion(client: OpenAI, *, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(payload),
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        suggestion, parse_metadata = parse_response_with_recovery(response, parse_model_json)
        if parse_metadata["parse_recovered"]:
            suggestion["__parse_recovered"] = True
            suggestion["__parse_recovery_source"] = parse_metadata["parse_recovery_source"]
        return suggestion
    except ValueError as exc:
        raw_output = getattr(exc, "raw_provider_output", None)
        raise ModelResponseError(str(exc), raw_provider_output=raw_output or _response_snapshot(response)) from exc


def response_text(response: Any) -> str:
    for source, text in response_text_candidates(response):
        if source in {"message.content", "output_text"}:
            return text
    raise ModelResponseError(
        "Provider response did not contain text content.",
        raw_provider_output=_response_snapshot(response),
    )


def response_text_candidates(response: Any) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            candidates.append(("message.content", content))

        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, (dict, list)):
            candidates.append(("structured_response", json.dumps(parsed, ensure_ascii=False)))
        elif isinstance(parsed, str) and parsed.strip():
            candidates.append(("structured_response", parsed))

        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            candidates.append(("reasoning_content", reasoning_content))

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        candidates.append(("output_text", output_text))

    snapshot = _response_snapshot(response)
    if snapshot.strip():
        candidates.append(("raw_provider_output", snapshot))
    return candidates


def parse_response_with_recovery(
    response: Any,
    parser: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = response_text_candidates(response)
    first_error: ValueError | None = None
    first_text: str | None = None
    first_source: str | None = None

    for source, text in candidates:
        for candidate_text in _candidate_json_texts(text, source=source):
            try:
                parsed = parser(candidate_text)
            except ValueError as exc:
                if first_error is None:
                    first_error = exc
                    first_text = candidate_text
                    first_source = source
                continue
            return parsed, {
                "parse_recovered": source != "message.content" or candidate_text != text,
                "parse_recovery_source": source,
            }

    if first_error is not None:
        error = ValueError(str(first_error))
        setattr(error, "raw_provider_output", first_text)
        setattr(error, "parse_recovery_source", first_source)
        raise error

    error = ValueError("Provider response did not contain parseable JSON content.")
    setattr(error, "raw_provider_output", _response_snapshot(response))
    raise error


def _candidate_json_texts(text: str, *, source: str) -> list[str]:
    candidates = [text]
    extracted = extract_final_json_object(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    if source == "raw_provider_output":
        for escaped in re.findall(r'"(\{(?:[^"\\]|\\.)+\})"', text, flags=re.DOTALL):
            try:
                decoded = json.loads(f'"{escaped}"')
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, str) and decoded.strip():
                candidates.append(decoded)
                nested = extract_final_json_object(decoded)
                if nested and nested != decoded:
                    candidates.append(nested)

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return unique


def extract_final_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder(object_pairs_hook=_strict_json_object)
    best: tuple[int, int, str] | None = None
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        candidate = text[start : start + end]
        candidate_length = len(candidate)
        if best is None or candidate_length > best[1] or (candidate_length == best[1] and start > best[0]):
            best = (start, end, candidate)
    return best[2] if best else None


def parse_model_json(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text, object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError("Model output was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")

    expected_keys = {"topic", "subtopic", "difficulty", "confidence", "rationale", "review_required"}
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Model output keys did not match the required schema. Expected {sorted(expected_keys)}, got {sorted(actual_keys)}."
        )

    topic = _require_non_empty_string(payload["topic"], "topic")
    difficulty = _require_difficulty_score(payload["difficulty"])
    confidence = _require_confidence_value(payload["confidence"])
    rationale = _require_non_empty_string(payload["rationale"], "rationale")

    subtopic_value = payload["subtopic"]
    if subtopic_value is None:
        subtopic = ""
    elif isinstance(subtopic_value, str):
        subtopic = subtopic_value.strip()
    else:
        raise ValueError("subtopic must be a string or null.")

    review_required = payload["review_required"]
    if not isinstance(review_required, bool):
        raise ValueError("review_required must be a boolean.")

    return {
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "confidence": confidence,
        "rationale": rationale,
        "review_required": review_required,
    }


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_difficulty_score(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("difficulty must be a numeric score from 0 to 100.")
    if not isinstance(value, (int, float)):
        raise ValueError("difficulty must be a numeric score from 0 to 100.")
    score = normalize_difficulty_score(value)
    if score is None:
        raise ValueError("difficulty must be a numeric score from 0 to 100.")
    return score


def _require_confidence_value(value: Any) -> str | int | float:
    if isinstance(value, bool):
        raise ValueError("confidence must be a non-empty string or numeric value.")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("confidence must be a non-empty string or numeric value.")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Model output contained a duplicate key: {key}.")
        payload[key] = value
    return payload


def _response_snapshot(response: Any) -> str:
    model_dump_json = getattr(response, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return str(model_dump_json(indent=2))
        except TypeError:
            return str(model_dump_json())
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return json.dumps(model_dump(), ensure_ascii=False, indent=2)
        except TypeError:
            return json.dumps(model_dump(), ensure_ascii=False)
    return repr(response)


def build_sidecar_success(
    record: dict[str, Any],
    suggestion: dict[str, Any],
    *,
    model: str,
    run_timestamp: str,
) -> dict[str, Any]:
    notes = record.get("notes")
    note_map = notes if isinstance(notes, dict) else {}

    raw_topic = _raw_or_none(suggestion.get("topic"))
    raw_subtopic = _raw_or_none(suggestion.get("subtopic"))
    raw_difficulty = _raw_or_none(suggestion.get("difficulty"))
    raw_confidence = _raw_or_none(suggestion.get("confidence"))
    normalized_confidence = normalize_confidence_value(raw_confidence)
    raw_rationale = _raw_or_none(suggestion.get("rationale"))
    raw_review_required = bool(suggestion.get("review_required"))

    local_topic = _raw_or_none(record.get("topic"))
    local_difficulty = _raw_or_none(record.get("difficulty")) or _raw_or_none(note_map.get("difficulty"))
    record_difficulty_score = _raw_or_none(record.get("difficulty_score"))
    note_difficulty_score = _raw_or_none(note_map.get("difficulty_score"))
    local_difficulty_score = normalize_difficulty_score(
        record_difficulty_score if record_difficulty_score is not None else note_difficulty_score
    )

    normalized_topic = normalize_topic_label(
        raw_topic,
        paper_family=record.get("paper_family"),
        raw_subtopic=raw_subtopic,
    )
    normalized_difficulty = normalize_difficulty_label(raw_difficulty)
    deepseek_difficulty_score = normalize_difficulty_score(raw_difficulty)

    local_topic_normalized = normalize_topic_label(
        local_topic,
        paper_family=record.get("paper_family"),
        raw_subtopic=note_map.get("subtopic"),
    ) if local_topic else None
    local_difficulty_normalized = (
        normalize_difficulty_label(local_difficulty)
        if local_difficulty
        else normalize_difficulty_label(local_difficulty_score)
    )

    topic_reconciliation_status = _reconciliation_status(
        raw_value=raw_topic,
        normalized_value=normalized_topic,
        local_value=local_topic_normalized,
    )
    difficulty_reconciliation_status = _reconciliation_status(
        raw_value=raw_difficulty,
        normalized_value=normalized_difficulty,
        local_value=local_difficulty_normalized,
    )

    final_review_reasons = _final_review_reasons(
        model_review_required=raw_review_required,
        validation_status=_raw_or_none(note_map.get("validation_status")),
        scope_quality_status=_raw_or_none(note_map.get("scope_quality_status")),
        text_fidelity_status=_raw_or_none(note_map.get("text_fidelity_status")),
        topic_trust_status=_raw_or_none(note_map.get("topic_trust_status")),
        topic_reconciliation_status=topic_reconciliation_status,
        difficulty_reconciliation_status=difficulty_reconciliation_status,
        local_difficulty_present=local_difficulty_normalized is not None or local_difficulty_score is not None,
    )
    question_image_path = record.get("question_image_path") or _first_path(record.get("question_image_paths"))
    mark_scheme_image_path = record.get("mark_scheme_image_path") or _first_path(record.get("mark_scheme_image_paths"))

    visual_required = bool(record.get("visual_required") or note_map.get("visual_required"))
    question_text_trust = _raw_or_none(record.get("question_text_trust")) or _raw_or_none(note_map.get("question_text_trust"))
    question_text_role = _raw_or_none(record.get("question_text_role")) or _raw_or_none(note_map.get("question_text_role"))

    ocr_text_trust = _raw_or_none(record.get("ocr_text_trust")) or _raw_or_none(note_map.get("ocr_text_trust"))
    ocr_text_role = _raw_or_none(record.get("ocr_text_role")) or _raw_or_none(note_map.get("ocr_text_role"))
    text_only_status = _raw_or_none(record.get("text_only_status")) or _raw_or_none(note_map.get("text_only_status"))
    visual_curation_status = _raw_or_none(record.get("visual_curation_status")) or _raw_or_none(note_map.get("visual_curation_status"))

    enrichment_risk = text_only_enrichment_risk(
        visual_required=visual_required,
        question_text_trust=question_text_trust,
        question_text_role=question_text_role,
    )

    extra_review_reasons = list(final_review_reasons)

    if enrichment_risk == "high":
        extra_review_reasons.append("text_only_enrichment_risk:high")
    if text_only_status and text_only_status != "ready":
        extra_review_reasons.append(f"text_only_status:{text_only_status}")
    if visual_curation_status == "fail":
        extra_review_reasons.append("visual_curation_status:fail")
    if ocr_text_trust in {"low", "unusable"}:
        extra_review_reasons.append(f"ocr_text_trust:{ocr_text_trust}")

    final_review_reasons = sorted(set(extra_review_reasons))

    sidecar = {
        "deepseek_topic": raw_topic,
        "deepseek_subtopic": raw_subtopic,
        "deepseek_difficulty": raw_difficulty,
        "deepseek_difficulty_score": deepseek_difficulty_score,
        "deepseek_confidence": raw_confidence,
        "deepseek_confidence_normalized": normalized_confidence,
        "deepseek_rationale": raw_rationale,
        "deepseek_review_required": raw_review_required,
        "deepseek_topic_raw": raw_topic,
        "deepseek_subtopic_raw": raw_subtopic,
        "deepseek_difficulty_raw": raw_difficulty,
        "deepseek_confidence_raw": raw_confidence,
        "deepseek_rationale_raw": raw_rationale,
        "deepseek_review_required_raw": raw_review_required,
        "deepseek_topic_normalized": normalized_topic,
        "deepseek_difficulty_normalized": normalized_difficulty,
        "local_topic": local_topic_normalized or local_topic,
        "local_difficulty": local_difficulty_normalized or local_difficulty,
        "local_difficulty_score": local_difficulty_score,
        "topic_reconciliation_status": topic_reconciliation_status,
        "difficulty_reconciliation_status": difficulty_reconciliation_status,
        "final_review_required": bool(final_review_reasons),
        "final_review_reasons": final_review_reasons,
        "enrichment_mode": "text_with_image_reference",
        "image_available": bool(question_image_path),
        "image_was_sent_to_model": False,
        "question_image_path": question_image_path or None,
        "mark_scheme_image_path": mark_scheme_image_path or None,
        "vision_model_required": visual_required,
        "question_text_role": question_text_role,
        "question_text_trust": question_text_trust,
        "ocr_text_role": ocr_text_role,
        "ocr_text_trust": ocr_text_trust,
        "text_only_status": text_only_status,
        "visual_curation_status": visual_curation_status,
        "text_only_enrichment_risk": enrichment_risk,
        "llm_provider": LLM_PROVIDER,
        "llm_model": model,
        "llm_prompt_version": PROMPT_VERSION,
        "llm_run_timestamp": run_timestamp,
    }
    if suggestion.get("__parse_recovered"):
        sidecar["parse_recovered"] = True
        sidecar["parse_recovery_source"] = suggestion.get("__parse_recovery_source")
    return sidecar


def build_sidecar_error(
    *,
    error_type: str,
    message: str,
    model: str,
    run_timestamp: str,
    raw_provider_output: str | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "type": error_type,
        "message": message,
    }
    if raw_provider_output:
        error["raw_provider_output"] = raw_provider_output
    return {
        "error": error,
        "llm_provider": LLM_PROVIDER,
        "llm_model": model,
        "llm_prompt_version": PROMPT_VERSION,
        "llm_run_timestamp": run_timestamp,
    }


def enrich_records(
    records: Sequence[dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    failure_log_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    run_timestamp = datetime.now(timezone.utc).isoformat()
    sidecar: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(record["question_id"])
        payload = build_enrichment_payload(record)
        try:
            suggestion = request_suggestion(client, model=model, payload=payload)
        except ModelResponseError as exc:
            error_record = build_sidecar_error(
                error_type="parse_error",
                message=str(exc),
                model=model,
                run_timestamp=run_timestamp,
                raw_provider_output=exc.raw_provider_output,
            )
            sidecar[question_id] = error_record
            _append_failure_log(
                failure_log_path,
                question_id=question_id,
                error_type="parse_error",
                error_message=str(exc),
                model=model,
                run_timestamp=run_timestamp,
                raw_provider_output=exc.raw_provider_output,
                request_payload=payload,
            )
            continue
        except ValueError as exc:
            error_record = build_sidecar_error(
                error_type="parse_error",
                message=str(exc),
                model=model,
                run_timestamp=run_timestamp,
            )
            sidecar[question_id] = error_record
            _append_failure_log(
                failure_log_path,
                question_id=question_id,
                error_type="parse_error",
                error_message=str(exc),
                model=model,
                run_timestamp=run_timestamp,
                raw_provider_output=None,
                request_payload=payload,
            )
            continue
        except Exception as exc:
            error_record = build_sidecar_error(
                error_type=DeepSeekErrorType.PROVIDER_ERROR,
                message=f"{exc.__class__.__name__}: {exc}",
                model=model,
                run_timestamp=run_timestamp,
            )
            sidecar[question_id] = error_record
            _append_failure_log(
                failure_log_path,
                question_id=question_id,
                error_type=DeepSeekErrorType.PROVIDER_ERROR,
                error_message=f"{exc.__class__.__name__}: {exc}",
                model=model,
                run_timestamp=run_timestamp,
                raw_provider_output=None,
                request_payload=payload,
            )
            continue

        sidecar[question_id] = build_sidecar_success(
            record,
            suggestion,
            model=model,
            run_timestamp=run_timestamp,
        )
    return sidecar


def write_sidecar(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_name": DEEPSEEK_SIDECAR_SCHEMA_NAME,
        "schema_version": DEEPSEEK_SIDECAR_SCHEMA_VERSION,
        "record_count": len(payload),
        "enrichments": payload,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_existing_sidecar(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    sidecar_path = Path(path)
    if not sidecar_path.exists():
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StartupConfigurationError(f"Existing sidecar is not valid JSON: {sidecar_path}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("enrichments"), dict):
        return {
            str(question_id): enrichment
            for question_id, enrichment in payload["enrichments"].items()
            if isinstance(enrichment, dict)
        }
    if isinstance(payload, dict):
        return {
            str(question_id): enrichment
            for question_id, enrichment in payload.items()
            if isinstance(enrichment, dict)
        }
    raise StartupConfigurationError(f"Existing sidecar format is not supported: {sidecar_path}")


def canonical_component_for_paper_family(value: Any) -> str:
    paper_family = str(value or "").strip().lower()
    return PAPER_FAMILY_TO_CANONICAL_COMPONENT.get(paper_family, paper_family)


def product_paper_family_for_component(value: Any) -> str:
    component = str(value or "").strip().lower()
    return CANONICAL_COMPONENT_TO_PAPER_FAMILY.get(component, component)


def load_canonical_taxonomy(taxonomy_root: str | Path, paper_family: Any) -> CanonicalTaxonomy:
    root = Path(taxonomy_root)
    component_key = canonical_component_for_paper_family(paper_family)
    if component_key not in {"p1", "p3", "m1", "s1"}:
        raise StartupConfigurationError(f"No canonical taxonomy is configured for paper family/component: {paper_family}")

    topic_map_path = root / "topic_filter_maps" / f"topic_filter_map_9709_{component_key}_v1.json"
    skill_map_path = root / "skill_maps" / f"skill_map_9709_{component_key}_v1.json"
    try:
        topic_map = json.loads(topic_map_path.read_text(encoding="utf-8"))
        skill_map = json.loads(skill_map_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StartupConfigurationError(
            f"Canonical taxonomy files are missing for component {component_key} under {root}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StartupConfigurationError(f"Canonical taxonomy file is not valid JSON for component {component_key}") from exc

    topics: dict[str, dict[str, Any]] = {}
    subtopics: dict[str, dict[str, Any]] = {}
    subtopic_to_topic: dict[str, str] = {}
    skill_to_subtopic: dict[str, str] = {}
    skill_to_topic: dict[str, str] = {}

    for topic in topic_map.get("topics", []):
        if not isinstance(topic, dict):
            continue
        topic_id = str(topic.get("topic_id", "")).strip()
        if not topic_id:
            continue
        topics[topic_id] = topic
        for subtopic in topic.get("subtopics", []):
            if not isinstance(subtopic, dict):
                continue
            subtopic_id = str(subtopic.get("subtopic_id", "")).strip()
            if not subtopic_id:
                continue
            subtopics[subtopic_id] = subtopic
            parent_topic_id = str(subtopic.get("parent_topic_id") or topic_id).strip()
            subtopic_to_topic[subtopic_id] = parent_topic_id
            for skill_id in subtopic.get("linked_skill_ids", []):
                if isinstance(skill_id, str) and skill_id.strip():
                    skill_to_subtopic[skill_id.strip()] = subtopic_id
                    skill_to_topic[skill_id.strip()] = parent_topic_id

    skills: dict[str, dict[str, Any]] = {}
    skill_to_asterion_region: dict[str, str] = {}
    for skill in skill_map.get("skills", []):
        if not isinstance(skill, dict):
            continue
        skill_id = str(skill.get("skill_id", "")).strip()
        if not skill_id:
            continue
        skills[skill_id] = skill
        region_id = str(skill.get("asterion_region_id", "")).strip()
        if region_id:
            skill_to_asterion_region[skill_id] = region_id

    return CanonicalTaxonomy(
        paper_family=product_paper_family_for_component(component_key),
        component_key=component_key,
        topic_map_path=topic_map_path,
        skill_map_path=skill_map_path,
        topics=topics,
        subtopics=subtopics,
        skills=skills,
        subtopic_to_topic=subtopic_to_topic,
        skill_to_subtopic=skill_to_subtopic,
        skill_to_topic=skill_to_topic,
        skill_to_asterion_region=skill_to_asterion_region,
    )


def taxonomy_prompt_summary(taxonomy: CanonicalTaxonomy) -> dict[str, Any]:
    topics: list[dict[str, Any]] = []
    for topic_id, topic in sorted(taxonomy.topics.items()):
        topics.append(
            {
                "topic_id": topic_id,
                "topic_name": topic.get("topic_name") or topic.get("official_section_name") or topic_id,
                "subtopic_ids": [
                    subtopic.get("subtopic_id")
                    for subtopic in topic.get("subtopics", [])
                    if isinstance(subtopic, dict) and subtopic.get("subtopic_id")
                ],
            }
        )

    subtopics = [
        {
            "subtopic_id": subtopic_id,
            "parent_topic_id": taxonomy.subtopic_to_topic.get(subtopic_id),
            "subtopic_name": subtopic.get("subtopic_name") or subtopic_id,
            "linked_skill_ids": subtopic.get("linked_skill_ids", []),
            "recognizer_signals": subtopic.get("recognizer_signals", []),
        }
        for subtopic_id, subtopic in sorted(taxonomy.subtopics.items())
    ]
    skills = [
        {
            "skill_id": skill_id,
            "name": skill.get("name") or skill_id,
            "description": skill.get("description", ""),
            "section": skill.get("section", ""),
            "asterion_region_id": skill.get("asterion_region_id", ""),
            "prerequisite_skill_ids": skill.get("prerequisite_skill_ids", []),
            "recognizer_signals": skill.get("recognizer_signals", []),
        }
        for skill_id, skill in sorted(taxonomy.skills.items())
    ]
    return {
        "component_key": taxonomy.component_key,
        "paper_family": taxonomy.paper_family,
        "allowed_topic_ids": sorted(taxonomy.topics),
        "allowed_subtopic_ids": sorted(taxonomy.subtopics),
        "allowed_skill_ids": sorted(taxonomy.skills),
        "topics": topics,
        "subtopics": subtopics,
        "skills": skills,
    }


def build_ai_assisted_question_payload(
    record: dict[str, Any],
    *,
    existing_enrichment: dict[str, Any] | None = None,
    include_subparts: bool = False,
) -> dict[str, Any]:
    payload = build_enrichment_payload(record)
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    local_difficulty = record.get("difficulty") if record.get("difficulty") is not None else notes.get("difficulty")
    local_difficulty_score = (
        record.get("difficulty_score")
        if record.get("difficulty_score") is not None
        else notes.get("difficulty_score")
    )
    payload.update(
        {
            "paper": record.get("paper"),
            "question_solution_marks": record.get("question_solution_marks"),
            "subparts": record.get("subparts", []) if include_subparts else [],
            "subparts_solution_marks": record.get("subparts_solution_marks", {}) if include_subparts else {},
            "page_refs": record.get("page_refs", {}),
            "local_topic": record.get("topic"),
            "local_difficulty": local_difficulty,
            "local_difficulty_score": local_difficulty_score,
        }
    )
    if existing_enrichment:
        payload["existing_deepseek_v1"] = {
            key: existing_enrichment.get(key)
            for key in [
                "deepseek_topic",
                "deepseek_subtopic",
                "deepseek_difficulty",
                "deepseek_difficulty_score",
                "deepseek_confidence_normalized",
                "deepseek_rationale",
                "deepseek_review_required",
                "topic_reconciliation_status",
                "final_review_required",
                "final_review_reasons",
            ]
            if key in existing_enrichment
        }
    return payload


def build_ai_assisted_batch_payload(
    records: Sequence[dict[str, Any]],
    *,
    taxonomy: CanonicalTaxonomy,
    existing_sidecar: dict[str, dict[str, Any]] | None = None,
    include_subparts: bool = False,
) -> dict[str, Any]:
    existing_sidecar = existing_sidecar or {}
    questions = [
        build_ai_assisted_question_payload(
            record,
            existing_enrichment=existing_sidecar.get(str(record["question_id"])),
            include_subparts=include_subparts,
        )
        for record in records
    ]
    return {
        "taxonomy": taxonomy_prompt_summary(taxonomy),
        "questions": questions,
    }


def build_ai_assisted_messages(payload: dict[str, Any], *, prompt_version: str = AI_ASSISTED_PROMPT_VERSION) -> list[dict[str, str]]:
    instructions = (
        "You are enriching CAIE 9709 exam-bank metadata against a fixed canonical taxonomy. "
        "The canonical taxonomy supplied in the user payload is the only allowed source for primary_topic_id, "
        "secondary_topic_ids, subtopic_ids, skill_ids, and prerequisite_skill_ids. "
        "Do not invent canonical topic IDs, subtopic IDs, or skill IDs. "
        "If the taxonomy is missing a useful category, use only needs_new_subtopic_candidate/suggested_new_subtopic "
        "or needs_new_skill_candidate/suggested_new_skill; suggestions are review-only and must not be inserted into ID fields. "
        "DeepSeek enriches and explains; deterministic code validates, normalizes, ranks, merges, and decides usability. "
        "Use image paths only as evidence references; the text/OCR and mark-scheme text may be lossy. "
        "Return one strict JSON object only, with no markdown and no prose. "
        "The top-level object must contain exactly one key named items. "
        "items must be an array containing one object for each whole-question mapping and, where useful, each subpart mapping. "
        f"Prompt version: {prompt_version}. "
        "Every item must contain exactly these keys: "
        + ", ".join(sorted(AI_ASSISTED_REQUIRED_ITEM_KEYS))
        + ". confidence and ai_difficulty_score must be JSON numbers from 0 to 1. "
        "ai_difficulty_estimate must be easy, average, or difficult. "
        "reviewed_status must be machine_candidate. mapping_source must be deepseek_ai_assisted. "
        "strict_filter_candidate may be true only when direct assessed evidence is strong, confidence is not low, "
        "review_required is false, evidence_missing has no required evidence, and the mapping is not prerequisite-only. "
        "Prerequisite skills should stay in prerequisite_skill_ids and should not become primary topics unless they are "
        "the main assessment target. "
        "worked_example_seed and warmup_seed must be concise lesson-generation seeds, not full lessons. "
        "Difficulty should consider method families, subparts, marks, algebraic transformation, non-obvious methods, "
        "mixed syllabus areas, common error risk, visual interpretation, and degraded text/OCR evidence."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def request_ai_assisted_batch(
    client: OpenAI,
    *,
    model: str,
    payload: dict[str, Any],
    taxonomy: CanonicalTaxonomy,
    expected_records: Sequence[dict[str, Any]],
    prompt_version: str = AI_ASSISTED_PROMPT_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    response = client.chat.completions.create(
        model=model,
        messages=build_ai_assisted_messages(payload, prompt_version=prompt_version),
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        parsed, parse_metadata = parse_response_with_recovery(
            response,
            lambda raw_text: parse_ai_assisted_model_json(
                raw_text,
                taxonomy=taxonomy,
                expected_records=expected_records,
            ),
        )
    except ValueError as exc:
        raw_output = getattr(exc, "raw_provider_output", None)
        raise ModelResponseError(str(exc), raw_provider_output=raw_output or _response_snapshot(response)) from exc
    return parsed["items"], parse_metadata, _response_snapshot(response)


def parse_ai_assisted_model_json(
    raw_text: str,
    *,
    taxonomy: CanonicalTaxonomy,
    expected_records: Sequence[dict[str, Any]] | None = None,
    allow_review_override: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(raw_text, object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError("AI-assisted model output was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("AI-assisted model output must be a JSON object.")
    if set(payload) != {"items"}:
        raise ValueError("AI-assisted model output must contain exactly one top-level key: items.")
    items = payload["items"]
    if not isinstance(items, list):
        raise ValueError("AI-assisted model output items must be an array.")

    expected_question_ids = {str(record.get("question_id")) for record in expected_records or []}
    expected_subpart_ids = _expected_subpart_ids(expected_records or [])
    validated_items = [
        validate_ai_assisted_item(
            item,
            taxonomy=taxonomy,
            expected_question_ids=expected_question_ids,
            expected_subpart_ids=expected_subpart_ids,
            allow_review_override=allow_review_override,
        )
        for item in items
    ]
    return {"items": validated_items}


def _expected_subpart_ids(records: Sequence[dict[str, Any]]) -> set[str]:
    subpart_ids: set[str] = set()
    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        for subpart in record.get("subparts", []) or []:
            label = str(subpart).strip()
            if question_id and label:
                subpart_ids.add(f"{question_id}_{label}")
    return subpart_ids


def validate_ai_assisted_item(
    item: Any,
    *,
    taxonomy: CanonicalTaxonomy,
    expected_question_ids: set[str] | None = None,
    expected_subpart_ids: set[str] | None = None,
    allow_review_override: bool = False,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("AI-assisted item must be a JSON object.")
    actual_keys = set(item)
    if actual_keys != AI_ASSISTED_REQUIRED_ITEM_KEYS:
        raise ValueError(
            "AI-assisted item keys did not match the required schema. "
            f"Expected {sorted(AI_ASSISTED_REQUIRED_ITEM_KEYS)}, got {sorted(actual_keys)}."
        )

    question_id = _require_non_empty_string(item["question_id"], "question_id")
    if expected_question_ids and question_id not in expected_question_ids:
        raise ValueError(f"AI-assisted item returned unexpected question_id: {question_id}")

    subpart_id_raw = item["subpart_id"]
    if subpart_id_raw is None:
        subpart_id = None
    elif isinstance(subpart_id_raw, str) and subpart_id_raw.strip():
        subpart_id = subpart_id_raw.strip()
        if expected_subpart_ids and subpart_id not in expected_subpart_ids:
            raise ValueError(f"AI-assisted item returned unexpected subpart_id: {subpart_id}")
    else:
        raise ValueError("subpart_id must be null or a non-empty string.")

    paper_family = str(item["paper_family"]).strip().lower()
    if paper_family != taxonomy.paper_family:
        raise ValueError(
            f"paper_family must be {taxonomy.paper_family} for component {taxonomy.component_key}, got {paper_family}."
        )

    primary_topic_id = _require_known_id(item["primary_topic_id"], "primary_topic_id", taxonomy.topics)
    secondary_topic_ids = _require_known_id_list(item["secondary_topic_ids"], "secondary_topic_ids", taxonomy.topics)
    subtopic_ids = _require_known_id_list(item["subtopic_ids"], "subtopic_ids", taxonomy.subtopics)
    skill_ids = _require_known_id_list(item["skill_ids"], "skill_ids", taxonomy.skills)
    prerequisite_skill_ids = _require_known_id_list(
        item["prerequisite_skill_ids"],
        "prerequisite_skill_ids",
        taxonomy.skills,
    )

    _validate_subtopic_topic_links(primary_topic_id, secondary_topic_ids, subtopic_ids, taxonomy=taxonomy)
    _validate_skill_topic_links(skill_ids, primary_topic_id, secondary_topic_ids, taxonomy=taxonomy)

    confidence_score = _require_probability(item["confidence"], "confidence")
    ai_difficulty_score = _require_probability(item["ai_difficulty_score"], "ai_difficulty_score")
    ai_difficulty_estimate = _require_one_of(
        item["ai_difficulty_estimate"],
        "ai_difficulty_estimate",
        {"easy", "average", "difficult"},
    )
    review_required = _require_bool(item["review_required"], "review_required")
    strict_filter_candidate = _require_bool(item["strict_filter_candidate"], "strict_filter_candidate")
    needs_new_subtopic_candidate = _require_bool(
        item["needs_new_subtopic_candidate"],
        "needs_new_subtopic_candidate",
    )
    needs_new_skill_candidate = _require_bool(item["needs_new_skill_candidate"], "needs_new_skill_candidate")

    normalized = {
        "question_id": question_id,
        "subpart_id": subpart_id,
        "paper_family": paper_family,
        "primary_topic_id": primary_topic_id,
        "secondary_topic_ids": secondary_topic_ids,
        "subtopic_ids": subtopic_ids,
        "skill_ids": skill_ids,
        "method_families": _require_string_list(item["method_families"], "method_families"),
        "prerequisite_skill_ids": prerequisite_skill_ids,
        "exam_techniques": _require_string_list(item["exam_techniques"], "exam_techniques"),
        "common_mistakes": _require_string_list(item["common_mistakes"], "common_mistakes"),
        "worked_example_seed": _require_non_empty_string(item["worked_example_seed"], "worked_example_seed"),
        "warmup_seed": _require_non_empty_string(item["warmup_seed"], "warmup_seed"),
        "strict_filter_candidate": strict_filter_candidate,
        "strict_filter_reason": _require_non_empty_string(item["strict_filter_reason"], "strict_filter_reason"),
        "evidence_used": _require_string_list(item["evidence_used"], "evidence_used"),
        "evidence_missing": _require_string_list(item["evidence_missing"], "evidence_missing"),
        "confidence": confidence_score,
        "review_required": review_required,
        "review_reasons": _require_string_list(item["review_reasons"], "review_reasons"),
        "ai_difficulty_estimate": ai_difficulty_estimate,
        "ai_difficulty_score": ai_difficulty_score,
        "ai_difficulty_factors": _require_string_list(item["ai_difficulty_factors"], "ai_difficulty_factors"),
        "needs_new_subtopic_candidate": needs_new_subtopic_candidate,
        "suggested_new_subtopic": _nullable_string(item["suggested_new_subtopic"], "suggested_new_subtopic"),
        "needs_new_skill_candidate": needs_new_skill_candidate,
        "suggested_new_skill": _nullable_string(item["suggested_new_skill"], "suggested_new_skill"),
        "mapping_source": _require_one_of(
            item["mapping_source"],
            "mapping_source",
            {"deepseek_ai_assisted"},
        ),
        "reviewed_status": _require_one_of(item["reviewed_status"], "reviewed_status", {"machine_candidate"}),
    }
    normalized["asterion_region_ids"] = [
        taxonomy.skill_to_asterion_region[skill_id]
        for skill_id in normalized["skill_ids"]
        if skill_id in taxonomy.skill_to_asterion_region
    ]
    _apply_strict_filter_fail_closed(normalized, allow_review_override=allow_review_override)
    return normalized


def _require_known_id(value: Any, field_name: str, allowed: dict[str, Any]) -> str:
    cleaned = _require_non_empty_string(value, field_name)
    if cleaned not in allowed:
        raise ValueError(f"{field_name} is not in the canonical taxonomy: {cleaned}")
    return cleaned


def _require_known_id_list(value: Any, field_name: str, allowed: dict[str, Any]) -> list[str]:
    ids = _require_string_list(value, field_name)
    unknown = [item for item in ids if item not in allowed]
    if unknown:
        raise ValueError(f"{field_name} contains unknown canonical IDs: {', '.join(unknown)}")
    return ids


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array.")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        cleaned = item.strip()
        if cleaned:
            strings.append(cleaned)
    return strings


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number from 0 to 1.")
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must be a JSON number from 0 to 1.")
    return round(numeric, 4)


def _require_one_of(value: Any, field_name: str, allowed: set[str]) -> str:
    cleaned = _require_non_empty_string(value, field_name)
    if cleaned not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}, got {cleaned}.")
    return cleaned


def _nullable_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    raise ValueError(f"{field_name} must be a string or null.")


def _validate_subtopic_topic_links(
    primary_topic_id: str,
    secondary_topic_ids: list[str],
    subtopic_ids: list[str],
    *,
    taxonomy: CanonicalTaxonomy,
) -> None:
    allowed_topics = {primary_topic_id, *secondary_topic_ids}
    mismatched = [
        subtopic_id
        for subtopic_id in subtopic_ids
        if taxonomy.subtopic_to_topic.get(subtopic_id) not in allowed_topics
    ]
    if mismatched:
        raise ValueError(
            "subtopic_ids must belong to primary_topic_id or secondary_topic_ids: "
            + ", ".join(mismatched)
        )


def _validate_skill_topic_links(
    skill_ids: list[str],
    primary_topic_id: str,
    secondary_topic_ids: list[str],
    *,
    taxonomy: CanonicalTaxonomy,
) -> None:
    allowed_topics = {primary_topic_id, *secondary_topic_ids}
    mismatched = [
        skill_id
        for skill_id in skill_ids
        if taxonomy.skill_to_topic.get(skill_id) and taxonomy.skill_to_topic.get(skill_id) not in allowed_topics
    ]
    if mismatched:
        raise ValueError(
            "skill IDs must belong to primary_topic_id or secondary_topic_ids: "
            + ", ".join(mismatched)
        )


def _apply_strict_filter_fail_closed(item: dict[str, Any], *, allow_review_override: bool) -> None:
    disqualifiers: list[str] = []
    if item["confidence"] < STRICT_FILTER_MIN_CONFIDENCE:
        disqualifiers.append("confidence_below_strict_filter_threshold")
    if item["review_required"] and not allow_review_override:
        disqualifiers.append("review_required")
    if item["evidence_missing"]:
        disqualifiers.append("required_evidence_missing")
    if not item["subtopic_ids"] or not item["skill_ids"]:
        disqualifiers.append("broad_or_prerequisite_only_mapping")
    if _looks_prerequisite_only(item):
        disqualifiers.append("prerequisite_only_mapping")
    if disqualifiers:
        item["strict_filter_candidate"] = False
        existing_reasons = list(item["review_reasons"])
        item["review_reasons"] = sorted(set(existing_reasons + disqualifiers))
        item["strict_filter_reason"] = (
            f"Excluded from strict filtering: {', '.join(disqualifiers)}."
        )


def _looks_prerequisite_only(item: dict[str, Any]) -> bool:
    if not item["skill_ids"] and item["prerequisite_skill_ids"]:
        return True
    combined_text = " ".join(
        str(value)
        for value in [
            item.get("strict_filter_reason", ""),
            *item.get("review_reasons", []),
            *item.get("method_families", []),
        ]
    ).lower()
    return "prerequisite-only" in combined_text or "prerequisite only" in combined_text


def is_human_reviewed(enrichment: dict[str, Any] | None) -> bool:
    if not isinstance(enrichment, dict):
        return False
    reviewed_status = str(enrichment.get("reviewed_status") or enrichment.get("human_review_status") or "").lower()
    if reviewed_status in {"reviewed", "human_reviewed", "accepted"}:
        return True
    return bool(enrichment.get("human_reviewed") is True or enrichment.get("reviewed_by_human") is True)


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_batch_id(records: Sequence[dict[str, Any]], *, batch_by_paper: bool = True) -> str:
    if not records:
        return "empty"
    first = records[0]
    paper_family = str(first.get("paper_family", "unknown")).strip().lower()
    paper = str(first.get("paper", "")).strip()
    if batch_by_paper and paper:
        return f"{paper_family}_{paper}"
    question_ids = [str(record.get("question_id")) for record in records]
    digest = stable_json_hash(question_ids)[:12]
    return f"{paper_family}_{paper or 'batch'}_{digest}"


def batch_records(
    records: Sequence[dict[str, Any]],
    *,
    batch_by_paper: bool = True,
    batch_size: int = 20,
) -> list[list[dict[str, Any]]]:
    if batch_by_paper:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[(str(record.get("paper_family", "")).lower(), str(record.get("paper", "")))].append(record)
        return [
            sorted(batch, key=lambda record: str(record.get("question_id", "")))
            for _, batch in sorted(grouped.items())
        ]

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (str(item.get("paper_family", "")), str(item.get("paper", "")), str(item.get("question_id", "")))):
        current.append(record)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def should_rerun_ai_assisted_record(
    record: dict[str, Any],
    existing_enrichment: dict[str, Any] | None,
    *,
    prompt_version: str,
    resume: bool = False,
    only_errors: bool = False,
    only_review_required: bool = False,
    only_topic_mismatch: bool = False,
    only_unmapped_labels: bool = False,
    explicit_question_ids: set[str] | None = None,
) -> bool:
    question_id = str(record.get("question_id", "")).strip()
    explicit = bool(explicit_question_ids and question_id in explicit_question_ids)
    error = existing_enrichment.get("error") if isinstance(existing_enrichment, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    missing = not existing_enrichment
    old_prompt = (
        isinstance(existing_enrichment, dict)
        and existing_enrichment.get("llm_prompt_version") != prompt_version
    )
    resumable_error = error_type in {DeepSeekErrorType.PROVIDER_ERROR, "parse_error"}

    if only_errors and not (missing or resumable_error):
        return False
    if only_review_required and not bool(existing_enrichment and existing_enrichment.get("final_review_required")):
        return False
    if only_topic_mismatch and not (
        existing_enrichment and existing_enrichment.get("topic_reconciliation_status") == ReconciliationStatus.MISMATCH
    ):
        return False
    if only_unmapped_labels and not (
        existing_enrichment and existing_enrichment.get("topic_reconciliation_status") == ReconciliationStatus.UNMAPPED_LABEL
    ):
        return False
    if is_human_reviewed(existing_enrichment):
        return False
    if explicit:
        return True
    if resume and isinstance(existing_enrichment, dict) and not error and not old_prompt:
        return False
    return missing or resumable_error or old_prompt or not resume


def select_ai_assisted_records(
    records: Sequence[dict[str, Any]],
    *,
    existing_sidecar: dict[str, dict[str, Any]] | None = None,
    component: str | None = None,
    paper: str | None = None,
    question_ids: Sequence[str] | None = None,
    limit: int = 25,
    resume: bool = False,
    prompt_version: str = AI_ASSISTED_PROMPT_VERSION,
    only_errors: bool = False,
    only_review_required: bool = False,
    only_topic_mismatch: bool = False,
    only_unmapped_labels: bool = False,
) -> list[dict[str, Any]]:
    selected = list(records)
    existing_sidecar = existing_sidecar or {}
    if component:
        wanted_family = product_paper_family_for_component(canonical_component_for_paper_family(component))
        selected = [record for record in selected if str(record.get("paper_family", "")).lower() == wanted_family]
    if paper:
        selected = [record for record in selected if str(record.get("paper", "")).strip() == paper]
    explicit_ids = {value.strip() for value in question_ids or [] if value.strip()}
    if explicit_ids:
        selected = [record for record in selected if str(record.get("question_id", "")).strip() in explicit_ids]

    selected = [
        record
        for record in selected
        if should_rerun_ai_assisted_record(
            record,
            existing_sidecar.get(str(record.get("question_id", ""))),
            prompt_version=prompt_version,
            resume=resume,
            only_errors=only_errors,
            only_review_required=only_review_required,
            only_topic_mismatch=only_topic_mismatch,
            only_unmapped_labels=only_unmapped_labels,
            explicit_question_ids=explicit_ids or None,
        )
    ]
    selected.sort(key=lambda record: (str(record.get("paper_family", "")), str(record.get("paper", "")), str(record.get("question_id", ""))))
    if limit >= 0:
        selected = selected[:limit]
    return selected


def build_ai_assisted_record(
    record: dict[str, Any],
    *,
    items: Sequence[dict[str, Any]],
    existing_enrichment: dict[str, Any] | None,
    model: str,
    prompt_version: str,
    run_timestamp: str,
    batch_id: str,
    batch_input_hash: str,
    taxonomy: CanonicalTaxonomy,
    parse_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if is_human_reviewed(existing_enrichment):
        preserved = dict(existing_enrichment or {})
        preserved["ai_assisted_preserved_human_review"] = True
        return preserved

    base = dict(existing_enrichment or {})
    question_items = [dict(item) for item in items if item.get("question_id") == record.get("question_id")]
    strict_items = [
        item for item in question_items if item.get("strict_filter_candidate") is True and item.get("review_required") is False
    ]
    final_review_reasons = sorted(
        set(
            str(reason)
            for item in question_items
            for reason in item.get("review_reasons", [])
            if str(reason).strip()
        )
    )
    if any(item.get("review_required") for item in question_items):
        final_review_reasons.append("ai_assisted_item_review_required")
    if not strict_items:
        final_review_reasons.append("no_default_strict_filter_candidate")

    payload_for_hash = build_ai_assisted_question_payload(record, existing_enrichment=existing_enrichment, include_subparts=True)
    base.update(
        {
            "question_id": record.get("question_id"),
            "paper": record.get("paper"),
            "paper_family": record.get("paper_family"),
            "ai_assisted_schema_version": AI_ASSISTED_SIDECAR_SCHEMA_VERSION,
            "ai_assisted_items": question_items,
            "strict_filter_candidates": [
                {
                    "subpart_id": item.get("subpart_id"),
                    "primary_topic_id": item.get("primary_topic_id"),
                    "secondary_topic_ids": item.get("secondary_topic_ids", []),
                    "subtopic_ids": item.get("subtopic_ids", []),
                    "skill_ids": item.get("skill_ids", []),
                    "asterion_region_ids": item.get("asterion_region_ids", []),
                    "confidence": item.get("confidence"),
                }
                for item in strict_items
            ],
            "ai_final_review_required": bool(final_review_reasons),
            "ai_final_review_reasons": sorted(set(final_review_reasons)),
            "mapping_source": "deepseek_ai_assisted",
            "reviewed_status": base.get("reviewed_status", "machine_candidate"),
            "llm_provider": LLM_PROVIDER,
            "llm_model": model,
            "llm_prompt_version": prompt_version,
            "llm_run_timestamp": run_timestamp,
            "input_hash": stable_json_hash(payload_for_hash),
            "batch_id": batch_id,
            "batch_input_hash": batch_input_hash,
            "taxonomy_component_key": taxonomy.component_key,
            "taxonomy_topic_map_path": str(taxonomy.topic_map_path),
            "taxonomy_skill_map_path": str(taxonomy.skill_map_path),
        }
    )
    if parse_metadata and parse_metadata.get("parse_recovered"):
        base["parse_recovered"] = True
        base["parse_recovery_source"] = parse_metadata.get("parse_recovery_source")
    return base


def build_ai_assisted_error_record(
    record: dict[str, Any],
    *,
    existing_enrichment: dict[str, Any] | None,
    error_type: str,
    message: str,
    model: str,
    prompt_version: str,
    run_timestamp: str,
    batch_id: str,
    batch_input_hash: str,
    raw_provider_output: str | None = None,
) -> dict[str, Any]:
    if is_human_reviewed(existing_enrichment):
        preserved = dict(existing_enrichment or {})
        preserved["ai_assisted_preserved_human_review"] = True
        return preserved
    base = dict(existing_enrichment or {})
    error: dict[str, Any] = {"type": error_type, "message": message}
    if raw_provider_output:
        error["raw_provider_output"] = raw_provider_output
    base.update(
        {
            "question_id": record.get("question_id"),
            "paper": record.get("paper"),
            "paper_family": record.get("paper_family"),
            "error": error,
            "ai_assisted_items": [],
            "ai_final_review_required": True,
            "ai_final_review_reasons": [f"ai_assisted_{error_type}"],
            "llm_provider": LLM_PROVIDER,
            "llm_model": model,
            "llm_prompt_version": prompt_version,
            "llm_run_timestamp": run_timestamp,
            "batch_id": batch_id,
            "batch_input_hash": batch_input_hash,
        }
    )
    return base


def calibrate_difficulty_by_paper_family(
    enrichments: dict[str, dict[str, Any]],
    question_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records_by_id = {str(record.get("question_id")): record for record in question_records}
    scored: dict[str, tuple[str, float, dict[str, Any]]] = {}
    for question_id, enrichment in enrichments.items():
        if not isinstance(enrichment, dict) or isinstance(enrichment.get("error"), dict):
            continue
        record = records_by_id.get(question_id, {})
        paper_family = str(enrichment.get("paper_family") or record.get("paper_family") or "").lower()
        if not paper_family:
            continue
        score, basis = deterministic_difficulty_score(enrichment, record)
        scored[question_id] = (paper_family, score, basis)

    by_family: dict[str, list[tuple[str, float, dict[str, Any]]]] = defaultdict(list)
    for question_id, (paper_family, score, basis) in scored.items():
        by_family[paper_family].append((question_id, score, basis))

    calibrated = {question_id: dict(enrichment) for question_id, enrichment in enrichments.items()}
    for paper_family, family_rows in by_family.items():
        family_rows.sort(key=lambda row: (-row[1], row[0]))
        count = len(family_rows)
        for index, (question_id, score, basis) in enumerate(family_rows):
            percentile = 50.0 if count == 1 else round(100.0 * (count - 1 - index) / (count - 1), 2)
            calibrated[question_id]["deterministic_difficulty_percentile"] = percentile
            calibrated[question_id]["deterministic_difficulty_band"] = difficulty_band_for_percentile(percentile)
            calibrated[question_id]["difficulty_rank_within_paper_family"] = index + 1
            calibrated[question_id]["difficulty_rank_basis"] = {
                "paper_family": paper_family,
                "rank_direction": "1 is most difficult within paper_family",
                "combined_score": round(score, 4),
                **basis,
            }
    return calibrated


def deterministic_difficulty_score(enrichment: dict[str, Any], record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    items = enrichment.get("ai_assisted_items") if isinstance(enrichment.get("ai_assisted_items"), list) else []
    ai_scores = [
        float(item["ai_difficulty_score"])
        for item in items
        if isinstance(item, dict) and isinstance(item.get("ai_difficulty_score"), (int, float))
    ]
    ai_score = max(ai_scores) if ai_scores else None

    mark_total = normalize_difficulty_score(record.get("question_solution_marks"))
    if mark_total is None and isinstance(record.get("notes"), dict):
        mark_total = normalize_difficulty_score(record["notes"].get("question_total_detected"))
    marks_component = min(float(mark_total or 0), 12.0) / 12.0

    subpart_count = len(record.get("subparts", []) or [])
    subpart_component = min(float(subpart_count), 5.0) / 5.0

    method_count = len({method for item in items if isinstance(item, dict) for method in item.get("method_families", [])})
    method_component = min(float(method_count), 5.0) / 5.0

    topic_links = {
        topic_id
        for item in items
        if isinstance(item, dict)
        for topic_id in [item.get("primary_topic_id"), *item.get("secondary_topic_ids", [])]
        if topic_id
    }
    topic_component = min(float(len(topic_links)), 4.0) / 4.0

    risk_flags = 0
    if enrichment.get("ai_final_review_required") or enrichment.get("final_review_required"):
        risk_flags += 1
    if record.get("visual_required") or (isinstance(record.get("notes"), dict) and record["notes"].get("visual_required")):
        risk_flags += 1
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    question_text_trust = record.get("question_text_trust") or notes.get("question_text_trust")
    if str(question_text_trust or "").lower() == "low":
        risk_flags += 1
    risk_component = min(float(risk_flags), 3.0) / 3.0

    local_score = normalize_difficulty_score(record.get("difficulty_score"))
    if local_score is None and isinstance(record.get("notes"), dict):
        local_score = normalize_difficulty_score(record["notes"].get("difficulty_score"))
    local_component = (local_score / 100.0) if local_score is not None else 0.5
    ai_component = ai_score if ai_score is not None else local_component

    combined = (
        0.5 * ai_component
        + 0.15 * marks_component
        + 0.12 * subpart_component
        + 0.12 * method_component
        + 0.06 * topic_component
        + 0.05 * risk_component
    )
    basis = {
        "ai_difficulty_score": ai_score,
        "local_difficulty_score": local_score,
        "mark_total": mark_total,
        "subpart_count": subpart_count,
        "method_family_count": method_count,
        "topic_link_count": len(topic_links),
        "review_or_evidence_risk_flags": risk_flags,
        "weights": {
            "ai_difficulty_score": 0.5,
            "mark_total": 0.15,
            "subpart_count": 0.12,
            "method_family_count": 0.12,
            "topic_link_count": 0.06,
            "review_or_evidence_risk_flags": 0.05,
        },
    }
    return combined, basis


def difficulty_band_for_percentile(percentile: float) -> str:
    if percentile < 25:
        return "foundation"
    if percentile < 60:
        return "standard"
    if percentile < 85:
        return "challenging"
    return "advanced"


def write_ai_assisted_sidecar(
    enrichments: dict[str, dict[str, Any]],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_name": AI_ASSISTED_SIDECAR_SCHEMA_NAME,
        "schema_version": AI_ASSISTED_SIDECAR_SCHEMA_VERSION,
        "record_count": len(enrichments),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "enrichments": enrichments,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def batch_cache_dir(output_path: str | Path) -> Path:
    output = Path(output_path)
    stem = output.stem
    return output.parent / f"{stem}.batches"


def batch_cache_path(output_path: str | Path, batch_id: str) -> Path:
    safe_batch_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_id)
    return batch_cache_dir(output_path) / f"{safe_batch_id}.json"


def write_batch_cache(output_path: str | Path, batch_payload: dict[str, Any]) -> Path:
    path = batch_cache_path(output_path, str(batch_payload["batch_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(batch_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_successful_batch_cache(output_path: str | Path, batch_id: str, *, batch_input_hash: str) -> dict[str, Any] | None:
    path = batch_cache_path(output_path, batch_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "success":
        return None
    if payload.get("batch_input_hash") != batch_input_hash:
        return None
    return payload


def enrich_ai_assisted_records(
    records: Sequence[dict[str, Any]],
    *,
    client: OpenAI,
    taxonomy_root: str | Path,
    existing_sidecar: dict[str, dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    prompt_version: str = AI_ASSISTED_PROMPT_VERSION,
    include_subparts: bool = False,
    batch_by_paper: bool = True,
    resume: bool = False,
    failure_log_path: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_timestamp = datetime.now(timezone.utc).isoformat()
    existing_sidecar = existing_sidecar or {}
    enrichments: dict[str, dict[str, Any]] = {}
    manifest_batches: list[dict[str, Any]] = []

    batches = batch_records(records, batch_by_paper=batch_by_paper)
    for batch in batches:
        batch_id = stable_batch_id(batch, batch_by_paper=batch_by_paper)
        taxonomy = load_canonical_taxonomy(taxonomy_root, batch[0].get("paper_family"))
        batch_payload = build_ai_assisted_batch_payload(
            batch,
            taxonomy=taxonomy,
            existing_sidecar=existing_sidecar,
            include_subparts=include_subparts,
        )
        batch_input_hash = stable_json_hash(batch_payload)
        question_ids = [str(record["question_id"]) for record in batch]

        cached = (
            read_successful_batch_cache(output_path, batch_id, batch_input_hash=batch_input_hash)
            if output_path and resume
            else None
        )
        if cached:
            items = cached.get("items", [])
            parse_metadata = cached.get("parse_metadata", {})
            status = "success_cached"
            raw_provider_output = cached.get("raw_provider_output")
        else:
            try:
                items, parse_metadata, raw_provider_output = request_ai_assisted_batch(
                    client,
                    model=model,
                    payload=batch_payload,
                    taxonomy=taxonomy,
                    expected_records=batch,
                    prompt_version=prompt_version,
                )
                status = "success"
            except ModelResponseError as exc:
                status = "parse_error"
                items = []
                parse_metadata = {}
                raw_provider_output = exc.raw_provider_output
                for record in batch:
                    question_id = str(record["question_id"])
                    enrichments[question_id] = build_ai_assisted_error_record(
                        record,
                        existing_enrichment=existing_sidecar.get(question_id),
                        error_type="parse_error",
                        message=str(exc),
                        model=model,
                        prompt_version=prompt_version,
                        run_timestamp=run_timestamp,
                        batch_id=batch_id,
                        batch_input_hash=batch_input_hash,
                        raw_provider_output=exc.raw_provider_output,
                    )
                    _append_failure_log(
                        failure_log_path,
                        question_id=question_id,
                        error_type="parse_error",
                        error_message=str(exc),
                        model=model,
                        run_timestamp=run_timestamp,
                        raw_provider_output=exc.raw_provider_output,
                        request_payload=batch_payload,
                    )
            except Exception as exc:
                status = DeepSeekErrorType.PROVIDER_ERROR
                items = []
                parse_metadata = {}
                raw_provider_output = None
                message = f"{exc.__class__.__name__}: {exc}"
                for record in batch:
                    question_id = str(record["question_id"])
                    enrichments[question_id] = build_ai_assisted_error_record(
                        record,
                        existing_enrichment=existing_sidecar.get(question_id),
                        error_type=DeepSeekErrorType.PROVIDER_ERROR,
                        message=message,
                        model=model,
                        prompt_version=prompt_version,
                        run_timestamp=run_timestamp,
                        batch_id=batch_id,
                        batch_input_hash=batch_input_hash,
                    )
                    _append_failure_log(
                        failure_log_path,
                        question_id=question_id,
                        error_type=DeepSeekErrorType.PROVIDER_ERROR,
                        error_message=message,
                        model=model,
                        run_timestamp=run_timestamp,
                        raw_provider_output=None,
                        request_payload=batch_payload,
                    )

        if status in {"success", "success_cached"}:
            for record in batch:
                question_id = str(record["question_id"])
                enrichments[question_id] = build_ai_assisted_record(
                    record,
                    items=items,
                    existing_enrichment=existing_sidecar.get(question_id),
                    model=model,
                    prompt_version=prompt_version,
                    run_timestamp=run_timestamp,
                    batch_id=batch_id,
                    batch_input_hash=batch_input_hash,
                    taxonomy=taxonomy,
                    parse_metadata=parse_metadata,
                )

        cache_payload = {
            "batch_id": batch_id,
            "status": "success" if status == "success_cached" else status,
            "question_ids": question_ids,
            "paper": batch[0].get("paper"),
            "paper_family": batch[0].get("paper_family"),
            "batch_input_hash": batch_input_hash,
            "llm_model": model,
            "llm_prompt_version": prompt_version,
            "llm_run_timestamp": run_timestamp,
            "items": items,
            "parse_metadata": parse_metadata,
            "raw_provider_output": raw_provider_output,
        }
        cache_path = write_batch_cache(output_path, cache_payload) if output_path else None
        manifest_batches.append(
            {
                "batch_id": batch_id,
                "status": status,
                "question_ids": question_ids,
                "paper": batch[0].get("paper"),
                "paper_family": batch[0].get("paper_family"),
                "batch_input_hash": batch_input_hash,
                "cache_path": str(cache_path) if cache_path else None,
            }
        )

    return enrichments, {
        "run_timestamp": run_timestamp,
        "batch_count": len(manifest_batches),
        "batches": manifest_batches,
        "model": model,
        "prompt_version": prompt_version,
    }


def parse_ai_assisted_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek AI-assisted canonical taxonomy enrichment.")
    add_ai_assisted_cli_arguments(parser)
    args = parser.parse_args(argv)
    finalize_ai_assisted_args(args)
    return args


def add_ai_assisted_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to question_bank.json.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("exam_bank_taxonomy/canonical"),
        help="Path to the active canonical taxonomy directory.",
    )
    parser.add_argument(
        "--existing-sidecar",
        type=Path,
        default=None,
        help="Optional existing DeepSeek sidecar to preserve and use as evidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/json/question_bank.ai_assisted.v2.json"),
        help="Path to write the v2 AI-assisted sidecar.",
    )
    parser.add_argument("--component", choices=["p1", "p3", "p4", "p5", "m1", "s1"], default=None)
    parser.add_argument("--paper", default=None)
    parser.add_argument("--question-id", action="append", default=None, help="Question ID to enrich. Repeatable.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=AI_ASSISTED_PROMPT_VERSION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--failure-log", type=Path, default=None)
    parser.add_argument("--only-errors", action="store_true")
    parser.add_argument("--only-review-required", action="store_true")
    parser.add_argument("--only-topic-mismatch", action="store_true")
    parser.add_argument("--only-unmapped-labels", action="store_true")
    parser.add_argument("--include-subparts", action="store_true")
    parser.add_argument("--batch-by-paper", action="store_true", default=True)
    parser.add_argument("--no-batch-by-paper", action="store_false", dest="batch_by_paper")
    parser.add_argument("--recompute-difficulty", action="store_true")
    parser.add_argument(
        "--allow-provider-failure",
        action="store_true",
        help="Exit 0 even if every attempted AI-assisted batch fails with a provider/API error.",
    )


def finalize_ai_assisted_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.limit < 0:
        raise StartupConfigurationError("--limit must be zero or greater.")
    args.question_ids = _parse_question_ids(args.question_id)
    return args


def run_ai_assisted(argv: Sequence[str] | None = None) -> int:
    args = parse_ai_assisted_args(argv)
    return run_ai_assisted_from_args(args)


def run_ai_assisted_from_args(args: argparse.Namespace) -> int:
    validate_paths(input_path=args.input, output_path=args.output)
    records = load_question_bank(args.input)
    existing_sidecar: dict[str, dict[str, Any]] = {}
    if args.existing_sidecar:
        existing_sidecar.update(load_existing_sidecar(args.existing_sidecar))
    if args.output.exists():
        existing_sidecar.update(load_existing_sidecar(args.output))

    selected = select_ai_assisted_records(
        records,
        existing_sidecar=existing_sidecar,
        component=args.component,
        paper=args.paper,
        question_ids=args.question_ids,
        limit=args.limit,
        resume=args.resume,
        prompt_version=args.prompt_version,
        only_errors=args.only_errors,
        only_review_required=args.only_review_required,
        only_topic_mismatch=args.only_topic_mismatch,
        only_unmapped_labels=args.only_unmapped_labels,
    )

    if args.dry_run:
        summary = {
            "input": str(args.input),
            "taxonomy": str(args.taxonomy),
            "existing_sidecar": str(args.existing_sidecar) if args.existing_sidecar else None,
            "output": str(args.output),
            "selected_count": len(selected),
            "question_ids": [record["question_id"] for record in selected],
            "batch_by_paper": args.batch_by_paper,
            "would_call_network": False,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    failure_log_path = args.failure_log or default_failure_log_path(args.output)
    if selected:
        client = create_client(base_url=args.base_url)
        enrichments, run_manifest = enrich_ai_assisted_records(
            selected,
            client=client,
            taxonomy_root=args.taxonomy,
            existing_sidecar=existing_sidecar,
            output_path=args.output,
            model=args.model,
            prompt_version=args.prompt_version,
            include_subparts=args.include_subparts,
            batch_by_paper=args.batch_by_paper,
            resume=args.resume,
            failure_log_path=failure_log_path,
        )
    else:
        enrichments = {}
        run_manifest = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_count": 0,
            "batches": [],
            "model": args.model,
            "prompt_version": args.prompt_version,
        }

    merged = dict(existing_sidecar)
    merged.update(enrichments)
    if args.recompute_difficulty or enrichments:
        merged = calibrate_difficulty_by_paper_family(merged, records)

    metadata = {
        "input_path": str(args.input),
        "taxonomy_path": str(args.taxonomy),
        "existing_sidecar_path": str(args.existing_sidecar) if args.existing_sidecar else None,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "selected_count": len(selected),
        "resume": bool(args.resume),
        "batch_by_paper": bool(args.batch_by_paper),
        "difficulty_calibration": {
            "scope": "within paper_family",
            "bands": {
                "foundation": "0 <= percentile < 25",
                "standard": "25 <= percentile < 60",
                "challenging": "60 <= percentile < 85",
                "advanced": "85 <= percentile <= 100",
            },
        },
        "run_manifest": run_manifest,
    }
    write_ai_assisted_sidecar(merged, args.output, metadata=metadata)
    print(f"Wrote {len(merged)} AI-assisted enrichment records to {args.output}")
    counts = enrichment_failure_counts(enrichments)
    if counts["failed"]:
        print(
            f"AI-assisted enrichment completed with {counts['succeeded']} successes and "
            f"{counts['failed']} failures ({counts['provider_failed']} provider/API failures)."
        )
        print(f"Logged failure details to {failure_log_path}")
    if should_fail_for_provider_errors(enrichments, allow_provider_failure=args.allow_provider_failure):
        print(
            "All attempted AI-assisted enrichments failed with provider/API errors. "
            "Use --allow-provider-failure to preserve the sidecar and exit 0."
        )
        return 1
    return 0


def enrichment_failure_counts(sidecar: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {
        "attempted": len(sidecar),
        "succeeded": 0,
        "failed": 0,
        "provider_failed": 0,
    }
    for value in sidecar.values():
        error = value.get("error")
        if not isinstance(error, dict):
            counts["succeeded"] += 1
            continue
        counts["failed"] += 1
        if error.get("type") == DeepSeekErrorType.PROVIDER_ERROR:
            counts["provider_failed"] += 1
    return counts


def should_fail_for_provider_errors(sidecar: dict[str, dict[str, Any]], *, allow_provider_failure: bool) -> bool:
    if allow_provider_failure:
        return False
    counts = enrichment_failure_counts(sidecar)
    return counts["attempted"] > 0 and counts["provider_failed"] == counts["attempted"]


def default_failure_log_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix == ".json":
        return output.with_suffix(".failures.jsonl")
    return output.parent / f"{output.name}.failures.jsonl"


def _append_failure_log(
    failure_log_path: str | Path | None,
    *,
    question_id: str,
    error_type: str,
    error_message: str,
    model: str,
    run_timestamp: str,
    raw_provider_output: str | None,
    request_payload: dict[str, Any],
) -> None:
    if failure_log_path is None:
        return
    path = Path(failure_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "question_id": question_id,
        "error_type": error_type,
        "error_message": error_message,
        "llm_provider": LLM_PROVIDER,
        "llm_model": model,
        "llm_prompt_version": PROMPT_VERSION,
        "llm_run_timestamp": run_timestamp,
        "raw_provider_output": raw_provider_output,
        "request_payload": request_payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def validate_paths(*, input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise StartupConfigurationError("Output path must be different from the input question bank path.")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_paths(input_path=args.input, output_path=args.output)
    failure_log_path = args.failure_log or default_failure_log_path(args.output)
    records = load_question_bank(args.input)
    selected = select_records(
        records,
        limit=args.limit,
        question_ids=args.question_ids,
        paper_family=args.paper_family,
    )

    if args.dry_run:
        summary = {
            "input": str(args.input),
            "output": str(args.output),
            "failure_log": str(failure_log_path),
            "selected_count": len(selected),
            "question_ids": [record["question_id"] for record in selected],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    client = create_client(base_url=args.base_url)
    sidecar = enrich_records(selected, client=client, model=args.model, failure_log_path=failure_log_path)
    write_sidecar(sidecar, args.output)
    print(f"Wrote {len(sidecar)} DeepSeek enrichment records to {args.output}")
    counts = enrichment_failure_counts(sidecar)
    if counts["failed"]:
        print(
            f"DeepSeek enrichment completed with {counts['succeeded']} successes and "
            f"{counts['failed']} failures ({counts['provider_failed']} provider/API failures)."
        )
        print(f"Logged failure details to {failure_log_path}")
    if should_fail_for_provider_errors(sidecar, allow_provider_failure=args.allow_provider_failure):
        print(
            "All attempted DeepSeek enrichments failed with provider/API errors. "
            "Use --allow-provider-failure to preserve the sidecar and exit 0."
        )
        return 1
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except StartupConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
