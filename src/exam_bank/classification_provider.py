from __future__ import annotations

import json
from typing import Any

from .config import AppConfig
from .models import ClassificationResult
from .trust import Confidence


def classify_with_openai(text: str, marks: int | None, config: AppConfig, local: ClassificationResult) -> ClassificationResult:
    from openai import OpenAI

    family_taxonomy = config.paper_family_taxonomy.get(local.paper_family, {}) if local.paper_family != "unknown" else config.paper_family_taxonomy
    schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "subtopic": {"type": "string"},
            "topic_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "topic_evidence": {"type": "string"},
            "difficulty": {"type": "string", "enum": config.difficulty_labels},
            "difficulty_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "difficulty_evidence": {"type": "string"},
        },
        "required": ["topic", "subtopic", "topic_confidence", "topic_evidence", "difficulty", "difficulty_confidence", "difficulty_evidence"],
        "additionalProperties": False,
    }
    client = OpenAI(timeout=config.classification.openai_timeout_seconds)
    prompt = (
        "Classify this CAIE 9709 maths question. Use only the supplied paper-family topic bank. "
        f"Paper family: {local.paper_family}. Topic bank:\n{json.dumps(family_taxonomy, indent=2)}\n\n"
        f"Marks: {marks if marks is not None else 'unknown'}\nQuestion:\n{text[:6000]}"
    )
    response = client.responses.create(
        model=config.classification.openai_model,
        input=prompt,
        text={"format": {"type": "json_schema", "name": "exam_question_classification", "schema": schema, "strict": True}},
    )
    data = json.loads(_response_text(response))
    if not _is_valid_topic_path(local.paper_family, str(data["topic"]), str(data["subtopic"]), config):
        return local
    local.topic = str(data["topic"])
    local.subtopic = str(data["subtopic"])
    local.topic_confidence = str(data["topic_confidence"])
    local.topic_evidence = str(data["topic_evidence"])
    local.difficulty = str(data["difficulty"])
    local.difficulty_confidence = str(data["difficulty_confidence"])
    local.difficulty_evidence = str(data["difficulty_evidence"])
    local.difficulty_uncertain = local.difficulty_confidence == Confidence.LOW
    return local


def _is_valid_topic_path(paper_family: str, topic: str, subtopic: str, config: AppConfig) -> bool:
    if paper_family == "unknown":
        return any(subtopic in topics.get(topic, []) for family, topics in config.paper_family_taxonomy.items() if family != "unknown")
    return subtopic in config.paper_family_taxonomy.get(paper_family, {}).get(topic, [])


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    output = getattr(response, "output", None)
    if output:
        chunks: list[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    raise ValueError("OpenAI response did not contain output text.")
