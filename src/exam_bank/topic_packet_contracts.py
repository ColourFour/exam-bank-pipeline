from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REVIEW_PROMOTION_SCHEMA = "exam_bank.review_promotion"
REVIEW_PROMOTION_VERSION = 1
PACKET_PROJECTION_FINGERPRINT_VERSION = 2
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def path_provenance(path: Path | None) -> dict[str, Any]:
    """Describe the file used now and, for promoted reviews, its logical source.

    Promotion wraps an unchanged decision payload with authority metadata.  The
    current path and checksum remain useful audit evidence, while the recorded
    source identifies an equivalent pre-promotion routing input for projection
    fingerprint compatibility.
    """

    if path is None:
        return {"path": "", "sha256": ""}
    candidate = Path(path)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else ""
    provenance: dict[str, Any] = {"path": str(candidate), "sha256": digest}
    promoted_from = _promoted_source_provenance(candidate)
    if promoted_from is not None:
        provenance["promoted_from"] = promoted_from
    return provenance


def packet_projection_fingerprint(manifest: dict[str, Any]) -> str:
    projection = {
        "fingerprint_version": PACKET_PROJECTION_FINGERPRINT_VERSION,
        "schema_version": manifest.get("schema_version"),
        "paper_family": manifest.get("paper_family"),
        "topic_id": manifest.get("topic_id"),
        "subtopic_id": manifest.get("subtopic_id") or "",
        "records": [
            {
                "question_id": row.get("question_id"),
                "primary_topic_id": row.get("primary_topic_id"),
                "secondary_topic_ids": row.get("secondary_topic_ids") or [],
                "section": row.get("section"),
            }
            for row in sorted(
                manifest.get("included_records") or [], key=lambda item: str(item.get("question_id") or "")
            )
        ],
    }
    raw = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _promoted_source_provenance(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    promotion = payload.get("promotion") if isinstance(payload, dict) else None
    if not isinstance(promotion, dict):
        return None
    source_path = str(promotion.get("source_artifact_path") or "").strip()
    source_sha256 = str(promotion.get("source_artifact_sha256") or "").strip().lower()
    if (
        promotion.get("schema_name") != REVIEW_PROMOTION_SCHEMA
        or promotion.get("schema_version") != REVIEW_PROMOTION_VERSION
        or not source_path
        or not _SHA256_PATTERN.fullmatch(source_sha256)
    ):
        return None
    return {"path": source_path, "sha256": source_sha256}
