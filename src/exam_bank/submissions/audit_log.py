from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: Path, assignment_id: str) -> None:
        self.path = path
        self.assignment_id = assignment_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event_type: str,
        *,
        student_id: str = "",
        source_filename: str = "",
        status: str = "",
        reasons: list[str] | None = None,
        **extra: Any,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "assignment_id": self.assignment_id,
            "student_id": student_id,
            "source_filename": source_filename,
            "status": status,
            "reasons": reasons or [],
        }
        event.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
