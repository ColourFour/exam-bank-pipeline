from __future__ import annotations

import re
from typing import Any

from .core.subject_contract import taxonomy_paper_family


def normalize_component_code(value: Any) -> str:
    text = str(value or "").strip().lower().removeprefix("p")
    match = re.search(r"\d+", text)
    if not match:
        return ""
    code = match.group(0)
    return code.zfill(2) if len(code) == 1 else code


def packet_family_for_component(
    component_code: Any,
    *,
    year: Any = None,
    session: Any = None,
    paper: Any = None,
) -> str:
    return taxonomy_paper_family(
        "",
        component=normalize_component_code(component_code),
        year=year,
        session=session,
        paper=paper,
    )
