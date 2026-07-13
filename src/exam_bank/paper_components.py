from __future__ import annotations

import re
from typing import Any

_COMPONENT_TO_PACKET_FAMILY = {
    "01": "p1",
    "11": "p1",
    "12": "p1",
    "13": "p1",
    "15": "p1",
    "03": "p3",
    "31": "p3",
    "32": "p3",
    "33": "p3",
    "35": "p3",
    "04": "p4",
    "41": "p4",
    "42": "p4",
    "43": "p4",
    "45": "p4",
    "06": "p5",
    "51": "p5",
    "52": "p5",
    "53": "p5",
    "55": "p5",
    "61": "p5",
    "62": "p5",
    "63": "p5",
    "65": "p5",
}


def normalize_component_code(value: Any) -> str:
    text = str(value or "").strip().lower().removeprefix("p")
    match = re.search(r"\d+", text)
    if not match:
        return ""
    code = match.group(0)
    return code.zfill(2) if len(code) == 1 else code


def packet_family_for_component(component_code: Any) -> str:
    return _COMPONENT_TO_PACKET_FAMILY.get(normalize_component_code(component_code), "")
