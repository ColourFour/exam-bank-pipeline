from __future__ import annotations

import csv
import re
from pathlib import Path

from .document_metadata import companion_candidates, parse_filename_metadata


def find_mark_scheme(
    question_pdf: str | Path,
    mark_schemes_dir: str | Path,
    mappings_dir: str | Path | None = None,
) -> Path | None:
    question_pdf = Path(question_pdf)
    mark_schemes_dir = Path(mark_schemes_dir)

    override = _find_mapping_override(question_pdf, mark_schemes_dir, Path(mappings_dir) if mappings_dir else None)
    if override:
        return override

    metadata = parse_filename_metadata(question_pdf)
    if metadata.canonical_key:
        candidates = companion_candidates(metadata, mark_schemes_dir, "MS")
        if candidates:
            return candidates[0]

    for candidate_name in _auto_candidate_names(question_pdf.name):
        candidate = mark_schemes_dir / candidate_name
        if candidate.exists():
            return candidate

    normalized_qp = _normalize_pair_key(question_pdf.stem)
    scored: list[tuple[int, Path]] = []
    for candidate in mark_schemes_dir.glob("*.pdf"):
        score = _pair_score(normalized_qp, _normalize_pair_key(candidate.stem))
        if score > 0:
            scored.append((score, candidate))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
    return None


def _find_mapping_override(question_pdf: Path, mark_schemes_dir: Path, mappings_dir: Path | None) -> Path | None:
    if mappings_dir is None or not mappings_dir.exists():
        return None

    question_keys = {question_pdf.name, question_pdf.stem, str(question_pdf)}
    for csv_path in mappings_dir.glob("*.csv"):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question_value = (row.get("question_pdf") or row.get("question") or "").strip()
                ms_value = (row.get("mark_scheme_pdf") or row.get("mark_scheme") or "").strip()
                if question_value in question_keys and ms_value:
                    return _resolve_mark_scheme_path(ms_value, mark_schemes_dir, mappings_dir)

    for yaml_path in list(mappings_dir.glob("*.yaml")) + list(mappings_dir.glob("*.yml")):
        try:
            import yaml
        except ImportError:
            continue
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        entries = raw.get("pairs", raw) if isinstance(raw, dict) else raw
        if isinstance(entries, dict):
            for question_value, ms_value in entries.items():
                if str(question_value) in question_keys:
                    return _resolve_mark_scheme_path(str(ms_value), mark_schemes_dir, mappings_dir)
        elif isinstance(entries, list):
            for row in entries:
                if not isinstance(row, dict):
                    continue
                question_value = str(row.get("question_pdf") or row.get("question") or "").strip()
                ms_value = str(row.get("mark_scheme_pdf") or row.get("mark_scheme") or "").strip()
                if question_value in question_keys and ms_value:
                    return _resolve_mark_scheme_path(ms_value, mark_schemes_dir, mappings_dir)
    return None


def _resolve_mark_scheme_path(value: str, mark_schemes_dir: Path, mappings_dir: Path) -> Path | None:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([mark_schemes_dir / path, mappings_dir / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _auto_candidate_names(question_name: str) -> list[str]:
    replacements = [
        ("_qp_", "_ms_"),
        ("-qp-", "-ms-"),
        (" qp ", " ms "),
        ("_question_", "_mark_scheme_"),
        ("question_paper", "mark_scheme"),
        ("Question Paper", "Mark Scheme"),
    ]
    candidates: list[str] = []
    for old, new in replacements:
        if old in question_name:
            candidates.append(question_name.replace(old, new))
    if "_qp" in question_name:
        candidates.append(question_name.replace("_qp", "_ms"))
    if " qp" in question_name:
        candidates.append(question_name.replace(" qp", " ms"))
    return list(dict.fromkeys(candidates))


def _normalize_pair_key(stem: str) -> str:
    lowered = stem.lower()
    lowered = re.sub(r"\b(qp|ms|question|paper|mark|scheme)\b", "", lowered)
    lowered = lowered.replace("_qp_", "_").replace("_ms_", "_")
    lowered = re.sub(r"[^a-z0-9]+", "", lowered)
    return lowered


def _pair_score(question_key: str, mark_scheme_key: str) -> int:
    if not question_key or not mark_scheme_key:
        return 0
    if question_key == mark_scheme_key:
        return 100
    shared = len(set(_tokenize(question_key)) & set(_tokenize(mark_scheme_key)))
    return shared


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", value)
