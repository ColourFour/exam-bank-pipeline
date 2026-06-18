from __future__ import annotations

import csv
from pathlib import Path

from .core.paper_identity import IdentityError, PaperIdentity, paper_identity_from_parts, session_for_source_path
from .document_metadata import companion_candidates, parse_filename_metadata


def find_mark_scheme(
    question_pdf: str | Path,
    mark_schemes_dir: str | Path,
    mappings_dir: str | Path | None = None,
) -> Path | None:
    question_pdf = Path(question_pdf)
    mark_schemes_dir = Path(mark_schemes_dir)
    question_identity = _identity_for_pdf(question_pdf)
    if question_identity is None:
        return None

    override = _find_mapping_override(question_pdf, mark_schemes_dir, Path(mappings_dir) if mappings_dir else None)
    if override:
        override_identity = _identity_for_pdf(override)
        return override if override_identity and _same_paper_identity(override_identity, question_identity) else None

    candidates = [
        candidate
        for candidate in sorted(mark_schemes_dir.glob("*.pdf"))
        if (candidate_identity := _identity_for_pdf(candidate))
        and _same_paper_identity(candidate_identity, question_identity)
        and parse_filename_metadata(candidate).document_type == "mark_scheme"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return None

    # Compatibility path for older normalized filenames with enough metadata
    # to form the same canonical key but not a complete PaperIdentity.
    metadata = parse_filename_metadata(question_pdf)
    if metadata.canonical_key:
        legacy_candidates = [
            candidate
            for candidate in companion_candidates(metadata, mark_schemes_dir, "MS")
            if (candidate_identity := _identity_for_pdf(candidate)) and _same_paper_identity(candidate_identity, question_identity)
        ]
        if len(legacy_candidates) == 1:
            return legacy_candidates[0]
    return None


def _identity_for_pdf(path: str | Path) -> PaperIdentity | None:
    metadata = parse_filename_metadata(path)
    try:
        return paper_identity_from_parts(
            syllabus=metadata.syllabus or "9709",
            subject_family=metadata.paper_family,
            year=metadata.year,
            session=session_for_source_path(
                path,
                year=metadata.year,
                fallback_session=metadata.normalized_session_key or metadata.session,
            ),
            component=metadata.component,
        )
    except IdentityError:
        return None


def _same_paper_identity(left: PaperIdentity, right: PaperIdentity) -> bool:
    return left.paper_id == right.paper_id and left.session_code == right.session_code


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
