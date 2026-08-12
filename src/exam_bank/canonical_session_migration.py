from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .atomic_json import write_atomic_json
from .corpus_session_identity import PdfSessionEvidence, detect_pdf_session


DEFAULT_QUESTION_BANK = Path("output/json/question_bank.json")
DEFAULT_ALIAS_MANIFEST = Path("manifests/migrations/canonical_session_identity_aliases.v1.json")
DEFAULT_AUTHORITATIVE_ARTIFACTS = (
    Path("data/topic_routing/question_bank.topic_routing.v1.json"),
    Path("data/review/canonical/asterion/content_lab_reviewed_decisions.v1.json"),
    Path("data/review/canonical/asterion/student_runtime_safe_decisions.v1.json"),
    Path("data/review/canonical/p3_exact_skill/reviewed_decisions.v1.json"),
    Path("data/review/canonical/p3_exact_skill/reviewed_mark_events.v1.json"),
    Path("data/review/canonical/text_fidelity/question_text_gold.v1.json"),
    Path("data/review/canonical/text_fidelity/review_state.v1.json"),
    Path("data/review/canonical/topic/topic_bank_reviewed_decisions.v1.json"),
    Path("data/review/canonical/topic/topic_overlap_review_current.v1.json"),
)

_SOURCE_PDF_RE = re.compile(
    r"(?:^|/)9709_(?P<season>[msw])(?P<yy>\d{2})_qp_(?P<component>\d{1,2})\.pdf(?:$|[?#])",
    re.IGNORECASE,
)
_QUESTION_ID_RE = re.compile(
    r"^(?P<component>\d{2})(?P<session>spring|summer|winter)(?P<yy>\d{2})(?P<suffix>_q\d{2}.*)$",
    re.IGNORECASE,
)
_PAPER_ID_RE = re.compile(
    r"^(?P<component>\d{2})(?P<session>spring|summer|winter)(?P<yy>\d{2})$",
    re.IGNORECASE,
)


class CanonicalSessionMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionIdentityAlias:
    legacy_question_id: str
    canonical_question_id: str
    legacy_paper_id: str
    canonical_paper_id: str
    source_session_code: str
    raw_source_session_code: str
    source_pdf: str
    source_sha256: str
    source_size_bytes: int
    session_evidence_status: str
    first_page_text_sha256: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate the legacy March-as-summer identity collision using the question bank's raw source PDF provenance."
        )
    )
    parser.add_argument("--question-bank", type=Path, default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--artifact", action="append", type=Path, dest="artifacts")
    parser.add_argument("--alias-manifest", type=Path, default=DEFAULT_ALIAS_MANIFEST)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Root used to resolve relative source_pdf provenance paths.",
    )
    parser.add_argument("--write", action="store_true", help="Apply the validated migration. The default is a dry run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = migrate_canonical_session_identity(
        question_bank_path=args.question_bank,
        artifact_paths=args.artifacts or DEFAULT_AUTHORITATIVE_ARTIFACTS,
        alias_manifest_path=args.alias_manifest,
        source_root=args.source_root,
        write=bool(args.write),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def migrate_canonical_session_identity(
    *,
    question_bank_path: str | Path,
    artifact_paths: Iterable[str | Path],
    alias_manifest_path: str | Path,
    source_root: str | Path = Path.cwd(),
    write: bool = False,
) -> dict[str, Any]:
    question_bank_path = Path(question_bank_path)
    alias_manifest_path = Path(alias_manifest_path)
    bank = _read_json_object(question_bank_path)
    questions = bank.get("questions")
    if not isinstance(questions, list) or not all(isinstance(record, dict) for record in questions):
        raise CanonicalSessionMigrationError("question bank must contain a questions list of objects")

    source_root = Path(source_root).resolve()
    source_question_bank_sha256 = _file_sha256(question_bank_path)
    aliases = build_session_identity_aliases(questions, source_root=source_root)
    migrated_bank = migrate_question_bank_payload(bank, aliases)
    alias_manifest = build_alias_manifest(
        aliases,
        question_bank_path=question_bank_path,
        source_root=source_root,
        source_question_bank_sha256=source_question_bank_sha256,
    )

    artifact_payloads: dict[Path, dict[str, Any]] = {}
    artifact_reports: list[dict[str, Any]] = []
    for raw_path in artifact_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise CanonicalSessionMigrationError(f"authoritative artifact does not exist: {path}")
        payload = _read_json_object(path)
        migrated, counters = migrate_authoritative_payload(payload, aliases)
        if payload.get("schema_name") == "exam_bank.topic_routing_sidecar":
            _annotate_topic_routes(migrated, aliases)
        artifact_payloads[path] = migrated
        artifact_reports.append(
            {
                "path": str(path),
                "changed": migrated != payload,
                "rewritten_keys": counters["keys"],
                "rewritten_values": counters["values"],
                "legacy_aliases_remaining": count_legacy_alias_references(migrated, aliases),
            }
        )

    remaining = sum(item["legacy_aliases_remaining"] for item in artifact_reports)
    if remaining:
        raise CanonicalSessionMigrationError(
            f"migration left {remaining} legacy March identity references in authoritative artifacts"
        )

    report = {
        "schema_name": "exam_bank.canonical_session_identity_migration",
        "schema_version": 1,
        "write": write,
        "question_bank": str(question_bank_path),
        "alias_manifest": str(alias_manifest_path),
        "source_root": str(source_root),
        "source_question_bank_sha256": source_question_bank_sha256,
        "question_bank_record_count": len(questions),
        "march_alias_count": len(aliases),
        "paper_alias_count": len({alias.legacy_paper_id for alias in aliases}),
        "question_bank_records_rewritten": sum(
            alias.legacy_question_id != alias.canonical_question_id for alias in aliases
        ),
        "artifacts": artifact_reports,
        "requires_downstream_refresh": [
            "question_bank_run_manifest",
            "topic_routing_evidence_packet_hashes",
            "topic_routing_release_manifest",
            "difficulty_and_mark_event_sidecars",
            "asterion_exports",
        ],
    }

    if write:
        write_atomic_json(migrated_bank, question_bank_path)
        for path, payload in artifact_payloads.items():
            write_atomic_json(payload, path)
            if payload.get("schema_name") == "exam_bank.topic_routing_sidecar":
                checksum_path = path.with_suffix(".sha256")
                checksum_path.write_text(
                    f"{_file_sha256(path)}  {path.name}\n",
                    encoding="utf-8",
                )
        write_atomic_json(alias_manifest, alias_manifest_path)
    return report


def build_session_identity_aliases(
    records: Sequence[dict[str, Any]],
    *,
    source_root: str | Path = Path.cwd(),
) -> list[SessionIdentityAlias]:
    aliases: list[SessionIdentityAlias] = []
    current_ids: dict[str, int] = {}
    paper_source_codes: dict[str, set[str]] = {}
    root = Path(source_root).resolve()
    evidence_cache: dict[Path, tuple[PdfSessionEvidence, str, int]] = {}

    for index, record in enumerate(records):
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            raise CanonicalSessionMigrationError(f"question bank record {index} is missing question_id")
        if question_id in current_ids:
            raise CanonicalSessionMigrationError(f"duplicate question_id before migration: {question_id}")
        current_ids[question_id] = index

        source_pdf = _source_pdf(record)
        source_match = _SOURCE_PDF_RE.search(source_pdf)
        if source_match is None:
            continue
        raw_source_code = f"{source_match.group('season').lower()}{source_match.group('yy')}"
        source_path = _resolve_source_pdf(source_pdf, root)
        cached = evidence_cache.get(source_path)
        if cached is None:
            evidence = detect_pdf_session(source_path)
            if evidence.session_code is None:
                raise CanonicalSessionMigrationError(
                    "source PDF does not provide unambiguous first-page session evidence: "
                    f"{source_pdf!r} ({evidence.status}: {evidence.error})"
                )
            cached = (evidence, _file_sha256(source_path), source_path.stat().st_size)
            evidence_cache[source_path] = cached
        evidence, source_sha256, source_size_bytes = cached
        assert evidence.session_code is not None
        source_code = f"{evidence.session_code}{source_match.group('yy')}"
        paper_id = str(record.get("paper") or record.get("canonical_paper_id") or "").strip()
        if paper_id:
            paper_source_codes.setdefault(paper_id, set()).add(source_code[0])
        if source_code[0] != "m":
            _validate_non_march_identity(question_id, paper_id, source_code)
            continue

        alias = _march_alias_for_record(
            record,
            source_pdf=source_pdf,
            source_session_code=source_code,
            raw_source_session_code=raw_source_code,
            source_component=source_match.group("component"),
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            evidence=evidence,
        )
        aliases.append(alias)

    for alias in aliases:
        seasons = paper_source_codes.get(alias.legacy_paper_id, set())
        if seasons - {"m"}:
            raise CanonicalSessionMigrationError(
                "legacy paper identity is shared by multiple raw sessions in the live bank; "
                f"cannot rewrite it generically: {alias.legacy_paper_id} -> {sorted(seasons)}"
            )
        target_owner = current_ids.get(alias.canonical_question_id)
        legacy_owner = current_ids.get(alias.legacy_question_id)
        if target_owner is not None and target_owner != legacy_owner:
            raise CanonicalSessionMigrationError(
                "canonical March question ID would collide with an existing record: "
                f"{alias.canonical_question_id}"
            )

    aliases.sort(key=lambda item: item.canonical_question_id)
    return aliases


def migrate_question_bank_payload(
    payload: dict[str, Any],
    aliases: Sequence[SessionIdentityAlias],
) -> dict[str, Any]:
    migrated = copy.deepcopy(payload)
    questions = migrated.get("questions")
    if not isinstance(questions, list):
        raise CanonicalSessionMigrationError("question bank must contain questions")
    aliases_by_current_id = {
        alias.legacy_question_id: alias for alias in aliases
    } | {
        alias.canonical_question_id: alias for alias in aliases
    }
    for index, record in enumerate(questions):
        if not isinstance(record, dict):
            raise CanonicalSessionMigrationError(f"question bank record {index} is not an object")
        question_id = str(record.get("question_id") or "").strip()
        alias = aliases_by_current_id.get(question_id)
        if alias is None:
            continue
        rewritten, _counters = _rewrite_json(
            record,
            _replacement_pairs([alias]),
        )
        if not isinstance(rewritten, dict):
            raise CanonicalSessionMigrationError("rewritten question record is not an object")
        rewritten["question_id"] = alias.canonical_question_id
        rewritten["paper"] = alias.canonical_paper_id
        if "canonical_paper_id" in rewritten:
            rewritten["canonical_paper_id"] = alias.canonical_paper_id
        rewritten["canonical_session"] = f"spring{alias.source_session_code[1:]}"
        if "session" in rewritten and str(rewritten.get("session") or "").lower().startswith("summer"):
            rewritten["session"] = f"spring{alias.source_session_code[1:]}"
        questions[index] = rewritten

    migrated_ids = [str(record.get("question_id") or "") for record in questions if isinstance(record, dict)]
    if len(migrated_ids) != len(set(migrated_ids)):
        duplicates = sorted({value for value in migrated_ids if migrated_ids.count(value) > 1})
        raise CanonicalSessionMigrationError(f"duplicate question IDs after migration: {duplicates[:10]}")
    if "record_count" in migrated:
        migrated["record_count"] = len(questions)
    return migrated


def migrate_authoritative_payload(
    payload: dict[str, Any],
    aliases: Sequence[SessionIdentityAlias],
) -> tuple[dict[str, Any], dict[str, int]]:
    rewritten, counters = _rewrite_json(payload, _replacement_pairs(aliases))
    if not isinstance(rewritten, dict):
        raise CanonicalSessionMigrationError("authoritative JSON root must remain an object")
    return rewritten, counters


def build_alias_manifest(
    aliases: Sequence[SessionIdentityAlias],
    *,
    question_bank_path: str | Path,
    source_root: str | Path = Path.cwd(),
    source_question_bank_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": "exam_bank.canonical_session_identity_aliases",
        "schema_version": 1,
        "migration_reason": "March mYY and May/June sYY previously collapsed to summerYY",
        "identity_contract": {
            "m": "spring",
            "s": "summer",
            "w": "winter",
        },
        "source_question_bank": str(question_bank_path),
        "source_question_bank_sha256": source_question_bank_sha256
        or _file_sha256(Path(question_bank_path)),
        "source_root": str(Path(source_root).resolve()),
        "alias_count": len(aliases),
        "aliases": [asdict(alias) for alias in aliases],
        "runtime_policy": (
            "Aliases are audit provenance only. Runtime joins must use canonical IDs and must not resolve "
            "legacy summerYY keys after an independent June record has been admitted."
        ),
    }


def count_legacy_alias_references(payload: Any, aliases: Sequence[SessionIdentityAlias]) -> int:
    legacy_tokens = tuple(
        sorted(
            {
                *(alias.legacy_question_id for alias in aliases if alias.legacy_question_id != alias.canonical_question_id),
                *(alias.legacy_paper_id for alias in aliases if alias.legacy_paper_id != alias.canonical_paper_id),
            },
            key=len,
            reverse=True,
        )
    )
    return sum(sum(value.count(token) for token in legacy_tokens) for value in _json_strings(payload))


def _march_alias_for_record(
    record: dict[str, Any],
    *,
    source_pdf: str,
    source_session_code: str,
    raw_source_session_code: str,
    source_component: str,
    source_sha256: str,
    source_size_bytes: int,
    evidence: PdfSessionEvidence,
) -> SessionIdentityAlias:
    question_id = str(record.get("question_id") or "").strip()
    question_match = _QUESTION_ID_RE.fullmatch(question_id)
    if question_match is None:
        raise CanonicalSessionMigrationError(f"unsupported March question ID: {question_id!r}")
    paper_id = str(record.get("paper") or record.get("canonical_paper_id") or "").strip()
    paper_match = _PAPER_ID_RE.fullmatch(paper_id)
    if paper_match is None:
        raise CanonicalSessionMigrationError(f"unsupported March paper ID: {paper_id!r}")

    component = source_component.zfill(2)
    yy = source_session_code[1:]
    if question_match.group("component") != component or paper_match.group("component") != component:
        raise CanonicalSessionMigrationError(
            f"March source/component mismatch: source={source_pdf!r} question_id={question_id!r} paper={paper_id!r}"
        )
    if question_match.group("yy") != yy or paper_match.group("yy") != yy:
        raise CanonicalSessionMigrationError(
            f"March source/year mismatch: source={source_pdf!r} question_id={question_id!r} paper={paper_id!r}"
        )
    if question_match.group("session").lower() not in {"summer", "spring"}:
        raise CanonicalSessionMigrationError(f"March source has incompatible question identity: {question_id!r}")
    if paper_match.group("session").lower() not in {"summer", "spring"}:
        raise CanonicalSessionMigrationError(f"March source has incompatible paper identity: {paper_id!r}")

    canonical_paper = f"{component}spring{yy}"
    canonical_question = f"{canonical_paper}{question_match.group('suffix')}"
    legacy_paper = f"{component}summer{yy}"
    legacy_question = f"{legacy_paper}{question_match.group('suffix')}"
    return SessionIdentityAlias(
        legacy_question_id=legacy_question,
        canonical_question_id=canonical_question,
        legacy_paper_id=legacy_paper,
        canonical_paper_id=canonical_paper,
        source_session_code=source_session_code,
        raw_source_session_code=raw_source_session_code,
        source_pdf=source_pdf,
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
        session_evidence_status=evidence.status,
        first_page_text_sha256=evidence.first_page_text_sha256,
    )


def _validate_non_march_identity(question_id: str, paper_id: str, source_session_code: str) -> None:
    expected_label = {"s": "summer", "w": "winter"}.get(source_session_code[0])
    if expected_label and (expected_label not in question_id.lower() or expected_label not in paper_id.lower()):
        raise CanonicalSessionMigrationError(
            "publisher session evidence disagrees with canonical identity: "
            f"source_session={source_session_code!r} question_id={question_id!r} paper={paper_id!r}"
        )


def _resolve_source_pdf(value: str, root: Path) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if path != root and root not in path.parents:
        raise CanonicalSessionMigrationError(f"source PDF escapes source root: {value!r}")
    if not path.is_file():
        raise CanonicalSessionMigrationError(f"source PDF does not exist: {value!r}")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise CanonicalSessionMigrationError(f"source PDF path contains a symbolic link: {value!r}")
    return path


def _replacement_pairs(aliases: Sequence[SessionIdentityAlias]) -> tuple[tuple[str, str], ...]:
    replacements: dict[str, str] = {}
    for alias in aliases:
        if alias.legacy_question_id != alias.canonical_question_id:
            replacements[alias.legacy_question_id] = alias.canonical_question_id
        if alias.legacy_paper_id != alias.canonical_paper_id:
            replacements[alias.legacy_paper_id] = alias.canonical_paper_id
    return tuple(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _rewrite_json(value: Any, replacements: Sequence[tuple[str, str]]) -> tuple[Any, dict[str, int]]:
    counters = {"keys": 0, "values": 0}

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            rewritten = _rewrite_string(item, replacements)
            if rewritten != item:
                counters["values"] += 1
            return rewritten
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, dict):
            rewritten_dict: dict[str, Any] = {}
            for key, child in item.items():
                rewritten_key = _rewrite_string(str(key), replacements)
                if rewritten_key != key:
                    counters["keys"] += 1
                rewritten_child = visit(child)
                if rewritten_key in rewritten_dict and rewritten_dict[rewritten_key] != rewritten_child:
                    raise CanonicalSessionMigrationError(
                        f"identity rewrite would merge distinct JSON keys: {key!r} -> {rewritten_key!r}"
                    )
                rewritten_dict[rewritten_key] = rewritten_child
            return rewritten_dict
        return item

    return visit(value), counters


def _rewrite_string(value: str, replacements: Sequence[tuple[str, str]]) -> str:
    rewritten = value
    for legacy, canonical in replacements:
        rewritten = rewritten.replace(legacy, canonical)
    return rewritten


def _annotate_topic_routes(payload: dict[str, Any], aliases: Sequence[SessionIdentityAlias]) -> None:
    records = payload.get("records")
    if not isinstance(records, dict):
        return
    by_canonical_id = {alias.canonical_question_id: alias for alias in aliases}
    for question_id, route in records.items():
        alias = by_canonical_id.get(str(question_id))
        if alias is not None and isinstance(route, dict):
            route["source_session_code"] = alias.source_session_code


def _source_pdf(record: dict[str, Any]) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    return str(notes.get("source_pdf") or record.get("source_pdf") or "").strip()


def _json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_strings(child)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalSessionMigrationError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CanonicalSessionMigrationError(f"JSON root must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
