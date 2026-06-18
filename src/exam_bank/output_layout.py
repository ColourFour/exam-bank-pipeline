from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import AppConfig
from .core.asset_paths import AssetPathResolver
from .core.paper_identity import (
    IdentityError,
    build_question_id,
    canonical_subject_family,
    paper_identity_from_parts,
)
from .document_metadata import DocumentMetadata, parse_filename_metadata


LEGACY_LAYOUT_PROFILE = "legacy"
CANONICAL_LAYOUT_PROFILE = "canonical"
OCR_CANDIDATE_LAYOUT_PROFILE = "ocr-candidate"
OUTPUT_LAYOUT_VERSION = 1

QUESTION_BANK_FILENAME = "question_bank.json"
TRIAGE_BASELINE_FILENAME = "baseline_question_bank.json"
TRIAGE_COMPARISONS_DIRNAME = "comparisons"

LEGACY_SUBJECT_DIRS: dict[str, str] = {
    "p1": "pm1",
    "p3": "pm3",
    "p4": "stats",
    "p5": "mechanics",
    "p6": "stats",
}
CANONICAL_SUBJECTS: tuple[str, ...] = ("pm1", "pm3", "stats", "mechanics")

PAPER_ARTIFACT_DIR_RE = re.compile(r"^(?:pm1|pm3|stats|mechanics)$")
ITERATION_DIR_RE = re.compile(r"^iteration_\d{3}$")


@dataclass(frozen=True)
class GeneratedOutputContract:
    """Central path contract for generated output roots."""

    root: Path

    def run_root(self, run_id: str) -> Path:
        return self.root / "runs" / _safe_segment(run_id)

    def canonical_current_dir(self) -> Path:
        return self.root / "current"

    def ocr_candidate_root(self, run_id: str = "latest") -> Path:
        return self.root / "candidates" / "ocr" / _safe_segment(run_id)

    def triage_root(self) -> Path:
        return self.root / "triage"

    def triage_iteration_dir(self, iteration: str) -> Path:
        return self.triage_root() / _safe_segment(iteration)

    def triage_comparisons_dir(self, iteration: str | Path) -> Path:
        iteration_path = Path(iteration)
        if iteration_path.is_absolute() or iteration_path.parent != Path("."):
            return iteration_path / TRIAGE_COMPARISONS_DIRNAME
        return self.triage_iteration_dir(str(iteration)) / TRIAGE_COMPARISONS_DIRNAME

    def audit_dir(self, audit_id: str) -> Path:
        return self.root / "audits" / _safe_segment(audit_id)

    def asterion_reports_dir(self, run_id: str = "latest") -> Path:
        return self.root / "asterion" / "reports" / _safe_segment(run_id)

    def asterion_exports_dir(self, run_id: str = "latest") -> Path:
        return self.root / "asterion" / "exports" / _safe_segment(run_id)


def generated_output_contract(root: str | Path = "output") -> GeneratedOutputContract:
    return GeneratedOutputContract(Path(root))


def default_asterion_export_path(
    input_path: str | Path,
    filename: str,
    *,
    run_id: str = "latest",
) -> Path:
    contract = generated_output_contract(infer_generated_root(input_path))
    return contract.asterion_exports_dir(run_id) / filename


def default_triage_comparison_path(iteration_dir: str | Path, comparison_name: str) -> Path:
    return Path(iteration_dir) / TRIAGE_COMPARISONS_DIRNAME / comparison_name


def infer_generated_root(path: str | Path) -> Path:
    """Infer the generated-output root from a file or folder inside that root."""

    path = Path(path)
    candidates = [path, *path.parents]
    for candidate in candidates:
        if candidate.name == "json":
            return candidate.parent
        if candidate.name in {"current", "artifacts", "audits", "run_status", "logs"}:
            return candidate.parent
        if candidate.name in {"exports", "reports"} and candidate.parent.name == "asterion":
            return candidate.parent.parent
        if candidate.name == "asterion":
            return candidate.parent
        if candidate.name == "triage":
            return candidate.parent
        if candidate.name == "ocr" and candidate.parent.name == "candidates":
            return candidate.parent.parent
        if candidate.parent.name == "ocr" and candidate.parent.parent.name == "candidates":
            return candidate.parent.parent.parent
        if candidate.parent.name == "runs":
            return candidate.parent.parent
    if path.suffix:
        return path.parent
    return path


def output_profile_for_root(root: str | Path) -> str:
    root_path = Path(root)
    parts = root_path.parts
    if len(parts) >= 3 and parts[-3:-1] == ("candidates", "ocr"):
        return OCR_CANDIDATE_LAYOUT_PROFILE
    if root_path.name == "output_ocr_candidate":
        return OCR_CANDIDATE_LAYOUT_PROFILE
    if root_path.parent.name == "runs" or root_path.name in {"output", "current"}:
        return CANONICAL_LAYOUT_PROFILE
    return LEGACY_LAYOUT_PROFILE


def question_image_output_path(
    question_pdf: str | Path,
    question_number: str,
    config: AppConfig,
) -> Path:
    metadata = parse_filename_metadata(question_pdf)
    identity = paper_identity_from_parts(
        syllabus=metadata.syllabus or "9709",
        subject_family=metadata.paper_family,
        year=metadata.year,
        session=_asset_session_from_path(question_pdf, metadata),
        component=metadata.component,
        question_number=question_number,
    )
    return AssetPathResolver(config.output.root_dir()).question_image(identity).absolute_path


def mark_scheme_image_output_path(
    mark_scheme_pdf: str | Path,
    question_number: str,
    config: AppConfig,
) -> Path:
    metadata = parse_filename_metadata(mark_scheme_pdf)
    identity = paper_identity_from_parts(
        syllabus=metadata.syllabus or "9709",
        subject_family=metadata.paper_family,
        year=metadata.year,
        session=_asset_session_from_path(mark_scheme_pdf, metadata),
        component=metadata.component,
        question_number=question_number,
    )
    return AssetPathResolver(config.output.root_dir()).mark_scheme_image(identity).absolute_path


def paper_output_dir(config: AppConfig, metadata: DocumentMetadata) -> Path:
    identity = paper_identity_from_parts(
        syllabus=metadata.syllabus or "9709",
        subject_family=metadata.paper_family,
        year=metadata.year,
        session=metadata.normalized_session_key or metadata.session,
        component=metadata.component,
    )
    return config.output.root_dir() / identity.subject_family / identity.paper_id


def paper_instance_id(component: str, session: str, year: str) -> str:
    return paper_identity_from_parts(
        syllabus="9709",
        year=year,
        session=session,
        component=component,
    ).paper_id


def question_id(paper: str, question_number: str) -> str:
    return build_question_id(paper, question_number)


def paper_family_dir_name(paper_family: str) -> str:
    return canonical_subject(paper_family)


def canonical_subject(paper_family: str) -> str:
    try:
        return canonical_subject_family(paper_family)
    except IdentityError:
        return "unknown"


def canonical_asset_filename(
    *,
    subject: str,
    year: str,
    session: str,
    paper_type: str,
    question_number: str,
    asset_type: str,
) -> str:
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family=subject,
        year=year,
        session=session,
        component=_component_from_subject(subject),
        question_number=question_number,
    )
    resolver = AssetPathResolver(Path("."))
    if paper_type == "qp" and asset_type == "question":
        return Path(resolver.question_image(identity).canonical_path).name
    if paper_type == "ms" and asset_type == "markscheme":
        return Path(resolver.mark_scheme_image(identity).canonical_path).name
    raise IdentityError(f"unsupported asset type: paper_type={paper_type!r} asset_type={asset_type!r}")


def compact_session_code(session: str, year: str) -> str:
    identity = paper_identity_from_parts(
        syllabus="9709",
        subject_family="pm1",
        year=year,
        session=session,
        component="11",
    )
    return identity.session_code


def _component_from_subject(subject: str) -> str:
    family = canonical_subject(subject)
    return {
        "pm1": "11",
        "pm3": "31",
        "stats": "41",
        "mechanics": "51",
    }.get(family, "11")


def _canonical_year(year: str) -> str:
    digits = "".join(char for char in str(year) if char.isdigit())
    if len(digits) == 2:
        return f"20{digits}"
    if len(digits) >= 4:
        return digits[-4:]
    return "0000"


def _asset_session_from_path(path: str | Path, metadata: DocumentMetadata) -> str:
    stem = Path(path).stem.lower()
    yy = _canonical_year(metadata.year)[-2:]
    if re.search(r"\b(?:march|mar)\b|(?:^|_)m\d{2}(?:_|$)", stem):
        return f"m{yy}"
    if re.search(r"\b(?:may|june|jun|summer)\b|(?:^|_)s\d{2}(?:_|$)", stem):
        return f"s{yy}"
    if re.search(r"\b(?:october|november|oct|nov|winter|autumn)\b|(?:^|_)w\d{2}(?:_|$)", stem):
        return f"w{yy}"
    return metadata.normalized_session_key or metadata.session


def component_code_from_values(component: str, source_paper_code: str = "") -> str:
    for value in [source_paper_code, component]:
        digits = "".join(char for char in str(value) if char.isdigit())
        if len(digits) >= 2:
            return digits[-2:]
        if len(digits) == 1:
            return digits.zfill(2)
    return "xx"


def relative_to_output_root(path: str | Path, config: AppConfig) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(config.output.root_dir()))
    except ValueError:
        return str(path)


def _question_png_name(question_number: str) -> str:
    return f"{_question_id_suffix(question_number)}.png"


def _question_id_suffix(question_number: str) -> str:
    digits = _question_number_digits(question_number)
    if digits != "00":
        return f"q{digits}"
    return f"q{_safe_segment(question_number)}"


def _question_number_digits(question_number: str) -> str:
    digits = "".join(char for char in question_number if char.isdigit())
    if digits:
        return f"{int(digits):02d}"
    return "00"


def _safe_segment(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_") or "unknown"
