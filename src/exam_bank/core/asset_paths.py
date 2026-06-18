from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .paper_identity import IdentityError, PaperIdentity


AssetKind = Literal["question_image", "mark_scheme_image"]


@dataclass(frozen=True)
class AssetPath:
    kind: AssetKind
    paper_id: str
    question_id: str
    component: str
    canonical_path: str
    absolute_path: Path


@dataclass(frozen=True)
class AssetPathResolver:
    output_root: Path

    def question_image(self, identity: PaperIdentity) -> AssetPath:
        return self._asset(identity, kind="question_image", paper_type="qp", asset_type="question")

    def mark_scheme_image(self, identity: PaperIdentity) -> AssetPath:
        return self._asset(identity, kind="mark_scheme_image", paper_type="ms", asset_type="markscheme")

    def _asset(self, identity: PaperIdentity, *, kind: AssetKind, paper_type: str, asset_type: str) -> AssetPath:
        _require_question_identity(identity)
        filename = (
            f"{identity.subject_family}_{identity.year}_{identity.session_code}_"
            f"{paper_type}_{identity.question_id.rsplit('_', 1)[-1]}_{asset_type}.png"
        )
        canonical_path = str(Path(identity.subject_family) / filename)
        return AssetPath(
            kind=kind,
            paper_id=identity.paper_id,
            question_id=identity.question_id,
            component=identity.component,
            canonical_path=canonical_path,
            absolute_path=self.output_root / canonical_path,
        )


def _require_question_identity(identity: PaperIdentity) -> None:
    if not identity.paper_id:
        raise IdentityError("asset path requires paper_id")
    if not identity.question_id:
        raise IdentityError("asset path requires question_id")
