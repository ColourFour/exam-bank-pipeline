from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

import fcntl


PIPELINE_LOCK_FILENAME = ".pipeline.lock"


class PublicationReadBlockedError(RuntimeError):
    pass


def infer_published_output_root(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.parent.name != "json":
        return None
    return candidate.parent.parent


@contextmanager
def publication_read_guard(path: str | Path) -> Iterator[None]:
    """Fail closed while a conventional published output is being changed or recovered."""

    input_path = Path(path)
    output_root = infer_published_output_root(input_path)
    if output_root is None:
        yield
        return

    lock_path = output_root / PIPELINE_LOCK_FILENAME
    if lock_path.is_symlink():
        raise PublicationReadBlockedError(f"Publication lock is an unsafe symbolic link: {lock_path}")
    handle = None
    acquired = False
    try:
        if lock_path.is_file():
            handle = lock_path.open("r", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as exc:
                raise PublicationReadBlockedError(
                    f"Published output is currently being updated: {output_root}"
                ) from exc
        active_journals = _active_publication_journals(output_root)
        if active_journals:
            raise PublicationReadBlockedError(
                "Published output has an interrupted transaction awaiting recovery: "
                f"{active_journals[0]}"
            )
        yield
    finally:
        if handle is not None:
            try:
                if acquired:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def read_json_under_publication_guard(path: str | Path) -> Any:
    input_path = Path(path)
    with publication_read_guard(input_path):
        return json.loads(input_path.read_text(encoding="utf-8"))


def publication_journal_prefix(output_root: Path) -> str:
    return f".{output_root.name}.rollback-"


def publication_committed_prefix(output_root: Path) -> str:
    return f".{output_root.name}.committed-"


def _active_publication_journals(output_root: Path) -> list[Path]:
    parent = output_root.parent
    if not parent.exists():
        return []
    prefix = publication_journal_prefix(output_root)
    try:
        return sorted(
            (path for path in parent.iterdir() if path.name.startswith(prefix)),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise PublicationReadBlockedError(
            f"Cannot verify publication state for {output_root}"
        ) from exc
