from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, TextIO
from uuid import uuid4


RUN_STATUS_FILENAME = "run_status.json"
BATCH_STATUS_FILENAME = "batch_status.jsonl"
RUN_MANIFEST_FILENAME = "run_manifest.json"

RUN_STATUS_VALUES = {"pending", "running", "completed", "failed", "interrupted"}
BATCH_STATUS_VALUES = {"pending", "running", "completed", "failed", "skipped"}


Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_status_root_for_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix and output.parent.name == "json":
        return output.parent.parent / "run_status"
    if output.suffix:
        return output.parent / "run_status"
    return output / "run_status"


def generate_run_id(run_type: str, now: datetime | None = None) -> str:
    timestamp = (now or utcnow()).strftime("%Y%m%dT%H%M%SZ")
    safe_type = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in run_type).strip("_")
    return f"{timestamp}-{safe_type or 'run'}-{uuid4().hex[:8]}"


def latest_run_id(status_root: str | Path, *, run_type: str | None = None) -> str | None:
    root = Path(status_root)
    if not root.exists():
        return None
    candidates: list[tuple[str, float, str]] = []
    for status_path in root.glob(f"*/{RUN_STATUS_FILENAME}"):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if run_type and payload.get("run_type") != run_type:
            continue
        updated_at = str(payload.get("updated_at") or payload.get("started_at") or "")
        try:
            timestamp = datetime.fromisoformat(updated_at).timestamp()
        except ValueError:
            timestamp = status_path.stat().st_mtime
        candidates.append((updated_at, timestamp, status_path.parent.name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    return candidates[-1][2]


def resolve_run_id(
    *,
    status_root: str | Path,
    run_type: str,
    requested_run_id: str | None = None,
    resume: bool = False,
    clock: Clock = utcnow,
) -> str:
    if requested_run_id:
        return requested_run_id
    if resume:
        existing = latest_run_id(status_root, run_type=run_type)
        if existing:
            return existing
    return generate_run_id(run_type, clock())


def read_latest_batch_statuses(status_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(status_dir) / BATCH_STATUS_FILENAME
    statuses: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return statuses
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        batch_id = str(payload.get("batch_id") or "").strip()
        if batch_id:
            statuses[batch_id] = payload
    return statuses


def completed_batch_ids(status_dir: str | Path) -> set[str]:
    return {
        batch_id
        for batch_id, payload in read_latest_batch_statuses(status_dir).items()
        if payload.get("status") in {"completed", "skipped"}
    }


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _seconds_between(started_at: datetime, now: datetime) -> float:
    return max(0.0, (now - started_at).total_seconds())


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class BatchHandle:
    batch_id: str
    paper: str | None
    paper_family: str | None
    record_count: int
    started_at: datetime


class TerminalProgressRenderer:
    def __init__(
        self,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        min_interval_seconds: float = 1.0,
        width: int = 20,
    ) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.min_interval_seconds = min_interval_seconds
        self.width = width
        self._last_render_monotonic = 0.0
        self._last_line = ""

    def render(self, status: dict[str, Any], *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render_monotonic < self.min_interval_seconds:
            return
        self._last_render_monotonic = now
        line = self._line(status)
        if line == self._last_line and not force:
            return
        self._last_line = line
        if self.stream.isatty():
            columns = shutil.get_terminal_size((100, 20)).columns
            rendered = line[: max(1, columns - 1)]
            self.stream.write("\r" + rendered.ljust(columns - 1))
            if status.get("status") in {"completed", "failed", "interrupted"}:
                self.stream.write("\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def _line(self, status: dict[str, Any]) -> str:
        percent = float(status.get("percent_complete") or 0.0)
        filled = int(round((percent / 100.0) * self.width))
        filled = max(0, min(self.width, filled))
        bar = "#" * filled + "-" * (self.width - filled)
        run_type = str(status.get("run_type") or "run").replace("_", " ")
        phase = str(status.get("current_phase") or "pending")
        parts = [f"[{bar}] {percent:.0f}%", run_type, phase]
        paper = status.get("current_paper")
        if paper:
            parts.append(f"paper {paper}")
        batch_id = status.get("current_batch_id")
        total_batches = int(status.get("total_batches") or 0)
        completed_batches = int(status.get("completed_batches") or 0)
        if total_batches:
            batch_label = f"batch {completed_batches}/{total_batches}"
            if batch_id:
                batch_label += f" ({batch_id})"
            parts.append(batch_label)
        total_records = int(status.get("total_records") or 0)
        completed_records = int(status.get("completed_records") or 0)
        if total_records:
            parts.append(f"{completed_records}/{total_records} records")
        elif completed_records:
            parts.append(f"{completed_records} records")
        parts.append(f"elapsed {_format_duration(status.get('elapsed_seconds'))}")
        eta = status.get("estimated_remaining_seconds")
        if eta is not None:
            parts.append(f"ETA {_format_duration(eta)}")
        retry_count = int(status.get("retry_count") or 0)
        if retry_count:
            parts.append(f"retry {retry_count}")
        success = int(status.get("successful_records") or 0)
        failed = int(status.get("failed_records") or 0)
        skipped = int(status.get("skipped_records") or 0)
        if success or failed or skipped:
            parts.append(f"ok {success} fail {failed} skip {skipped}")
        review_required = int(status.get("review_required_records") or 0)
        provider_failures = int(status.get("provider_failure_records") or 0)
        if status.get("run_type") == "topic_routing" or review_required or provider_failures:
            parts.append(f"review {review_required}")
            parts.append(f"provider_fail {provider_failures}")
        output_path = status.get("output_path")
        if output_path:
            parts.append(f"output {output_path}")
        return " | ".join(parts)


class RunStatusTracker:
    def __init__(
        self,
        *,
        run_id: str,
        run_type: str,
        status_root: str | Path,
        command: str,
        input_paths: list[str | Path] | None = None,
        output_paths: list[str | Path] | None = None,
        config_paths: list[str | Path] | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
        progress: bool = True,
        stream: TextIO | None = None,
        clock: Clock = utcnow,
    ) -> None:
        self.run_id = run_id
        self.run_type = run_type
        self.status_root = Path(status_root)
        self.status_dir = self.status_root / run_id
        self.command = command
        self.input_paths = [str(path) for path in input_paths or [] if path]
        self.output_paths = [str(path) for path in output_paths or [] if path]
        self.config_paths = [str(path) for path in config_paths or [] if path]
        self.model = model
        self.prompt_version = prompt_version
        self.clock = clock
        self.renderer = TerminalProgressRenderer(enabled=progress, stream=stream)
        self.started_at = clock()
        self.finished_at: datetime | None = None
        self._current_batch: BatchHandle | None = None
        self._batch_starts: dict[str, datetime] = {}
        self._status: dict[str, Any] = {
            "run_id": run_id,
            "run_type": run_type,
            "command": command,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(self.started_at),
            "finished_at": None,
            "status": "pending",
            "current_phase": "pending",
            "current_batch_id": None,
            "current_paper": None,
            "current_paper_family": None,
            "completed_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "skipped_batches": 0,
            "total_batches": 0,
            "completed_records": 0,
            "total_records": 0,
            "percent_complete": 0.0,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "successful_records": 0,
            "failed_records": 0,
            "skipped_records": 0,
            "review_required_records": 0,
            "provider_failure_records": 0,
            "retry_count": 0,
            "output_path": self.output_paths[0] if self.output_paths else "",
            "error_summary": "",
        }

    @property
    def run_status_path(self) -> Path:
        return self.status_dir / RUN_STATUS_FILENAME

    @property
    def batch_status_path(self) -> Path:
        return self.status_dir / BATCH_STATUS_FILENAME

    @property
    def run_manifest_path(self) -> Path:
        return self.status_dir / RUN_MANIFEST_FILENAME

    def start(self, *, phase: str, total_batches: int = 0, total_records: int = 0) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self._status["status"] = "running"
        self._status["current_phase"] = phase
        self._status["total_batches"] = max(0, int(total_batches))
        self._status["total_records"] = max(0, int(total_records))
        self._write_manifest()
        self._write_status(force_render=True)

    def set_totals(self, *, total_batches: int | None = None, total_records: int | None = None) -> None:
        if total_batches is not None:
            self._status["total_batches"] = max(0, int(total_batches))
        if total_records is not None:
            self._status["total_records"] = max(0, int(total_records))
        self._write_status()

    def update_phase(
        self,
        phase: str,
        *,
        current_batch_id: str | None = None,
        current_paper: str | None = None,
        current_paper_family: str | None = None,
        output_path: str | Path | None = None,
        retry_count: int | None = None,
        force_render: bool = False,
    ) -> None:
        self._status["current_phase"] = phase
        if current_batch_id is not None:
            self._status["current_batch_id"] = current_batch_id
        if current_paper is not None:
            self._status["current_paper"] = current_paper
        if current_paper_family is not None:
            self._status["current_paper_family"] = current_paper_family
        if output_path is not None:
            self._status["output_path"] = str(output_path)
        if retry_count is not None:
            self._status["retry_count"] = max(0, int(retry_count))
        self._write_status(force_render=force_render)

    def update_extra_counts(
        self,
        *,
        review_required_records: int | None = None,
        provider_failure_records: int | None = None,
        force_render: bool = False,
    ) -> None:
        if review_required_records is not None:
            self._status["review_required_records"] = max(0, int(review_required_records))
        if provider_failure_records is not None:
            self._status["provider_failure_records"] = max(0, int(provider_failure_records))
        self._write_status(force_render=force_render)

    def start_batch(
        self,
        *,
        batch_id: str,
        paper: str | None = None,
        paper_family: str | None = None,
        record_count: int = 0,
        phase: str = "running",
    ) -> None:
        now = self.clock()
        self._current_batch = BatchHandle(
            batch_id=batch_id,
            paper=paper,
            paper_family=paper_family,
            record_count=max(0, int(record_count)),
            started_at=now,
        )
        self._batch_starts[batch_id] = now
        self.update_phase(
            phase,
            current_batch_id=batch_id,
            current_paper=paper,
            current_paper_family=paper_family,
            force_render=True,
        )

    def complete_batch(
        self,
        *,
        batch_id: str,
        paper: str | None = None,
        paper_family: str | None = None,
        record_count: int = 0,
        successful_records: int | None = None,
        failed_records: int = 0,
        skipped_records: int = 0,
    ) -> None:
        successful = record_count if successful_records is None else successful_records
        self._append_batch_status(
            batch_id=batch_id,
            paper=paper,
            paper_family=paper_family,
            status="completed",
            record_count=record_count,
            successful_records=successful,
            failed_records=failed_records,
            skipped_records=skipped_records,
            error_message=None,
        )
        self._status["completed_batches"] = int(self._status["completed_batches"]) + 1
        self._status["successful_batches"] = int(self._status["successful_batches"]) + 1
        self._status["completed_records"] = int(self._status["completed_records"]) + max(
            0,
            int(successful) + int(failed_records) + int(skipped_records),
        )
        self._status["successful_records"] = int(self._status["successful_records"]) + max(0, int(successful))
        self._status["failed_records"] = int(self._status["failed_records"]) + max(0, int(failed_records))
        self._status["skipped_records"] = int(self._status["skipped_records"]) + max(0, int(skipped_records))
        self._clear_current_batch_if(batch_id)
        self._write_status(force_render=True)

    def skip_batch(
        self,
        *,
        batch_id: str,
        paper: str | None = None,
        paper_family: str | None = None,
        record_count: int = 0,
        skipped_records: int | None = None,
    ) -> None:
        skipped = record_count if skipped_records is None else skipped_records
        self._append_batch_status(
            batch_id=batch_id,
            paper=paper,
            paper_family=paper_family,
            status="skipped",
            record_count=record_count,
            successful_records=0,
            failed_records=0,
            skipped_records=skipped,
            error_message=None,
        )
        self._status["completed_batches"] = int(self._status["completed_batches"]) + 1
        self._status["skipped_batches"] = int(self._status["skipped_batches"]) + 1
        self._status["completed_records"] = int(self._status["completed_records"]) + max(0, int(skipped))
        self._status["skipped_records"] = int(self._status["skipped_records"]) + max(0, int(skipped))
        self._clear_current_batch_if(batch_id)
        self._write_status(force_render=True)

    def fail_batch(
        self,
        *,
        batch_id: str,
        paper: str | None = None,
        paper_family: str | None = None,
        record_count: int = 0,
        failed_records: int | None = None,
        successful_records: int = 0,
        skipped_records: int = 0,
        error_message: str | None = None,
    ) -> None:
        failed = record_count if failed_records is None else failed_records
        self._append_batch_status(
            batch_id=batch_id,
            paper=paper,
            paper_family=paper_family,
            status="failed",
            record_count=record_count,
            successful_records=successful_records,
            failed_records=failed,
            skipped_records=skipped_records,
            error_message=error_message,
        )
        self._status["completed_batches"] = int(self._status["completed_batches"]) + 1
        self._status["failed_batches"] = int(self._status["failed_batches"]) + 1
        self._status["completed_records"] = int(self._status["completed_records"]) + max(
            0,
            int(successful_records) + int(failed) + int(skipped_records),
        )
        self._status["successful_records"] = int(self._status["successful_records"]) + max(0, int(successful_records))
        self._status["failed_records"] = int(self._status["failed_records"]) + max(0, int(failed))
        self._status["skipped_records"] = int(self._status["skipped_records"]) + max(0, int(skipped_records))
        if error_message:
            self._status["error_summary"] = error_message
        self._clear_current_batch_if(batch_id)
        self._write_status(force_render=True)

    def finish(self, final_status: str = "completed", *, error_summary: str | None = None) -> None:
        if final_status not in RUN_STATUS_VALUES:
            raise ValueError(f"Unsupported run status: {final_status}")
        self.finished_at = self.clock()
        self._status["status"] = final_status
        self._status["finished_at"] = _iso(self.finished_at)
        self._status["current_phase"] = "finished" if final_status == "completed" else final_status
        self._status["current_batch_id"] = None
        if error_summary is not None:
            self._status["error_summary"] = error_summary
        if final_status == "completed" and not int(self._status["total_records"] or 0):
            self._status["total_records"] = int(self._status["completed_records"] or 0)
        self._write_status(force_render=True)
        self._write_manifest()

    def batch_artifact_path(self, batch_id: str, filename: str) -> Path:
        safe_batch_id = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in batch_id)
        return self.status_dir / "batch_artifacts" / safe_batch_id / filename

    def write_batch_artifact(self, batch_id: str, filename: str, payload: Any) -> Path:
        path = self.batch_artifact_path(batch_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_batch_artifact(self, batch_id: str, filename: str) -> Any | None:
        path = self.batch_artifact_path(batch_id, filename)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def status_snapshot(self) -> dict[str, Any]:
        self._refresh_computed_fields()
        return dict(self._status)

    def final_summary(self) -> str:
        status = self.status_snapshot()
        lines = [
            "Run summary:",
            f"  run type: {status['run_type']}",
            f"  final status: {status['status']}",
            f"  completed batches/papers: {status['completed_batches']}/{status['total_batches']}",
            f"  failed batches/papers: {status.get('failed_batches', 0)}",
            f"  skipped batches/papers: {status.get('skipped_batches', 0)}",
            f"  successful records: {status['successful_records']}",
            f"  failed records: {status['failed_records']}",
            f"  skipped records: {status['skipped_records']}",
            f"  review-required records: {status.get('review_required_records', 0)}",
            f"  provider/API failures: {status.get('provider_failure_records', 0)}",
            f"  total elapsed time: {_format_duration(status['elapsed_seconds'])}",
            f"  final output path: {status.get('output_path') or 'n/a'}",
            f"  status file path: {self.run_status_path}",
            f"  batch status path: {self.batch_status_path}",
            f"  manifest path: {self.run_manifest_path}",
            f"  retry/recovery count: {status['retry_count']}",
        ]
        if status["status"] in {"failed", "interrupted"}:
            lines.append(f"  suggested next command: {self.command} --resume --run-id {self.run_id}")
        return "\n".join(lines)

    def _append_batch_status(
        self,
        *,
        batch_id: str,
        paper: str | None,
        paper_family: str | None,
        status: str,
        record_count: int,
        successful_records: int,
        failed_records: int,
        skipped_records: int,
        error_message: str | None,
    ) -> None:
        if status not in BATCH_STATUS_VALUES:
            raise ValueError(f"Unsupported batch status: {status}")
        now = self.clock()
        started_at = self._batch_starts.pop(batch_id, self._current_batch.started_at if self._current_batch else now)
        payload = {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "batch_id": batch_id,
            "paper": paper,
            "paper_family": paper_family,
            "status": status,
            "started_at": _iso(started_at),
            "finished_at": _iso(now),
            "elapsed_seconds": _seconds_between(started_at, now),
            "record_count": max(0, int(record_count)),
            "successful_records": max(0, int(successful_records)),
            "failed_records": max(0, int(failed_records)),
            "skipped_records": max(0, int(skipped_records)),
            "error_message": error_message,
        }
        self.status_dir.mkdir(parents=True, exist_ok=True)
        with self.batch_status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _clear_current_batch_if(self, batch_id: str) -> None:
        if self._current_batch and self._current_batch.batch_id == batch_id:
            self._current_batch = None
            self._status["current_batch_id"] = None

    def _refresh_computed_fields(self) -> None:
        now = self.finished_at or self.clock()
        self._status["updated_at"] = _iso(now)
        self._status["elapsed_seconds"] = _seconds_between(self.started_at, now)
        completed_units, total_units = self._progress_units()
        if total_units:
            self._status["percent_complete"] = round(min(100.0, (completed_units / total_units) * 100.0), 2)
        elif self._status["status"] == "completed":
            self._status["percent_complete"] = 100.0
        else:
            self._status["percent_complete"] = 0.0
        self._status["estimated_remaining_seconds"] = self._estimated_remaining_seconds(completed_units, total_units)

    def _progress_units(self) -> tuple[int, int]:
        completed_records = int(self._status["completed_records"] or 0)
        total_records = int(self._status["total_records"] or 0)
        if total_records:
            return completed_records, total_records
        return int(self._status["completed_batches"] or 0), int(self._status["total_batches"] or 0)

    def _estimated_remaining_seconds(self, completed_units: int, total_units: int) -> float | None:
        if not total_units or completed_units <= 0 or completed_units >= total_units:
            return None
        elapsed = float(self._status["elapsed_seconds"] or 0.0)
        if elapsed <= 0:
            return None
        rate = elapsed / completed_units
        return max(0.0, rate * (total_units - completed_units))

    def _write_status(self, *, force_render: bool = False) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_computed_fields()
        tmp = self.run_status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._status, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.run_status_path)
        self.renderer.render(self._status, force=force_render)

    def _write_manifest(self) -> None:
        now = self.finished_at or self.clock()
        payload = {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "command": self.command,
            "input_paths": self.input_paths,
            "output_paths": self.output_paths,
            "config_paths": self.config_paths,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "git_commit": _git_commit(),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "total_elapsed_seconds": _seconds_between(self.started_at, now),
            "final_status": self._status["status"],
        }
        self.status_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.run_manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.run_manifest_path)
