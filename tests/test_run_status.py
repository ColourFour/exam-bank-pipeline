from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from exam_bank.cli import cmd_process
from exam_bank.config import AppConfig
from exam_bank.pipeline import PipelineResult
from exam_bank.run_status import RunStatusTracker, completed_batch_ids


REQUIRED_RUN_STATUS_KEYS = {
    "run_id",
    "run_type",
    "command",
    "started_at",
    "updated_at",
    "finished_at",
    "status",
    "current_phase",
    "current_batch_id",
    "current_paper",
    "current_paper_family",
    "completed_batches",
    "total_batches",
    "completed_records",
    "total_records",
    "percent_complete",
    "elapsed_seconds",
    "estimated_remaining_seconds",
    "successful_records",
    "failed_records",
    "skipped_records",
    "retry_count",
    "output_path",
    "error_summary",
}


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 5, 11, 8, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_tracker_writes_status_manifest_batches_and_progress_math(tmp_path: Path) -> None:
    clock = FakeClock()
    tracker = RunStatusTracker(
        run_id="run-1",
        run_type="other",
        status_root=tmp_path / "run_status",
        command="fake command",
        input_paths=["input.json"],
        output_paths=["output.json"],
        progress=False,
        clock=clock,
    )

    tracker.start(phase="preparing_batches", total_batches=2, total_records=4)
    pending = _read_json(tracker.run_status_path)
    assert REQUIRED_RUN_STATUS_KEYS <= set(pending)
    assert pending["percent_complete"] == 0
    assert pending["estimated_remaining_seconds"] is None

    tracker.start_batch(batch_id="batch-a", paper="12spring24", paper_family="p1", record_count=2)
    clock.advance(10)
    tracker.complete_batch(batch_id="batch-a", paper="12spring24", paper_family="p1", record_count=2)
    halfway = _read_json(tracker.run_status_path)
    assert halfway["completed_records"] == 2
    assert halfway["percent_complete"] == 50
    assert halfway["elapsed_seconds"] == 10
    assert halfway["estimated_remaining_seconds"] == 10

    tracker.start_batch(batch_id="batch-b", paper="13spring24", paper_family="p1", record_count=2)
    clock.advance(5)
    tracker.fail_batch(
        batch_id="batch-b",
        paper="13spring24",
        paper_family="p1",
        record_count=2,
        error_message="provider_error",
    )
    tracker.finish("failed", error_summary="provider_error")

    final = _read_json(tracker.run_status_path)
    manifest = _read_json(tracker.run_manifest_path)
    batches = _read_jsonl(tracker.batch_status_path)

    assert final["status"] == "failed"
    assert final["failed_records"] == 2
    assert final["percent_complete"] == 100
    assert manifest["final_status"] == "failed"
    assert [batch["status"] for batch in batches] == ["completed", "failed"]
    assert completed_batch_ids(tracker.status_dir) == {"batch-a"}


def test_standard_process_cli_writes_run_status_during_fake_run(monkeypatch, tmp_path: Path, capsys) -> None:
    output_root = tmp_path / "output"

    def fake_process_inputs(
        input_path: str,
        config: AppConfig,
        *,
        progress: RunStatusTracker,
        resume_completed_batch_ids: set[str],
        force_rerun: bool,
    ) -> PipelineResult:
        assert input_path == str(tmp_path / "input")
        assert resume_completed_batch_ids == set()
        assert force_rerun is False
        progress.set_totals(total_batches=2, total_records=4)
        progress.start_batch(batch_id="p1_12spring24", paper="12spring24", paper_family="p1", record_count=2)
        progress.complete_batch(batch_id="p1_12spring24", paper="12spring24", paper_family="p1", record_count=2)
        progress.start_batch(batch_id="p1_13spring24", paper="13spring24", paper_family="p1", record_count=2)
        progress.skip_batch(batch_id="p1_13spring24", paper="13spring24", paper_family="p1", record_count=2)
        json_path = config.output.json_dir / config.naming.json_name
        return PipelineResult([], json_path, config.output.root_dir(), question_count=4, paper_count=2)

    monkeypatch.setattr("exam_bank.cli.process_inputs", fake_process_inputs)

    exit_code = cmd_process(
        Namespace(
            command="process",
            input=str(tmp_path / "input"),
            output=str(output_root),
            config="config.yaml",
            enable_ocr=False,
            ocr_language="",
            tesseract_cmd="",
            progress=False,
            status_dir=output_root / "run_status",
            run_id="standard-test",
            resume=False,
            force_rerun=False,
        )
    )

    assert exit_code == 0
    status_path = output_root / "run_status" / "standard-test" / "run_status.json"
    batch_path = output_root / "run_status" / "standard-test" / "batch_status.jsonl"
    status = _read_json(status_path)
    batches = _read_jsonl(batch_path)
    summary = capsys.readouterr().out

    assert REQUIRED_RUN_STATUS_KEYS <= set(status)
    assert status["run_type"] == "standard"
    assert status["status"] == "completed"
    assert status["completed_records"] == 4
    assert status["skipped_records"] == 2
    assert status["percent_complete"] == 100
    assert [batch["status"] for batch in batches] == ["completed", "skipped"]
    assert "total elapsed time" in summary
    assert str(status_path) in summary


def test_standard_resume_reads_completed_batches(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    previous = output_root / "run_status" / "resume-test"
    previous.mkdir(parents=True)
    with (previous / "batch_status.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "run_id": "resume-test",
                    "run_type": "standard",
                    "batch_id": "p1_12spring24",
                    "paper": "12spring24",
                    "paper_family": "p1",
                    "status": "completed",
                    "started_at": "2026-05-11T08:00:00+00:00",
                    "finished_at": "2026-05-11T08:01:00+00:00",
                    "elapsed_seconds": 60,
                    "record_count": 2,
                    "successful_records": 2,
                    "failed_records": 0,
                    "skipped_records": 0,
                    "error_message": None,
                }
            )
            + "\n"
        )

    captured: dict[str, object] = {}

    def fake_process_inputs(
        _input_path: str,
        config: AppConfig,
        *,
        progress: RunStatusTracker,
        resume_completed_batch_ids: set[str],
        force_rerun: bool,
    ) -> PipelineResult:
        captured["resume_completed_batch_ids"] = resume_completed_batch_ids
        captured["force_rerun"] = force_rerun
        progress.set_totals(total_batches=1, total_records=2)
        progress.skip_batch(batch_id="p1_12spring24", paper="12spring24", paper_family="p1", record_count=2)
        return PipelineResult([], config.output.json_dir / config.naming.json_name, config.output.root_dir(), question_count=2, paper_count=1)

    monkeypatch.setattr("exam_bank.cli.process_inputs", fake_process_inputs)

    exit_code = cmd_process(
        Namespace(
            command="process",
            input=str(tmp_path / "input"),
            output=str(output_root),
            config="config.yaml",
            enable_ocr=False,
            ocr_language="",
            tesseract_cmd="",
            progress=False,
            status_dir=output_root / "run_status",
            run_id="resume-test",
            resume=True,
            force_rerun=False,
        )
    )

    assert exit_code == 0
    assert captured["resume_completed_batch_ids"] == {"p1_12spring24"}
    assert captured["force_rerun"] is False
