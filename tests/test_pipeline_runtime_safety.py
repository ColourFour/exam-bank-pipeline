from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from exam_bank import pipeline
from exam_bank.config import AppConfig
from exam_bank.document_metadata import DocumentMetadata
from exam_bank.document_registry import build_document_registry
from exam_bank.exporters import records_to_output_questions
from exam_bank.models import QuestionRecord
from exam_bank.output_layout import question_image_output_path
from exam_bank.publication_safety import (
    PublicationReadBlockedError,
    read_json_under_publication_guard,
)
from exam_bank.trust import PaperTotalStatus, RescanResult


class _MemoryProgress:
    def __init__(self) -> None:
        self.artifacts: dict[tuple[str, str], Any] = {}
        self.skipped: list[str] = []

    def read_batch_artifact(self, batch_id: str, name: str) -> Any:
        return self.artifacts.get((batch_id, name))

    def write_batch_artifact(self, batch_id: str, name: str, payload: Any) -> None:
        self.artifacts[(batch_id, name)] = payload

    def skip_batch(self, *, batch_id: str, **_kwargs: Any) -> None:
        self.skipped.append(batch_id)

    def set_totals(self, **_kwargs: Any) -> None:
        pass

    def start_batch(self, **_kwargs: Any) -> None:
        pass

    def complete_batch(self, **_kwargs: Any) -> None:
        pass

    def fail_batch(self, **_kwargs: Any) -> None:
        pass

    def update_phase(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _question_record(source_pdf: Path, config: AppConfig, *, component: str) -> QuestionRecord:
    image_path = question_image_output_path(source_pdf, "1", config)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(f"question-{component}".encode())
    return QuestionRecord(
        source_pdf=str(source_pdf),
        paper_name=f"{component}winter25",
        question_number="1",
        full_question_label="1",
        screenshot_path=str(image_path),
        combined_question_text="Find x.",
        body_text_raw="Find x.",
        body_text_normalized="Find x.",
        math_lines=[],
        diagram_text=[],
        extraction_quality_score=0.95,
        extraction_quality_flags=[],
        part_texts=[],
        answer_text="x = 2",
        paper_family="P1",
        source_paper_family="P1",
        inferred_paper_family="P1",
        paper_family_confidence="high",
        topic="quadratics",
        subtopic="general",
        topic_confidence="high",
        topic_evidence="fixture",
        secondary_topics=[],
        topic_uncertain=False,
        topic_trust_status="normal",
        difficulty="easy",
        difficulty_confidence="high",
        difficulty_evidence="fixture",
        difficulty_uncertain=False,
        marks=2,
        marks_if_available=2,
        page_numbers=[1],
        review_flags=[],
        confidence=0.9,
        syllabus_code="9709",
        session="winter25",
        year="2025",
        document_type="question_paper",
        component=component,
        source_paper_code=component,
        scope_quality_status="clean",
        text_source_profile="native_pdf",
        text_fidelity_status="clean",
        question_text_role="readable_text",
        question_text_trust="high",
        visual_curation_status="ready",
        text_only_status="ready",
        validation_status="pass",
    )


def _touch_question_paper(root: Path, component: str) -> Path:
    path = root / f"9709 Mathematics November 2025 Question Paper {component}.pdf"
    path.write_bytes(f"%PDF-{component}\n".encode())
    return path


def _prepare_publication_crash_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[Path], list[dict[str, Any]]]:
    output_root = tmp_path / "output"
    stage_root = tmp_path / "stage"
    (output_root / "json").mkdir(parents=True)
    (stage_root / "json").mkdir(parents=True)
    (output_root / "artifact.bin").write_bytes(b"old-artifact")
    (output_root / "json" / "question_bank.json").write_text("old-json", encoding="utf-8")
    (stage_root / "artifact.bin").write_bytes(b"new-artifact")
    (stage_root / "introduced.bin").write_bytes(b"new-only")
    (stage_root / "json" / "question_bank.json").write_text("new-json", encoding="utf-8")
    files = [
        stage_root / "artifact.bin",
        stage_root / "introduced.bin",
        stage_root / "json" / "question_bank.json",
    ]
    journal_root, entries = pipeline._create_publication_journal(
        files,
        stage_root=stage_root,
        output_root=output_root,
        final_json_relative=Path("json/question_bank.json"),
    )
    assert (journal_root / pipeline._PUBLICATION_JOURNAL_FILENAME).is_file()
    return output_root, stage_root, journal_root, files, entries


def test_parallel_interrupt_cancels_queued_futures_before_executor_shutdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _touch_question_paper(tmp_path, "12")
    _touch_question_paper(tmp_path, "13")
    _touch_question_paper(tmp_path, "31")
    _touch_question_paper(tmp_path, "32")
    registry = build_document_registry(tmp_path)
    config = AppConfig()
    config.output.apply_root(tmp_path / "isolated-output")
    events: list[str] = []
    submitted: list[FakeFuture] = []

    class FakeFuture:
        def __init__(self, index: int) -> None:
            self.index = index
            self.cancelled = False

        def result(self):
            events.append(f"result:{self.index}")
            raise KeyboardInterrupt

        def cancel(self) -> bool:
            self.cancelled = True
            events.append(f"cancel:{self.index}")
            return True

    class FakeExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _traceback) -> bool:
            assert all(future.cancelled for future in submitted)
            events.append("shutdown")
            return False

        def submit(self, *_args: Any, **_kwargs: Any) -> FakeFuture:
            future = FakeFuture(len(submitted))
            submitted.append(future)
            return future

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", lambda _futures: iter(submitted[:1]))

    with pytest.raises(KeyboardInterrupt):
        pipeline._process_registry_entries_parallel(
            registry,
            config,
            progress=None,
            resume_completed_batch_ids=None,
            force_rerun=False,
            workers=2,
            publication_config=config,
        )

    assert len(submitted) == 2
    assert events == ["result:0", "cancel:0", "cancel:1", "shutdown"]


def test_parallel_submission_window_never_exceeds_worker_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for component in ("11", "12", "13", "31", "32"):
        _touch_question_paper(tmp_path, component)
    registry = build_document_registry(tmp_path)
    first_entry = registry.question_paper_entries()[0]
    cached_batch_id = pipeline._entry_progress_context(first_entry)["batch_id"]
    config = AppConfig()
    output_root = tmp_path / "isolated-output"
    config.output.apply_root(output_root)
    progress = _MemoryProgress()
    progress.started = []
    progress.completed = []
    submitted: list[FakeFuture] = []
    completion_windows: list[tuple[int, int]] = []
    cache_writes: list[str] = []
    outstanding = 0

    class FakeFuture:
        def __init__(self, index: int) -> None:
            self.index = index

        def result(self):
            nonlocal outstanding
            outstanding -= 1
            return [self.index], output_root

        def cancel(self) -> bool:
            return False

    class FakeExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback) -> bool:
            assert outstanding == 0
            return False

        def submit(self, *_args: Any, **_kwargs: Any) -> FakeFuture:
            nonlocal outstanding
            assert outstanding < 2
            future = FakeFuture(len(submitted))
            submitted.append(future)
            outstanding += 1
            return future

    def start_batch(**kwargs: Any) -> None:
        progress.started.append(kwargs["batch_id"])
        assert len(progress.started) - len(progress.completed) <= 2

    def complete_batch(**kwargs: Any) -> None:
        progress.completed.append(kwargs["batch_id"])

    def bounded_as_completed(active_futures):
        active = tuple(active_futures)
        completion_windows.append((len(active), len(submitted)))
        return iter(active[:1])

    finalized_records: list[int] = []

    def finalize(records, *_args: Any, **_kwargs: Any):
        finalized_records.extend(records)
        return "finished"

    monkeypatch.setattr(progress, "start_batch", start_batch)
    monkeypatch.setattr(progress, "complete_batch", complete_batch)
    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", bounded_as_completed)
    monkeypatch.setattr(
        pipeline,
        "_load_valid_cached_batch_records",
        lambda *_args, **kwargs: [-1] if kwargs["batch_id"] == cached_batch_id else None,
    )
    monkeypatch.setattr(pipeline, "records_to_output_questions", lambda *_args: [])
    monkeypatch.setattr(
        pipeline,
        "_write_batch_cache",
        lambda *_args, **kwargs: cache_writes.append(kwargs["batch_id"]),
    )
    monkeypatch.setattr(pipeline, "_promote_worker_artifacts", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_finalize_registry_result", finalize)

    result = pipeline._process_registry_entries_parallel(
        registry,
        config,
        progress=progress,
        resume_completed_batch_ids={cached_batch_id},
        force_rerun=False,
        workers=2,
        publication_config=config,
    )

    assert result == "finished"
    assert len(submitted) == 4
    assert completion_windows == [(2, 2), (2, 3), (2, 4), (1, 4)]
    assert len(progress.started) == len(progress.completed) == 4
    assert progress.skipped == [cached_batch_id]
    assert len(cache_writes) == 4
    assert finalized_records == [-1, 0, 1, 2, 3]


def test_output_lock_rejects_overlapping_publishers(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    with pipeline._output_root_lock(output_root):
        with pytest.raises(pipeline.PipelineOutputLockedError, match="already publishing"):
            with pipeline._output_root_lock(output_root):
                pass


def test_publication_read_guard_blocks_writer_and_incomplete_journal(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    question_bank_path = output_root / "json" / "question_bank.json"
    question_bank_path.parent.mkdir(parents=True)
    question_bank_path.write_text('{"questions": []}', encoding="utf-8")
    with pipeline._output_root_lock(output_root):
        with pytest.raises(PublicationReadBlockedError, match="currently being updated"):
            read_json_under_publication_guard(question_bank_path)

    interrupted_journal = tmp_path / ".output.rollback-crashed"
    interrupted_journal.mkdir()
    with pytest.raises(PublicationReadBlockedError, match="awaiting recovery"):
        read_json_under_publication_guard(question_bank_path)


def test_publication_read_guard_allows_committed_cleanup_marker(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    question_bank_path = output_root / "json" / "question_bank.json"
    question_bank_path.parent.mkdir(parents=True)
    question_bank_path.write_text('{"questions": []}', encoding="utf-8")
    (tmp_path / ".output.committed-awaiting-cleanup").mkdir()

    assert read_json_under_publication_guard(question_bank_path) == {"questions": []}


def test_process_sample_uses_locked_transactional_publication(tmp_path: Path, monkeypatch) -> None:
    question_pdf = _touch_question_paper(tmp_path, "12")
    config = AppConfig()
    output_root = tmp_path / "output"
    config.output.apply_root(output_root)
    output_root.mkdir()
    (output_root / "artifact.bin").write_bytes(b"old")

    def build_sample(
        source_pdf: Path,
        stage_config: AppConfig,
        **_kwargs: Any,
    ) -> list[QuestionRecord]:
        (stage_config.output.root_dir() / "artifact.bin").write_bytes(b"new")
        return [_question_record(Path(source_pdf), stage_config, component="12")]

    monkeypatch.setattr(pipeline, "build_records_for_pdf", build_sample)
    result = pipeline.process_sample(question_pdf, config)

    assert result.output_root == output_root
    assert result.json_path.is_file()
    assert json.loads(result.json_path.read_text(encoding="utf-8"))["record_count"] == 1
    assert (output_root / "artifact.bin").read_bytes() == b"new"
    assert Path(result.records[0].screenshot_path).is_relative_to(output_root)
    assert Path(result.records[0].screenshot_path).is_file()
    assert not list(tmp_path.glob(".output.sample-*"))
    assert not list(tmp_path.glob(".output.rollback-*"))
    assert not list(tmp_path.glob(".output.committed-*"))


def test_process_sample_rejects_overlap_and_never_publishes_failed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    question_pdf = _touch_question_paper(tmp_path, "12")
    config = AppConfig()
    output_root = tmp_path / "output"
    config.output.apply_root(output_root)
    output_root.mkdir()
    (output_root / "artifact.bin").write_bytes(b"old")
    calls = 0

    def fail_sample(
        _source_pdf: Path,
        stage_config: AppConfig,
        **_kwargs: Any,
    ) -> list[QuestionRecord]:
        nonlocal calls
        calls += 1
        (stage_config.output.root_dir() / "artifact.bin").write_bytes(b"partial")
        raise RuntimeError("sample extraction failed")

    monkeypatch.setattr(pipeline, "build_records_for_pdf", fail_sample)
    with pipeline._output_root_lock(output_root):
        with pytest.raises(pipeline.PipelineOutputLockedError):
            pipeline.process_sample(question_pdf, config)
    assert calls == 0

    with pytest.raises(RuntimeError, match="sample extraction failed"):
        pipeline.process_sample(question_pdf, config)

    assert calls == 1
    assert (output_root / "artifact.bin").read_bytes() == b"old"
    assert not list(tmp_path.glob(".output.sample-*"))


def test_transactional_promotion_rolls_back_all_replaced_files(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "output"
    stage_root = tmp_path / "stage"
    (output_root / "json").mkdir(parents=True)
    (stage_root / "json").mkdir(parents=True)
    (output_root / "a.bin").write_bytes(b"old-a")
    (output_root / "b.bin").write_bytes(b"old-b")
    (output_root / "json" / "question_bank.json").write_text("old-json", encoding="utf-8")
    (stage_root / "a.bin").write_bytes(b"new-a")
    (stage_root / "b.bin").write_bytes(b"new-b")
    (stage_root / "json" / "question_bank.json").write_text("new-json", encoding="utf-8")

    real_replace = os.replace

    def fail_second_artifact(source: str | Path, destination: str | Path) -> None:
        if Path(source) == stage_root / "b.bin" and Path(destination) == output_root / "b.bin":
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_second_artifact)
    with pytest.raises(OSError, match="injected publication failure"):
        pipeline._promote_run_artifacts_transactionally(
            stage_root,
            output_root,
            final_json_relative=Path("json/question_bank.json"),
        )

    assert (output_root / "a.bin").read_bytes() == b"old-a"
    assert (output_root / "b.bin").read_bytes() == b"old-b"
    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "old-json"


def test_transactional_promotion_requires_final_json_before_mutating_output(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    stage_root = tmp_path / "stage"
    output_root.mkdir()
    stage_root.mkdir()
    (output_root / "artifact.bin").write_bytes(b"old")
    (stage_root / "artifact.bin").write_bytes(b"new")

    with pytest.raises(FileNotFoundError, match="Staged final JSON is missing"):
        pipeline._promote_run_artifacts_transactionally(
            stage_root,
            output_root,
            final_json_relative=Path("json/question_bank.json"),
        )

    assert (output_root / "artifact.bin").read_bytes() == b"old"
    assert (stage_root / "artifact.bin").read_bytes() == b"new"


def test_transactional_promotion_rolls_back_interruption_at_final_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "output"
    stage_root = tmp_path / "stage"
    (output_root / "json").mkdir(parents=True)
    (stage_root / "json").mkdir(parents=True)
    output_artifact = output_root / "artifact.bin"
    staged_artifact = stage_root / "artifact.bin"
    output_json = output_root / "json" / "question_bank.json"
    staged_json = stage_root / "json" / "question_bank.json"
    output_artifact.write_bytes(b"old-artifact")
    output_json.write_text("old-json", encoding="utf-8")
    staged_artifact.write_bytes(b"new-artifact")
    staged_json.write_text("new-json", encoding="utf-8")
    staged_move_order: list[Path] = []
    real_replace = os.replace

    def interrupt_final_json(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.is_relative_to(stage_root):
            staged_move_order.append(source_path.relative_to(stage_root))
        if source_path == staged_json and destination_path == output_json:
            raise KeyboardInterrupt
        real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", interrupt_final_json)
    with pytest.raises(KeyboardInterrupt):
        pipeline._promote_run_artifacts_transactionally(
            stage_root,
            output_root,
            final_json_relative=Path("json/question_bank.json"),
        )

    assert staged_move_order == [Path("artifact.bin"), Path("json/question_bank.json")]
    assert output_artifact.read_bytes() == b"old-artifact"
    assert output_json.read_text(encoding="utf-8") == "old-json"


def test_transactional_promotion_ignores_stale_sibling_stage(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    stage_root = tmp_path / ".output.run-current"
    stale_stage = tmp_path / ".output.run-stale"
    (stage_root / "json").mkdir(parents=True)
    stale_stage.mkdir()
    (stage_root / "artifact.bin").write_bytes(b"current")
    (stage_root / "json" / "question_bank.json").write_text("current-json", encoding="utf-8")
    (stale_stage / "poison.bin").write_bytes(b"stale")

    pipeline._promote_run_artifacts_transactionally(
        stage_root,
        output_root,
        final_json_relative=Path("json/question_bank.json"),
    )

    assert (output_root / "artifact.bin").read_bytes() == b"current"
    assert not (output_root / "poison.bin").exists()
    assert (stale_stage / "poison.bin").read_bytes() == b"stale"


def test_transactional_promotion_never_replaces_pipeline_lock(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    stage_root = tmp_path / "stage"
    (stage_root / "json").mkdir(parents=True)
    output_root.mkdir()
    live_lock = output_root / ".pipeline.lock"
    live_lock.write_text("live-lock", encoding="utf-8")
    (stage_root / ".pipeline.lock").write_text("staged-lock", encoding="utf-8")
    (stage_root / "nested").mkdir()
    (stage_root / "nested" / ".pipeline.lock").write_text("nested-lock", encoding="utf-8")
    (stage_root / "artifact.bin").write_bytes(b"current")
    (stage_root / "json" / "question_bank.json").write_text("current-json", encoding="utf-8")

    pipeline._promote_run_artifacts_transactionally(
        stage_root,
        output_root,
        final_json_relative=Path("json/question_bank.json"),
    )

    assert live_lock.read_text(encoding="utf-8") == "live-lock"
    assert not (output_root / "nested" / ".pipeline.lock").exists()
    assert (output_root / "artifact.bin").read_bytes() == b"current"


def test_next_lock_recovers_crash_after_asset_moves(tmp_path: Path) -> None:
    output_root, _stage_root, journal_root, files, entries = _prepare_publication_crash_fixture(tmp_path)
    for source, entry in zip(files[:2], entries[:2], strict=True):
        pipeline._promote_publication_file(
            source,
            output_root=output_root,
            journal_root=journal_root,
            entry=entry,
        )

    assert (output_root / "artifact.bin").read_bytes() == b"new-artifact"
    assert (output_root / "introduced.bin").read_bytes() == b"new-only"
    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "old-json"
    with pipeline._output_root_lock(output_root):
        pass

    assert (output_root / "artifact.bin").read_bytes() == b"old-artifact"
    assert not (output_root / "introduced.bin").exists()
    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "old-json"
    assert not journal_root.exists()


def test_next_lock_rolls_back_crash_after_final_json_before_commit(tmp_path: Path) -> None:
    output_root, _stage_root, journal_root, files, entries = _prepare_publication_crash_fixture(tmp_path)
    for source, entry in zip(files, entries, strict=True):
        pipeline._promote_publication_file(
            source,
            output_root=output_root,
            journal_root=journal_root,
            entry=entry,
        )

    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "new-json"
    with pipeline._output_root_lock(output_root):
        pass

    assert (output_root / "artifact.bin").read_bytes() == b"old-artifact"
    assert not (output_root / "introduced.bin").exists()
    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "old-json"
    assert not journal_root.exists()


def test_next_lock_keeps_committed_publication_when_cleanup_was_interrupted(tmp_path: Path) -> None:
    output_root, _stage_root, journal_root, files, entries = _prepare_publication_crash_fixture(tmp_path)
    for source, entry in zip(files, entries, strict=True):
        pipeline._promote_publication_file(
            source,
            output_root=output_root,
            journal_root=journal_root,
            entry=entry,
        )
    committed_root = pipeline._mark_publication_committed(journal_root, output_root=output_root)

    assert committed_root.is_dir()
    with pipeline._output_root_lock(output_root):
        pass

    assert (output_root / "artifact.bin").read_bytes() == b"new-artifact"
    assert (output_root / "introduced.bin").read_bytes() == b"new-only"
    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "new-json"
    assert not committed_root.exists()


def test_publication_recovery_is_idempotent_before_journal_cleanup(tmp_path: Path) -> None:
    output_root, _stage_root, journal_root, files, entries = _prepare_publication_crash_fixture(tmp_path)
    for source, entry in zip(files[:2], entries[:2], strict=True):
        pipeline._promote_publication_file(
            source,
            output_root=output_root,
            journal_root=journal_root,
            entry=entry,
        )
    pipeline._rollback_publication_journal(
        journal_root,
        output_root=output_root,
        entries=entries,
    )

    assert journal_root.is_dir()
    assert (output_root / "artifact.bin").read_bytes() == b"old-artifact"
    assert not (output_root / "introduced.bin").exists()
    with pipeline._output_root_lock(output_root):
        pass

    assert (output_root / "artifact.bin").read_bytes() == b"old-artifact"
    assert (output_root / "json" / "question_bank.json").read_text(encoding="utf-8") == "old-json"
    assert not journal_root.exists()


def test_failed_run_never_publishes_partial_stage(tmp_path: Path, monkeypatch) -> None:
    _touch_question_paper(tmp_path, "12")
    _touch_question_paper(tmp_path, "13")
    registry = build_document_registry(tmp_path)
    config = AppConfig()
    output_root = tmp_path / "output"
    config.output.apply_root(output_root)
    output_root.mkdir()
    (output_root / "published.bin").write_bytes(b"old")
    calls = 0

    def fail_on_second(_question_pdf: Path, stage_config: AppConfig, **_kwargs: Any) -> list[QuestionRecord]:
        nonlocal calls
        calls += 1
        (stage_config.output.root_dir() / "published.bin").write_bytes(b"partial")
        if calls == 2:
            raise RuntimeError("second paper failed")
        return [_question_record(_question_pdf, stage_config, component="12")]

    monkeypatch.setattr(pipeline, "build_records_for_pdf", fail_on_second)
    with pytest.raises(RuntimeError, match="second paper failed"):
        pipeline._process_registry_entries_transactionally(registry, config)

    assert (output_root / "published.bin").read_bytes() == b"old"
    assert not list(tmp_path.glob(".output.run-*"))


@pytest.mark.parametrize("workers", [1, 2])
def test_zero_record_paper_fails_closed_with_source_and_batch_context(
    tmp_path: Path,
    monkeypatch,
    workers: int,
) -> None:
    question_pdf = _touch_question_paper(tmp_path, "12")
    registry = build_document_registry(tmp_path)
    entry = registry.question_paper_entries()[0]
    batch_id = pipeline._entry_progress_context(entry)["batch_id"]
    config = AppConfig()
    output_root = tmp_path / "output"
    config.output.apply_root(output_root)
    output_root.mkdir()
    (output_root / "published.bin").write_bytes(b"old")
    progress = _MemoryProgress()
    failed: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    monkeypatch.setattr(progress, "fail_batch", lambda **kwargs: failed.append(kwargs))
    monkeypatch.setattr(progress, "complete_batch", lambda **kwargs: completed.append(kwargs))
    monkeypatch.setattr(pipeline, "build_records_for_pdf", lambda *_args, **_kwargs: [])

    with pytest.raises(
        pipeline.EmptyPaperExtractionError,
        match=rf"source={question_pdf} batch_id={batch_id}",
    ):
        pipeline._process_registry_entries_transactionally(
            registry,
            config,
            progress=progress,
            workers=workers,
        )

    assert completed == []
    assert len(failed) == 1
    assert failed[0]["batch_id"] == batch_id
    assert str(question_pdf) in failed[0]["error_message"]
    assert (output_root / "published.bin").read_bytes() == b"old"


def test_resume_restores_full_records_and_verifies_live_assets(tmp_path: Path, monkeypatch) -> None:
    first_pdf = _touch_question_paper(tmp_path, "12")
    second_pdf = _touch_question_paper(tmp_path, "13")
    registry = build_document_registry(tmp_path)
    config = AppConfig()
    output_root = tmp_path / "output"
    config.output.apply_root(output_root)
    config.ensure_output_dirs()
    progress = _MemoryProgress()

    entries = registry.question_paper_entries()
    first_entry = next(entry for entry in entries if entry.question_paper == first_pdf)
    first_context = pipeline._entry_progress_context(first_entry)
    cached_record = _question_record(first_pdf, config, component="12")
    cached_payload = records_to_output_questions([cached_record], output_root)
    pipeline._write_batch_cache(
        progress,
        batch_id=first_context["batch_id"],
        cache_key=pipeline._extraction_batch_cache_key(first_entry, config),
        records=[cached_record],
        question_payload=cached_payload,
        rendered_root=output_root,
        publication_root=output_root,
    )

    built_sources: list[Path] = []

    def build_uncached(question_pdf: Path, stage_config: AppConfig, **_kwargs: Any) -> list[QuestionRecord]:
        built_sources.append(Path(question_pdf))
        return [_question_record(Path(question_pdf), stage_config, component="13")]

    monkeypatch.setattr(pipeline, "build_records_for_pdf", build_uncached)
    result = pipeline._process_registry_entries_transactionally(
        registry,
        config,
        progress=progress,
        resume_completed_batch_ids={first_context["batch_id"]},
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    expected_sources = [
        {
            "role": "question_paper",
            "path": str(path),
            "exists": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (first_pdf, second_pdf)
    ]
    expected_manifest = json.dumps(
        {"schema_version": 1, "sources": expected_sources},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert built_sources == [second_pdf]
    assert progress.skipped == [first_context["batch_id"]]
    assert len(result.records) == 2
    assert payload["record_count"] == 2
    assert payload["run_manifest"]["artifact_root"] == str(output_root)
    assert payload["run_manifest"]["input_manifest_sha256"] == hashlib.sha256(expected_manifest).hexdigest()
    assert {question["paper"] for question in payload["questions"]} == {"12winter25", "13winter25"}

    cached_asset = output_root / cached_payload[0]["question_image_path"]
    cached_asset.write_bytes(b"corrupt")
    manifest = progress.read_batch_artifact(first_context["batch_id"], "assets.json")
    assert not pipeline._cached_asset_manifest_is_current(manifest, publication_root=output_root)


def test_cache_key_includes_algorithm_fingerprint(tmp_path: Path, monkeypatch) -> None:
    question_pdf = _touch_question_paper(tmp_path, "12")
    entry = build_document_registry(tmp_path).question_paper_entries()[0]
    config = AppConfig()

    monkeypatch.setattr(pipeline, "_pipeline_code_fingerprint", lambda: "algorithm-one")
    first = pipeline._extraction_batch_cache_key(entry, config)
    monkeypatch.setattr(pipeline, "_pipeline_code_fingerprint", lambda: "algorithm-two")
    second = pipeline._extraction_batch_cache_key(entry, config)

    assert question_pdf.is_file()
    assert first != second


def test_resume_rejects_mixed_cache_generation(tmp_path: Path) -> None:
    question_pdf = _touch_question_paper(tmp_path, "12")
    entry = build_document_registry(tmp_path).question_paper_entries()[0]
    config = AppConfig()
    output_root = tmp_path / "output"
    config.output.apply_root(output_root)
    progress = _MemoryProgress()
    context = pipeline._entry_progress_context(entry)
    record = _question_record(question_pdf, config, component="12")
    question_payload = records_to_output_questions([record], output_root)
    cache_key = pipeline._extraction_batch_cache_key(entry, config)
    pipeline._write_batch_cache(
        progress,
        batch_id=context["batch_id"],
        cache_key=cache_key,
        records=[record],
        question_payload=question_payload,
        rendered_root=output_root,
        publication_root=output_root,
    )
    replacement_records = progress.read_batch_artifact(context["batch_id"], "records.json")
    replacement_records[0]["answer_text"] = "partially rewritten generation"
    progress.write_batch_artifact(context["batch_id"], "records.json", replacement_records)

    assert pipeline._load_valid_cached_batch_records(
        progress,
        batch_id=context["batch_id"],
        expected_cache_key=cache_key,
        publication_root=output_root,
    ) is None


def test_rejected_rescan_artifacts_cannot_overwrite_selected_pass(tmp_path: Path, monkeypatch) -> None:
    question_pdf = _touch_question_paper(tmp_path, "12")
    config = AppConfig()
    config.output.apply_root(tmp_path / "output")
    metadata = DocumentMetadata(
        syllabus="9709",
        subject="Mathematics",
        year="2025",
        session="winter25",
        normalized_session_key="winter25",
        document_type="question_paper",
        component="12",
        source="fixture",
    )
    layout_calls = 0

    def fake_layouts(_path: Path, pass_config: AppConfig) -> list[str]:
        nonlocal layout_calls
        layout_calls += 1
        layout_artifact = pass_config.output.root_dir() / "layout.bin"
        layout_artifact.parent.mkdir(parents=True, exist_ok=True)
        layout_artifact.write_bytes(pass_config.output.root_dir().name.encode())
        return [f"layout-{layout_calls}"]

    def fake_records(**kwargs: Any) -> list[QuestionRecord]:
        pass_config = kwargs["config"]
        records = [_question_record(question_pdf, pass_config, component="12")]
        Path(records[0].screenshot_path).write_bytes(pass_config.output.root_dir().name.encode())
        return records

    mismatch = {
        "expected_total": 75,
        "detected_total": 2,
        "status": PaperTotalStatus.MISMATCH,
        "difference": -73,
    }
    monkeypatch.setattr(pipeline, "extract_pdf_layout", fake_layouts)
    monkeypatch.setattr(pipeline, "parse_internal_document_metadata", lambda _layouts: metadata)
    monkeypatch.setattr(pipeline, "reconcile_document_metadata", lambda filename, _internal: filename)
    monkeypatch.setattr(pipeline, "detect_question_spans", lambda layouts, *_args: [layouts[0]])
    monkeypatch.setattr(pipeline, "_build_records_from_spans", fake_records)
    monkeypatch.setattr(pipeline, "_paper_total_check", lambda *_args, **_kwargs: mismatch)
    monkeypatch.setattr(
        pipeline,
        "_select_preferred_detection_pass",
        lambda **kwargs: (
            kwargs["initial_spans"],
            kwargs["initial_records"],
            kwargs["initial_total_check"],
            RescanResult.NO_IMPROVEMENT,
        ),
    )
    monkeypatch.setattr(pipeline, "_reconcile_paper_topics", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_paper_total_focus", lambda *_args: {})
    monkeypatch.setattr(pipeline, "_apply_paper_total_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_reconcile_question_mark_total_mismatches", lambda *_args: None)

    records = pipeline.build_records_for_pdf(question_pdf, config, filename_metadata=metadata)
    selected_path = Path(records[0].screenshot_path)

    assert selected_path.read_bytes() == b"initial"
    assert (config.output.root_dir() / "layout.bin").read_bytes() == b"initial"
    assert selected_path.is_relative_to(config.output.root_dir())
    assert layout_calls == 2
    assert not list(config.output.root_dir().glob(".detection-passes-*"))
