from __future__ import annotations

import json
from pathlib import Path

from exam_bank.image_alignment_controller import AlignmentExtractionOutput, ImageBindingContext, resolve_image_to_question, run_image_alignment_loop


def test_alignment_loop_converges_when_data_is_clean(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_image(output / _q_path(1))
    _write_image(output / _ms_path(1))
    bank = _question_bank([_question(1)])

    result = run_image_alignment_loop(
        lambda _context: AlignmentExtractionOutput(bank, artifact_root=output),
        artifact_root=output,
        report_path=tmp_path / "report.json",
        max_iterations=3,
    )

    assert result.initial_alignment_score == 1.0
    assert result.final_alignment_score == 1.0
    assert result.iterations_run == 1
    assert result.reached_threshold is True
    assert result.failure_breakdown == {
        "missing_image": 0,
        "orphan_image": 0,
        "misaligned_image": 0,
        "weak_crop": 0,
        "duplicate_mapping": 0,
        "legacy_segmentation_failure": 0,
    }
    assert json.loads(result.report_path.read_text(encoding="utf-8"))["final_convergence_status"] == "converged"


def test_alignment_loop_stops_at_max_iterations_without_infinite_loop(tmp_path: Path) -> None:
    output = tmp_path / "output"
    calls: list[int] = []
    bank = _question_bank([_question(1, question_path="", mark_scheme_path="")])

    def runner(context):
        calls.append(context.iteration)
        return AlignmentExtractionOutput(bank, artifact_root=output)

    result = run_image_alignment_loop(
        runner,
        artifact_root=output,
        report_path=tmp_path / "report.json",
        max_iterations=3,
    )

    assert calls == [1, 2, 3]
    assert result.initial_alignment_score == 0.0
    assert result.final_alignment_score == 0.0
    assert result.iterations_run == 3
    assert result.reached_threshold is False
    assert result.failure_breakdown["missing_image"] == 2


def test_alignment_loop_preserves_previously_correct_mappings(tmp_path: Path) -> None:
    output = tmp_path / "output"
    for index in [1, 2]:
        _write_image(output / _q_path(index))
        _write_image(output / _ms_path(index))
    first_pass = _question_bank(
        [
            _question(1),
            _question(2, question_path="", mark_scheme_path=""),
        ]
    )
    second_pass = _question_bank(
        [
            _question(1, question_path=_q_path(2), mark_scheme_path=_ms_path(2)),
            _question(2),
        ]
    )

    def runner(context):
        return AlignmentExtractionOutput(first_pass if context.iteration == 1 else second_pass, artifact_root=output)

    result = run_image_alignment_loop(
        runner,
        artifact_root=output,
        report_path=tmp_path / "report.json",
        max_iterations=2,
    )

    assert result.reached_threshold is True
    events = result.report["safety"]["events"]
    assert any(event["event"] == "protected_mapping_restored" and event["question_id"] == "12summer21_q01" for event in events)
    assert any(action["strategy"] == "restore_previous_correct_mapping" for action in result.report["repair_log"])


def test_alignment_loop_enforces_image_count_stability(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_image(output / _q_path(1))
    _write_image(output / _ms_path(1))
    first_pass = _question_bank(
        [
            _question(1),
            _question(2, question_path="", mark_scheme_path=""),
        ]
    )
    second_pass = _question_bank([_question(1)])

    def runner(context):
        return AlignmentExtractionOutput(first_pass if context.iteration == 1 else second_pass, artifact_root=output)

    result = run_image_alignment_loop(
        runner,
        artifact_root=output,
        report_path=tmp_path / "report.json",
        max_iterations=2,
    )

    assert result.final_alignment_score == 1.0
    assert result.reached_threshold is False
    assert result.report["safety"]["image_count_stability_enforced"] is True
    assert result.report["safety"]["image_count_stability_ok"] is False
    assert any(event["event"] == "expected_image_count_decreased" for event in result.report["safety"]["events"])


def test_alignment_loop_rebinds_orphan_images_to_paper_identity_question_id(tmp_path: Path) -> None:
    output = tmp_path / "output"
    for index in [1, 2]:
        _write_image(output / _q_path(index))
        _write_image(output / _ms_path(index))
    bank = _question_bank(
        [
            _question(1),
            _question(2, question_path="", mark_scheme_path=""),
        ]
    )

    result = run_image_alignment_loop(
        lambda _context: AlignmentExtractionOutput(bank, artifact_root=output),
        artifact_root=output,
        report_path=tmp_path / "report.json",
        binding_report_path=tmp_path / "binding_report.json",
        max_iterations=3,
    )

    assert result.reached_threshold is True
    assert result.report["orphan_images"] == 0
    assert result.binding_report["initial_orphan_count"] == 2
    assert result.binding_report["final_orphan_count"] == 0
    assert result.binding_report["rebinding_success_count"] == 2
    assert result.binding_report["primary_resolution_method_used"] == "textual_anchor_proximity"


def test_alignment_loop_fails_when_orphans_remain_after_final_pass(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_image(output / _q_path(1))
    _write_image(output / "pm1" / "unmapped_diagram.png")
    bank = _question_bank([_question(1, mark_scheme_path="")])

    result = run_image_alignment_loop(
        lambda _context: AlignmentExtractionOutput(bank, artifact_root=output),
        artifact_root=output,
        report_path=tmp_path / "report.json",
        binding_report_path=tmp_path / "binding_report.json",
        max_iterations=2,
        expected_image_kinds=("question_image",),
    )

    assert result.final_alignment_score == 1.0
    assert result.reached_threshold is False
    assert result.report["alignment_status"] == "FAIL"
    assert result.binding_report["final_orphan_count"] == 1
    assert result.binding_report["retry"]["retry_queue_final_count"] == 1


def test_resolve_image_to_question_is_deterministic_by_bbox_then_identity(tmp_path: Path) -> None:
    bank = _question_bank(
        [
            _question(2) | {"question_bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 100}, "page_number": 1},
            _question(3) | {"question_bbox": {"x0": 0, "y0": 150, "x1": 100, "y1": 250}, "page_number": 1},
        ]
    )
    context = ImageBindingContext(question_bank=bank, artifact_root=tmp_path)
    image = {
        "path": _q_path(3),
        "paper_id": "12summer21",
        "question_number": "3",
        "bbox": {"x0": 10, "y0": 160, "x1": 80, "y1": 220},
        "page_number": 1,
    }

    assert [resolve_image_to_question(image, context) for _ in range(5)] == ["12summer21_q03"] * 5


def test_orphan_rebinding_does_not_create_duplicate_slot_bindings(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write_image(output / _q_path(1))
    _write_image(output / "pm1" / "pm1_2021_m21_12_qp_q01_question_v2.png")
    bank = _question_bank([_question(1)])

    result = run_image_alignment_loop(
        lambda _context: AlignmentExtractionOutput(bank, artifact_root=output),
        artifact_root=output,
        report_path=tmp_path / "report.json",
        binding_report_path=tmp_path / "binding_report.json",
        max_iterations=2,
        expected_image_kinds=("question_image",),
    )

    assert result.reached_threshold is False
    assert result.binding_report["rebinding_success_count"] == 0
    assert result.binding_report["failure_reasons_distribution"]["matching slot is protected as previously correct"] >= 1


def _question_bank(questions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_name": "exam_bank.question_bank",
        "schema_version": 2,
        "record_count": len(questions),
        "questions": questions,
    }


def _question(
    number: int,
    *,
    question_path: str | None = None,
    mark_scheme_path: str | None = None,
    question_crop_confidence: str = "high",
    mark_scheme_crop_confidence: str = "high",
    mapping_status: str = "pass",
) -> dict[str, object]:
    question_id = f"12summer21_q{number:02d}"
    q_path = _q_path(number) if question_path is None else question_path
    ms_path = _ms_path(number) if mark_scheme_path is None else mark_scheme_path
    return {
        "question_id": question_id,
        "paper": "12summer21",
        "canonical_paper_id": "12summer21",
        "paper_family": "pm1",
        "question_number": str(number),
        "canonical_question_artifact": q_path,
        "canonical_mark_scheme_artifact": ms_path,
        "question_image_path": q_path,
        "mark_scheme_image_path": ms_path,
        "question_image_paths": [q_path] if q_path else [],
        "mark_scheme_image_paths": [ms_path] if ms_path else [],
        "notes": {
            "question_crop_confidence": question_crop_confidence,
            "mark_scheme_crop_confidence": mark_scheme_crop_confidence,
            "mapping_status": mapping_status,
            "validation_status": "pass",
        },
    }


def _q_path(number: int) -> str:
    return f"pm1/pm1_2021_m21_12_qp_q{number:02d}_question.png"


def _ms_path(number: int) -> str:
    return f"pm1/pm1_2021_m21_12_ms_q{number:02d}_markscheme.png"


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
