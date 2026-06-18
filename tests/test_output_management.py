import json
from pathlib import Path

from exam_bank.cli import main
from exam_bank.output_management import build_cleanup_plan, build_output_inventory


def test_output_inventory_finds_contract_paths(tmp_path: Path, capsys) -> None:
    root = tmp_path / "output"
    _write(root / "json" / "question_bank.json", {"questions": []})
    _write(root / "pm1" / "pm1_2021_m21_qp_q01_question.png", "png")
    _write(root / "triage" / "iteration_001" / "baseline_question_bank.json", {"questions": []})
    _write(root / "triage" / "iteration_001" / "comparisons" / "comparison.auto-iteration-001.json", {"ok": True})
    _write(
        root / "asterion" / "exports" / "latest" / "asterion_question_bank_v1.json",
        {"generated_at": "2026-05-27T15:20:00+00:00", "questions": []},
    )
    _write(root / "audits" / "iteration_001" / "audit_summary.json", {"ok": True})
    _write(root / "run_status" / "run-1" / "run_status.json", {"run_id": "run-1"})

    report = build_output_inventory([root], include_size=True)

    assert report["output_roots_found"] == [str(root)]
    assert report["run_ids_found"] == ["run-1"]
    assert [item["path"] for item in report["question_bank_files"]] == [str(root / "json" / "question_bank.json")]
    assert [item["path"] for item in report["artifact_trees"]] == [str(root / "pm1")]
    assert [item["path"] for item in report["frozen_baselines"]] == [
        str(root / "triage" / "iteration_001" / "baseline_question_bank.json")
    ]
    assert [item["path"] for item in report["auto_triage_comparisons"]] == [
        str(root / "triage" / "iteration_001" / "comparisons" / "comparison.auto-iteration-001.json")
    ]
    assert report["asterion_outputs"][0]["last_run_at"] == "2026-05-27T15:20:00+00:00"

    output_json = root / "output_inventory.json"
    assert main(["output-inventory", "--root", str(root), "--json", str(output_json)]) == 0
    captured = capsys.readouterr()
    assert "Generated Output Inventory" in captured.out
    assert "last run" in captured.out
    assert json.loads(output_json.read_text(encoding="utf-8"))["schema_name"] == "exam_bank.output_inventory"


def test_cleanup_plan_is_dry_run_and_preserves_frozen_baselines(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write(root / "json" / "question_bank.json", {"questions": []})
    _write(root / "triage" / "iteration_001" / "baseline_question_bank.json", {"questions": []})
    _write(root / "triage" / "iteration_001" / "comparisons" / "comparison.auto-iteration-001.json", {"ok": True})
    _write(root / "candidates" / "ocr" / "old" / "json" / "question_bank.json", {"questions": []})
    _write(root / "candidates" / "ocr" / "latest" / "json" / "question_bank.json", {"questions": []})

    plan = build_cleanup_plan([root])
    by_path = {entry["path"]: entry["action"] for entry in plan["entries"]}

    assert plan["dry_run"] is True
    assert by_path[str(root / "json" / "question_bank.json")] == "keep: canonical/current"
    assert by_path[str(root / "triage" / "iteration_001" / "baseline_question_bank.json")] == "keep: frozen baseline"
    assert by_path[str(root / "triage" / "iteration_001" / "comparisons" / "comparison.auto-iteration-001.json")] == (
        "safe delete generated report"
    )

    output_md = root / "output_cleanup_plan.md"
    assert main(["output-cleanup-plan", "--root", str(root), "--write", str(output_md)]) == 0
    assert "Dry run only" in output_md.read_text(encoding="utf-8")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
