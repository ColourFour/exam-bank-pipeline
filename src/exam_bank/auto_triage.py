from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .output_layout import default_triage_comparison_path
from .triage import ISSUE_SET_HARD_FAILURES, compare_iteration, issue_counts, is_hard_failure, load_question_bank


AUTO_TRIAGE_PROMPTS: dict[str, str] = {
    "supervisor_prompt.md": """# Auto-Triage Supervisor Prompt

You are supervising an evidence-gated extraction-quality iteration for the exam-bank pipeline.

Non-negotiable constraints:
- The supported runtime is the extraction pipeline.
- Output is image-first. Question and mark-scheme crops are the source of truth.
- Extracted text is metadata and remains trust-gated.
- `question_bank.json` is metadata and must not be treated as stronger evidence than image crops.
- DeepSeek and topic labels are sidecar metadata only.
- Use OCR-enabled exports for canonical production comparisons.
- No-OCR comparisons may support debugging, but must not be reported as production improvement.
- Do not delete, rewrite, or regenerate old `output/triage/iteration_*` baselines.
- Do not broadly loosen validation, scope, text-fidelity, topic-trust, or visual-curation gates.
- Every accepted iteration must report metrics before and after, the comparison path, and exact test results.

Workflow:
1. Confirm the current iteration folder contains `metrics_before.json` and `selected_target.json`.
2. Keep the implementation scope to the selected dominant failure cluster.
3. Route work through Agent 1 through Agent 5 in order.
4. Stop if the target hard-failure threshold has already been met.
5. Reject or pause if evidence is mixed, OCR mode is wrong, tests fail, or flags appear suppressed without extraction evidence.
""",
    "agent1_planner_prompt.md": """# Agent 1 Planner Prompt

Read `metrics_before.json`, `selected_target.json`, and the triage sample for this iteration.

Produce a narrow implementation plan:
- Identify the selected issue cluster and why it is the target for this pass.
- Inspect sampled question and mark-scheme image crops before trusting metadata.
- Define likely files to change.
- Define tests to add or update from reviewed examples.
- Define exact acceptance criteria and stop criteria.
- Define what must not change.

Constraints:
- Tests-first where a focused regression test is practical.
- Scope one dominant issue per iteration.
- Do not loosen validation or trust gates broadly.
- Do not delete or mutate triage baselines.
- OCR-enabled comparison is required for production scoring.
- Extracted text remains metadata.
- DeepSeek/topic labels remain sidecar only.
""",
    "agent2_builder_prompt.md": """# Agent 2 Builder Prompt

Implement only Agent 1's scoped plan.

Rules:
- Add or update regression tests for the reviewed examples.
- Keep changes limited to the files named by Agent 1 unless new evidence requires a small documented expansion.
- Avoid unrelated cleanup and formatting churn.
- Preserve image-first behavior and trust gating.
- Do not make failures disappear by suppressing validation, review, text-fidelity, or topic-trust flags.
- Do not modify or delete frozen `output/triage` baselines.

Report:
- Files changed.
- Tests added or updated.
- Any deviation from Agent 1's plan and the evidence for it.
""",
    "agent3_tests_prompt.md": """# Agent 3 Test Gatekeeper Prompt

Verify the implementation before any quality claim is made.

Required:
- Run focused regression tests for this iteration.
- Run the full test suite: `.venv/bin/python -m pytest`.
- Report exact commands and exact pass/fail results.
- Block the iteration if tests fail.

Constraints:
- Do not accept partial test evidence as a pass.
- Do not edit code unless the user explicitly assigns you a test-fix role.
- Do not delete or mutate triage baselines.
""",
    "agent4_integration_prompt.md": """# Agent 4 Integration Prompt

Verify production-style evidence after tests pass.

Required:
- Run or verify a full OCR-enabled export.
- Verify the current output is OCR-enabled with `auto-triage-status`.
- Run the auto-triage comparison against the frozen OCR-enabled baseline.
- Check `metrics_after.json`, `decision.json`, and the comparison report.
- Inspect `worsened_records` and any status regressions.

Constraints:
- OCR/no-OCR comparisons must not be mixed for production scoring.
- Image crops remain the source of truth.
- Extracted text and topic labels remain metadata.
- Do not replace canonical output until comparison evidence is understood.
""",
    "agent5_review_prompt.md": """# Agent 5 Adversarial Review Prompt

Review the completed iteration as if it might be a false improvement.

Look for:
- Flag suppression instead of extraction fixes.
- Broad validation or trust-gate loosening.
- OCR/no-OCR baseline mistakes.
- Overfitting to sampled records.
- Regressions hidden outside the target issue.
- Missing regression tests for reviewed examples.
- Deleted or altered triage baselines.

Final verdict must be one of:
- PASS
- PASS WITH RISKS
- BLOCKED

Include metrics before/after, the comparison path, `worsened_records` summary, and exact full-test result.
""",
}

PROMPT_DIR_NAME = "Prompt"
ITERATION_RE = re.compile(r"^iteration_(\d{3})$")

STATUS_COUNT_FIELDS = {
    "text_source_profile": ("text_source_profile",),
    "validation_status": ("validation_status",),
    "markscheme_mapping_status": ("markscheme_mapping_status", "mapping_status"),
    "text_fidelity_status": ("text_fidelity_status",),
    "scope_quality_status": ("scope_quality_status",),
    "topic_trust_status": ("topic_trust_status",),
    "visual_curation_status": ("visual_curation_status",),
}

FLAG_COUNT_FIELDS = {
    "validation_flag_counts": "validation_flags",
    "review_flag_counts": "review_flags",
    "text_fidelity_flag_counts": "text_fidelity_flags",
}

BAD_STATUS_VALUES = {
    "validation_status_counts": {"fail", "review"},
    "markscheme_mapping_status_counts": {"fail", "review"},
    "scope_quality_status_counts": {"fail", "review"},
    "text_fidelity_status_counts": {"unusable", "degraded"},
    "topic_trust_status_counts": {"review_required", "degraded_text"},
    "visual_curation_status_counts": {"fail", "review"},
}


def build_status_report(input_path: str | Path, *, top_limit: int = 20) -> dict[str, Any]:
    payload, records = load_question_bank(input_path)
    hard_failure_count = sum(1 for record in records if is_hard_failure(record))
    clusters = issue_counts(records, issue_set=ISSUE_SET_HARD_FAILURES)
    top_issue_clusters = [
        {"issue": issue, "count": count} for issue, count in list(clusters.items())[:top_limit]
    ]
    dominant = top_issue_clusters[0] if top_issue_clusters else {"issue": "none", "count": 0}

    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "input_path": str(input_path),
        "record_count": len(records),
        "declared_record_count": payload.get("record_count") if isinstance(payload, dict) else None,
        "ocr_counts": _ocr_counts(records),
        "ocr_profile": _ocr_profile(records),
        "hard_failure_count": hard_failure_count,
        "dominant_failure_cluster": dominant,
        "top_issue_clusters": top_issue_clusters,
        "issue_counts": clusters,
    }

    for output_key, candidates in STATUS_COUNT_FIELDS.items():
        report[f"{output_key}_counts"] = _status_counts(records, candidates)
    for output_key, field_name in FLAG_COUNT_FIELDS.items():
        report[output_key] = _flag_counts(records, field_name)

    return report


def write_status_report(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    report = build_status_report(input_path)
    if output_path is not None:
        _write_json(output_path, report)
    return report


def create_auto_triage_plan(
    input_path: str | Path,
    *,
    handoff_root: str | Path = "agent_handoffs/auto_triage",
    target_max_hard_failures: int,
    sample_size: int = 30,
    seed: int = 1,
    triage_root: str | Path | None = None,
    candidate_output: str | Path = "output_ocr_candidate",
) -> dict[str, Any]:
    if target_max_hard_failures < 0:
        raise ValueError("target_max_hard_failures must be non-negative.")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    input_path = Path(input_path)
    handoff_root = Path(handoff_root)
    output_root = _infer_output_root(input_path)
    triage_root = Path(triage_root) if triage_root is not None else output_root / "triage"
    ensure_auto_triage_prompts(handoff_root)

    metrics = build_status_report(input_path)
    hard_failure_count = int(metrics["hard_failure_count"])
    if hard_failure_count <= target_max_hard_failures:
        return {
            "stopped": True,
            "reason": "target_threshold_met",
            "current_hard_failure_count": hard_failure_count,
            "target_max_hard_failures": target_max_hard_failures,
            "handoff_root": str(handoff_root),
            "metrics": metrics,
        }

    iteration_name = _next_iteration_name(handoff_root, triage_root)
    iteration_dir = handoff_root / iteration_name
    iteration_dir.mkdir(parents=True)

    dominant = metrics["dominant_failure_cluster"]
    selected_target = {
        "issue": dominant["issue"],
        "count": dominant["count"],
        "issue_set": ISSUE_SET_HARD_FAILURES,
        "sample_size": sample_size,
        "seed": seed,
        "current_hard_failure_count": hard_failure_count,
        "target_max_hard_failures": target_max_hard_failures,
        "minimum_required_improvement": (
            "hard failures decrease, or selected target issue decreases by at least 1 without unrelated regressions"
        ),
    }
    commands = build_runbook_commands(
        input_path=input_path,
        handoff_iteration=iteration_dir,
        triage_iteration=triage_root / iteration_name,
        target=str(dominant["issue"]),
        sample_size=sample_size,
        seed=seed,
        candidate_output=candidate_output,
    )

    metrics_before_path = iteration_dir / "metrics_before.json"
    selected_target_path = iteration_dir / "selected_target.json"
    commands_path = iteration_dir / "commands.json"
    agent1_path = iteration_dir / "agent1_request.md"

    _write_json(metrics_before_path, metrics)
    _write_json(selected_target_path, selected_target)
    _write_json(commands_path, commands)
    agent1_path.write_text(
        _agent1_request_markdown(
            iteration_name=iteration_name,
            metrics=metrics,
            selected_target=selected_target,
            commands=commands,
        ),
        encoding="utf-8",
    )

    return {
        "stopped": False,
        "iteration": iteration_name,
        "iteration_dir": str(iteration_dir),
        "prompt_dir": str(handoff_root / PROMPT_DIR_NAME),
        "metrics_before_path": str(metrics_before_path),
        "selected_target_path": str(selected_target_path),
        "commands_path": str(commands_path),
        "agent1_request_path": str(agent1_path),
        "selected_target": selected_target,
        "top_issue_clusters": metrics["top_issue_clusters"],
        "commands": commands,
    }


def ensure_auto_triage_prompts(handoff_root: str | Path) -> dict[str, Any]:
    prompt_dir = Path(handoff_root) / PROMPT_DIR_NAME
    prompt_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    existing: list[str] = []
    for filename, body in AUTO_TRIAGE_PROMPTS.items():
        path = prompt_dir / filename
        if path.exists():
            existing.append(str(path))
            continue
        path.write_text(body, encoding="utf-8")
        written.append(str(path))
    return {"prompt_dir": str(prompt_dir), "written": written, "existing": existing}


def build_runbook_commands(
    *,
    input_path: str | Path = "output/json/question_bank.json",
    handoff_iteration: str | Path,
    triage_iteration: str | Path,
    target: str,
    sample_size: int = 30,
    seed: int = 1,
    candidate_output: str | Path = "output_ocr_candidate",
) -> dict[str, str]:
    input_path = Path(input_path)
    handoff_iteration = Path(handoff_iteration)
    triage_iteration = Path(triage_iteration)
    candidate_output = Path(candidate_output)
    iteration_name = triage_iteration.name
    comparison_name = f"comparison.auto-{iteration_name.replace('_', '-')}.json"
    comparison_output = default_triage_comparison_path(triage_iteration, comparison_name)
    current_candidate = candidate_output / "json" / "question_bank.json"

    return {
        "triage_sample": (
            ".venv/bin/python -m exam_bank.cli triage-sample "
            f"--input {input_path} --output-root {triage_iteration.parent} --iteration {iteration_name} "
            f"--issue-set {ISSUE_SET_HARD_FAILURES} --target {target} --sample-size {sample_size} --seed {seed}"
        ),
        "triage_serve": f".venv/bin/python -m exam_bank.cli triage-serve --iteration {triage_iteration}",
        "full_ocr_rerun": (
            f".venv/bin/python -m exam_bank.cli process --input input --output {candidate_output} --enable-ocr"
        ),
        "ocr_verification": f".venv/bin/python -m exam_bank.cli auto-triage-status --input {current_candidate}",
        "full_tests": ".venv/bin/python -m pytest",
        "triage_comparison": (
            ".venv/bin/python -m exam_bank.cli auto-triage-compare "
            f"--iteration {handoff_iteration} --baseline-triage {triage_iteration} "
            f"--current {current_candidate} --output {comparison_output} --test-status pass"
        ),
    }


def build_auto_triage_runbook(
    *,
    input_path: str | Path = "output/json/question_bank.json",
    handoff_root: str | Path = "agent_handoffs/auto_triage",
    iteration: str | Path | None = None,
    baseline_triage: str | Path | None = None,
    candidate_output: str | Path = "output_ocr_candidate",
    sample_size: int = 30,
    seed: int = 1,
) -> dict[str, Any]:
    handoff_iteration = _resolve_handoff_iteration(handoff_root, iteration)
    selected_target = _load_selected_target(handoff_iteration)
    target = str(selected_target.get("issue") or "auto")
    sample_size = int(selected_target.get("sample_size") or sample_size)
    seed = int(selected_target.get("seed") or seed)
    triage_iteration = Path(baseline_triage) if baseline_triage is not None else _infer_triage_iteration(
        input_path, handoff_iteration.name
    )
    commands = build_runbook_commands(
        input_path=input_path,
        handoff_iteration=handoff_iteration,
        triage_iteration=triage_iteration,
        target=target,
        sample_size=sample_size,
        seed=seed,
        candidate_output=candidate_output,
    )
    return {
        "iteration": str(handoff_iteration),
        "baseline_triage": str(triage_iteration),
        "selected_target": selected_target,
        "commands": commands,
    }


def compare_auto_triage_iteration(
    *,
    iteration: str | Path,
    baseline_triage: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
    test_status: str = "unknown",
    max_worsened_records: int = 0,
    target_material_decrease: int = 1,
    max_hard_failure_increase: int = 0,
    max_status_regression: int = 10,
) -> dict[str, Any]:
    iteration_dir = Path(iteration)
    if not iteration_dir.exists():
        raise FileNotFoundError(f"Auto-triage iteration does not exist: {iteration_dir}")
    if max_worsened_records < 0:
        raise ValueError("max_worsened_records must be non-negative.")
    if target_material_decrease < 1:
        raise ValueError("target_material_decrease must be at least 1.")

    output = Path(output_path) if output_path is not None else None
    comparison = compare_iteration(baseline_triage, current_path=current_path, output_path=output)
    baseline_metrics = build_status_report(_baseline_question_bank_path(baseline_triage))
    metrics_after = build_status_report(current_path)
    selected_target = _load_selected_target(iteration_dir)
    previous_decision = _load_previous_decision(iteration_dir)
    decision = decide_auto_triage_iteration(
        comparison=comparison,
        metrics_before=baseline_metrics,
        metrics_after=metrics_after,
        selected_target=selected_target,
        test_status=test_status,
        max_worsened_records=max_worsened_records,
        target_material_decrease=target_material_decrease,
        max_hard_failure_increase=max_hard_failure_increase,
        max_status_regression=max_status_regression,
        previous_decision=previous_decision,
    )
    decision.update(
        {
            "generated_at": _utc_now(),
            "iteration_dir": str(iteration_dir),
            "baseline_triage": str(baseline_triage),
            "current_path": str(current_path),
            "comparison_output": str(output) if output is not None else "",
            "metrics_after_path": str(iteration_dir / "metrics_after.json"),
            "decision_path": str(iteration_dir / "decision.json"),
        }
    )

    _write_json(iteration_dir / "metrics_after.json", metrics_after)
    _write_json(iteration_dir / "decision.json", decision)
    return {"comparison": comparison, "metrics_after": metrics_after, "decision": decision}


def decide_auto_triage_iteration(
    *,
    comparison: dict[str, Any],
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    selected_target: dict[str, Any] | None = None,
    test_status: str = "unknown",
    max_worsened_records: int = 0,
    target_material_decrease: int = 1,
    max_hard_failure_increase: int = 0,
    max_status_regression: int = 10,
    previous_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = str((selected_target or {}).get("issue") or comparison.get("target") or "")
    hard_failure_delta = int(comparison.get("hard_failure_delta") or 0)
    target_issue_delta = int(comparison.get("target_issue_delta") or 0)
    worsened_records = comparison.get("worsened_records") if isinstance(comparison.get("worsened_records"), list) else []
    hard_failures_decreased = hard_failure_delta < 0
    target_issue_decreased = target_issue_delta <= -target_material_decrease
    material_improvement = hard_failures_decreased or target_issue_decreased
    current_ocr_enabled = bool(metrics_after.get("ocr_profile", {}).get("ocr_enabled"))
    baseline_ocr_enabled = bool(metrics_before.get("ocr_profile", {}).get("ocr_enabled"))
    normalized_test_status = _normalize_test_status(test_status)

    rejected_reasons: list[str] = []
    review_reasons: list[str] = []

    if normalized_test_status == "fail":
        rejected_reasons.append("tests_failed")
    elif normalized_test_status != "pass":
        review_reasons.append("tests_not_confirmed")

    if not current_ocr_enabled:
        rejected_reasons.append("current_output_not_ocr_enabled")
    if not baseline_ocr_enabled:
        review_reasons.append("baseline_not_ocr_enabled")

    if hard_failure_delta > max_hard_failure_increase:
        rejected_reasons.append(
            f"hard_failures_increased:{hard_failure_delta}>allowed:{max_hard_failure_increase}"
        )

    if len(worsened_records) > max_worsened_records:
        rejected_reasons.append(
            f"worsened_records:{len(worsened_records)}>allowed:{max_worsened_records}"
        )

    if not material_improvement:
        if _same_target_failed_to_improve(previous_decision, target):
            rejected_reasons.append("selected_target_not_improved_for_two_consecutive_iterations")
        else:
            review_reasons.append("no_material_improvement")

    status_regressions = _status_regressions(metrics_before, metrics_after, max_status_regression)
    if status_regressions:
        rejected_reasons.append("major_status_regression")

    if _flag_suppression_without_evidence(metrics_before, metrics_after, material_improvement):
        rejected_reasons.append("validation_flags_suppressed_without_evidence")
    if _trust_gate_loosening_without_evidence(metrics_before, metrics_after, material_improvement):
        rejected_reasons.append("trust_gates_loosened_without_evidence")

    decision = "accepted"
    if rejected_reasons:
        decision = "rejected"
    elif review_reasons:
        decision = "needs-human-review"

    return {
        "decision": decision,
        "accepted": decision == "accepted",
        "rejected_reasons": rejected_reasons,
        "review_reasons": review_reasons,
        "selected_target": target,
        "test_status": normalized_test_status,
        "current_ocr_enabled": current_ocr_enabled,
        "baseline_ocr_enabled": baseline_ocr_enabled,
        "hard_failures_decreased": hard_failures_decreased,
        "target_issue_decreased": target_issue_decreased,
        "material_improvement": material_improvement,
        "hard_failure_delta": hard_failure_delta,
        "target_issue_delta": target_issue_delta,
        "worsened_record_count": len(worsened_records),
        "max_worsened_records": max_worsened_records,
        "status_regressions": status_regressions,
        "acceptance_criteria": [
            "full tests pass",
            "current and baseline outputs are OCR-enabled for canonical scoring",
            "hard failures decrease or selected target issue decreases materially",
            "worsened_records stays under threshold",
            "no major unrelated status regression",
            "no broad validation/trust-gate loosening without evidence",
        ],
    }


def _agent1_request_markdown(
    *,
    iteration_name: str,
    metrics: dict[str, Any],
    selected_target: dict[str, Any],
    commands: dict[str, str],
) -> str:
    top_issues = "\n".join(
        f"- `{item['issue']}`: {item['count']}" for item in metrics.get("top_issue_clusters", [])[:10]
    )
    commands_md = "\n".join(f"- `{name}`:\n\n```bash\n{command}\n```" for name, command in commands.items())
    return f"""# Agent 1 Request - {iteration_name}

Plan one narrow extraction-quality improvement pass.

Selected target:
- Issue: `{selected_target['issue']}`
- Current count: `{selected_target['count']}`
- Current hard failures: `{selected_target['current_hard_failure_count']}`
- Stop threshold: `{selected_target['target_max_hard_failures']}`

Top issue counts:
{top_issues}

Stop criteria:
- Stop immediately if current hard failures are at or below the configured threshold.
- Stop if the selected target is not actionable from the visual sample.
- Stop if the work would require broad validation or trust-gate loosening.

Acceptance criteria:
- Focused regression tests are added or updated for reviewed examples where practical.
- Full `.venv/bin/python -m pytest` passes.
- Canonical comparison uses an OCR-enabled current output against an OCR-enabled baseline.
- Hard failures decrease, or `{selected_target['issue']}` decreases by at least one.
- `worsened_records` stays under the configured threshold.
- No broad status regression or flag suppression without extraction evidence.

What not to change:
- Do not delete or rewrite existing `output/triage` baselines.
- Do not make `question_bank.json` the source of truth over image crops.
- Do not treat extracted text, DeepSeek labels, or topic labels as canonical evidence.
- Do not do unrelated cleanup.

Commands:
{commands_md}
"""


def _status_counts(records: list[dict[str, Any]], field_candidates: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = _first_present_field(record, field_candidates)
        counts[_count_key(value)] += 1
    return dict(counts.most_common())


def _flag_counts(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_list_field(record, field_name))
    return dict(counts.most_common())


def _ocr_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("ocr_ran") is True:
            counts["true"] += 1
        elif record.get("ocr_ran") is False:
            counts["false"] += 1
        else:
            counts["missing"] += 1
    return dict(counts)


def _ocr_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _ocr_counts(records)
    true_count = counts.get("true", 0)
    false_count = counts.get("false", 0)
    missing_count = counts.get("missing", 0)
    return {
        "ocr_enabled": bool(records) and true_count == len(records),
        "ocr_disabled": bool(records) and false_count == len(records),
        "ocr_mixed": true_count > 0 and (false_count > 0 or missing_count > 0),
        "ocr_missing": missing_count,
    }


def _first_present_field(record: dict[str, Any], field_candidates: tuple[str, ...]) -> Any:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    for field_name in field_candidates:
        if field_name in record and record.get(field_name) not in (None, ""):
            return record.get(field_name)
        if field_name in notes and notes.get(field_name) not in (None, ""):
            return notes.get(field_name)
    return None


def _field(record: dict[str, Any], field_name: str) -> Any:
    return _first_present_field(record, (field_name,))


def _list_field(record: dict[str, Any], field_name: str) -> list[str]:
    value = _field(record, field_name)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _count_key(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _next_iteration_name(*roots: str | Path) -> str:
    max_seen = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            match = ITERATION_RE.match(child.name)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
    return f"iteration_{max_seen + 1:03d}"


def _resolve_handoff_iteration(handoff_root: str | Path, iteration: str | Path | None) -> Path:
    if iteration is not None and str(iteration):
        path = Path(iteration)
        if path.exists() or path.is_absolute() or path.parent != Path("."):
            return path
        return Path(handoff_root) / path
    root = Path(handoff_root)
    existing = sorted(
        (child for child in root.glob("iteration_[0-9][0-9][0-9]") if child.is_dir()),
        key=lambda child: child.name,
    )
    if not existing:
        raise FileNotFoundError(f"No auto-triage iterations found under {root}")
    return existing[-1]


def _infer_triage_iteration(input_path: str | Path, iteration_name: str) -> Path:
    return _infer_output_root(Path(input_path)) / "triage" / iteration_name


def _infer_output_root(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if input_path.parent.name == "json":
        return input_path.parent.parent
    return input_path.parent


def _baseline_question_bank_path(iteration_dir: str | Path) -> Path:
    iteration_dir = Path(iteration_dir)
    summary_path = iteration_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline_path = summary.get("baseline_path")
        if baseline_path:
            return Path(baseline_path)
    return iteration_dir / "baseline_question_bank.json"


def _load_selected_target(iteration_dir: str | Path) -> dict[str, Any]:
    path = Path(iteration_dir) / "selected_target.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_previous_decision(iteration_dir: Path) -> dict[str, Any] | None:
    match = ITERATION_RE.match(iteration_dir.name)
    if not match:
        return None
    previous_number = int(match.group(1)) - 1
    if previous_number < 1:
        return None
    path = iteration_dir.parent / f"iteration_{previous_number:03d}" / "decision.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _same_target_failed_to_improve(previous_decision: dict[str, Any] | None, target: str) -> bool:
    if not previous_decision:
        return False
    return (
        str(previous_decision.get("selected_target") or "") == target
        and previous_decision.get("material_improvement") is False
    )


def _status_regressions(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    max_status_regression: int,
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for counts_key, bad_values in BAD_STATUS_VALUES.items():
        before_counts = metrics_before.get(counts_key) if isinstance(metrics_before.get(counts_key), dict) else {}
        after_counts = metrics_after.get(counts_key) if isinstance(metrics_after.get(counts_key), dict) else {}
        before_bad_total = sum(int(before_counts.get(status, 0)) for status in bad_values)
        after_bad_total = sum(int(after_counts.get(status, 0)) for status in bad_values)
        bad_total_delta = after_bad_total - before_bad_total
        if bad_total_delta <= max_status_regression:
            continue
        for status in sorted(bad_values):
            delta = int(after_counts.get(status, 0)) - int(before_counts.get(status, 0))
            if delta > max_status_regression:
                regressions.append(
                    {
                        "field": counts_key,
                        "status": status,
                        "delta": delta,
                        "bad_status_delta": bad_total_delta,
                    }
                )
    return regressions


def _flag_suppression_without_evidence(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    material_improvement: bool,
) -> bool:
    if material_improvement:
        return False
    before_flags = _flag_total(metrics_before.get("validation_flag_counts"))
    after_flags = _flag_total(metrics_after.get("validation_flag_counts"))
    return after_flags < before_flags


def _trust_gate_loosening_without_evidence(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    material_improvement: bool,
) -> bool:
    if material_improvement:
        return False
    return _bad_status_total(metrics_after) < _bad_status_total(metrics_before)


def _bad_status_total(metrics: dict[str, Any]) -> int:
    total = 0
    for counts_key, bad_values in BAD_STATUS_VALUES.items():
        counts = metrics.get(counts_key)
        if not isinstance(counts, dict):
            continue
        total += sum(int(counts.get(status, 0)) for status in bad_values)
    return total


def _flag_total(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(int(count) for count in value.values())


def _normalize_test_status(value: str) -> str:
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"passed", "pass", "ok", "success", "true"}:
        return "pass"
    if normalized in {"failed", "fail", "error", "false"}:
        return "fail"
    return "unknown"


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
