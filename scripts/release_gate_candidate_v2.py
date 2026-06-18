from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exam_bank.asterion_export import build_asterion_export, build_content_lab_candidates
from exam_bank.audit import audit_current_output_integrity


DEFAULT_CANONICAL = "output/json/question_bank.json"
DEFAULT_CANDIDATE = "output/codex_text_extraction_candidate_v2/json/question_bank.json"
DEFAULT_CANONICAL_READINESS = "output/audits/codex_text_extraction_baseline/audit_summary.json"
DEFAULT_CANDIDATE_READINESS = "output/audits/codex_text_extraction_candidate_v2/audit_summary.json"
DEFAULT_CANDIDATE_BASELINE = "output/audits/codex_text_extraction_candidate_v2/baseline_comparison_summary.json"
DEFAULT_TOPIC_SIDECAR = "output/json/question_bank.topic_routing.v1.json"
DEFAULT_OUT_DIR = "output/audits/codex_text_extraction_candidate_v2_release_gate"


DECISION_PROMOTE = "PROMOTE_CANDIDATE_V2"
DECISION_DO_NOT_PROMOTE = "DO_NOT_PROMOTE"
DECISION_REPAIR = "NEEDS_TARGETED_REPAIR_BEFORE_PROMOTION"


ROLE_KEYS = (
    "canonical_practice",
    "quick_check_source",
    "warmup_generator_source",
    "guardian_candidate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate canonical vs candidate-v2 release-gate evidence without promoting the candidate.",
    )
    parser.add_argument("--canonical", default=DEFAULT_CANONICAL, help="Current canonical question_bank.json.")
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE, help="Candidate question_bank.json.")
    parser.add_argument("--artifact-root", default="output", help="Canonical/promotion artifact root for image integrity checks.")
    parser.add_argument(
        "--candidate-artifact-root",
        default="",
        help="Candidate source artifact root. Defaults to the candidate question bank's parent output root.",
    )
    parser.add_argument("--canonical-readiness", default=DEFAULT_CANONICAL_READINESS, help="Canonical readiness audit summary.")
    parser.add_argument("--candidate-readiness", default=DEFAULT_CANDIDATE_READINESS, help="Candidate readiness audit summary.")
    parser.add_argument("--candidate-baseline", default=DEFAULT_CANDIDATE_BASELINE, help="Candidate baseline comparison summary.")
    parser.add_argument("--topic-sidecar", default=DEFAULT_TOPIC_SIDECAR, help="Topic routing sidecar to assess strict-filter safety.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for release-gate JSON and Markdown reports.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_release_gate_report(
        canonical_path=Path(args.canonical),
        candidate_path=Path(args.candidate),
        artifact_root=Path(args.artifact_root),
        candidate_artifact_root=Path(args.candidate_artifact_root) if args.candidate_artifact_root else None,
        canonical_readiness_path=Path(args.canonical_readiness),
        candidate_readiness_path=Path(args.candidate_readiness),
        candidate_baseline_path=Path(args.candidate_baseline),
        topic_sidecar_path=Path(args.topic_sidecar),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "release_gate_candidate_v2.json", report)
    (out_dir / "release_gate_candidate_v2.md").write_text(render_markdown(report), encoding="utf-8")

    print(f"Decision: {report['decision']}")
    print(f"Wrote {out_dir / 'release_gate_candidate_v2.md'}")
    print(f"Wrote {out_dir / 'release_gate_candidate_v2.json'}")
    return 0


def build_release_gate_report(
    *,
    canonical_path: Path,
    candidate_path: Path,
    artifact_root: Path,
    candidate_artifact_root: Path | None,
    canonical_readiness_path: Path,
    candidate_readiness_path: Path,
    candidate_baseline_path: Path,
    topic_sidecar_path: Path,
) -> dict[str, Any]:
    canonical_payload = load_json(canonical_path)
    candidate_payload = load_json(candidate_path)
    canonical_records = records_from_payload(canonical_payload)
    candidate_records = records_from_payload(candidate_payload)
    candidate_source_root = candidate_artifact_root or infer_output_root(candidate_path)

    canonical_readiness = load_json(canonical_readiness_path)
    candidate_readiness = load_json(candidate_readiness_path)
    candidate_baseline = load_json(candidate_baseline_path)
    topic_sidecar = load_json(topic_sidecar_path)

    canonical_integrity = audit_current_output_integrity(canonical_path, artifact_root=artifact_root)
    candidate_source_integrity = audit_current_output_integrity(candidate_path, artifact_root=candidate_source_root)
    candidate_promotion_integrity = audit_current_output_integrity(candidate_path, artifact_root=artifact_root)
    canonical_asterion = build_asterion_export(canonical_payload, artifact_root=artifact_root, base_dir=Path.cwd())
    candidate_asterion = build_asterion_export(candidate_payload, artifact_root=candidate_source_root, base_dir=Path.cwd())
    canonical_content_lab = build_content_lab_candidates(canonical_asterion)
    candidate_content_lab = build_content_lab_candidates(candidate_asterion)

    metrics = {
        "record_counts": {
            "canonical_declared": canonical_payload.get("record_count"),
            "canonical_actual": len(canonical_records),
            "candidate_declared": candidate_payload.get("record_count"),
            "candidate_actual": len(candidate_records),
        },
        "identity": {
            "canonical_duplicate_ids": duplicate_ids(canonical_records),
            "candidate_duplicate_ids": duplicate_ids(candidate_records),
            "records_added": candidate_baseline.get("records_added"),
            "records_removed": candidate_baseline.get("records_removed"),
            "records_present_in_both": candidate_baseline.get("records_present_in_both"),
        },
        "image_integrity_source_root": compare_counts(
            canonical_integrity.get("counts", {}),
            candidate_source_integrity.get("counts", {}),
            [
                "missing_question_image_path_count",
                "missing_question_image_file_count",
                "missing_mark_scheme_image_path_count",
                "allowed_missing_mark_scheme_image_path_count",
                "unexpected_missing_mark_scheme_image_path_count",
                "missing_mark_scheme_image_file_count",
            ],
        ),
        "image_integrity_promotion_target": compare_counts(
            canonical_integrity.get("counts", {}),
            candidate_promotion_integrity.get("counts", {}),
            [
                "missing_question_image_path_count",
                "missing_question_image_file_count",
                "missing_mark_scheme_image_path_count",
                "allowed_missing_mark_scheme_image_path_count",
                "unexpected_missing_mark_scheme_image_path_count",
                "missing_mark_scheme_image_file_count",
            ],
        ),
        "hard_blockers": compare_counts_from_paths(
            canonical_readiness,
            candidate_readiness,
            [("readiness", "hard_blocker_count")],
        ),
        "mapping_validation": compare_counts(
            canonical_readiness.get("mapping_validation", {}),
            candidate_readiness.get("mapping_validation", {}),
            [
                "missing_mark_scheme_image_path_count",
                "missing_mark_scheme_text_count",
                "mapping_fail_but_validation_pass_count",
                "missing_mark_scheme_but_validation_pass_count",
                "mark_total_mismatch_count",
                "paper_total_matched_but_local_validation_failed_count",
            ],
        )
        | {
            "mapping_status_distribution": compare_distribution(
                canonical_readiness.get("mapping_validation", {}).get("mapping_status_distribution", {}),
                candidate_readiness.get("mapping_validation", {}).get("mapping_status_distribution", {}),
            ),
            "validation_status_distribution": compare_distribution(
                canonical_readiness.get("mapping_validation", {}).get("validation_status_distribution", {}),
                candidate_readiness.get("mapping_validation", {}).get("validation_status_distribution", {}),
            ),
        },
        "text_ocr": {
            "text_fidelity_status_distribution": compare_distribution(
                nested(canonical_readiness, "ocr_candidate_measurement", "text_fidelity_status_distribution"),
                nested(candidate_readiness, "ocr_candidate_measurement", "text_fidelity_status_distribution"),
            ),
            "question_text_trust_distribution": compare_distribution(
                nested(canonical_readiness, "ocr_candidate_measurement", "question_text_trust_distribution"),
                nested(candidate_readiness, "ocr_candidate_measurement", "question_text_trust_distribution"),
            ),
            "text_only_status_distribution": compare_distribution(
                nested(canonical_readiness, "ocr_candidate_measurement", "text_only_status_distribution"),
                nested(candidate_readiness, "ocr_candidate_measurement", "text_only_status_distribution"),
            ),
            "ocr_selected_count": compare_scalar(
                nested(canonical_readiness, "ocr_candidate_measurement", "ocr_selected_count"),
                nested(candidate_readiness, "ocr_candidate_measurement", "ocr_selected_count"),
            ),
            "possible_ocr_false_negative_count": compare_scalar(
                nested(canonical_readiness, "ocr_candidate_measurement", "possible_ocr_false_negative_count"),
                nested(candidate_readiness, "ocr_candidate_measurement", "possible_ocr_false_negative_count"),
            ),
            "suspicious_ocr_selected_count": compare_scalar(
                nested(canonical_readiness, "ocr_candidate_measurement", "suspicious_ocr_selected_count"),
                nested(candidate_readiness, "ocr_candidate_measurement", "suspicious_ocr_selected_count"),
            ),
        },
        "mark_totals_and_subparts": compare_counts(
            canonical_readiness.get("mark_totals_and_subparts", {}),
            candidate_readiness.get("mark_totals_and_subparts", {}),
            [
                "records_question_total_not_equal_mark_scheme_total",
                "records_question_solution_marks_not_equal_question_total_detected",
                "records_with_mark_scheme_total_detected_missing",
                "records_with_question_solution_marks_missing",
                "records_with_question_total_detected_missing",
                "records_with_subparts_present_but_all_subpart_marks_null",
                "total_null_subpart_mark_entries",
                "total_subpart_entries",
            ],
        ),
        "asterion": {
            "question_bank_role_counts": {
                "canonical": role_counts(canonical_asterion),
                "candidate": role_counts(candidate_asterion),
                "delta": counter_delta(role_counts(canonical_asterion), role_counts(candidate_asterion)),
            },
            "content_lab": {
                "canonical_record_count": canonical_content_lab.get("record_count"),
                "candidate_record_count": candidate_content_lab.get("record_count"),
                "role_status_counts": {
                    "canonical": content_lab_role_counts(canonical_content_lab),
                    "candidate": content_lab_role_counts(candidate_content_lab),
                    "delta": counter_delta(
                        content_lab_role_counts(canonical_content_lab),
                        content_lab_role_counts(candidate_content_lab),
                    ),
                },
                "generation_gate_status_counts": {
                    "canonical": generation_gate_counts(canonical_content_lab),
                    "candidate": generation_gate_counts(candidate_content_lab),
                    "delta": counter_delta(
                        generation_gate_counts(canonical_content_lab),
                        generation_gate_counts(candidate_content_lab),
                    ),
                },
            },
        },
        "topic_sidecar": topic_sidecar_summary(topic_sidecar, candidate_record_count=len(candidate_records)),
        "baseline_status_movement": {
            "improved_status_counts": candidate_baseline.get("improved_status_counts", {}),
            "worsened_status_counts": candidate_baseline.get("worsened_status_counts", {}),
            "asterion_tier_change_counts": candidate_baseline.get("asterion_tier_change_counts", {}),
            "field_change_counts": candidate_baseline.get("field_change_counts", {}),
            "reliable": candidate_baseline.get("reliable"),
            "unreliable_reason": candidate_baseline.get("unreliable_reason"),
        },
    }

    blockers = release_blockers(
        candidate_source_integrity=candidate_source_integrity,
        candidate_promotion_integrity=candidate_promotion_integrity,
        candidate_readiness=candidate_readiness,
        candidate_baseline=candidate_baseline,
        topic_summary=metrics["topic_sidecar"],
    )
    decision = decide(blockers)

    return {
        "schema_name": "exam_bank.candidate_release_gate",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "inputs": {
            "canonical": str(canonical_path),
            "candidate": str(candidate_path),
            "artifact_root": str(artifact_root),
            "candidate_artifact_root": str(candidate_source_root),
            "canonical_readiness": str(canonical_readiness_path),
            "candidate_readiness": str(candidate_readiness_path),
            "candidate_baseline": str(candidate_baseline_path),
            "topic_sidecar": str(topic_sidecar_path),
        },
        "policy": {
            "image_first": True,
            "candidate_text_is_advisory": True,
            "topic_sidecar_requires_safe_for_strict_filters": True,
            "no_automatic_promotion": True,
        },
        "metrics": metrics,
        "blockers": blockers,
        "promotion_commands_if_approved": promotion_commands(),
    }


def release_blockers(
    *,
    candidate_source_integrity: dict[str, Any],
    candidate_promotion_integrity: dict[str, Any],
    candidate_readiness: dict[str, Any],
    candidate_baseline: dict[str, Any],
    topic_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    source_counts = candidate_source_integrity.get("counts", {})
    promotion_counts = candidate_promotion_integrity.get("counts", {})
    readiness = candidate_readiness.get("readiness", {})
    mapping = candidate_readiness.get("mapping_validation", {})
    marks = candidate_readiness.get("mark_totals_and_subparts", {})
    worsened = candidate_baseline.get("worsened_status_counts", {}) if isinstance(candidate_baseline, dict) else {}

    if not candidate_source_integrity.get("ok"):
        blockers.append(blocker("P0", "candidate_source_integrity_failed", "Candidate source output-integrity audit is not ok."))
    if source_counts.get("duplicate_question_id_record_count", 0):
        blockers.append(blocker("P0", "duplicate_question_ids", "Candidate has duplicate question_id records."))
    if source_counts.get("missing_question_image_path_count", 0) or source_counts.get("missing_question_image_file_count", 0):
        blockers.append(blocker("P0", "missing_question_images", "Candidate has missing or unresolved question images."))
    if source_counts.get("unexpected_missing_mark_scheme_image_path_count", 0) or source_counts.get("missing_mark_scheme_image_file_count", 0):
        blockers.append(
            blocker("P0", "unexpected_missing_mark_scheme_images", "Candidate source has unexpected missing or unresolved mark-scheme images.")
        )
    if not candidate_promotion_integrity.get("ok"):
        blockers.append(
            blocker(
                "P1",
                "promotion_target_artifacts_missing",
                "Candidate JSON does not fully resolve against the canonical output artifact root; copy candidate artifacts before or during promotion.",
                {
                    "missing_question_image_file_count": promotion_counts.get("missing_question_image_file_count", 0),
                    "missing_mark_scheme_image_file_count": promotion_counts.get("missing_mark_scheme_image_file_count", 0),
                },
            )
        )

    hard_blockers = int(readiness.get("hard_blocker_count") or 0)
    if hard_blockers:
        blockers.append(
            blocker(
                "P1",
                "candidate_hard_blockers_remain",
                f"Candidate still has {hard_blockers} hard blockers.",
                readiness.get("hard_blocker_reason_counts", {}),
            )
        )
    validation_failures = int(mapping.get("validation_status_distribution", {}).get("fail") or 0)
    if validation_failures:
        blockers.append(blocker("P1", "validation_failures_remain", f"Candidate still has {validation_failures} validation failures."))
    mapping_failures = int(mapping.get("mapping_status_distribution", {}).get("fail") or 0)
    if mapping_failures:
        blockers.append(blocker("P1", "mapping_failures_remain", f"Candidate still has {mapping_failures} mapping failures."))
    mark_mismatches = int(marks.get("records_question_total_not_equal_mark_scheme_total") or 0)
    if mark_mismatches:
        blockers.append(blocker("P1", "mark_total_mismatches_remain", f"Candidate still has {mark_mismatches} question/mark-scheme total mismatches."))
    null_subpart_entries = int(marks.get("total_null_subpart_mark_entries") or 0)
    if null_subpart_entries:
        blockers.append(blocker("P2", "subpart_marks_incomplete", f"Candidate still has {null_subpart_entries} null subpart mark entries."))
    if worsened:
        blockers.append(blocker("P1", "status_regressions_present", "Candidate has worsened status movements against canonical.", worsened))
    if not topic_summary.get("safe_for_strict_filters"):
        blockers.append(
            blocker(
                "P2",
                "topic_sidecar_not_strict_filter_safe",
                "Topic sidecar is not safe for strict filters; strict topic filters must remain disabled after any promotion.",
                {
                    "failed_records": topic_summary.get("failed_records"),
                    "schema_validation_error_count": topic_summary.get("schema_validation_error_count"),
                },
            )
        )
    return blockers


def decide(blockers: list[dict[str, Any]]) -> str:
    p0 = [item for item in blockers if item["priority"] == "P0"]
    p1 = [item for item in blockers if item["priority"] == "P1"]
    if p0:
        return DECISION_DO_NOT_PROMOTE
    if p1:
        return DECISION_REPAIR
    return DECISION_PROMOTE


def promotion_commands() -> list[str]:
    return [
        "rsync -a output/codex_text_extraction_candidate_v2/pm1/ output/pm1/",
        "rsync -a output/codex_text_extraction_candidate_v2/pm3/ output/pm3/",
        "rsync -a output/codex_text_extraction_candidate_v2/stats/ output/stats/",
        "rsync -a output/codex_text_extraction_candidate_v2/mechanics/ output/mechanics/",
        "cp output/codex_text_extraction_candidate_v2/json/question_bank.json output/json/question_bank.json",
        ".venv/bin/python -m exam_bank.cli audit --input output/json/question_bank.json --output output/json/audit.current.json",
        ".venv/bin/python -m exam_bank.cli output-integrity-audit --input output/json/question_bank.json --artifact-root output --output output/json/audit.current.integrity.json",
        ".venv/bin/python scripts/audit_question_bank_readiness.py --input output/json/question_bank.json --artifact-root output --out-dir output/audits/current",
        ".venv/bin/python scripts/audit_ocr_candidates.py --input output/json/question_bank.json --json-output output/audits/current/ocr_candidate_audit.json",
        ".venv/bin/python -m exam_bank.cli asterion-export --input output/json/question_bank.json --artifact-root output",
        ".venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --input output/json/question_bank.json --artifact-root output",
        ".venv/bin/python -m pytest -q",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Candidate v2 Release Gate",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "This report does not promote the candidate. Canonical images remain source of truth; extracted text, OCR, topics, and AI sidecars remain advisory unless their gates allow use.",
        "",
        "## Inputs",
    ]
    for key, value in report["inputs"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Before / After Metrics",
            "",
            "| Metric | Canonical | Candidate v2 | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    add_row(lines, "records", metrics["record_counts"]["canonical_actual"], metrics["record_counts"]["candidate_actual"])
    add_row(lines, "duplicate question IDs", len(metrics["identity"]["canonical_duplicate_ids"]), len(metrics["identity"]["candidate_duplicate_ids"]))
    for label, data in metrics["image_integrity_source_root"].items():
        add_compared_row(lines, f"source-root {label}", data)
    for label, data in metrics["image_integrity_promotion_target"].items():
        add_compared_row(lines, f"promotion-target {label}", data)
    add_compared_row(lines, "hard blockers", metrics["hard_blockers"]["readiness.hard_blocker_count"])
    add_status_row(lines, "mapping failures", metrics["mapping_validation"]["mapping_status_distribution"], "fail")
    add_status_row(lines, "validation failures", metrics["mapping_validation"]["validation_status_distribution"], "fail")
    add_compared_row(lines, "possible OCR false negatives", metrics["text_ocr"]["possible_ocr_false_negative_count"])
    add_compared_row(lines, "OCR selected", metrics["text_ocr"]["ocr_selected_count"])
    add_compared_row(lines, "suspicious OCR selected", metrics["text_ocr"]["suspicious_ocr_selected_count"])
    add_status_row(lines, "degraded text", metrics["text_ocr"]["text_fidelity_status_distribution"], "degraded")
    add_status_row(lines, "low-trust text", metrics["text_ocr"]["question_text_trust_distribution"], "low")
    add_status_row(lines, "text-only fail", metrics["text_ocr"]["text_only_status_distribution"], "fail")
    add_compared_row(lines, "mark total mismatches", metrics["mark_totals_and_subparts"]["records_question_total_not_equal_mark_scheme_total"])
    add_compared_row(lines, "null subpart mark entries", metrics["mark_totals_and_subparts"]["total_null_subpart_mark_entries"])

    lines.extend(["", "## Asterion Role Counts", "", "| Role status | Canonical | Candidate v2 | Delta |", "| --- | ---: | ---: | ---: |"])
    for key, data in sorted(metrics["asterion"]["question_bank_role_counts"]["delta"].items()):
        before = metrics["asterion"]["question_bank_role_counts"]["canonical"].get(key, 0)
        after = metrics["asterion"]["question_bank_role_counts"]["candidate"].get(key, 0)
        lines.append(f"| `{key}` | {before} | {after} | {format_delta(after - before)} |")

    lines.extend(["", "## Content Lab Counts", "", "| Count | Canonical | Candidate v2 | Delta |", "| --- | ---: | ---: | ---: |"])
    content_lab = metrics["asterion"]["content_lab"]
    add_row(lines, "candidate records", content_lab["canonical_record_count"], content_lab["candidate_record_count"])
    for key, delta in sorted(content_lab["role_status_counts"]["delta"].items()):
        before = content_lab["role_status_counts"]["canonical"].get(key, 0)
        after = content_lab["role_status_counts"]["candidate"].get(key, 0)
        lines.append(f"| `role_statuses.{key}` | {before} | {after} | {format_delta(delta)} |")
    for key, delta in sorted(content_lab["generation_gate_status_counts"]["delta"].items()):
        before = content_lab["generation_gate_status_counts"]["canonical"].get(key, 0)
        after = content_lab["generation_gate_status_counts"]["candidate"].get(key, 0)
        lines.append(f"| `generation_gate.status.{key}` | {before} | {after} | {format_delta(delta)} |")

    lines.extend(["", "## Topic Sidecar Safety"])
    topic = metrics["topic_sidecar"]
    lines.extend(
        [
            f"- schema: `{topic.get('schema_name')}` v{topic.get('schema_version')}",
            f"- record_count: {topic.get('record_count')} sidecar / {topic.get('candidate_record_count')} candidate",
            f"- safe_for_strict_filters: `{topic.get('safe_for_strict_filters')}`",
            f"- failed_records: {topic.get('failed_records')}",
            f"- schema_validation_error_count: {topic.get('schema_validation_error_count')}",
        ]
    )

    lines.extend(["", "## Blockers"])
    if report["blockers"]:
        for item in report["blockers"]:
            detail = f" Details: `{json.dumps(item['details'], sort_keys=True)}`" if item.get("details") else ""
            lines.append(f"- {item['priority']} `{item['code']}`: {item['message']}{detail}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Deliberate Promotion Commands"])
    if report["decision"] == DECISION_PROMOTE:
        lines.append("Run only after human approval:")
    else:
        lines.append("Do not run these until blockers are repaired or explicitly waived:")
    lines.append("")
    lines.append("```bash")
    lines.extend(report["promotion_commands_if_approved"])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def add_row(lines: list[str], label: str, before: Any, after: Any) -> None:
    lines.append(f"| {label} | {before} | {after} | {format_delta(num(after) - num(before))} |")


def add_compared_row(lines: list[str], label: str, compared: dict[str, Any]) -> None:
    before = compared.get("canonical")
    after = compared.get("candidate")
    lines.append(f"| {label} | {before} | {after} | {format_delta(num(after) - num(before))} |")


def add_status_row(lines: list[str], label: str, distribution: dict[str, dict[str, Any]], status: str) -> None:
    add_compared_row(lines, label, distribution.get(status, {"canonical": 0, "candidate": 0}))


def compare_counts_from_paths(before: dict[str, Any], after: dict[str, Any], paths: list[tuple[str, ...]]) -> dict[str, dict[str, Any]]:
    return {".".join(path): compare_scalar(nested(before, *path), nested(after, *path)) for path in paths}


def compare_counts(before: dict[str, Any], after: dict[str, Any], keys: list[str]) -> dict[str, dict[str, Any]]:
    return {key: compare_scalar(before.get(key, 0), after.get(key, 0)) for key in keys}


def compare_distribution(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: compare_scalar(before.get(key, 0), after.get(key, 0)) for key in sorted(set(before) | set(after))}


def compare_scalar(before: Any, after: Any) -> dict[str, Any]:
    return {"canonical": before, "candidate": after, "delta": num(after) - num(before)}


def role_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in payload.get("questions", []):
        roles = record.get("usage_roles") if isinstance(record.get("usage_roles"), dict) else {}
        for role in ROLE_KEYS:
            counts[f"{role}.{roles.get(role, 'missing')}"] += 1
    return dict(sorted(counts.items()))


def content_lab_role_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in payload.get("candidates", []):
        roles = candidate.get("role_statuses") if isinstance(candidate.get("role_statuses"), dict) else {}
        for role in ROLE_KEYS:
            if role in roles:
                counts[f"{role}.{roles[role]}"] += 1
    return dict(sorted(counts.items()))


def generation_gate_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in payload.get("candidates", []):
        gate = candidate.get("generation_gate") if isinstance(candidate.get("generation_gate"), dict) else {}
        counts[str(gate.get("status") or "missing")] += 1
    return dict(sorted(counts.items()))


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(set(before) | set(after))}


def topic_sidecar_summary(payload: dict[str, Any], *, candidate_record_count: int) -> dict[str, Any]:
    run_summary = nested(payload, "metadata", "run_summary")
    failures = run_summary.get("failures_by_reason", {}) if isinstance(run_summary, dict) else {}
    return {
        "schema_name": payload.get("schema_name"),
        "schema_version": payload.get("schema_version"),
        "record_count": payload.get("record_count"),
        "candidate_record_count": candidate_record_count,
        "records_object_count": len(payload.get("records", {})) if isinstance(payload.get("records"), dict) else None,
        "safe_for_strict_filters": run_summary.get("safe_for_strict_filters") if isinstance(run_summary, dict) else None,
        "failed_records": run_summary.get("failed_records") if isinstance(run_summary, dict) else None,
        "review_required_records": run_summary.get("review_required_records") if isinstance(run_summary, dict) else None,
        "strict_filter_records": run_summary.get("strict_filter_records") if isinstance(run_summary, dict) else None,
        "schema_validation_error_count": failures.get("schema_validation_error") if isinstance(failures, dict) else None,
        "failures_by_reason": failures,
    }


def duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(record.get("question_id") or "") for record in records if record.get("question_id"))
    return sorted(value for value, count in counts.items() if count > 1)


def records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("questions")
    if not isinstance(records, list):
        raise ValueError("Question-bank payload must contain a questions list.")
    return [record for record in records if isinstance(record, dict)]


def infer_output_root(question_bank_path: Path) -> Path:
    if question_bank_path.parent.name == "json":
        return question_bank_path.parent.parent
    return question_bank_path.parent


def nested(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current


def blocker(priority: str, code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    return {"priority": priority, "code": code, "message": message, "details": details or {}}


def format_delta(delta: int | float) -> str:
    if delta > 0:
        return f"+{delta:g}"
    return f"{delta:g}"


def num(value: Any) -> float:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return value
    return 0


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
