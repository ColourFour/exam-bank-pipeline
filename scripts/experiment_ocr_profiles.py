from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from exam_bank.ocr_profiles import OCRProfileRun, available_ocr_profiles, run_profile_ocr
from exam_bank.text_fidelity import (
    issue_key,
    load_fixture_manifest,
    measurable_improvement_targets,
    score_text_issues,
    text_value,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "text_fidelity" / "bad_text_records.json"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "reports" / "ocr_profile_experiment.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / "output" / "reports" / "ocr_profile_experiment.md"
DEFAULT_ROUTING_JSON_OUT = REPO_ROOT / "output" / "reports" / "ocr_profile_routing_experiment.json"
DEFAULT_ROUTING_MARKDOWN_OUT = REPO_ROOT / "output" / "reports" / "ocr_profile_routing_experiment.md"
DEFAULT_ROUTING_DOC_OUT = REPO_ROOT / "docs" / "text_extraction" / "OCR_PROFILE_ROUTING_EXPERIMENT.md"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "output"
SCHEMA_NAME = "exam_bank.ocr_profile_experiment"
SCHEMA_VERSION = 1
ROUTING_SCHEMA_NAME = "exam_bank.ocr_profile_routing_experiment"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimentally compare OCR/preprocessing profiles against frozen bad-text fixtures only."
    )
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_PATH), help="Frozen bad-text fixture manifest path.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="Root containing canonical question images.")
    parser.add_argument("--json-out", default=None, help="JSON report output path.")
    parser.add_argument("--markdown-out", default=None, help="Markdown report output path.")
    parser.add_argument(
        "--routing-analysis",
        action="store_true",
        help="Write layout/paper/failure-family routing analysis reports without changing production selection.",
    )
    parser.add_argument(
        "--routing-doc-out",
        default=str(DEFAULT_ROUTING_DOC_OUT),
        help="Documentation copy path for --routing-analysis markdown.",
    )
    parser.add_argument("--language", default="eng", help="Tesseract language.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per-image Tesseract timeout.")
    args = parser.parse_args()

    manifest = load_fixture_manifest(Path(args.fixtures))
    report = build_experiment_report(
        manifest,
        image_root=Path(args.image_root),
        language=args.language,
        timeout_seconds=args.timeout_seconds,
        include_routing_analysis=args.routing_analysis,
    )

    json_out = Path(args.json_out) if args.json_out else DEFAULT_ROUTING_JSON_OUT if args.routing_analysis else DEFAULT_JSON_OUT
    markdown_out = (
        Path(args.markdown_out) if args.markdown_out else DEFAULT_ROUTING_MARKDOWN_OUT if args.routing_analysis else DEFAULT_MARKDOWN_OUT
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    if args.routing_analysis:
        routing_doc_out = Path(args.routing_doc_out)
        routing_doc_out.parent.mkdir(parents=True, exist_ok=True)
        routing_doc_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
    if args.routing_analysis:
        print(f"Wrote {args.routing_doc_out}")
    print(f"Best profile: {report['summary']['best_profile_by_average_score']}")
    print(f"Worst profile: {report['summary']['worst_profile_by_average_score']}")
    if report["summary"]["runtime_blockers"]:
        print(f"Runtime blockers: {report['summary']['runtime_blockers']}")
    return 0


def build_experiment_report(
    manifest: dict[str, Any],
    *,
    image_root: Path,
    language: str = "eng",
    timeout_seconds: int = 30,
    include_routing_analysis: bool = False,
) -> dict[str, Any]:
    profiles = available_ocr_profiles()
    records: list[dict[str, Any]] = []
    for fixture in manifest["records"]:
        baseline = score_profile_text(fixture, text_value(fixture.get("currently_selected_text")), "baseline_current")
        profile_rows = [baseline]
        image_path = resolve_image_path(image_root, fixture)
        for profile in profiles:
            if profile.name == "baseline_current":
                continue
            if image_path is None:
                run = OCRProfileRun(
                    profile=profile.name,
                    text="",
                    runtime_seconds=0.0,
                    ok=False,
                    error=f"missing image under {image_root}",
                )
            else:
                run = run_profile_ocr(
                    image_path,
                    profile,
                    language=language,
                    timeout_seconds=timeout_seconds,
                )
            profile_rows.append(score_profile_text(fixture, run.text, profile.name, run=run))
        records.append(build_record_result(fixture, image_path, baseline, profile_rows))

    profile_summary = summarize_profiles(records)
    family_summary = summarize_group(records, "paper_family")
    layout_summary = summarize_layouts(records)
    report = {
        "schema_name": ROUTING_SCHEMA_NAME if include_routing_analysis else SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "fixture_schema_name": manifest.get("schema_name"),
        "fixture_schema_version": manifest.get("schema_version"),
        "record_count": len(records),
        "scope": "frozen_bad_text_fixtures_only",
        "production_behavior_unchanged": True,
        "writes_question_bank": False,
        "profiles": [profile.__dict__ for profile in profiles],
        "summary": build_summary(profile_summary),
        "profile_summary": profile_summary,
        "paper_family_summary": family_summary,
        "layout_type_summary": layout_summary,
        "records": records,
    }
    if include_routing_analysis:
        routing_slices = summarize_routing_slices(records)
        report["routing_analysis"] = {
            "scope": "report_only_candidate_routing",
            "production_behavior_unchanged": True,
            "writes_selected_text": False,
            "writes_asterion_exports": False,
            "treats_profile_output_as_canonical": False,
            "safety_rule": "A profile is marked safely_better for a slice only when it has at least one improvement, positive average delta, and zero regressions versus baseline in that slice.",
            "slice_summary": routing_slices,
            "candidate_slices": [row for row in routing_slices if row["best_safe_profile"]],
            "unsafe_or_no_safe_slices": [row for row in routing_slices if not row["best_safe_profile"]],
        }
    return report


def score_profile_text(
    fixture: dict[str, Any],
    text: str,
    profile_name: str,
    *,
    run: OCRProfileRun | None = None,
) -> dict[str, Any]:
    expectations = list(
        (fixture.get("expected_normalized_text_or_structural_expectations") or {}).get("expectations") or []
    )
    issues = score_text_issues(
        fixture,
        text,
        text_value(fixture.get("native_pdf_text_raw")),
        text,
        str(fixture.get("question_number") or "").strip(),
        expectations,
        [str(tag) for tag in fixture.get("failure_tags") or []],
        [],
        include_disagreement=False,
        include_structural_rejection=False,
    )
    fail_count = sum(1 for issue in issues if issue["severity"] == "fail")
    warn_count = sum(1 for issue in issues if issue["severity"] == "warn")
    fixture_score = max(0, 100 - (fail_count * 20) - (warn_count * 5))
    return {
        "profile": profile_name,
        "ok": True if run is None else run.ok,
        "error": "" if run is None else run.error,
        "runtime_seconds": 0.0 if run is None else round(run.runtime_seconds, 4),
        "text_length": len(text),
        "text_snippet": text[:240],
        "fixture_score": fixture_score,
        "status": "fail" if fail_count else "warn" if warn_count else "pass",
        "issue_count": len(issues),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "issue_keys": [issue_key(issue) for issue in issues],
        "measurable_targets": measurable_improvement_targets(issues),
    }


def build_record_result(
    fixture: dict[str, Any],
    image_path: Path | None,
    baseline: dict[str, Any],
    profile_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_keys = set(baseline["issue_keys"])
    for row in profile_rows:
        row_keys = set(row["issue_keys"])
        row["score_delta_vs_baseline"] = row["fixture_score"] - baseline["fixture_score"]
        row["resolved_issue_keys_vs_baseline"] = sorted(baseline_keys - row_keys)
        row["introduced_issue_keys_vs_baseline"] = sorted(row_keys - baseline_keys)
        if row["score_delta_vs_baseline"] > 0:
            row["comparison_vs_baseline"] = "improved"
        elif row["score_delta_vs_baseline"] < 0:
            row["comparison_vs_baseline"] = "regressed"
        else:
            row["comparison_vs_baseline"] = "unchanged"

    best = max(profile_rows, key=lambda row: (row["fixture_score"], -row["runtime_seconds"], row["profile"]))
    worst = min(profile_rows, key=lambda row: (row["fixture_score"], -row["runtime_seconds"], row["profile"]))
    return {
        "record_id": fixture.get("record_id"),
        "paper_id": fixture.get("paper_id"),
        "paper_family": fixture.get("paper_family"),
        "layout_types": infer_layout_types(fixture),
        "routing_slices": infer_routing_slices(fixture),
        "failure_types": infer_failure_types(fixture),
        "question_number": fixture.get("question_number"),
        "question_image_path": fixture.get("question_image_path"),
        "resolved_image_path": str(image_path) if image_path else "",
        "best_profile": best["profile"],
        "best_fixture_score": best["fixture_score"],
        "worst_profile": worst["profile"],
        "worst_fixture_score": worst["fixture_score"],
        "profiles": profile_rows,
    }


def resolve_image_path(image_root: Path, fixture: dict[str, Any]) -> Path | None:
    relative = Path(str(fixture.get("question_image_path") or ""))
    if not relative:
        return None
    direct = image_root / relative
    if direct.exists():
        return direct
    matches = sorted(image_root.glob(f"**/{relative.as_posix()}"))
    return matches[0] if matches else None


def infer_layout_types(fixture: dict[str, Any]) -> list[str]:
    tags = set(str(tag) for tag in fixture.get("failure_tags") or [])
    expectations = " ".join(
        (fixture.get("expected_normalized_text_or_structural_expectations") or {}).get("expectations") or []
    ).lower()
    layouts: set[str] = set()
    if tags & {"fraction_structure", "rational_expression", "integral_bounds", "derivative_layout", "radical_or_power_structure"}:
        layouts.add("formula_heavy")
    if "table" in expectations or tags & {"table_layout", "probability_table"}:
        layouts.add("table_preserving")
    if "diagram" in expectations or tags & {"diagram_prompt", "graph_or_diagram"}:
        layouts.add("diagram_safe")
    if tags & {"question_anchor_presence", "question_number_not_at_start"} or "question number" in expectations:
        layouts.add("anchor_sensitive")
    if not layouts:
        layouts.add("general_text")
    return sorted(layouts)


def infer_failure_types(fixture: dict[str, Any]) -> list[str]:
    tags = set(str(tag) for tag in fixture.get("failure_tags") or [])
    expectations = " ".join(
        (fixture.get("expected_normalized_text_or_structural_expectations") or {}).get("expectations") or []
    ).lower()
    failure_types: set[str] = set()
    if tags & {"mark_bracket_missing", "mark_bracket_order"} or "mark bracket" in expectations:
        failure_types.add("mark_bracket_recovery")
    if tags & {"question_anchor_missing", "question_anchor_displaced"} or "starts with question number" in expectations:
        failure_types.add("question_anchor_recovery")
    if tags & {"truncated_text", "crop_boundary_or_contamination", "page_furniture_contamination"}:
        failure_types.add("crop_or_contamination")
    if tags & {"ocr_noise", "severe_ocr_noise", "bullet_list_ocr", "table_furniture_noise"}:
        failure_types.add("ocr_noise")
    if tags & {"clean_crop_degraded_text", "spacing_degradation"}:
        failure_types.add("clean_crop_degraded_text")
    if tags & {"math_notation", "greek_symbol", "integral_bounds", "trig_symbol_fidelity", "vector_matrix_layout"}:
        failure_types.add("math_symbol_loss")
    if not failure_types:
        failure_types.add("other")
    return sorted(failure_types)


def infer_routing_slices(fixture: dict[str, Any]) -> list[dict[str, str]]:
    tags = set(str(tag) for tag in fixture.get("failure_tags") or [])
    expectations = " ".join(
        (fixture.get("expected_normalized_text_or_structural_expectations") or {}).get("expectations") or []
    ).lower()
    slices: set[tuple[str, str]] = set()

    paper_family = str(fixture.get("paper_family") or "unknown").upper()
    if paper_family in {"P1", "P3", "P4", "P5"}:
        slices.add(("paper_family", paper_family))

    if tags & {"fraction_structure", "rational_expression", "polynomial_reading_order", "radical_or_power_structure", "complex_number_layout"}:
        slices.add(("layout_family", "dense_algebra"))
    if tags & {"calculus_expression", "integral_bounds", "derivative_layout"} or any(
        token in expectations for token in ("integral", "dy/dx", "derivative", "ln(", "exponential")
    ):
        slices.add(("layout_family", "calculus_integrals"))
    if tags & {"trig_symbol_fidelity", "greek_symbol"} or any(
        token in expectations for token in ("theta", "cos", "sin", "sigma", "standard deviation")
    ):
        slices.add(("layout_family", "trig_log_notation"))
    if tags & {"vector_matrix_layout", "transformation_diagram_order"} or "vector" in expectations:
        slices.add(("layout_family", "vectors_matrices"))
    if tags & {
        "diagram_reading_order",
        "mechanics_diagram",
        "mechanics_graph_reading_order",
        "statistics_table_structure",
        "table_furniture_noise",
    } or any(token in expectations for token in ("diagram", "table", "graph", "histogram", "spinner")):
        slices.add(("layout_family", "diagrams_tables"))
    if tags & {"mark_bracket_missing", "mark_bracket_order"} or "mark bracket" in expectations:
        slices.add(("failure_type", "mark_bracket_recovery"))
    if tags & {"question_anchor_missing", "question_anchor_displaced"} or "starts with question number" in expectations:
        slices.add(("failure_type", "question_anchor_recovery"))
    if tags & {"math_notation", "greek_symbol", "integral_bounds", "trig_symbol_fidelity", "vector_matrix_layout"}:
        slices.add(("failure_type", "symbol_heavy_cases"))

    return [{"slice_type": slice_type, "slice": name} for slice_type, name in sorted(slices)]


def summarize_profiles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for row in record["profiles"]:
            by_profile[row["profile"]].append(row)
    summaries = []
    for profile, rows in by_profile.items():
        ok_rows = [row for row in rows if row["ok"]]
        summaries.append(
            {
                "profile": profile,
                "record_count": len(rows),
                "ok_count": len(ok_rows),
                "error_count": len(rows) - len(ok_rows),
                "average_fixture_score": round(mean(row["fixture_score"] for row in rows), 2),
                "pass_count": sum(1 for row in rows if row["status"] == "pass"),
                "warn_count": sum(1 for row in rows if row["status"] == "warn"),
                "fail_count": sum(1 for row in rows if row["status"] == "fail"),
                "improved_count": sum(1 for row in rows if row["comparison_vs_baseline"] == "improved"),
                "regressed_count": sum(1 for row in rows if row["comparison_vs_baseline"] == "regressed"),
                "unchanged_count": sum(1 for row in rows if row["comparison_vs_baseline"] == "unchanged"),
                "total_runtime_seconds": round(sum(row["runtime_seconds"] for row in rows), 4),
                "average_runtime_seconds": round(mean(row["runtime_seconds"] for row in rows), 4),
                "top_errors": dict(Counter(row["error"] for row in rows if row["error"]).most_common(3)),
            }
        )
    return sorted(summaries, key=lambda row: (-row["average_fixture_score"], row["average_runtime_seconds"], row["profile"]))


def summarize_group(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_key) or "unknown")].append(record)
    return [summarize_record_group(group, rows) for group, rows in sorted(grouped.items())]


def summarize_layouts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for layout in record["layout_types"]:
            grouped[layout].append(record)
    return [summarize_record_group(group, rows) for group, rows in sorted(grouped.items())]


def summarize_record_group(group: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    profile_scores: dict[str, list[int]] = defaultdict(list)
    for record in records:
        for row in record["profiles"]:
            profile_scores[row["profile"]].append(row["fixture_score"])
    averages = {
        profile: round(mean(scores), 2)
        for profile, scores in profile_scores.items()
    }
    best_profile = max(averages, key=lambda profile: (averages[profile], profile)) if averages else ""
    return {
        "group": group,
        "record_count": len(records),
        "best_profile": best_profile,
        "average_scores": dict(sorted(averages.items(), key=lambda item: (-item[1], item[0]))),
    }


def summarize_routing_slices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for slice_ref in record["routing_slices"]:
            grouped[(slice_ref["slice_type"], slice_ref["slice"])].append(record)

    summaries = []
    for (slice_type, slice_name), slice_records in sorted(grouped.items()):
        profile_rows = [summarize_profile_for_slice(profile, rows) for profile, rows in rows_by_profile(slice_records).items()]
        profile_rows = sorted(
            profile_rows,
            key=lambda row: (
                -row["average_score"],
                -row["average_delta_vs_baseline"],
                row["regressed_count"],
                -row["improved_count"],
                row["average_runtime_seconds"],
                row["profile"],
            ),
        )
        safe_profiles = [row for row in profile_rows if row["safety_classification"] == "safely_better"]
        best_safe = safe_profiles[0]["profile"] if safe_profiles else ""
        no_safe_reason = ""
        if not best_safe:
            no_safe_reason = summarize_no_safe_reason(profile_rows)
        summaries.append(
            {
                "slice_type": slice_type,
                "slice": slice_name,
                "record_count": len(slice_records),
                "record_ids": [str(record["record_id"]) for record in slice_records],
                "best_safe_profile": best_safe,
                "no_safe_profile_reason": no_safe_reason,
                "profile_summary": profile_rows,
                "regressions": collect_slice_regressions(slice_records),
                "improvements": collect_slice_improvements(slice_records),
            }
        )
    return summaries


def rows_by_profile(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for row in record["profiles"]:
            if row["profile"] == "baseline_current":
                continue
            grouped[row["profile"]].append(row)
    return grouped


def summarize_profile_for_slice(profile: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["score_delta_vs_baseline"] for row in rows]
    improved_count = sum(1 for delta in deltas if delta > 0)
    regressed_count = sum(1 for delta in deltas if delta < 0)
    average_delta = round(mean(deltas), 2)
    if improved_count and average_delta > 0 and regressed_count == 0:
        safety = "safely_better"
    elif regressed_count:
        safety = "unsafe_regressions"
    elif improved_count:
        safety = "mixed_no_net_gain"
    else:
        safety = "no_measured_gain"
    return {
        "profile": profile,
        "record_count": len(rows),
        "average_score": round(mean(row["fixture_score"] for row in rows), 2),
        "average_delta_vs_baseline": average_delta,
        "min_delta_vs_baseline": min(deltas),
        "max_delta_vs_baseline": max(deltas),
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "unchanged_count": sum(1 for delta in deltas if delta == 0),
        "total_runtime_seconds": round(sum(row["runtime_seconds"] for row in rows), 4),
        "average_runtime_seconds": round(mean(row["runtime_seconds"] for row in rows), 4),
        "safety_classification": safety,
    }


def summarize_no_safe_reason(profile_rows: list[dict[str, Any]]) -> str:
    if not profile_rows:
        return "no non-baseline profile rows"
    if all(row["regressed_count"] for row in profile_rows):
        return "all profiles regressed on at least one fixture in this slice"
    if not any(row["improved_count"] for row in profile_rows):
        return "no profile improved over baseline in this slice"
    return "candidate profiles either regressed or lacked positive net gain"


def collect_slice_regressions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regressions = []
    for record in records:
        for row in record["profiles"]:
            if row["profile"] == "baseline_current" or row["score_delta_vs_baseline"] >= 0:
                continue
            regressions.append(
                {
                    "record_id": record["record_id"],
                    "profile": row["profile"],
                    "delta": row["score_delta_vs_baseline"],
                    "introduced_issue_keys": row["introduced_issue_keys_vs_baseline"],
                }
            )
    return sorted(regressions, key=lambda row: (row["record_id"], row["profile"]))


def collect_slice_improvements(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    improvements = []
    for record in records:
        for row in record["profiles"]:
            if row["profile"] == "baseline_current" or row["score_delta_vs_baseline"] <= 0:
                continue
            improvements.append(
                {
                    "record_id": record["record_id"],
                    "profile": row["profile"],
                    "delta": row["score_delta_vs_baseline"],
                    "resolved_issue_keys": row["resolved_issue_keys_vs_baseline"],
                }
            )
    return sorted(improvements, key=lambda row: (row["record_id"], row["profile"]))


def build_summary(profile_summary: list[dict[str, Any]]) -> dict[str, Any]:
    if not profile_summary:
        return {
            "best_profile_by_average_score": "",
            "worst_profile_by_average_score": "",
            "runtime_blockers": [],
        }
    best = profile_summary[0]["profile"]
    worst = sorted(profile_summary, key=lambda row: (row["average_fixture_score"], -row["average_runtime_seconds"], row["profile"]))[0]["profile"]
    blockers = [
        {"profile": row["profile"], "top_errors": row["top_errors"]}
        for row in profile_summary
        if row["error_count"]
    ]
    return {
        "best_profile_by_average_score": best,
        "worst_profile_by_average_score": worst,
        "runtime_blockers": blockers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    title = "OCR Profile Routing Experiment" if report.get("routing_analysis") else "OCR Profile Experiment"
    lines = [
        f"# {title}",
        "",
        "Experimental fixture-only OCR/preprocessing comparison. Canonical images remain the source of truth.",
        "",
        "## Scope",
        "",
        f"- Records: {report['record_count']}",
        f"- Fixture scope: {report['scope']}",
        f"- Production behavior unchanged: {report['production_behavior_unchanged']}",
        f"- Writes question_bank.json: {report['writes_question_bank']}",
        "",
        "## Profile Summary",
        "",
        "| Profile | Avg score | Pass | Warn | Fail | Improved | Regressed | Runtime total | Runtime avg | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["profile_summary"]:
        lines.append(
            f"| {row['profile']} | {row['average_fixture_score']:.2f} | {row['pass_count']} | {row['warn_count']} | "
            f"{row['fail_count']} | {row['improved_count']} | {row['regressed_count']} | "
            f"{row['total_runtime_seconds']:.2f}s | {row['average_runtime_seconds']:.2f}s | {row['error_count']} |"
        )

    lines.extend(
        [
            "",
            "## Family/Layout Signals",
            "",
            "| Group type | Group | Records | Best profile |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in report["paper_family_summary"]:
        lines.append(f"| paper_family | {row['group']} | {row['record_count']} | {row['best_profile']} |")
    for row in report["layout_type_summary"]:
        lines.append(f"| layout_type | {row['group']} | {row['record_count']} | {row['best_profile']} |")

    if report.get("routing_analysis"):
        routing = report["routing_analysis"]
        lines.extend(
            [
                "",
                "## Routing Safety Summary",
                "",
                f"- Routing scope: {routing['scope']}",
                f"- Writes selected text: {routing['writes_selected_text']}",
                f"- Writes Asterion exports: {routing['writes_asterion_exports']}",
                f"- Treats profile output as canonical: {routing['treats_profile_output_as_canonical']}",
                f"- Safety rule: {routing['safety_rule']}",
                "",
                "| Slice type | Slice | Records | Best safe profile | No-safe reason |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for row in routing["slice_summary"]:
            best_safe = row["best_safe_profile"] or "none"
            reason = row["no_safe_profile_reason"] or "none"
            lines.append(f"| {row['slice_type']} | {row['slice']} | {row['record_count']} | {best_safe} | {reason} |")

        lines.extend(
            [
                "",
                "## Routing Slice Profile Detail",
                "",
                "| Slice | Profile | Avg score | Avg delta | Min delta | Max delta | Improved | Regressed | Runtime total | Safety |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in routing["slice_summary"]:
            slice_label = f"{row['slice_type']}:{row['slice']}"
            for profile in row["profile_summary"]:
                lines.append(
                    f"| {slice_label} | {profile['profile']} | {profile['average_score']:.2f} | "
                    f"{profile['average_delta_vs_baseline']:.2f} | {profile['min_delta_vs_baseline']} | "
                    f"{profile['max_delta_vs_baseline']} | {profile['improved_count']} | "
                    f"{profile['regressed_count']} | {profile['total_runtime_seconds']:.2f}s | "
                    f"{profile['safety_classification']} |"
                )

        lines.extend(
            [
                "",
                "## Routing Slice Regressions",
                "",
                "| Slice | Record | Profile | Delta | Introduced issues |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        regression_count = 0
        for row in routing["slice_summary"]:
            slice_label = f"{row['slice_type']}:{row['slice']}"
            for regression in row["regressions"]:
                regression_count += 1
                introduced = ", ".join(regression["introduced_issue_keys"]) or "score-only regression"
                lines.append(
                    f"| {slice_label} | {regression['record_id']} | {regression['profile']} | "
                    f"{regression['delta']} | {introduced} |"
                )
        if regression_count == 0:
            lines.append("| none | none | none | 0 | none |")

    lines.extend(
        [
            "",
            "## Per-Record Best/Worst",
            "",
            "| Record | Family | Layouts | Best | Best score | Worst | Worst score |",
            "| --- | --- | --- | --- | ---: | --- | ---: |",
        ]
    )
    for record in report["records"]:
        lines.append(
            f"| {record['record_id']} | {record['paper_family']} | {', '.join(record['layout_types'])} | "
            f"{record['best_profile']} | {record['best_fixture_score']} | {record['worst_profile']} | {record['worst_fixture_score']} |"
        )

    lines.extend(
        [
            "",
            "## Improvements",
            "",
            "| Record | Profile | Delta | Resolved issues |",
            "| --- | --- | ---: | --- |",
        ]
    )
    improvement_count = 0
    for record in report["records"]:
        for row in record["profiles"]:
            if row["comparison_vs_baseline"] != "improved":
                continue
            improvement_count += 1
            resolved = ", ".join(row["resolved_issue_keys_vs_baseline"]) or "score-only improvement"
            lines.append(f"| {record['record_id']} | {row['profile']} | +{row['score_delta_vs_baseline']} | {resolved} |")
    if improvement_count == 0:
        lines.append("| none | none | 0 | none |")

    lines.extend(
        [
            "",
            "## Regressions",
            "",
            "| Record | Profile | Delta | Introduced issues |",
            "| --- | --- | ---: | --- |",
        ]
    )
    regression_count = 0
    for record in report["records"]:
        for row in record["profiles"]:
            if row["comparison_vs_baseline"] != "regressed":
                continue
            regression_count += 1
            introduced = ", ".join(row["introduced_issue_keys_vs_baseline"]) or "score-only regression"
            lines.append(f"| {record['record_id']} | {row['profile']} | {row['score_delta_vs_baseline']} | {introduced} |")
    if regression_count == 0:
        lines.append("| none | none | 0 | none |")

    if report["summary"]["runtime_blockers"]:
        lines.extend(["", "## Runtime Blockers", ""])
        for blocker in report["summary"]["runtime_blockers"]:
            lines.append(f"- {blocker['profile']}: {blocker['top_errors']}")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
