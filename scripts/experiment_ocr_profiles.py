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
DEFAULT_IMAGE_ROOT = REPO_ROOT / "output"
SCHEMA_NAME = "exam_bank.ocr_profile_experiment"
SCHEMA_VERSION = 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimentally compare OCR/preprocessing profiles against frozen bad-text fixtures only."
    )
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_PATH), help="Frozen bad-text fixture manifest path.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="Root containing canonical question images.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON report output path.")
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT), help="Markdown report output path.")
    parser.add_argument("--language", default="eng", help="Tesseract language.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per-image Tesseract timeout.")
    args = parser.parse_args()

    manifest = load_fixture_manifest(Path(args.fixtures))
    report = build_experiment_report(
        manifest,
        image_root=Path(args.image_root),
        language=args.language,
        timeout_seconds=args.timeout_seconds,
    )

    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {markdown_out}")
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
    return {
        "schema_name": SCHEMA_NAME,
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
    lines = [
        "# OCR Profile Experiment",
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
