#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from exam_bank.topic_packets import load_packet_taxonomy, normalize_packet_topic, normalize_paper_family


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Localize topic packet missing-answer-image exclusions before repair."
    )
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--routing", type=Path, default=Path("data/topic_routing/question_bank.topic_routing.v1.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("output"))
    parser.add_argument("--report", type=Path, required=True, help="Report path prefix, without .json/.md suffix.")
    args = parser.parse_args()

    report = build_report(
        question_bank_path=args.question_bank,
        packets_root=args.packets,
        taxonomy_path=args.taxonomy,
        routing_path=args.routing,
        artifact_root=args.artifact_root,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return 0


def build_report(
    *,
    question_bank_path: Path,
    packets_root: Path,
    taxonomy_path: Path,
    routing_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    question_bank = json.loads(question_bank_path.read_text(encoding="utf-8"))
    records = question_bank.get("questions") or []
    records_by_id = {str(record.get("question_id") or ""): record for record in records}
    taxonomy = load_packet_taxonomy(taxonomy_path)
    routing_records = _load_routing_records(routing_path)

    summary_path = packets_root / "topic_packet_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing_skips = [
        item for item in summary.get("skipped_records", [])
        if str(item.get("reason") or "") == "missing_answer_image"
    ]
    missing_ids = [str(item.get("question_id") or "") for item in missing_skips]
    missing_id_set = set(missing_ids)
    included = _load_included_records(packets_root)
    mark_scheme_image_index = _mark_scheme_image_index(artifact_root)

    analyses = [
        _analyze_missing_record(
            records_by_id[question_id],
            skip,
            taxonomy=taxonomy,
            artifact_root=artifact_root,
            routing_records=routing_records,
            mark_scheme_image_index=mark_scheme_image_index,
        )
        for question_id, skip in zip(missing_ids, missing_skips, strict=True)
        if question_id in records_by_id
    ]

    all_record_context = {
        question_id: _record_context(record, taxonomy=taxonomy, routing_records=routing_records)
        for question_id, record in records_by_id.items()
    }
    paper_table = _paper_impact_table(
        records_by_id=records_by_id,
        missing_id_set=missing_id_set,
        included=included,
        analyses=analyses,
        artifact_root=artifact_root,
        mark_scheme_image_index=mark_scheme_image_index,
    )

    failure_class_counts = Counter(analysis["failure_class"] for analysis in analyses)
    ranked_clusters = {
        "by_year": _rank_counts(analysis["year"] for analysis in analyses),
        "by_session": _rank_counts(analysis["session"] for analysis in analyses),
        "by_component": _rank_counts(analysis["component"] for analysis in analyses),
        "by_paper_family": _rank_counts(analysis["paper_family"] for analysis in analyses),
        "by_normalized_packet_family": _rank_counts(analysis["normalized_family"] for analysis in analyses),
        "by_normalized_topic": _rank_counts(analysis["normalized_topic"] for analysis in analyses),
        "by_source_question_paper_path": _rank_counts(analysis["source_question_paper_path"] for analysis in analyses),
        "by_source_mark_scheme_path": _rank_counts(analysis["source_mark_scheme_path"] for analysis in analyses),
        "by_expected_answer_image_path_pattern": _rank_counts(
            _path_pattern(analysis["expected_answer_image_path"]) for analysis in analyses
        ),
        "by_actual_answer_crop_directory": _rank_counts(
            analysis["actual_answer_crop_directory"] or "not_detected" for analysis in analyses
        ),
        "by_packet_output_family_topic": _rank_counts(
            f"{analysis['packet_output_family']}/{analysis['packet_output_topic']}" for analysis in analyses
        ),
        "by_mapping_status": _rank_counts(analysis["mapping_status"] for analysis in analyses),
        "by_validation_status": _rank_counts(analysis["validation_status"] for analysis in analyses),
        "by_failure_class": _rank_counts(analysis["failure_class"] for analysis in analyses),
    }

    top_papers = sorted(
        paper_table,
        key=lambda item: (int(item["missing_answer_image_records"]), float(item["missing_rate"])),
        reverse=True,
    )
    high_impact_samples = _sample_records(analyses, top_papers)
    clustering_answers = _clustering_answers(analyses, paper_table)
    recommendation = _repair_recommendation(analyses, paper_table)

    accounted = len(analyses)
    return {
        "schema_name": "exam_bank.topic_packet_missing_answer_concentration",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_bank_path": str(question_bank_path),
        "packets_root": str(packets_root),
        "summary_path": str(summary_path),
        "taxonomy_path": str(taxonomy_path),
        "routing_path": str(routing_path) if routing_path.exists() else "",
        "total_question_bank_records": len(records_by_id),
        "missing_answer_image_records": len(missing_ids),
        "records_analyzed": accounted,
        "analysis_accounting_ok": accounted == len(missing_ids),
        "classification_totals": dict(sorted(failure_class_counts.items())),
        "concentration_tables": ranked_clusters,
        "paper_impact_table": top_papers,
        "top_10_impacted_papers": top_papers[:10],
        "clustering_answers": clustering_answers,
        "samples": high_impact_samples,
        "repair_strategy_recommendation": recommendation,
        "detected_answer_crop_elsewhere_count": sum(1 for item in analyses if item["detected_existing_answer_crop_path"]),
        "true_segmentation_failure_count": sum(1 for item in analyses if item["failure_class"] == "B_segmentation_failed"),
        "path_linking_failure_count": sum(1 for item in analyses if item["failure_class"] == "C_crop_exists_linking_failed"),
        "records": analyses,
        "all_record_context_counts": {
            "included_records_from_manifests": len(included),
            "question_bank_records": len(all_record_context),
        },
    }


def _analyze_missing_record(
    record: dict[str, Any],
    skip: dict[str, Any],
    *,
    taxonomy: dict[str, Any],
    artifact_root: Path,
    routing_records: dict[str, dict[str, Any]],
    mark_scheme_image_index: list[str],
) -> dict[str, Any]:
    context = _record_context(record, taxonomy=taxonomy, routing_records=routing_records)
    expected_paths = _top_level_mark_scheme_paths(record)
    nested_candidates = _nested_mark_scheme_candidates(record)
    existing_top_level = _existing_paths(expected_paths, artifact_root)
    existing_nested = _existing_paths(nested_candidates, artifact_root)
    same_paper_candidates = _same_paper_crop_candidates(record, mark_scheme_image_index)
    detected_path = (existing_top_level + existing_nested + same_paper_candidates)[:1]
    source_ms = context["source_mark_scheme_path"]
    source_qp = context["source_question_paper_path"]
    source_ms_exists = bool(source_ms) and Path(source_ms).is_file()
    if not source_ms_exists:
        failure_class = "A_no_mark_scheme_source"
        why_missed = "No mark-scheme source PDF was recorded or found on disk."
    elif existing_top_level:
        failure_class = "C_crop_exists_linking_failed"
        why_missed = "A top-level answer crop path exists on disk but packet lookup still reported it missing."
    elif existing_nested:
        failure_class = "C_crop_exists_linking_failed"
        why_missed = (
            "Answer crop exists under nested mark_scheme_structure_detected.asset_identity.canonical_path, "
            "but top-level mark_scheme_image_path(s) are empty."
        )
    elif _legacy_field_candidates(record):
        failure_class = "E_builder_field_name_mismatch"
        why_missed = "Record has answer-like legacy fields that packet builder does not inspect."
    elif same_paper_candidates:
        failure_class = "D_legacy_id_mismatch"
        why_missed = "Answer crop exists for the same paper/question pattern but not under the current question ID path."
    else:
        failure_class = "B_segmentation_failed"
        why_missed = "Mark-scheme source exists, but no answer crop file was found at top-level or nested canonical paths."

    detected_existing = detected_path[0] if detected_path else ""
    expected_display = expected_paths[0] if expected_paths else (nested_candidates[0] if nested_candidates else "")
    actual_dir = str(Path(detected_existing).parent) if detected_existing else ""
    return {
        **context,
        "skip_reason": str(skip.get("reason") or ""),
        "expected_answer_image_path": expected_display,
        "top_level_answer_image_paths": expected_paths,
        "nested_candidate_answer_image_paths": nested_candidates,
        "detected_existing_answer_crop_path": detected_existing,
        "actual_answer_crop_directory": actual_dir,
        "source_question_paper_exists": bool(source_qp) and Path(source_qp).is_file(),
        "source_mark_scheme_exists": source_ms_exists,
        "packet_output_family": context["normalized_family"],
        "packet_output_topic": str(skip.get("assigned_topic_id") or context["normalized_topic"]),
        "failure_class": failure_class,
        "why_current_packet_builder_misses_it": why_missed,
    }


def _record_context(
    record: dict[str, Any],
    *,
    taxonomy: dict[str, Any],
    routing_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    question_id = str(record.get("question_id") or "")
    component = _component(record)
    normalization = normalize_packet_topic(
        component_code=component,
        current_family=record.get("paper_family"),
        raw_topic=record.get("topic"),
        taxonomy=taxonomy,
    )
    routing = routing_records.get(question_id, {})
    return {
        "question_id": question_id,
        "paper": str(record.get("paper") or ""),
        "year": _year(record),
        "session": _session(record),
        "component": component,
        "paper_family": normalize_paper_family(record.get("paper_family")),
        "raw_paper_family": str(record.get("paper_family") or ""),
        "raw_topic": str(record.get("topic") or ""),
        "normalized_family": normalization.expected_family if normalization.resolved else normalization.current_family,
        "normalized_topic": normalization.expected_topic if normalization.resolved else "",
        "normalization_status": normalization.status,
        "source_question_paper_path": str(notes.get("source_pdf") or ""),
        "source_mark_scheme_path": str(notes.get("mark_scheme_source_pdf") or ""),
        "mapping_status": _status(record, "mapping_status"),
        "validation_status": _status(record, "validation_status"),
        "routing_status": str(routing.get("status") or routing.get("routing_status") or ""),
    }


def _paper_impact_table(
    *,
    records_by_id: dict[str, dict[str, Any]],
    missing_id_set: set[str],
    included: dict[str, dict[str, Any]],
    analyses: list[dict[str, Any]],
    artifact_root: Path,
    mark_scheme_image_index: list[str],
) -> list[dict[str, Any]]:
    analyses_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for analysis in analyses:
        analyses_by_paper[analysis["paper"]].append(analysis)

    records_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records_by_id.values():
        records_by_paper[str(record.get("paper") or "")].append(record)

    rows = []
    for paper, paper_records in records_by_paper.items():
        if paper not in analyses_by_paper:
            continue
        missing_rows = analyses_by_paper[paper]
        total = len(paper_records)
        missing = len(missing_rows)
        included_count = sum(1 for record in paper_records if str(record.get("question_id") or "") in included)
        source_ms_paths = {row["source_mark_scheme_path"] for row in missing_rows if row["source_mark_scheme_path"]}
        source_exists = bool(source_ms_paths) and all(Path(path).is_file() for path in source_ms_paths)
        detected_paths = [
            path for record in paper_records
            for path in (
                _existing_paths(_top_level_mark_scheme_paths(record) + _nested_mark_scheme_candidates(record), artifact_root)
                or _same_paper_crop_candidates(record, mark_scheme_image_index)
            )
        ]
        detected_count = len(set(detected_paths))
        if detected_count == 0:
            crop_state = "no"
        elif detected_count < total:
            crop_state = "partial"
        else:
            crop_state = "yes"
        classes = Counter(row["failure_class"] for row in missing_rows)
        first = missing_rows[0]
        rows.append(
            {
                "paper": paper,
                "year": first["year"],
                "session": first["session"],
                "component": first["component"],
                "paper_family": first["paper_family"],
                "normalized_packet_family": first["normalized_family"],
                "total_question_bank_records": total,
                "included_records": included_count,
                "missing_answer_image_records": missing,
                "missing_rate": round(missing / total, 4) if total else 0,
                "source_mark_scheme_path_exists": source_exists,
                "source_mark_scheme_paths": sorted(source_ms_paths),
                "answer_crops_exist_for_paper": crop_state,
                "detected_answer_crop_count": detected_count,
                "likely_failure_class": classes.most_common(1)[0][0],
                "failure_class_counts": dict(sorted(classes.items())),
            }
        )
    return rows


def _sample_records(analyses: list[dict[str, Any]], top_papers: list[dict[str, Any]]) -> dict[str, Any]:
    by_paper = {row["paper"]: row for row in top_papers}
    highest_impact = []
    for paper in [row["paper"] for row in top_papers[:10]]:
        paper_rows = [row for row in analyses if row["paper"] == paper]
        highest_impact.extend(paper_rows[: max(1, min(3, 20 - len(highest_impact)))])
        if len(highest_impact) >= 20:
            break
    if len(highest_impact) < 20:
        highest_impact.extend(analyses[: 20 - len(highest_impact)])

    cluster_samples: dict[str, list[dict[str, Any]]] = {}
    cluster_counts = Counter((row["year"], row["component"]) for row in analyses)
    for (year, component), _count in cluster_counts.most_common(8):
        rows = [row for row in analyses if row["year"] == year and row["component"] == component]
        cluster_samples[f"{year}/{component}"] = [_sample_projection(row) for row in rows[:10]]

    return {
        "highest_impact_missing_records_overall": [_sample_projection(row) for row in highest_impact[:20]],
        "by_high_impact_year_component_cluster": cluster_samples,
        "top_paper_context": {
            paper: {
                "missing_answer_image_records": by_paper[paper]["missing_answer_image_records"],
                "likely_failure_class": by_paper[paper]["likely_failure_class"],
            }
            for paper in [row["paper"] for row in top_papers[:10]]
        },
    }


def _sample_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "year": row["year"],
        "component": row["component"],
        "family": row["paper_family"],
        "normalized_family": row["normalized_family"],
        "normalized_topic": row["normalized_topic"],
        "source_mark_scheme_path": row["source_mark_scheme_path"],
        "expected_answer_image_path": row["expected_answer_image_path"],
        "detected_existing_answer_crop_path": row["detected_existing_answer_crop_path"],
        "why_current_packet_builder_misses_it": row["why_current_packet_builder_misses_it"],
    }


def _clustering_answers(analyses: list[dict[str, Any]], paper_table: list[dict[str, Any]]) -> dict[str, Any]:
    years = Counter(row["year"] for row in analyses)
    components = Counter(row["component"] for row in analyses)
    families = Counter(row["normalized_family"] for row in analyses)
    old_legacy = sum(1 for row in analyses if row["component"] in {"01", "03", "04", "06"})
    modern = sum(1 for row in analyses if row["year"].isdigit() and int(row["year"]) >= 2021)
    old = sum(1 for row in analyses if row["year"].isdigit() and 2008 <= int(row["year"]) <= 2020)
    all_missing_papers = [
        row for row in paper_table
        if row["missing_answer_image_records"] == row["total_question_bank_records"]
    ]
    naming_linking = sum(1 for row in analyses if row["failure_class"] == "C_crop_exists_linking_failed")
    top_dirs = Counter(row["actual_answer_crop_directory"] or "not_detected" for row in analyses).most_common(10)
    return {
        "concentrated_in_2008_2020": old,
        "also_modern_2021_2025": modern,
        "top_years": dict(years.most_common(10)),
        "top_components": dict(components.most_common(10)),
        "top_normalized_families": dict(families.most_common()),
        "legacy_component_01_03_04_06_count": old_legacy,
        "top_actual_crop_directories": dict(top_dirs),
        "different_naming_or_linking_concentration_count": naming_linking,
        "papers_where_all_records_missing_from_packet_lookup_count": len(all_missing_papers),
        "top_papers_where_all_records_missing_from_packet_lookup": all_missing_papers[:20],
    }


def _repair_recommendation(analyses: list[dict[str, Any]], paper_table: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(row["failure_class"] for row in analyses)
    nested_link_count = sum(
        1 for row in analyses
        if row["failure_class"] == "C_crop_exists_linking_failed"
        and not row["top_level_answer_image_paths"]
        and row["nested_candidate_answer_image_paths"]
    )
    recommendation = (
        "Add a narrow mark-scheme asset linking pass that promotes existing nested "
        "notes.mark_scheme_structure_detected.asset_identity.canonical_path values into "
        "top-level mark_scheme_image_path / mark_scheme_image_paths when the file exists. "
        "Handle the remaining true segmentation failures separately."
    )
    return {
        "dominant_failure_class": classes.most_common(1)[0][0] if classes else "",
        "failure_class_counts": dict(sorted(classes.items())),
        "nested_canonical_crop_linking_count": nested_link_count,
        "recommended_smallest_next_repair": recommendation,
        "not_recommended_for_this_cluster": [
            "broad resolver rewrite",
            "topic routing refresh",
            "packet regeneration before question_bank asset links are repaired",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Topic Packet Missing Answer Concentration - 2026-06-27",
        "",
        "## Summary",
        "",
        f"- Missing-answer-image records analyzed: {report['records_analyzed']}",
        f"- Accounting OK: {report['analysis_accounting_ok']}",
        f"- Path/linking failures with detected crops: {report['path_linking_failure_count']}",
        f"- True segmentation failures: {report['true_segmentation_failure_count']}",
        "",
        "## Classification Totals",
        "",
    ]
    for key, value in report["classification_totals"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Impacted Papers", "", "| Paper | Year | Component | Family | Missing | Total | Rate | Crops | Class |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |"])
    for row in report["top_10_impacted_papers"]:
        lines.append(
            f"| {row['paper']} | {row['year']} | {row['component']} | {row['paper_family']} | "
            f"{row['missing_answer_image_records']} | {row['total_question_bank_records']} | "
            f"{row['missing_rate']:.2%} | {row['answer_crops_exist_for_paper']} ({row['detected_answer_crop_count']}) | "
            f"{row['likely_failure_class']} |"
        )
    lines.extend(["", "## Top Years", ""])
    for row in report["concentration_tables"]["by_year"][:10]:
        lines.append(f"- {row['key']}: {row['count']}")
    lines.extend(["", "## Top Components", ""])
    for row in report["concentration_tables"]["by_component"][:10]:
        lines.append(f"- {row['key']}: {row['count']}")
    lines.extend(["", "## Families", ""])
    for row in report["concentration_tables"]["by_normalized_packet_family"]:
        lines.append(f"- {row['key']}: {row['count']}")
    lines.extend(["", "## Clustering Answers", ""])
    ca = report["clustering_answers"]
    lines.append(f"- 2008-2020 records: {ca['concentrated_in_2008_2020']}")
    lines.append(f"- 2021-2025 records: {ca['also_modern_2021_2025']}")
    lines.append(f"- Legacy components 01/03/04/06: {ca['legacy_component_01_03_04_06_count']}")
    lines.append(
        "- Papers where all records are missing from packet lookup: "
        f"{ca['papers_where_all_records_missing_from_packet_lookup_count']}"
    )
    lines.append(
        "- Records where crops exist but use an unlinked/nested path convention: "
        f"{ca['different_naming_or_linking_concentration_count']}"
    )
    lines.extend(["", "## Top Actual Crop Directories", ""])
    for directory, count in ca["top_actual_crop_directories"].items():
        lines.append(f"- `{directory}`: {count}")
    lines.extend(["", "## Sample Missing Records", ""])
    for sample in report["samples"]["highest_impact_missing_records_overall"][:20]:
        lines.append(
            f"- {sample['question_id']}: expected `{sample['expected_answer_image_path']}`, "
            f"detected `{sample['detected_existing_answer_crop_path']}`; "
            f"{sample['why_current_packet_builder_misses_it']}"
        )
    lines.extend(["", "## Repair Strategy Recommendation", ""])
    rec = report["repair_strategy_recommendation"]
    lines.append(f"- Dominant failure class: {rec['dominant_failure_class']}")
    lines.append(f"- Nested canonical crop linking count: {rec['nested_canonical_crop_linking_count']}")
    lines.append(f"- Recommendation: {rec['recommended_smallest_next_repair']}")
    lines.append("")
    return "\n".join(lines)


def _load_included_records(packets_root: Path) -> dict[str, dict[str, Any]]:
    included = {}
    for manifest_path in sorted(packets_root.glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("included_records") or []:
            question_id = str(item.get("question_id") or "")
            included[question_id] = {
                "manifest_path": str(manifest_path),
                "family": str(manifest.get("paper_family") or ""),
                "topic": str(manifest.get("topic_id") or ""),
                "section": str(item.get("section") or ""),
            }
    return included


def _load_routing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    return {str(item.get("question_id") or ""): item for item in records if isinstance(item, dict)}


def _top_level_mark_scheme_paths(record: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ["mark_scheme_image_path", "canonical_mark_scheme_artifact"]:
        value = str(record.get(key) or "").strip()
        if value:
            values.append(value)
    raw_paths = record.get("mark_scheme_image_paths")
    if isinstance(raw_paths, list):
        values.extend(str(value).strip() for value in raw_paths if str(value or "").strip())
    return _dedupe(values)


def _nested_mark_scheme_candidates(record: dict[str, Any]) -> list[str]:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    structure = notes.get("mark_scheme_structure_detected")
    candidates: list[str] = []
    if isinstance(structure, dict):
        identity = structure.get("asset_identity")
        if isinstance(identity, dict) and identity.get("canonical_path"):
            candidates.append(str(identity["canonical_path"]))
    return _dedupe(candidates)


def _legacy_field_candidates(record: dict[str, Any]) -> list[str]:
    keys = [
        "answer_image_path",
        "answer_image_paths",
        "solution_image_path",
        "solution_image_paths",
        "markscheme_image",
        "markscheme_image_path",
        "markscheme_image_paths",
    ]
    return [key for key in keys if record.get(key)]


def _existing_paths(values: Iterable[str], artifact_root: Path) -> list[str]:
    found = []
    for value in values:
        path = Path(value)
        candidates = [path] if path.is_absolute() else [artifact_root / path, path]
        for candidate in candidates:
            if candidate.is_file():
                found.append(str(candidate))
                break
    return _dedupe(found)


def _mark_scheme_image_index(artifact_root: Path) -> list[str]:
    paths: list[str] = []
    for path in artifact_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        text = str(path).lower()
        if any(token in text for token in ["mark_scheme", "markscheme", "_ms_", "/ms/"]):
            paths.append(str(path))
    return paths


def _same_paper_crop_candidates(record: dict[str, Any], mark_scheme_image_index: list[str]) -> list[str]:
    # Conservative legacy-ID check: only count files that share paper and q-number tokens.
    paper = str(record.get("paper") or "").lower()
    question_number = str(record.get("question_number") or "").strip()
    if not paper or not question_number:
        return []
    q_tokens = {f"q{question_number.zfill(2)}", f"q{question_number}"}
    found = []
    for path in mark_scheme_image_index:
        text = path.lower()
        if paper in text and any(token in text for token in q_tokens) and ("ms" in text or "mark" in text):
            found.append(path)
            if len(found) >= 5:
                break
    return _dedupe(found)


def _rank_counts(values: Iterable[str]) -> list[dict[str, Any]]:
    counts = Counter(str(value or "unknown") for value in values)
    return [{"key": key, "count": count} for key, count in counts.most_common()]


def _status(record: dict[str, Any], key: str) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    return str(record.get(key) or notes.get(key) or "").strip() or "missing"


def _year(record: dict[str, Any]) -> str:
    text = str(record.get("paper") or record.get("question_id") or "")
    match = re.search(r"(\d{2})(?!.*\d)", text)
    if not match:
        return "unknown"
    year = int(match.group(1))
    return str(2000 + year if year < 70 else 1900 + year)


def _session(record: dict[str, Any]) -> str:
    text = str(record.get("paper") or record.get("question_id") or "").lower()
    for session in ["spring", "summer", "winter", "autumn"]:
        if session in text:
            return session
    return "unknown"


def _component(record: dict[str, Any]) -> str:
    notes = record.get("notes") if isinstance(record.get("notes"), dict) else {}
    for value in [notes.get("source_paper_code"), record.get("source_paper_code"), record.get("component")]:
        if value:
            return _normalize_component(value)
    paper = str(record.get("paper") or record.get("question_id") or "")
    match = re.match(r"(\d{2})", paper)
    return _normalize_component(match.group(1)) if match else ""


def _normalize_component(value: Any) -> str:
    text = str(value or "").strip().removeprefix("p")
    match = re.search(r"\d+", text)
    if not match:
        return ""
    code = match.group(0)
    return code.zfill(2) if len(code) == 1 else code


def _path_pattern(value: str) -> str:
    if not value:
        return "empty_top_level_path"
    return re.sub(r"q\d+", "qNN", value)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "records_analyzed": report["records_analyzed"],
        "analysis_accounting_ok": report["analysis_accounting_ok"],
        "classification_totals": report["classification_totals"],
        "top_10_impacted_papers": report["top_10_impacted_papers"],
        "recommendation": report["repair_strategy_recommendation"]["recommended_smallest_next_repair"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
