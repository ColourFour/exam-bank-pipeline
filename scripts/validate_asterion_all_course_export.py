from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json


VALID_COURSE_IDS = {"p1", "p3", "m1", "s1"}
PAPER_FAMILY_TO_COURSE_ID = {
    "p1": "p1",
    "p3": "p3",
    "p4": "m1",
    "m1": "m1",
    "p5": "s1",
    "s1": "s1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the all-course Asterion catalog/runtime export contract."
    )
    parser.add_argument("--question-bank", default="output/json/question_bank.json")
    parser.add_argument(
        "--catalog",
        default="output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json",
    )
    parser.add_argument(
        "--runtime",
        default="output/asterion/exports/latest/asterion_question_bank_v1.json",
    )
    parser.add_argument(
        "--content-lab",
        default="output/asterion/exports/latest/asterion_content_lab_candidates_v1.json",
    )
    parser.add_argument(
        "--topic-routing",
        default="output/json/question_bank.topic_routing.v1.json",
    )
    parser.add_argument("--artifact-root", default="output")
    parser.add_argument(
        "--student-runtime-target-rate",
        type=float,
        default=0.15,
        help="Minimum reviewed student-runtime-safe share required for P1, M1, and S1.",
    )
    parser.add_argument(
        "--student-runtime-topic-target-rate",
        type=float,
        default=0.35,
        help="Report-only minimum reviewed learning-runtime-safe share for each named P1, M1, and S1 topic.",
    )
    parser.add_argument(
        "--enforce-student-runtime-targets",
        action="store_true",
        help="Fail when P1/M1/S1 learning-runtime targets are not met. Defaults off because non-P3 advisory/image gates are not reviewed learning runtime.",
    )
    parser.add_argument(
        "--output",
        default="output/asterion/exports/latest/asterion_all_course_export_validation_2026_06_01.json",
    )
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_format_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "contains_newlines": "\n" in text,
        "appears_indented": "\n  " in text,
    }


def course_id_for_paper_family(paper_family: Any) -> str | None:
    if not isinstance(paper_family, str):
        return None
    return PAPER_FAMILY_TO_COURSE_ID.get(paper_family.lower())


def course_counts_from_questions(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(record.get("course_id", "")) for record in records)


def source_course_counts(question_bank: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in question_bank.get("questions", []):
        course_id = course_id_for_paper_family(record.get("paper_family"))
        if course_id:
            counts[course_id] += 1
    return counts


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key, "")) for record in records).items()))


def missing_image_report(
    records: list[dict[str, Any]], artifact_root: Path
) -> dict[str, Any]:
    missing_reference_counts: Counter[str] = Counter()
    missing_file_counts: Counter[str] = Counter()
    sample_missing_files: list[dict[str, str]] = []
    fields = ("question_image_path", "mark_scheme_image_path")

    for record in records:
        question_id = str(record.get("question_id", ""))
        for field in fields:
            path_value = record.get(field)
            if not path_value:
                missing_reference_counts[field] += 1
                continue
            path = Path(str(path_value))
            candidate = path if path.is_absolute() else artifact_root / path
            if not candidate.exists():
                missing_file_counts[field] += 1
                if len(sample_missing_files) < 20:
                    sample_missing_files.append(
                        {
                            "question_id": question_id,
                            "field": field,
                            "path": str(path_value),
                        }
                    )

    return {
        "missing_reference_counts": dict(sorted(missing_reference_counts.items())),
        "missing_file_counts": dict(sorted(missing_file_counts.items())),
        "sample_missing_files": sample_missing_files,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "question_bank": Path(args.question_bank),
        "catalog": Path(args.catalog),
        "runtime": Path(args.runtime),
        "content_lab": Path(args.content_lab),
        "topic_routing": Path(args.topic_routing),
    }
    artifact_root = Path(args.artifact_root)
    errors: list[str] = []
    warnings: list[str] = []

    missing_inputs = [name for name, path in paths.items() if not path.exists()]
    if missing_inputs:
        return {
            "ok": False,
            "errors": [f"Missing input file: {name}={paths[name]}" for name in missing_inputs],
            "warnings": [],
        }

    question_bank = load_json(paths["question_bank"])
    catalog = load_json(paths["catalog"])
    runtime = load_json(paths["runtime"])
    content_lab = load_json(paths["content_lab"])
    topic_routing = load_json(paths["topic_routing"])

    catalog_questions = catalog.get("questions", [])
    runtime_questions = runtime.get("questions", [])
    candidates = content_lab.get("candidates", [])
    topic_records = topic_routing.get("records", {})

    expected_schemas = {
        "catalog": "asterion.exam_bank_catalog",
        "runtime": "asterion.question_bank",
        "content_lab": "asterion.content_lab_candidates",
        "topic_routing": "exam_bank.topic_routing_sidecar",
    }
    observed_schemas = {
        "catalog": catalog.get("schema_name"),
        "runtime": runtime.get("schema_name"),
        "content_lab": content_lab.get("schema_name"),
        "topic_routing": topic_routing.get("schema_name"),
    }
    for name, expected in expected_schemas.items():
        if observed_schemas[name] != expected:
            errors.append(f"{name} schema_name is {observed_schemas[name]!r}; expected {expected!r}.")

    if catalog.get("record_count") != len(catalog_questions):
        errors.append("Catalog record_count does not match questions length.")
    if runtime.get("record_count") != len(runtime_questions):
        errors.append("Runtime record_count does not match questions length.")
    if content_lab.get("record_count") != len(candidates):
        errors.append("Content Lab record_count does not match candidates length.")
    if topic_routing.get("record_count") != len(topic_records):
        errors.append("Topic-routing record_count does not match records length.")

    source_counts = source_course_counts(question_bank)
    catalog_course_counts = course_counts_from_questions(catalog_questions)
    runtime_course_counts = course_counts_from_questions(runtime_questions)
    catalog_safe_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("student_runtime_safe") is True
    )
    catalog_reviewed_safe_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("student_runtime_safe") is True and record.get("review_status") == "reviewed"
    )
    candidate_course_counts = Counter(
        course_id
        for candidate in candidates
        if (course_id := course_id_for_paper_family(candidate.get("paper_family")))
    )
    catalog_visible_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("catalog_visible") is True
    )
    image_practice_safe_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("image_practice_safe") is True
    )
    advisory_topic_filter_ok_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("advisory_topic_filter_ok") is True
    )
    reviewed_topic_filter_safe_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("reviewed_topic_filter_safe") is True
    )
    learning_runtime_safe_counts = Counter(
        str(record.get("course_id", ""))
        for record in catalog_questions
        if record.get("learning_runtime_safe") is True
    )

    catalog_course_ids = set(catalog_course_counts)
    invalid_catalog_course_ids = sorted(catalog_course_ids - VALID_COURSE_IDS)
    invalid_runtime_course_ids = sorted(set(runtime_course_counts) - VALID_COURSE_IDS)
    if invalid_catalog_course_ids:
        errors.append(f"Catalog has invalid course_id values: {invalid_catalog_course_ids}.")
    if invalid_runtime_course_ids:
        errors.append(f"Runtime has invalid course_id values: {invalid_runtime_course_ids}.")

    catalog_metadata_courses = {
        str(course.get("course_id")): course for course in catalog.get("courses", [])
    }
    catalog_metadata_components = {
        str(component.get("course_id")): component for component in catalog.get("components", [])
    }
    for course_id, count in sorted(source_counts.items()):
        if count > 0 and course_id not in catalog_metadata_courses:
            errors.append(f"Source data has {course_id} records but catalog courses metadata omits it.")
        if count > 0 and course_id not in catalog_metadata_components:
            errors.append(
                f"Source data has {course_id} records but catalog components metadata omits it."
            )

    for course_id, count in sorted(catalog_course_counts.items()):
        metadata = catalog_metadata_courses.get(course_id)
        component = catalog_metadata_components.get(course_id)
        if metadata and metadata.get("record_count") != count:
            errors.append(f"Catalog courses metadata count mismatch for {course_id}.")
        if component and component.get("record_count") != count:
            errors.append(f"Catalog components metadata count mismatch for {course_id}.")
        if metadata and metadata.get("student_runtime_safe_record_count") != catalog_safe_counts[course_id]:
            errors.append(f"Catalog runtime-safe metadata count mismatch for {course_id}.")
        if component and component.get("student_runtime_safe_record_count") != catalog_safe_counts[course_id]:
            errors.append(f"Catalog component runtime-safe count mismatch for {course_id}.")

    unsafe_runtime_records = [
        str(record.get("question_id", ""))
        for record in runtime_questions
        if record.get("student_runtime_safe") is not True or record.get("review_status") != "reviewed"
    ]
    if unsafe_runtime_records:
        errors.append(
            "Runtime contains records that are not both student_runtime_safe=true and review_status=reviewed."
        )

    if runtime_course_counts != catalog_reviewed_safe_counts:
        errors.append("Runtime course counts do not match reviewed student-runtime-safe catalog counts.")

    unsafe_catalog_topic_routes = [
        str(record.get("question_id", ""))
        for record in catalog_questions
        if record.get("topic_id")
        and isinstance(record.get("topic_route"), dict)
        and record["topic_route"].get("filter_ok") is not True
    ]
    if unsafe_catalog_topic_routes:
        errors.append("Catalog has topic_id values backed by non-filterable topic routes.")

    unsafe_runtime_topic_routes = [
        str(record.get("question_id", ""))
        for record in runtime_questions
        if record.get("topic_id")
        and isinstance(record.get("topic_route"), dict)
        and record["topic_route"].get("filter_ok") is not True
    ]
    if unsafe_runtime_topic_routes:
        errors.append("Runtime has topic_id values backed by non-filterable topic routes.")

    runtime_records_with_candidate_fields = [
        str(record.get("question_id", ""))
        for record in runtime_questions
        if "candidate_id" in record or "generation_gate" in record or "possible_content_lab_roles" in record
    ]
    if runtime_records_with_candidate_fields:
        errors.append("Runtime records contain Content Lab candidate-only fields.")
    runtime_content_lab_ids = [
        str(record.get("question_id", ""))
        for record in runtime_questions
        if str(record.get("question_id", "")).startswith("content_lab_")
    ]
    if runtime_content_lab_ids:
        errors.append("Runtime contains content_lab_* question ids.")

    candidate_ids = {str(candidate.get("candidate_id", "")) for candidate in candidates}
    candidate_ids_in_runtime = sorted(candidate_ids & {str(record.get("question_id", "")) for record in runtime_questions})
    if candidate_ids_in_runtime:
        errors.append("Runtime question ids overlap Content Lab candidate ids.")

    p3_runtime_count = runtime_course_counts.get("p3", 0)
    p3_catalog_count = catalog_course_counts.get("p3", 0)
    min_expected_p3 = max(1, int(p3_catalog_count * 0.1)) if p3_catalog_count else 0
    if p3_catalog_count and p3_runtime_count < min_expected_p3:
        errors.append(
            f"P3 runtime-safe count is {p3_runtime_count}, below collapse guard {min_expected_p3}."
        )

    target_course_ids = ("p1", "m1", "s1")
    target_rate = float(args.student_runtime_target_rate)
    runtime_target_report: dict[str, dict[str, Any]] = {}
    for course_id in target_course_ids:
        total = catalog_course_counts.get(course_id, 0)
        runtime_safe_count = runtime_course_counts.get(course_id, 0)
        minimum = math.ceil(total * target_rate) if total else 0
        rate = runtime_safe_count / total if total else 0.0
        target_met = runtime_safe_count >= minimum
        runtime_target_report[course_id] = {
            "catalog_count": total,
            "runtime_safe_count": runtime_safe_count,
            "target_rate": target_rate,
            "minimum_runtime_safe_count": minimum,
            "runtime_safe_rate": round(rate, 6),
            "target_met": target_met,
        }
        if not target_met and args.enforce_student_runtime_targets:
            errors.append(
                f"{course_id} runtime-safe count is {runtime_safe_count}/{total} "
                f"({rate:.2%}), below target {minimum}/{total} ({target_rate:.2%})."
            )

    topic_target_rate = float(args.student_runtime_topic_target_rate)
    topic_totals: Counter[str] = Counter()
    topic_runtime_safe_counts: Counter[str] = Counter()
    missing_topic_counts: Counter[str] = Counter()
    missing_topic_runtime_safe_counts: Counter[str] = Counter()
    for record in catalog_questions:
        course_id = str(record.get("course_id", ""))
        if course_id not in target_course_ids:
            continue
        topic_id = str(record.get("topic_id") or "").strip()
        runtime_safe = record.get("student_runtime_safe") is True and record.get("review_status") == "reviewed"
        if not topic_id:
            missing_topic_counts[course_id] += 1
            if runtime_safe:
                missing_topic_runtime_safe_counts[course_id] += 1
            continue
        topic_totals[topic_id] += 1
        if runtime_safe:
            topic_runtime_safe_counts[topic_id] += 1

    topic_target_report: dict[str, dict[str, Any]] = {}
    for topic_id, total in sorted(topic_totals.items()):
        runtime_safe_count = topic_runtime_safe_counts[topic_id]
        minimum = math.ceil(total * topic_target_rate) if total else 0
        rate = runtime_safe_count / total if total else 0.0
        target_met = runtime_safe_count >= minimum
        topic_target_report[topic_id] = {
            "catalog_count": total,
            "runtime_safe_count": runtime_safe_count,
            "target_rate": topic_target_rate,
            "minimum_runtime_safe_count": minimum,
            "runtime_safe_rate": round(rate, 6),
            "target_met": target_met,
        }
        if not target_met and args.enforce_student_runtime_targets:
            errors.append(
                f"{topic_id} runtime-safe count is {runtime_safe_count}/{total} "
                f"({rate:.2%}), below topic target {minimum}/{total} ({topic_target_rate:.2%})."
            )

    if missing_topic_counts:
        warnings.append("Catalog has P1/M1/S1 records without topic_id; see student_runtime_missing_topic_counts.")
    if not args.enforce_student_runtime_targets:
        warnings.append("P1/M1/S1 learning-runtime targets are report-only; non-P3 image/advisory records are not reviewed learning runtime.")

    catalog_image_report = missing_image_report(catalog_questions, artifact_root)
    runtime_image_report = missing_image_report(runtime_questions, artifact_root)
    if runtime_image_report["missing_reference_counts"] or runtime_image_report["missing_file_counts"]:
        errors.append("Runtime has missing canonical image references or files.")
    if catalog_image_report["missing_reference_counts"] or catalog_image_report["missing_file_counts"]:
        warnings.append("Catalog has missing canonical image references or files; see image_reference_report.")

    topic_paper_families = Counter()
    for record in topic_records.values():
        course_id = course_id_for_paper_family(record.get("paper_family"))
        if course_id:
            topic_paper_families[course_id] += 1
        else:
            warnings.append("Topic routing contains a record with unmapped paper_family.")
            break

    format_reports = {
        name: json_format_report(path)
        for name, path in paths.items()
        if name != "question_bank"
    }
    for name, format_report in format_reports.items():
        if not format_report["contains_newlines"] or not format_report["appears_indented"]:
            warnings.append(f"{name} JSON does not appear to be indented/multi-line.")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "paths": {name: str(path) for name, path in paths.items()},
        "schemas": observed_schemas,
        "record_counts": {
            "source_question_bank": len(question_bank.get("questions", [])),
            "catalog": len(catalog_questions),
            "runtime": len(runtime_questions),
            "content_lab_candidates": len(candidates),
            "topic_routing": len(topic_records),
        },
        "source_course_counts": dict(sorted(source_counts.items())),
        "catalog_course_counts": dict(sorted(catalog_course_counts.items())),
        "catalog_runtime_safe_counts": dict(sorted(catalog_safe_counts.items())),
        "catalog_reviewed_runtime_safe_counts": dict(sorted(catalog_reviewed_safe_counts.items())),
        "safety_level_counts": {
            "catalog_visible": dict(sorted(catalog_visible_counts.items())),
            "image_practice_safe": dict(sorted(image_practice_safe_counts.items())),
            "advisory_topic_filter_ok": dict(sorted(advisory_topic_filter_ok_counts.items())),
            "reviewed_topic_filter_safe": dict(sorted(reviewed_topic_filter_safe_counts.items())),
            "learning_runtime_safe": dict(sorted(learning_runtime_safe_counts.items())),
        },
        "runtime_course_counts": dict(sorted(runtime_course_counts.items())),
        "catalog_review_status_counts": count_by(catalog_questions, "review_status"),
        "runtime_review_status_counts": count_by(runtime_questions, "review_status"),
        "content_lab_candidate_course_counts": dict(sorted(candidate_course_counts.items())),
        "content_lab_candidate_review_status_counts": count_by(candidates, "review_status"),
        "content_lab_generation_gate_status_counts": {
            status: count
            for status, count in sorted(
                Counter(
                    str(candidate.get("generation_gate", {}).get("status", ""))
                    for candidate in candidates
                ).items()
            )
        },
        "content_lab_candidate_source_question_overlap_with_runtime": len(
            {str(candidate.get("question_id", "")) for candidate in candidates}
            & {str(record.get("question_id", "")) for record in runtime_questions}
        ),
        "p3_behavior": {
            "catalog_count": p3_catalog_count,
            "runtime_safe_count": p3_runtime_count,
            "collapse_guard_minimum": min_expected_p3,
            "appears_preserved": p3_runtime_count >= min_expected_p3,
        },
        "student_runtime_target": runtime_target_report,
        "student_runtime_topic_target": topic_target_report,
        "student_runtime_missing_topic_counts": {
            course_id: {
                "catalog_count": missing_topic_counts[course_id],
                "runtime_safe_count": missing_topic_runtime_safe_counts[course_id],
            }
            for course_id in sorted(missing_topic_counts)
        },
        "all_course_runtime_gate_check": {
            "target_course_ids": list(target_course_ids),
            "target_rate": target_rate,
            "target_met": all(row["target_met"] for row in runtime_target_report.values()),
            "topic_target_rate": topic_target_rate,
            "topic_target_met": all(row["target_met"] for row in topic_target_report.values()),
        },
        "image_reference_report": {
            "catalog": catalog_image_report,
            "runtime": runtime_image_report,
        },
        "topic_routing_course_counts": dict(sorted(topic_paper_families.items())),
        "topic_routing_metadata": {
            "model": topic_routing.get("model"),
            "run_summary": topic_routing.get("metadata", {}).get("run_summary", {}),
        },
        "json_format": format_reports,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate(args)
    output = Path(args.output)
    write_atomic_json(report, output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
