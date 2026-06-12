from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .topic_routing_artifact import (
    DEFAULT_LOCAL_SIDECAR_PATH,
    file_sha256,
    verify_topic_routing_artifact,
)


DEFAULT_CATALOG_PATH = Path("output/asterion/exports/latest/asterion_exam_bank_catalog_v1.json")
DEFAULT_RUNTIME_PATH = Path("output/asterion/exports/latest/asterion_question_bank_v1.json")
DEFAULT_CONTENT_LAB_PATH = Path("output/asterion/exports/latest/asterion_content_lab_candidates_v1.json")
DEFAULT_VALIDATION_REPORT_PATH = Path("/tmp/asterion_export_release_provenance_pr15_validation.json")
DEFAULT_EXPECTED_PROVENANCE_PATH = Path("reports/asterion_export_release_provenance_pr15_2026_06_11.json")
DEFAULT_OUTPUT_PATH = Path("reports/asterion_export_release_manifest_pr16_2026_06_11.json")

ARTIFACT_KEYS = {
    "catalog": DEFAULT_CATALOG_PATH,
    "student_runtime": DEFAULT_RUNTIME_PATH,
    "content_lab_candidates": DEFAULT_CONTENT_LAB_PATH,
}


class AsterionReleaseBundleError(RuntimeError):
    pass


def file_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise AsterionReleaseBundleError(f"Missing release artifact: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_asterion_release_manifest(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    validation_report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
    expected_provenance_path: str | Path | None = DEFAULT_EXPECTED_PROVENANCE_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    runtime_path: str | Path = DEFAULT_RUNTIME_PATH,
    content_lab_path: str | Path = DEFAULT_CONTENT_LAB_PATH,
    refresh_expected: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    validation_report_path = Path(validation_report_path)
    if not validation_report_path.exists():
        raise AsterionReleaseBundleError(f"Missing validation report: {validation_report_path}")
    validation_report = _read_json_object(validation_report_path)
    if validation_report.get("ok") is not True:
        raise AsterionReleaseBundleError("Validation report is not ok:true.")

    sidecar_report = verify_topic_routing_artifact()
    artifacts = {
        "catalog": file_report(catalog_path),
        "student_runtime": file_report(runtime_path),
        "content_lab_candidates": file_report(content_lab_path),
    }
    validation_artifact = file_report(validation_report_path)
    counts = _counts_from_validation(validation_report)

    manifest = {
        "schema_name": "exam_bank.asterion_export_release_manifest",
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "release_inputs": {
            "durable_sidecar": {
                "path": sidecar_report["paths"]["durable_sidecar"],
                "sha256": sidecar_report["sha256"]["durable_sidecar"],
            },
            "local_sidecar": {
                "path": sidecar_report["paths"]["local_sidecar"],
                "sha256": sidecar_report["sha256"]["local_sidecar"],
                "matches_durable_sidecar": sidecar_report["sha256"]["local_matches_durable"],
                "ignored_local_output": True,
            },
        },
        "export_artifacts": artifacts,
        "validation": {
            "path": str(validation_report_path),
            "sha256": validation_artifact["sha256"],
            "size_bytes": validation_artifact["size_bytes"],
            "ok": True,
            "warnings": validation_report.get("warnings", []),
        },
        "counts": counts,
        "auto_grade_eligibility_changed": False,
        "asterion_runtime_behavior_changed": False,
        "p1_m1_s1_became_student_facing": counts["non_p3_runtime_records"] > 0,
        "handoff": {
            "export_json_files_remain_ignored_generated": True,
            "manifest_is_tracked_release_evidence": True,
            "deployment_must_fetch_or_receive_files_matching_these_hashes": True,
        },
    }

    expected = _load_expected_provenance(expected_provenance_path)
    if expected and not refresh_expected:
        _verify_expected_hashes(manifest, expected)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_asterion_release_manifest(
    *,
    manifest_path: str | Path,
    exports_dir: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    exports_dir = Path(exports_dir)
    if not manifest_path.exists():
        raise AsterionReleaseBundleError(f"Missing release manifest: {manifest_path}")
    if not exports_dir.exists():
        raise AsterionReleaseBundleError(f"Missing exports directory: {exports_dir}")

    manifest = _read_json_object(manifest_path)
    export_artifacts = manifest.get("export_artifacts")
    if not isinstance(export_artifacts, dict):
        raise AsterionReleaseBundleError("Release manifest is missing export_artifacts.")

    verified: dict[str, dict[str, Any]] = {}
    for role, info in sorted(export_artifacts.items()):
        if not isinstance(info, dict):
            raise AsterionReleaseBundleError(f"Invalid export artifact entry: {role}")
        manifest_path_value = info.get("path")
        expected_sha = info.get("sha256")
        expected_size = info.get("size_bytes")
        if not manifest_path_value or not expected_sha or expected_size is None:
            raise AsterionReleaseBundleError(f"Incomplete export artifact entry: {role}")
        artifact_path = exports_dir / Path(str(manifest_path_value)).name
        if not artifact_path.exists():
            raise AsterionReleaseBundleError(f"Missing export artifact for {role}: {artifact_path}")
        actual_sha = file_sha256(artifact_path)
        actual_size = artifact_path.stat().st_size
        if actual_sha != expected_sha:
            raise AsterionReleaseBundleError(f"{role} SHA-256 mismatch: {actual_sha} != {expected_sha}")
        if actual_size != expected_size:
            raise AsterionReleaseBundleError(f"{role} size mismatch: {actual_size} != {expected_size}")
        verified[role] = {
            "path": str(artifact_path),
            "sha256": actual_sha,
            "size_bytes": actual_size,
        }

    required = {"catalog", "student_runtime", "content_lab_candidates"}
    missing_roles = sorted(required - set(verified))
    if missing_roles:
        raise AsterionReleaseBundleError(f"Release manifest missing required artifact roles: {missing_roles}")

    return {
        "ok": True,
        "manifest": str(manifest_path),
        "exports_dir": str(exports_dir),
        "verified_artifacts": verified,
        "counts": manifest.get("counts", {}),
    }


def _counts_from_validation(validation_report: dict[str, Any]) -> dict[str, int]:
    record_counts = validation_report.get("record_counts", {})
    runtime_course_counts = validation_report.get("runtime_course_counts", {})
    safety_counts = validation_report.get("safety_level_counts", {})
    advisory_counts = safety_counts.get("advisory_topic_filter_ok", {})
    learning_counts = safety_counts.get("learning_runtime_safe", {})
    return {
        "catalog_records": int(record_counts.get("catalog") or 0),
        "student_runtime_records": int(record_counts.get("runtime") or 0),
        "p3_runtime_records": int(runtime_course_counts.get("p3") or 0),
        "non_p3_runtime_records": sum(
            int(count) for course_id, count in runtime_course_counts.items() if course_id != "p3"
        ),
        "content_lab_candidates": int(record_counts.get("content_lab_candidates") or 0),
        "advisory_topic_filter_ok": sum(int(count) for count in advisory_counts.values()),
        "learning_runtime_safe": sum(int(count) for count in learning_counts.values()),
    }


def _verify_expected_hashes(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    expected_sidecar = expected.get("durable_sidecar", {}).get("sha256")
    actual_sidecar = manifest["release_inputs"]["durable_sidecar"]["sha256"]
    if expected_sidecar and actual_sidecar != expected_sidecar:
        raise AsterionReleaseBundleError("Durable sidecar SHA-256 does not match expected provenance.")

    expected_exports = expected.get("export_artifacts", {})
    key_map = {
        "catalog": "catalog",
        "student_runtime": "student_runtime",
        "content_lab_candidates": "content_lab_candidates",
    }
    for expected_key, manifest_key in key_map.items():
        expected_sha = expected_exports.get(expected_key, {}).get("sha256")
        actual_sha = manifest["export_artifacts"][manifest_key]["sha256"]
        if expected_sha and actual_sha != expected_sha:
            raise AsterionReleaseBundleError(
                f"{manifest_key} SHA-256 does not match expected provenance."
            )


def _load_expected_provenance(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return _read_json_object(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AsterionReleaseBundleError(f"Expected JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package Asterion export release provenance as a small manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT_PATH)
    parser.add_argument("--expected-provenance", type=Path, default=DEFAULT_EXPECTED_PROVENANCE_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME_PATH)
    parser.add_argument("--content-lab", type=Path, default=DEFAULT_CONTENT_LAB_PATH)
    parser.add_argument(
        "--refresh-expected",
        action="store_true",
        help="Write the manifest without failing when current export SHAs differ from the expected provenance file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_asterion_release_manifest(
            output_path=args.output,
            validation_report_path=args.validation_report,
            expected_provenance_path=args.expected_provenance,
            catalog_path=args.catalog,
            runtime_path=args.runtime,
            content_lab_path=args.content_lab,
            refresh_expected=args.refresh_expected,
        )
    except AsterionReleaseBundleError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "output": str(args.output), "counts": manifest["counts"]}, indent=2, ensure_ascii=False))
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Asterion export files against a release manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exports-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_asterion_release_manifest(
            manifest_path=args.manifest,
            exports_dir=args.exports_dir,
        )
    except AsterionReleaseBundleError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
