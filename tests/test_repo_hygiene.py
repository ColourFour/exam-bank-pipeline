from pathlib import Path
import re
import subprocess


README_PATH = Path("README.md")
CONFIG_PATH = Path("config.yaml")
PYPROJECT_PATH = Path("pyproject.toml")
JUNK_PATHS = [
    ".DS_Store",
    "input/.DS_Store",
    "__MACOSX/",
    "input/__MACOSX/",
    "__pycache__/",
    "src/exam_bank/__pycache__/",
    "module.pyc",
    "src/exam_bank/module.pyc",
]


def test_readme_centers_supported_namespaced_process_command() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "exam-bank extract run --input input/pastpapers/9709 --output output" in readme
    assert "process-folder" not in readme
    assert "topic-pdfs" not in readme
    assert "practice-page" not in readme
    assert "manual-review" not in readme
    assert "open-qa" not in readme


def test_config_yaml_only_advertises_active_operational_sections() -> None:
    config_yaml = CONFIG_PATH.read_text(encoding="utf-8")

    for section in ["topic_pdfs:", "practice_page:", "manual_review:", "images_dir:", "csv_dir:", "review_dir:"]:
        assert section not in config_yaml


def test_package_metadata_matches_full_local_platform() -> None:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert 'description = "Image-first CAIE 9709 exam-bank extraction and local teaching platform."' in pyproject
    assert '"pandas>=2.0.0"' not in pyproject
    assert '"reportlab>=4.0.0"' not in pyproject


def test_package_data_includes_runtime_and_dashboard_assets() -> None:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert '"runtime_profile.json"' in pyproject
    assert '"classroom_dashboard/static/*"' in pyproject


def test_generated_inventory_files_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for pattern in [
        "repo_file_inventory.txt",
        "generated_output_inventory.txt",
        "output_inventory.json",
        "output_inventory.md",
        "output_cleanup_plan.md",
        "output_ocr_candidate/",
        "reports/*",
        ".agent-runs/",
    ]:
        assert pattern in gitignore


def test_generated_agent_and_report_artifacts_are_ignored() -> None:
    ignored_check = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            ".agent-runs/latest",
            ".agent-runs/2026-06-20T00-00-00-000Z/iteration-01/01-plan.json",
            "reports/output_storage_duplicate_audit.v1.json",
            "reports/debug/smoke.png",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(ignored_check.stdout.splitlines()) == {
        ".agent-runs/latest",
        ".agent-runs/2026-06-20T00-00-00-000Z/iteration-01/01-plan.json",
        "reports/output_storage_duplicate_audit.v1.json",
        "reports/debug/smoke.png",
    }

    manifest_check = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "manifests/releases/asterion_export_release_manifest.v1.json",
            "manifests/releases/asterion_export_release_provenance.v1.json",
        ],
        capture_output=True,
        text=True,
    )
    assert manifest_check.returncode == 1
    assert manifest_check.stdout == ""


def test_submission_private_roots_are_gitignored() -> None:
    ignored_check = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "data/submissions/roster.csv",
            "data/submissions/student-work.pdf",
            "output/submissions/completion.csv",
            "output/submissions/drafts/reminder.txt",
            "reports/submissions/audit.jsonl",
            "reports/submissions/run-summary.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(ignored_check.stdout.splitlines()) == {
        "data/submissions/roster.csv",
        "data/submissions/student-work.pdf",
        "output/submissions/completion.csv",
        "output/submissions/drafts/reminder.txt",
        "reports/submissions/audit.jsonl",
        "reports/submissions/run-summary.json",
    }

    placeholder_check = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "data/submissions/.gitkeep",
            "output/submissions/.gitkeep",
            "reports/submissions/.gitkeep",
        ],
        capture_output=True,
        text=True,
    )
    assert placeholder_check.returncode == 1
    assert placeholder_check.stdout == ""


def test_submission_contract_docs_exist() -> None:
    assert Path("docs/SUBMISSION_TRACKING_CONTRACT.md").is_file()
    assert Path("docs/SUBMISSION_PRIVACY_BOUNDARIES.md").is_file()


def test_os_and_python_cache_junk_is_absent_and_ignored() -> None:
    visible_files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    visible_junk = [
        path
        for path in visible_files
        if path.endswith("/.DS_Store")
        or path == ".DS_Store"
        or path.endswith(".pyc")
        or "__pycache__/" in path
        or path.startswith("__MACOSX/")
        or "/__MACOSX/" in path
    ]
    assert visible_junk == []

    check_ignore = subprocess.run(
        ["git", "check-ignore", "--no-index", *JUNK_PATHS],
        check=True,
        capture_output=True,
        text=True,
    )
    ignored_paths = set(check_ignore.stdout.splitlines())

    assert ignored_paths == set(JUNK_PATHS)


def test_generated_and_external_data_are_not_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    forbidden = [
        path
        for path in tracked
        if path.startswith(("input/", "tmp/", "agent_handoffs/"))
        or (path.startswith("data/review/") and not path.startswith("data/review/canonical/"))
        or (path.startswith("reports/") and path != "reports/.gitkeep")
        or (path.startswith("output/") and path not in {"output/json/.gitkeep", "output/submissions/.gitkeep"})
    ]
    assert forbidden == []


def test_large_tracked_files_are_confined_to_explicit_data_contracts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed_prefixes = (
        "data/review/canonical/",
        "data/topic_routing/",
        "exam_bank_taxonomy/",
        "tests/fixtures/",
    )
    unexpected = [
        path
        for path in tracked
        if Path(path).is_file()
        and Path(path).stat().st_size > 1024 * 1024
        and not path.startswith(allowed_prefixes)
    ]
    assert unexpected == []


def test_thin_top_level_script_wrappers_do_not_return() -> None:
    thin_scripts = [
        path.as_posix()
        for path in Path("scripts").glob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) <= 20
    ]
    assert thin_scripts == []


def test_tests_do_not_skip_when_live_output_or_reports_are_absent() -> None:
    live_state_skip = re.compile(r"if\s+not\s+Path\([\"'](?:output|reports)/.*?pytest\.skip", re.DOTALL)
    offenders = [
        path.as_posix()
        for path in Path("tests").glob("test_*.py")
        if live_state_skip.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
