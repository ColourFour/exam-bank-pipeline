import subprocess
from pathlib import Path


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


def test_readme_centers_supported_process_command() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert (
        ".venv/bin/python -m exam_bank.cli process \\\n"
        "  --input input/pastpapers/9709 \\\n"
        "  --output output \\\n"
        "  --enable-ocr"
    ) in readme
    assert "process-folder" not in readme
    assert "topic-pdfs" not in readme
    assert "practice-page" not in readme
    assert "manual-review" not in readme
    assert "open-qa" not in readme


def test_config_yaml_only_advertises_active_operational_sections() -> None:
    config_yaml = CONFIG_PATH.read_text(encoding="utf-8")

    for section in ["topic_pdfs:", "practice_page:", "manual_review:", "images_dir:", "csv_dir:", "review_dir:"]:
        assert section not in config_yaml


def test_package_metadata_matches_extraction_only_runtime() -> None:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert 'description = "CAIE 9709 question-paper and mark-scheme extraction pipeline."' in pyproject
    assert '"pandas>=2.0.0"' not in pyproject
    assert '"reportlab>=4.0.0"' not in pyproject


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
            "reports/asterion_export_release_manifest_pr16_2026_06_11.json",
            "reports/asterion_export_release_provenance_pr15_2026_06_11.json",
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
