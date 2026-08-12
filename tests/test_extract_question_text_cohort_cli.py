from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_question_text_cohort import main


def test_main_rejects_parallel_workers_before_reading_the_cohort(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--cohort",
                str(tmp_path / "does-not-exist.json"),
                "--output",
                str(tmp_path / "candidate.json"),
                "--workers",
                "2",
            ]
        )

    assert error.value.code == 2
    assert (
        "--workers must be 1 because native PDF layout extraction is not thread-safe"
        in capsys.readouterr().err
    )
