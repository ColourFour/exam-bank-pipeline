import json
import os
from pathlib import Path

import pytest

from exam_bank import atomic_json
from exam_bank.atomic_json import write_atomic_json


def test_write_atomic_json_preserves_indented_utf8_content(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "payload.json"

    result = write_atomic_json({"name": "Café", "items": [1, 2]}, output_path)

    assert result == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"name": "Café", "items": [1, 2]}
    assert output_path.read_text(encoding="utf-8") == '{\n  "name": "Café",\n  "items": [\n    1,\n    2\n  ]\n}'
    assert list(output_path.parent.glob("*.tmp")) == []


def test_write_atomic_json_uses_same_directory_non_json_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "question_bank.json"
    original_replace = os.replace
    observed_tmp_path: Path | None = None

    def replace_and_observe(src: str | Path, dst: str | Path) -> None:
        nonlocal observed_tmp_path
        observed_tmp_path = Path(src)
        assert Path(dst) == output_path
        assert observed_tmp_path.parent == output_path.parent
        assert observed_tmp_path.name.endswith(".tmp")
        assert observed_tmp_path not in output_path.parent.glob("*.json")
        original_replace(src, dst)

    monkeypatch.setattr(atomic_json.os, "replace", replace_and_observe)

    write_atomic_json({"ok": True}, output_path)

    assert observed_tmp_path is not None
    assert not observed_tmp_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}


def test_write_atomic_json_keeps_existing_target_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "payload.json"
    output_path.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(src: str | Path, dst: str | Path) -> None:
        raise RuntimeError("replace failed")

    monkeypatch.setattr(atomic_json.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        write_atomic_json({"new": True}, output_path)

    assert output_path.read_text(encoding="utf-8") == '{"old": true}'
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"old": True}
