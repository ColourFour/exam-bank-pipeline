from types import SimpleNamespace

from exam_bank.config import AppConfig
from exam_bank.pipeline import _extraction_batch_cache_key


def test_extraction_cache_key_changes_with_source_config_and_ocr_profile(tmp_path) -> None:
    question = tmp_path / "question.pdf"
    mark_scheme = tmp_path / "mark_scheme.pdf"
    question.write_bytes(b"question-v1")
    mark_scheme.write_bytes(b"marks-v1")
    entry = SimpleNamespace(question_paper=question, mark_scheme=mark_scheme, examiner_reports=[])
    config = AppConfig()

    baseline = _extraction_batch_cache_key(entry, config)
    assert baseline == _extraction_batch_cache_key(entry, config)

    question.write_bytes(b"question-v2")
    assert baseline != _extraction_batch_cache_key(entry, config)
    question.write_bytes(b"question-v1")

    config.ocr.enabled = True
    assert baseline != _extraction_batch_cache_key(entry, config)
