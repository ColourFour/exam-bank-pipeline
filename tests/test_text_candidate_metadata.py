from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image

from exam_bank.text_candidate_metadata import build_metadata_coverage_report, extract_record_metadata


def make_record(**overrides):
    record = {
        "question_id": "sample_q01",
        "paper": "sample",
        "paper_family": "p1",
        "question_number": "1",
        "canonical_question_artifact": "p1/sample/questions/q01.png",
        "question_image_path": "p1/sample/questions/q01.png",
        "mark_scheme_image_path": "p1/sample/mark_scheme/q01.png",
        "question_text": "1 Find x. [2]",
        "ocr_text": "1 Find x. [2]",
        "ocr_engine": "tesseract",
        "notes": {
            "question_crop_confidence": "high",
            "question_crop_diagnostics": {
                "page_width": 600,
                "page_height": 800,
                "regions": [
                    {
                        "page_number": 1,
                        "final_crop_bbox": {"x0": 0, "y0": 0, "x1": 300, "y1": 200},
                        "text_line_bboxes": [
                            {"x0": 10, "y0": 20, "x1": 240, "y1": 38},
                        ],
                    }
                ],
            },
            "text_candidate_source": "native",
            "text_source_profile": "native_pdf",
            "text_candidate_decision": "native_retained",
            "text_candidate_decision_reasons": ["native_retained"],
            "ocr_rejected_reasons": ["ocr_not_clearly_better"],
            "review_flags": ["weak_question_text"],
            "validation_flags": [],
            "extraction_quality_flags": ["heavy_math_density"],
            "text_fidelity_flags": [],
        },
    }
    record.update(overrides)
    return record


def write_png(root: Path, relative: str, size: tuple[int, int] = (320, 180)) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_extract_record_metadata_is_deterministic_and_non_mutating(tmp_path: Path) -> None:
    record = make_record()
    write_png(tmp_path, record["question_image_path"], size=(320, 180))
    original = deepcopy(record)

    first = extract_record_metadata(record, artifact_root=tmp_path)
    second = extract_record_metadata(record, artifact_root=tmp_path)

    assert first == second
    assert record == original
    assert first["fields"]["crop_pixel_dimensions"]["value"] == {"width": 320, "height": 180}
    assert first["fields"]["normalized_crop_area"]["value"] == 0.125
    assert first["fields"]["raw_native_text_candidate_window"]["present"] is True
    assert first["fields"]["raw_ocr_text_candidate_window"]["present"] is True
    assert first["fields"]["text_line_bounding_boxes"]["value"]["line_bbox_count"] == 1
    assert first["candidate_provenance"]["selected_source"] == "native"


def test_ocr_selected_record_reports_missing_independent_native_window(tmp_path: Path) -> None:
    record = make_record(
        notes={
            **make_record()["notes"],
            "text_candidate_source": "ocr",
            "text_source_profile": "ocr",
            "ocr_rejected_reasons": [],
        }
    )
    write_png(tmp_path, record["question_image_path"])

    metadata = extract_record_metadata(record, artifact_root=tmp_path)

    native_field = metadata["fields"]["raw_native_text_candidate_window"]
    assert native_field["present"] is False
    assert native_field["reason"] == "raw native candidate text is not stored independently of selected text"
    assert metadata["fields"]["rejected_candidate_reasons"]["present"] is True
    assert metadata["fields"]["rejected_candidate_reasons"]["populated"] is False


def test_build_metadata_coverage_counts_present_missing_fields(tmp_path: Path) -> None:
    native_record = make_record(question_id="sample_q01")
    ocr_record = make_record(
        question_id="sample_q02",
        question_image_path="p1/sample/questions/q02.png",
        canonical_question_artifact="p1/sample/questions/q02.png",
        notes={**make_record()["notes"], "text_candidate_source": "ocr", "text_source_profile": "ocr"},
    )
    write_png(tmp_path, native_record["question_image_path"])
    write_png(tmp_path, ocr_record["question_image_path"])

    report = build_metadata_coverage_report([native_record, ocr_record], artifact_root=tmp_path)

    assert report["schema_name"] == "exam_bank.text_candidate_metadata.coverage"
    assert report["record_count"] == 2
    assert report["field_counts"]["crop_pixel_dimensions"] == {"present": 2, "missing": 0}
    assert report["field_counts"]["raw_native_text_candidate_window"] == {"present": 1, "missing": 1}
    assert report["field_counts"]["candidate_provenance"] == {"present": 2, "missing": 0}
    assert report["candidate_source_counts"] == {"native": 1, "ocr": 1}
