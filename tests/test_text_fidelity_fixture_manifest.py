import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "text_fidelity" / "bad_text_records.json"
QUESTION_BANK_PATH = REPO_ROOT / "output" / "json" / "question_bank.json"
OUTPUT_ROOT = REPO_ROOT / "output"

MATH_TAGS = {
    "calculus_expression",
    "complex_number_layout",
    "derivative_layout",
    "fraction_structure",
    "greek_symbol",
    "inequality_direction",
    "integral_bounds",
    "math_notation",
    "polynomial_reading_order",
    "radical_or_power_structure",
    "rational_expression",
    "short_math_prompt_review",
    "trig_symbol_fidelity",
    "units_symbol",
    "vector_matrix_layout",
}

CROP_OR_CONTAMINATION_TAGS = {
    "crop_boundary_or_contamination",
    "diagram_reading_order",
    "mechanics_graph_reading_order",
    "page_furniture_contamination",
    "table_furniture_noise",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bad_text_fixture_manifest_is_deterministic_and_actionable() -> None:
    manifest = _load_json(FIXTURE_PATH)
    question_bank = _load_json(QUESTION_BANK_PATH)
    bank_records = {record["question_id"]: record for record in question_bank["questions"]}

    assert manifest["schema_name"] == "text_fidelity_bad_text_fixture_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["record_count"] == len(manifest["records"])
    assert 30 <= len(manifest["records"]) <= 50

    seen_ids: set[str] = set()
    all_tags: set[str] = set()
    math_records = 0
    crop_or_contamination_records = 0

    required_fields = {
        "record_id",
        "paper_id",
        "paper_family",
        "session",
        "question_number",
        "question_image_path",
        "currently_selected_text",
        "expected_normalized_text_or_structural_expectations",
        "failure_tags",
        "review_notes",
    }

    for fixture in manifest["records"]:
        assert required_fields <= set(fixture)
        assert fixture["record_id"] not in seen_ids
        seen_ids.add(fixture["record_id"])

        source_record = bank_records[fixture["record_id"]]
        assert fixture["paper_id"] == source_record["paper"]
        assert fixture["paper_family"] == source_record["paper_family"]
        assert fixture["question_number"] == source_record["question_number"]
        assert fixture["currently_selected_text"] == source_record["question_text"]

        question_image = OUTPUT_ROOT / fixture["question_image_path"]
        assert question_image.is_file(), fixture["question_image_path"]

        mark_scheme_image_path = fixture.get("mark_scheme_image_path")
        if mark_scheme_image_path:
            assert (OUTPUT_ROOT / mark_scheme_image_path).is_file(), mark_scheme_image_path

        expectations = fixture["expected_normalized_text_or_structural_expectations"]
        assert expectations["type"] == "structural_expectations"
        assert len(expectations["expectations"]) >= 3

        tags = set(fixture["failure_tags"])
        assert tags
        all_tags.update(tags)
        if tags & MATH_TAGS:
            math_records += 1
        if tags & CROP_OR_CONTAMINATION_TAGS:
            crop_or_contamination_records += 1

    assert len(all_tags) >= 6
    assert math_records >= 10
    assert crop_or_contamination_records >= 5
