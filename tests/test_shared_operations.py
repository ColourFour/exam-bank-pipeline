from PIL import Image

from exam_bank.image_operations import stitch_images
from exam_bank.models import BoundingBox, PageLayout
from exam_bank.paper_components import normalize_component_code, packet_family_for_component
from exam_bank.question_detection_graphics import is_answer_rule_like
from exam_bank.regeneration_selection import normalize_requested_ids


def test_stitch_images_preserves_order_and_gap() -> None:
    first = Image.new("RGB", (4, 2), "red")
    second = Image.new("RGB", (2, 3), "blue")

    stitched = stitch_images([first, second], gap_px=2)

    assert stitched.size == (4, 7)
    assert stitched.getpixel((0, 0)) == (255, 0, 0)
    assert stitched.getpixel((0, 3)) == (255, 255, 255)
    assert stitched.getpixel((0, 6)) == (0, 0, 255)


def test_component_normalization_and_packet_family_mapping() -> None:
    assert normalize_component_code("P3") == "03"
    assert packet_family_for_component("32") == "p3"
    assert packet_family_for_component("P41") == "p4"
    assert packet_family_for_component("52", paper="52winter19") == ""
    assert packet_family_for_component("52", paper="52spring20") == "p5"
    assert packet_family_for_component("52") == ""
    assert packet_family_for_component("62", paper="62winter19") == "p5"
    assert packet_family_for_component("62", paper="62spring20") == "p6"
    assert packet_family_for_component("62") == ""
    assert packet_family_for_component("unknown") == ""


def test_requested_id_normalization_handles_files_and_csv_values() -> None:
    assert normalize_requested_ids(["q1,output/q2.png", "q3\nq4"]) == {"q1", "q2", "q3", "q4"}


def test_answer_rule_geometry_is_shared() -> None:
    layout = PageLayout(page_number=1, width=100, height=200, blocks=[], graphics=[])

    assert is_answer_rule_like(BoundingBox(10, 20, 40, 21), layout)
    assert not is_answer_rule_like(BoundingBox(10, 20, 20, 21), layout)
