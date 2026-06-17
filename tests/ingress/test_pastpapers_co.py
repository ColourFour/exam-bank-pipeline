import unittest

from exam_bank.ingress import pastpapers_co as ingress


class PastPapersCoIngressTests(unittest.TestCase):
    def test_parse_legacy_pure_math_1(self):
        resource = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_qp_1.pdf",
            source_page="session-page",
        )
        self.assertIsNotNone(resource)
        self.assertEqual(resource.year, 2008)
        self.assertEqual(resource.paper, "pure_math_1")
        self.assertEqual(resource.component, "1")
        self.assertEqual(resource.session, "Oct/Nov")
        self.assertEqual(resource.doc_type, "qp")

    def test_parse_modern_pure_math_1_variant(self):
        resource = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2025-may-june/9709_s25_ms_12.pdf"
        )
        self.assertIsNotNone(resource)
        self.assertEqual(resource.year, 2025)
        self.assertEqual(resource.paper, "pure_math_1")
        self.assertEqual(resource.component, "12")
        self.assertEqual(resource.doc_type, "ms")

    def test_statistics_1_component_switch(self):
        old_s1 = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2019/2019-nov/9709_w19_qp_61.pdf"
        )
        new_s1 = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2020-may-june/9709_s20_qp_52.pdf"
        )
        new_s2 = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2020-may-june/9709_s20_qp_62.pdf"
        )
        self.assertIsNotNone(old_s1)
        self.assertEqual(old_s1.paper, "statistics_1")
        self.assertIsNotNone(new_s1)
        self.assertEqual(new_s1.paper, "statistics_1")
        self.assertIsNone(new_s2)

    def test_extract_links_handles_relative_pdfs_and_json_paths(self):
        html = """
        <html><body>
          <a href="9709_w08_qp_1.pdf">P1</a>
          <script>{"href":"/caie/a-level/mathematics-9709/2025-may-june/9709_s25_ms_52.pdf"}</script>
        </body></html>
        """
        links = ingress.extract_links(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov",
            html,
        )
        self.assertIn(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_qp_1.pdf",
            links,
        )
        self.assertIn(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2025-may-june/9709_s25_ms_52.pdf",
            links,
        )

    def test_build_exam_records_pairs_qp_and_ms(self):
        resources = [
            ingress.parse_pdf_resource(
                "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_qp_1.pdf",
                source_page="session-page",
            ),
            ingress.parse_pdf_resource(
                "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_ms_1.pdf",
                source_page="session-page",
            ),
        ]
        records = ingress.build_exam_records([r for r in resources if r])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["id"], "caie-9709-pure_math_1-w08-1")
        self.assertTrue(record["question_paper_url"].endswith("9709_w08_qp_1.pdf"))
        self.assertTrue(record["mark_scheme_url"].endswith("9709_w08_ms_1.pdf"))
        self.assertIsNone(record["variant"])


if __name__ == "__main__":
    unittest.main()
