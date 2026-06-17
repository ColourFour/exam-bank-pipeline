import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
        self.assertIn("asset_paths", record)

    def test_download_pdf_streams_mocked_response(self):
        class Response:
            status_code = 200

            def iter_content(self, chunk_size):
                yield b"%PDF-"
                yield b"body"

            def close(self):
                pass

        calls = []
        original_get = ingress.requests.get
        ingress.requests.get = lambda *args, **kwargs: calls.append((args, kwargs)) or Response()
        try:
            with TemporaryDirectory() as tmp:
                destination = Path(tmp) / "nested" / "paper.pdf"
                self.assertTrue(ingress.download_pdf("https://pastpapers.co/paper.pdf", destination))
                self.assertEqual(destination.read_bytes(), b"%PDF-body")
        finally:
            ingress.requests.get = original_get

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]["stream"])

    def test_asset_classification_mapping_and_destination_structure(self):
        qp = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_qp_1.pdf"
        )
        ms = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_ms_1.pdf"
        )

        self.assertEqual(ingress.classify_asset_type(qp), "question_paper")
        self.assertEqual(ingress.classify_asset_type(ms), "mark_scheme")

        destination = ingress.pdf_destination_path(qp, storage_root=Path("input/pastpapers"))
        self.assertEqual(
            destination,
            Path("input/pastpapers/question_papers/2008/oct-nov/9709_w08_qp_1.pdf"),
        )

    def test_duplicate_downloads_are_skipped(self):
        resource = ingress.parse_pdf_resource(
            "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_qp_1.pdf"
        )

        with TemporaryDirectory() as tmp:
            storage_root = Path(tmp) / "input" / "pastpapers"
            destination = ingress.pdf_destination_path(resource, storage_root=storage_root)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"existing")

            summary = ingress.CrawlSummary(min_year=2008, max_year=2008)
            local_paths = ingress.download_resource_pdfs(
                [resource],
                summary,
                storage_root=storage_root,
                downloader=lambda url, path: self.fail("duplicate should not download"),
            )

        self.assertEqual(summary.pdf_downloaded_count, 0)
        self.assertEqual(summary.pdf_skipped_duplicates, 1)
        self.assertEqual(summary.skipped_by_reason["skipped_duplicate"], 1)
        self.assertEqual(local_paths[resource.url], str(destination))

    def test_ingestion_records_include_local_asset_paths_and_write_jsonl(self):
        resources = [
            ingress.parse_pdf_resource(
                "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_qp_1.pdf"
            ),
            ingress.parse_pdf_resource(
                "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_ms_1.pdf"
            ),
        ]
        asset_paths = {
            resources[0].url: "input/pastpapers/question_papers/2008/oct-nov/9709_w08_qp_1.pdf",
            resources[1].url: "input/pastpapers/mark_schemes/2008/oct-nov/9709_w08_ms_1.pdf",
        }

        records = ingress.build_exam_records(resources, asset_local_paths=asset_paths)

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "exam_bank_input.jsonl"
            ingress.write_output(output, records)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(records[0]["asset_paths"]["question_paper_local_path"], asset_paths[resources[0].url])
        self.assertEqual(records[0]["asset_paths"]["mark_scheme_local_path"], asset_paths[resources[1].url])

    def test_html_fallback_without_pdf_links_still_produces_empty_valid_records(self):
        resources, summary = ingress.discover_resources_with_summary(
            base_url="https://pastpapers.co/caie/a-level/mathematics-9709",
            min_year=2008,
            max_year=2008,
            delay_seconds=0,
            fetcher=lambda url: "<html><body>No PDF links yet</body></html>",
            include_session_candidates=False,
        )

        self.assertEqual(resources, [])
        self.assertEqual(ingress.build_exam_records(resources), [])
        self.assertEqual(summary.total_pages_discovered, 1)

    def test_discovery_summary_dedupes_same_paper_across_pages(self):
        root = "https://pastpapers.co/caie/a-level/mathematics-9709"
        session = f"{root}/2008/2008-nov"

        def fetcher(url):
            if url == root:
                return f'<a href="{session}">Nov 2008</a>'
            if url == session:
                return """
                <a href="9709_w08_qp_1.pdf">P1 QP</a>
                <a href="/caie/a-level/mathematics-9709/mirror/9709_w08_qp_1.pdf">P1 QP mirror</a>
                <a href="9709_w08_ms_1.pdf">P1 MS</a>
                <a href="9709_w08_qp_2.pdf">Unsupported P2</a>
                """
            return None

        resources, summary = ingress.discover_resources_with_summary(
            base_url=root,
            min_year=2008,
            max_year=2008,
            delay_seconds=0,
            fetcher=fetcher,
            include_session_candidates=False,
        )
        records = ingress.build_exam_records(resources)
        ingress.populate_record_summary(summary, records)

        self.assertEqual(len(records), 1)
        self.assertEqual(summary.total_pages_discovered, 2)
        self.assertEqual(summary.total_papers_ingested, 1)
        self.assertEqual(summary.per_year_counts, {"2008": 1})
        self.assertEqual(summary.skipped_by_reason["duplicate_resource"], 1)
        self.assertEqual(summary.skipped_by_reason["unsupported_pdf"], 1)

    def test_strict_parse_fails_on_unknown_9709_pdf_format(self):
        with self.assertRaises(ValueError):
            ingress.parse_pdf_resource(
                "https://pastpapers.co/caie/a-level/mathematics-9709/2008/2008-nov/9709_w08_question_1.pdf",
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
