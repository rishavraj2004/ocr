"""Unit tests for Phase 12: Interactive Research Inspection Dashboard."""

import os
import unittest
from fastapi.testclient import TestClient
from src.dashboard.app import create_app


class TestDashboardEndpoints(unittest.TestCase):
    """Test suite validating Dashboard FastAPI endpoints and responses."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_root_index_html(self):
        """Verify root endpoint serves the HTML5 dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Multilingual RAG", response.text)
        self.assertIn("OCR Heatmap Inspector", response.text)

    def test_api_kpis(self):
        """Verify /api/kpis returns valid aggregate metrics."""
        response = self.client.get("/api/kpis")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("overall", data)
        self.assertIn("by_language", data)
        self.assertIn("efficiency", data)
        self.assertIn("optimal_threshold", data)

    def test_api_documents(self):
        """Verify /api/documents lists all benchmark document pages."""
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 10)
        first_doc = data[0]
        self.assertIn("page_id", first_doc)
        self.assertIn("language", first_doc)
        self.assertIn("word_count", first_doc)

    def test_api_document_detail(self):
        """Verify /api/document/{page_id} returns words, bboxes, and ground truth."""
        response = self.client.get("/api/document/doc_en_001_page_001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["metadata"]["page_id"], "doc_en_001_page_001")
        self.assertIn("words", data)
        self.assertIn("ground_truth_text", data)
        self.assertGreater(len(data["words"]), 0)
        first_word = data["words"][0]
        self.assertIn("confidence", first_word)
        self.assertIn("bbox", first_word)

    def test_api_document_detail_404(self):
        """Verify unknown page_id returns 404."""
        response = self.client.get("/api/document/non_existent_page_999")
        self.assertEqual(response.status_code, 404)

    def test_api_questions(self):
        """Verify /api/questions returns triangulated 3-variant evaluation."""
        response = self.client.get("/api/questions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 38)
        first_q = data[0]
        self.assertIn("variant_a_ground_truth", first_q)
        self.assertIn("variant_b_raw_ocr", first_q)
        self.assertIn("variant_c_corrected_ocr", first_q)
        self.assertIn("status", first_q)

    def test_api_sweeps(self):
        """Verify /api/sweeps returns sweep points."""
        response = self.client.get("/api/sweeps")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("points", data)
        self.assertIn("optimal_threshold", data)

    def test_api_tables(self):
        """Verify /api/tables returns exported LaTeX and Markdown tables."""
        response = self.client.get("/api/tables")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("triangulation_table.tex", data)
        self.assertIn("summary_tables.md", data)


if __name__ == "__main__":
    unittest.main()
