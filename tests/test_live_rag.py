"""Unit tests for Live RAG Q&A and Interactive Layer Testing System."""

import os
import unittest
from fastapi.testclient import TestClient

from src.dashboard.app import create_app
from src.rag.pipeline import get_live_rag_service


class TestLiveRAGAndLayers(unittest.TestCase):
    """Test suite validating live 3-variant RAG querying and layer testing."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.live_service = get_live_rag_service()

    def test_live_rag_service_query(self):
        """Verify LiveTriangulatedRAG direct query across Variants A, B, C."""
        question = "What is the peak solar capacity target?"
        res = self.live_service.query(question=question, top_k=2)

        self.assertIn("question", res)
        self.assertIn("analysis", res)
        self.assertIn("variants", res)
        self.assertIn("ground_truth", res["variants"])
        self.assertIn("raw_ocr", res["variants"])
        self.assertIn("corrected_ocr", res["variants"])

        # Check answers and chunks
        for variant_key in ["ground_truth", "raw_ocr", "corrected_ocr"]:
            v = res["variants"][variant_key]
            self.assertIn("answer", v)
            self.assertIn("retrieved_chunks", v)
            self.assertGreaterEqual(len(v["retrieved_chunks"]), 1)

    def test_api_live_samples(self):
        """Verify /api/live/samples returns quick-test preset questions."""
        response = self.client.get("/api/live/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        sample = data[0]
        self.assertIn("question", sample)
        self.assertIn("language", sample)

    def test_api_live_query_endpoint(self):
        """Verify POST /api/live/query returns 3-variant triangulation payload."""
        payload = {
            "question": "What is the targeted capacity?",
            "top_k": 2,
            "language": "en",
        }
        response = self.client.post("/api/live/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("variants", data)
        self.assertIn("analysis", data)
        self.assertIn("status", data["analysis"])

    def test_api_live_query_hindi(self):
        """Verify POST /api/live/query handles Hindi Devanagari questions."""
        payload = {
            "question": "मिशन का लक्ष्य क्या है?",
            "top_k": 2,
            "language": "hi",
        }
        response = self.client.post("/api/live/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("variants", data)
        self.assertEqual(data["analysis"]["language"], "hi")

    def test_api_live_test_correction(self):
        """Verify POST /api/live/test-correction returns simulation metrics and token economics."""
        payload = {
            "page_id": "doc_en_001_page_001",
            "threshold": 70.0,
            "context_window": 5,
        }
        response = self.client.post("/api/live/test-correction", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["page_id"], "doc_en_001_page_001")
        self.assertIn("total_words", data)
        self.assertIn("flagged_words", data)
        self.assertIn("flagged_percentage", data)
        self.assertIn("efficiency", data)
        self.assertIn("token_savings_pct", data["efficiency"])
        self.assertGreater(data["efficiency"]["token_savings_pct"], 0)

    def test_api_live_test_preprocessing(self):
        """Verify POST /api/live/test-preprocessing executes OpenCV pipeline."""
        payload = {
            "page_id": "doc_en_001_page_001",
            "denoise": True,
            "denoise_h": 10.0,
            "deskew": True,
            "threshold_method": "otsu",
        }
        response = self.client.post("/api/live/test-preprocessing", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["page_id"], "doc_en_001_page_001")
        self.assertIn("operations_applied", data)
        self.assertIn("comparison_url", data)


if __name__ == "__main__":
    unittest.main()
