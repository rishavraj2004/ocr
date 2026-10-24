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

    @classmethod
    def tearDownClass(cls):
        import glob
        from pathlib import Path
        for p in glob.glob("data/*/custom_*"):
            try:
                os.remove(p)
            except OSError:
                pass
        custom_meta = Path("data/metadata/custom_documents.json")
        if custom_meta.exists():
            try:
                os.remove(custom_meta)
            except OSError:
                pass

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

    def test_api_live_upload_text(self):
        """Verify POST /api/live/upload-text ingests text and makes it immediately searchable."""
        payload = {
            "title": "Solar Research Gazette",
            "text": "The Western Grid initiative deployed 450 gigawatts of high-efficiency photovoltaic capacity.",
            "language": "en",
            "threshold": 70.0,
        }
        response = self.client.post("/api/live/upload-text", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("page_id", data)
        self.assertIn("chunks_indexed", data)
        self.assertGreater(data["word_count"], 5)

        self.assertIn("sample_questions", data)
        self.assertGreater(len(data["sample_questions"]), 0)
        uploaded_page_id = data["page_id"]

        # 1. Immediately query specifically scoped to this uploaded document
        q_resp = self.client.post(
            "/api/live/query",
            json={
                "question": "How many gigawatts were deployed in the Western Grid?",
                "top_k": 2,
                "page_id": uploaded_page_id,
            },
        )
        self.assertEqual(q_resp.status_code, 200)
        q_data = q_resp.json()
        self.assertIn("variants", q_data)
        self.assertIn("ground_truth", q_data["variants"])
        gt_chunks = q_data["variants"]["ground_truth"]["retrieved_chunks"]
        if gt_chunks:
            self.assertEqual(gt_chunks[0]["page_id"], uploaded_page_id)

        # 2. Verify GET /api/live/custom-documents lists this document
        custom_docs_resp = self.client.get("/api/live/custom-documents")
        self.assertEqual(custom_docs_resp.status_code, 200)
        c_docs = custom_docs_resp.json()
        self.assertTrue(any(d["page_id"] == uploaded_page_id for d in c_docs))

    def test_api_live_upload_file(self):
        """Verify POST /api/live/upload-file accepts multipart text/image files."""
        file_content = b"Agricultural report: total wheat yield reached 85 million metric tons."
        response = self.client.post(
            "/api/live/upload-file",
            files={"file": ("wheat_report.txt", file_content, "text/plain")},
            data={"title": "Wheat Report", "language": "en", "threshold": 70.0},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("page_id", data)
        self.assertIn("sample_questions", data)


if __name__ == "__main__":
    unittest.main()

