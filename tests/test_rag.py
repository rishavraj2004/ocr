"""Unit tests for Phase 5: RAG Components (Chunker, Vector Store, Retriever, Generator, Pipeline, Metrics)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.core.config import ExperimentConfig
from src.core.schemas import DocumentVariant, Language, Question, QuestionType, TextChunk
from src.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    compute_answer_metrics,
    compute_exact_match,
    compute_token_f1,
    normalize_answer,
)
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_question,
)
from src.rag.chunker import SlidingWindowChunker
from src.rag.embeddings import MockEmbeddingModel
from src.rag.generator import MockGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import DenseRetriever
from src.rag.vector_store import FAISSVectorStore


class TestRAGComponents(unittest.TestCase):
    """Test suite for RAG subsystem components."""

    def test_sliding_window_chunker(self):
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=3)
        words = [f"word_{i}" for i in range(25)]
        text = " ".join(words)

        chunks = chunker.chunk_document(
            text=text,
            document_id="doc01",
            page_id="doc01_page01",
            variant=DocumentVariant.GROUND_TRUTH,
        )

        # 25 words with step_size=7:
        # chunk 0: 0..10
        # chunk 1: 7..17
        # chunk 2: 14..24
        # chunk 3: 21..25
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].variant, DocumentVariant.GROUND_TRUTH)
        self.assertTrue(chunks[0].chunk_id.startswith("ground_truth_doc01_page01"))

        # Empty text test
        self.assertEqual(chunker.chunk_document("", "doc01", "page01", DocumentVariant.GROUND_TRUTH), [])

    def test_faiss_vector_store(self):
        store = FAISSVectorStore(dimension=8)
        self.assertEqual(store.count(), 0)

        chunks = [
            TextChunk(
                chunk_id="c1",
                document_id="d1",
                page_id="p1",
                variant=DocumentVariant.GROUND_TRUTH,
                text="sample text one",
                chunk_index=0,
            ),
            TextChunk(
                chunk_id="c2",
                document_id="d1",
                page_id="p2",
                variant=DocumentVariant.GROUND_TRUTH,
                text="sample text two",
                chunk_index=1,
            ),
        ]
        # Random 8-dim embeddings
        emb = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        store.add_chunks(chunks, emb)
        self.assertEqual(store.count(), 2)

        # Search matching first chunk exactly
        q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(q, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertAlmostEqual(results[0].score, 1.0, places=5)
        self.assertEqual(results[0].rank, 1)

    def test_dense_retriever_and_metrics(self):
        emb_model = MockEmbeddingModel(dimension=64, seed=42)
        retriever = DenseRetriever(embedding_model=emb_model, top_k=3)

        chunks = [
            TextChunk(
                chunk_id="c_solar",
                document_id="doc_en_001",
                page_id="doc_en_001_page_001",
                variant=DocumentVariant.GROUND_TRUTH,
                text="Solar photovoltaic expansion reached 300 gigawatts in Rajasthan.",
                chunk_index=0,
            ),
            TextChunk(
                chunk_id="c_soil",
                document_id="doc_en_002",
                page_id="doc_en_002_page_001",
                variant=DocumentVariant.GROUND_TRUTH,
                text="Soil health card scheme analyzes organic carbon and micronutrients.",
                chunk_index=0,
            ),
        ]
        retriever.index_chunks(chunks)

        question = Question(
            question_id="q_test_01",
            document_id="doc_en_001",
            page_id="doc_en_001_page_001",
            language=Language.ENGLISH,
            question="What is the targeted total solar energy capacity in Rajasthan?",
            expected_answer="300 gigawatts",
            source_page=1,
            question_type=QuestionType.NUMERICAL,
        )

        eval_res = retriever.evaluate_question(question, variant=DocumentVariant.GROUND_TRUTH, top_k=2)
        self.assertEqual(len(eval_res.retrieved_chunks), 2)
        # Solar chunk should be ranked 1st due to word/n-gram overlap
        self.assertEqual(eval_res.retrieved_chunks[0].chunk_id, "c_solar")
        self.assertEqual(eval_res.recall_at_1, 1.0)
        self.assertEqual(eval_res.recall_at_3, 1.0)
        self.assertEqual(eval_res.recall_at_5, 1.0)
        self.assertEqual(eval_res.mrr, 1.0)

    def test_multilingual_answer_metrics(self):
        # English tests
        self.assertAlmostEqual(compute_exact_match("300 gigawatts", "300 Gigawatts", language="en"), 1.0)
        self.assertAlmostEqual(compute_exact_match("Rajasthan and Gujarat.", "rajasthan and gujarat", language="en"), 1.0)
        self.assertAlmostEqual(compute_exact_match("300 GW", "300 gigawatts", language="en"), 0.0)

        # Hindi tests
        self.assertAlmostEqual(compute_exact_match("५० गीगावाट।", "५० गीगावाट", language="hi"), 1.0)
        self.assertAlmostEqual(compute_exact_match("राजस्थान तथा गुजरात", "राजस्थान तथा गुजरात।", language="hi"), 1.0)

        # Token F1
        f1_full = compute_token_f1("300 gigawatts solar", "300 gigawatts solar", language="en")
        self.assertAlmostEqual(f1_full, 1.0)

        f1_partial = compute_token_f1("300 gigawatts", "300 gigawatts solar energy", language="en")
        # Precision: 2/2 = 1.0, Recall: 2/4 = 0.5, F1: 2*1*0.5/(1.5) = 2/3 = 0.6667
        self.assertAlmostEqual(f1_partial, 2.0 / 3.0, places=4)

        f1_zero = compute_token_f1("completely unrelated", "300 gigawatts", language="en")
        self.assertAlmostEqual(f1_zero, 0.0)

    def test_mock_generator(self):
        gen = MockGenerator()
        gen.register_expected_answer(
            question="What is the targeted solar capacity?",
            expected_answer="300 gigawatts",
        )

        # Context containing answer
        context = "National Renewable Mission: targeted total solar energy capacity is 300 gigawatts by 2030."
        ans, lat, p_tok, c_tok = gen.generate(context, "What is the targeted solar capacity?")
        self.assertEqual(ans, "300 gigawatts")
        self.assertGreater(lat, 0.0)
        self.assertGreater(p_tok, 0)
        self.assertGreater(c_tok, 0)

        # Empty context
        ans_empty, _, _, _ = gen.generate("", "What is the targeted solar capacity?")
        self.assertIn("No context provided", ans_empty)

    def test_raw_ocr_pipeline_and_degradation(self):
        from src.core.config import get_default_config

        cfg = get_default_config()
        cfg.rag.embedding_model = "mock"

        pipeline = RAGPipeline(variant=DocumentVariant.RAW_OCR, config=cfg)
        chunks = pipeline.index_documents()
        self.assertEqual(len(chunks), 10)
        self.assertEqual(chunks[0].variant, DocumentVariant.RAW_OCR)

        report = pipeline.run_evaluation()
        self.assertEqual(report["variant"], "raw_ocr")
        self.assertEqual(report["summary"]["num_questions"], 38)
        self.assertIn("by_question_type", report)
        self.assertIn("numerical", report["by_question_type"])

        # Empirical expectation: raw OCR introduces degradations, so EM < 1.0
        self.assertLess(report["overall"]["mean_exact_match"], 1.0)
        self.assertLess(report["overall"]["mean_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

