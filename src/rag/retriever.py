"""Retriever orchestrating query embedding, FAISS similarity search, and provenance metrics."""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple
from src.core.schemas import (
    DocumentVariant,
    Question,
    QuestionRetrievalResult,
    RetrievalItem,
    TextChunk,
)
from src.evaluation.retrieval_metrics import compute_retrieval_metrics_for_question
from src.rag.embeddings import BaseEmbeddingModel
from src.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Orchestrates dense text encoding and top-K chunk retrieval via FAISS."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: int = 5,
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.vector_store = vector_store or FAISSVectorStore(dimension=embedding_model.dimension)

    def index_chunks(self, chunks: List[TextChunk], batch_size: int = 32) -> None:
        """Encode chunks and load them into the FAISS index."""
        if not chunks:
            return

        logger.info("Encoding and indexing %d chunks...", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode_documents(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        logger.info("Indexed %d chunks successfully.", len(chunks))

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[RetrievalItem], float]:
        """Retrieve top_k chunks for a query string.

        Returns:
            Tuple of (retrieved_items, latency_seconds)
        """
        k = top_k or self.top_k
        t0 = time.perf_counter()
        q_vec = self.embedding_model.encode_queries([query])[0]
        items = self.vector_store.search(q_vec, top_k=k)
        latency = time.perf_counter() - t0
        return items, latency

    def evaluate_question(
        self,
        question: Question,
        variant: DocumentVariant,
        top_k: Optional[int] = None,
    ) -> QuestionRetrievalResult:
        """Execute retrieval for an evaluation question and compute Recall@K and MRR."""
        k = top_k or self.top_k
        items, latency = self.retrieve(question.question, top_k=k)
        r1, r3, r5, mrr = compute_retrieval_metrics_for_question(
            retrieved_items=items,
            target_page_id=question.page_id,
        )

        return QuestionRetrievalResult(
            question_id=question.question_id,
            variant=variant,
            retrieved_chunks=items,
            recall_at_1=r1,
            recall_at_3=r3,
            recall_at_5=r5,
            mrr=mrr,
            retrieval_latency_sec=latency,
        )
