"""FAISS-based vector index for dense semantic retrieval."""

from __future__ import annotations

import logging
from typing import List, Optional
import numpy as np
import faiss

from src.core.schemas import RetrievalItem, TextChunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """In-memory FAISS Vector Store using Inner Product (cosine similarity on normalized vectors)."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[TextChunk] = []

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """Add document chunks and their dense embeddings to the index.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Float32 numpy array of shape (N, dimension).
        """
        if len(chunks) == 0:
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
            )

        # Ensure float32
        vecs = embeddings.astype(np.float32)

        # L2-normalize vectors so Inner Product equals Cosine Similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        vecs = vecs / norms

        self.index.add(vecs)
        self.chunks.extend(chunks)
        logger.debug("Added %d chunks to FAISS index. Total count: %d", len(chunks), len(self.chunks))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalItem]:
        """Retrieve top_k most similar chunks for a given query embedding.

        Args:
            query_embedding: 1D array of shape (dimension,) or 2D of shape (1, dimension).
            top_k: Number of candidates to return.

        Returns:
            List of RetrievalItem ranked from highest to lowest similarity.
        """
        if self.index.ntotal == 0:
            return []

        q_vec = query_embedding.astype(np.float32).reshape(1, -1)
        # Normalize
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, actual_k)

        items: List[RetrievalItem] = []
        for rank_idx, (score, chunk_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                continue
            chunk = self.chunks[chunk_idx]
            items.append(
                RetrievalItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_id=chunk.page_id,
                    score=float(score),
                    rank=rank_idx,
                    text=chunk.text,
                )
            )

        return items

    def count(self) -> int:
        """Return total indexed chunks."""
        return self.index.ntotal

    def clear(self) -> None:
        """Reset the index."""
        self.index.reset()
        self.chunks.clear()
