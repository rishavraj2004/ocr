"""Embedding models for dense retrieval in multilingual RAG."""

from __future__ import annotations

import abc
import hashlib
import logging
import re
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(abc.ABC):
    """Abstract interface for dense multilingual embedding models."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the output embedding dimensionality."""
        pass

    @abc.abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts into an L2-normalized float32 2D array of shape (N, dimension)."""
        pass

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode search queries (allows model-specific query prefixes if applicable)."""
        return self.encode(queries)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode passage documents (allows model-specific doc prefixes if applicable)."""
        return self.encode(documents)


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Production multilingual dense embedding model using sentence-transformers (e.g., BGE-M3)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dim = 1024  # Default BGE-M3 dimension

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model '%s'...", self._model_name)
                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._dim = self._model.get_sentence_embedding_dimension()
                logger.info("Loaded '%s' with dimension %d.", self._model_name, self._dim)
            except Exception as e:
                logger.error("Failed to load SentenceTransformer '%s': %s", self._model_name, e)
                raise

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic hashing-based dense embedding model for reproducible, zero-overhead testing.

    Projects words and character n-grams into a fixed-dimension vector using feature hashing
    (signed hashing trick) followed by L2 normalization. Retains lexical and subword
    similarity across both English and Hindi scripts without external weights.
    """

    def __init__(self, dimension: int = 1024, seed: int = 42):
        self._dim = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Hash a token into a (bucket_index, sign) pair."""
        digest = hashlib.md5(f"{self._seed}:{token}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % self._dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        return bucket, sign

    def _embed_single(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        clean = text.strip().lower()
        if not clean:
            return vec

        # Words
        words = re.findall(r"\S+", clean)
        for w in words:
            b, s = self._hash_token(w)
            vec[b] += s * 2.0  # Full words get higher weight

            # Character 3-grams and 4-grams for subword / morph matching (Hindi and English)
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tri = w[i : i + 3]
                    b_tri, s_tri = self._hash_token(tri)
                    vec[b_tri] += s_tri * 0.5

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = [self._embed_single(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)


def create_embedding_model(
    model_name: str = "BAAI/bge-m3",
    dimension: int = 1024,
    use_mock: bool = False,
    device: Optional[str] = None,
) -> BaseEmbeddingModel:
    """Factory creating an embedding model instance based on configuration."""
    if use_mock or model_name.lower().startswith("mock"):
        logger.info("Instantiating MockEmbeddingModel (dim=%d)", dimension)
        return MockEmbeddingModel(dimension=dimension)

    try:
        import sentence_transformers  # noqa: F401
        logger.info("Instantiating SentenceTransformerEmbedding with model '%s'", model_name)
        return SentenceTransformerEmbedding(model_name=model_name, device=device)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Falling back to MockEmbeddingModel (dim=%d)",
            dimension,
        )
        return MockEmbeddingModel(dimension=dimension)
