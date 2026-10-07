"""Sliding window document chunker preserving document and page provenance."""

from __future__ import annotations

import re
from typing import List
from src.core.schemas import DocumentVariant, TextChunk


class SlidingWindowChunker:
    """Splits document text into overlapping token/word chunks with exact character offsets."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        unit: str = "words",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit
        self.step_size = chunk_size - chunk_overlap

    def chunk_document(
        self,
        text: str,
        document_id: str,
        page_id: str,
        variant: DocumentVariant,
    ) -> List[TextChunk]:
        """Slice document text into sequential overlapping TextChunk instances."""
        clean_text = text.strip()
        if not clean_text:
            return []

        # Find word tokens along with their character spans in original text
        word_matches = list(re.finditer(r"\S+", clean_text))
        if not word_matches:
            return []

        total_words = len(word_matches)
        chunks: List[TextChunk] = []
        chunk_idx = 0

        # If document has fewer words than one chunk size, return single complete chunk
        if total_words <= self.chunk_size:
            start_char = word_matches[0].start()
            end_char = word_matches[-1].end()
            chunk_text = clean_text[start_char:end_char]
            c = TextChunk(
                chunk_id=f"{variant.value}_{page_id}_c000",
                document_id=document_id,
                page_id=page_id,
                variant=variant,
                text=chunk_text,
                chunk_index=0,
                start_char=start_char,
                end_char=end_char,
            )
            return [c]

        start_word = 0
        while start_word < total_words:
            end_word = min(total_words, start_word + self.chunk_size)
            start_char = word_matches[start_word].start()
            end_char = word_matches[end_word - 1].end()
            chunk_text = clean_text[start_char:end_char]

            c = TextChunk(
                chunk_id=f"{variant.value}_{page_id}_c{chunk_idx:03d}",
                document_id=document_id,
                page_id=page_id,
                variant=variant,
                text=chunk_text,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(c)
            chunk_idx += 1

            if end_word >= total_words:
                break
            start_word += self.step_size

        return chunks
