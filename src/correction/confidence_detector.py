"""Confidence-guided span detector for identifying uncertain OCR tokens."""

from __future__ import annotations

from typing import List, Tuple
from pydantic import BaseModel, Field

from src.core.schemas import WordOCR


class RawSpanCandidate(BaseModel):
    """Contiguous range of low-confidence word tokens flagged for correction."""
    start_word_idx: int = Field(..., ge=0, description="Start index in page word list (inclusive)")
    end_word_idx: int = Field(..., ge=0, description="End index in page word list (exclusive)")
    words: List[WordOCR] = Field(default_factory=list)

    @property
    def original_text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(sum(w.confidence for w in self.words) / len(self.words))

    @property
    def min_confidence(self) -> float:
        if not self.words:
            return 0.0
        return float(min(w.confidence for w in self.words))


class ConfidenceDetector:
    """Identifies and groups contiguous words falling below a confidence threshold."""

    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold

    def detect_low_confidence_spans(self, words: List[WordOCR]) -> List[RawSpanCandidate]:
        """Group consecutive words where word.confidence < threshold into spans."""
        spans: List[RawSpanCandidate] = []
        current_span_words: List[WordOCR] = []
        current_start_idx = -1

        for idx, word in enumerate(words):
            if word.confidence < self.threshold:
                if not current_span_words:
                    current_start_idx = idx
                current_span_words.append(word)
            else:
                if current_span_words:
                    spans.append(
                        RawSpanCandidate(
                            start_word_idx=current_start_idx,
                            end_word_idx=idx,
                            words=current_span_words,
                        )
                    )
                    current_span_words = []
                    current_start_idx = -1

        # Final span if ends with low-confidence word
        if current_span_words:
            spans.append(
                RawSpanCandidate(
                    start_word_idx=current_start_idx,
                    end_word_idx=len(words),
                    words=current_span_words,
                )
            )

        return spans
