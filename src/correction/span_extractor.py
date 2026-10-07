"""Extracts contextual windows around low-confidence OCR spans."""

from __future__ import annotations

from typing import List
from src.core.schemas import CorrectionSpan, WordOCR
from src.correction.confidence_detector import RawSpanCandidate


class SpanExtractor:
    """Attaches preceding and succeeding local context words to detected low-confidence spans."""

    def __init__(self, context_window_words: int = 5):
        self.context_window_words = max(1, context_window_words)

    def extract_contextual_spans(
        self,
        page_id: str,
        words: List[WordOCR],
        raw_candidates: List[RawSpanCandidate],
    ) -> List[CorrectionSpan]:
        """Convert raw span candidates into CorrectionSpan instances with surrounding context."""
        contextual_spans: List[CorrectionSpan] = []

        for idx, candidate in enumerate(raw_candidates, 1):
            # Extract context before
            start_ctx = max(0, candidate.start_word_idx - self.context_window_words)
            before_words = words[start_ctx : candidate.start_word_idx]
            context_before = " ".join(w.text for w in before_words)

            # Extract context after
            end_ctx = min(len(words), candidate.end_word_idx + self.context_window_words)
            after_words = words[candidate.end_word_idx : end_ctx]
            context_after = " ".join(w.text for w in after_words)

            span_obj = CorrectionSpan(
                span_id=f"{page_id}_span_{idx:03d}",
                page_id=page_id,
                start_word_idx=candidate.start_word_idx,
                end_word_idx=candidate.end_word_idx,
                original_text=candidate.original_text,
                avg_confidence=round(candidate.avg_confidence, 1),
                min_confidence=round(candidate.min_confidence, 1),
                context_before=context_before,
                context_after=context_after,
                corrected_text=candidate.original_text,  # Initial placeholder
                changed=False,
                latency_sec=0.0,
            )
            contextual_spans.append(span_obj)

        return contextual_spans
