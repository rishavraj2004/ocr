"""Unit tests for confidence-guided selective correction."""

import unittest
from src.core.schemas import Language, WordOCR
from src.correction.confidence_detector import ConfidenceDetector
from src.correction.span_extractor import SpanExtractor
from src.correction.corrector import MockCorrector
from src.correction.runner import CorrectionRunner, reconstruct_corrected_text


class TestCorrection(unittest.TestCase):
    def setUp(self):
        # 7-word sequence where words 2 and 3 are low-confidence
        self.sample_words = [
            WordOCR(text="भारत", confidence=96.0, bbox=[0, 0, 10, 10]),
            WordOCR(text="सरकार", confidence=94.0, bbox=[10, 0, 20, 10]),
            WordOCR(text="सरचालित", confidence=42.0, bbox=[20, 0, 30, 10]),  # Low
            WordOCR(text="दरश", confidence=45.0, bbox=[30, 0, 40, 10]),      # Low (adjacent)
            WordOCR(text="भर", confidence=92.0, bbox=[40, 0, 50, 10]),
            WordOCR(text="में", confidence=95.0, bbox=[50, 0, 60, 10]),
            WordOCR(text="विकास", confidence=91.0, bbox=[60, 0, 70, 10]),
        ]

    def test_confidence_detector_contiguous_grouping(self):
        """Contiguous low-confidence words must be grouped into a single span candidate."""
        detector = ConfidenceDetector(threshold=70.0)
        spans = detector.detect_low_confidence_spans(self.sample_words)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].start_word_idx, 2)
        self.assertEqual(spans[0].end_word_idx, 4)
        self.assertEqual(spans[0].original_text, "सरचालित दरश")
        self.assertAlmostEqual(spans[0].avg_confidence, 43.5, places=1)

    def test_span_extractor_context_windows(self):
        """SpanExtractor must capture preceding and succeeding local context words."""
        detector = ConfidenceDetector(threshold=70.0)
        raw_spans = detector.detect_low_confidence_spans(self.sample_words)

        extractor = SpanExtractor(context_window_words=2)
        ctx_spans = extractor.extract_contextual_spans("p01", self.sample_words, raw_spans)

        self.assertEqual(len(ctx_spans), 1)
        span = ctx_spans[0]
        self.assertEqual(span.context_before, "भारत सरकार")
        self.assertEqual(span.context_after, "भर में")

    def test_mock_corrector_hindi(self):
        """MockCorrector fixes known Devanagari OCR confusion pairs."""
        detector = ConfidenceDetector(threshold=70.0)
        raw_spans = detector.detect_low_confidence_spans(self.sample_words)
        extractor = SpanExtractor(context_window_words=2)
        ctx_spans = extractor.extract_contextual_spans("p01", self.sample_words, raw_spans)

        corrector = MockCorrector()
        corr_text, changed, lat, p_tok, c_tok = corrector.correct_span(ctx_spans[0], Language.HINDI)

        self.assertTrue(changed)
        self.assertEqual(corr_text, "संचालित देश")

    def test_mock_corrector_english(self):
        """MockCorrector fixes typical Latin OCR substitutions (e.g. 1 -> l)."""
        en_words = [
            WordOCR(text="National", confidence=95.0, bbox=[0, 0, 10, 10]),
            WordOCR(text="tronsition", confidence=40.0, bbox=[10, 0, 20, 10]),
            WordOCR(text="energy", confidence=95.0, bbox=[20, 0, 30, 10]),     # High confidence separator
            WordOCR(text="officio1", confidence=45.0, bbox=[30, 0, 40, 10]),
            WordOCR(text="report", confidence=96.0, bbox=[40, 0, 50, 10]),
        ]
        detector = ConfidenceDetector(threshold=70.0)
        raw_spans = detector.detect_low_confidence_spans(en_words)
        extractor = SpanExtractor(context_window_words=1)
        ctx_spans = extractor.extract_contextual_spans("en01", en_words, raw_spans)

        self.assertEqual(len(ctx_spans), 2)
        corrector = MockCorrector()
        corr1, changed1, _, _, _ = corrector.correct_span(ctx_spans[0], Language.ENGLISH)
        self.assertTrue(changed1)
        self.assertEqual(corr1, "transition")

        corr2, changed2, _, _, _ = corrector.correct_span(ctx_spans[1], Language.ENGLISH)
        self.assertTrue(changed2)
        self.assertEqual(corr2, "official")

    def test_reconstruct_corrected_text(self):
        """Reconstruction preserves clean words while replacing low-confidence spans."""
        detector = ConfidenceDetector(threshold=70.0)
        raw_spans = detector.detect_low_confidence_spans(self.sample_words)
        extractor = SpanExtractor(context_window_words=2)
        ctx_spans = extractor.extract_contextual_spans("p01", self.sample_words, raw_spans)

        ctx_spans[0].corrected_text = "संचालित देश"
        reconstructed = reconstruct_corrected_text(self.sample_words, ctx_spans)

        expected = "भारत सरकार संचालित देश भर में विकास"
        self.assertEqual(reconstructed, expected)

    def test_correction_runner_page_evaluation(self):
        """CorrectionRunner processes a page, generates PageCorrectionResult, and computes recovery."""
        runner = CorrectionRunner()
        corr_res, raw_ev, corr_ev = runner.process_single_page("doc_hi_001_page_001", threshold=70.0)

        self.assertEqual(corr_res.page_id, "doc_hi_001_page_001")
        self.assertGreater(corr_res.total_words, 100)
        self.assertGreater(len(corr_res.spans), 0)
        self.assertLessEqual(corr_ev.cer, raw_ev.cer)  # CER should improve or equal
        self.assertLessEqual(corr_ev.wer, raw_ev.wer)  # WER should improve or equal


if __name__ == "__main__":
    unittest.main()
