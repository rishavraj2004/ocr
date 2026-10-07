"""Unit tests for OCR extraction, mock simulation, and CER/WER evaluation."""

import tempfile
import unicodedata
import unittest
from pathlib import Path
import numpy as np

from src.core.schemas import Language, DocumentVariant
from src.evaluation.ocr_metrics import calculate_cer, calculate_wer, evaluate_ocr_page
from src.ocr.mock_ocr import MockOCREngine
from src.ocr.tesseract_ocr import TesseractOCREngine, resolve_tesseract_cmd
from src.ocr.runner import OCRRunner


class TestOCR(unittest.TestCase):
    def test_cer_exact_match(self):
        """Identical strings must yield CER = 0.0."""
        text = "National Renewable Energy Transition Report 2025"
        self.assertEqual(calculate_cer(text, text), 0.0)

        hi_text = "राष्ट्रीय ग्रामीण आजीविका मिशन २०२४"
        self.assertEqual(calculate_cer(hi_text, hi_text), 0.0)

    def test_cer_edits(self):
        """Single character substitution in a 10-char string yields CER = 0.1."""
        ref = "abcdefghij"
        hyp = "abcdefghix"
        self.assertAlmostEqual(calculate_cer(ref, hyp), 0.1, places=3)

    def test_wer_exact_match(self):
        """Identical word sequences must yield WER = 0.0."""
        text = "Offshore wind turbines deliver clean renewable power"
        self.assertEqual(calculate_wer(text, text), 0.0)

    def test_wer_word_substitution(self):
        """One word substitution in 4 words yields WER = 0.25."""
        ref = "solar power generation capacity"
        hyp = "solar wind generation capacity"
        self.assertAlmostEqual(calculate_wer(ref, hyp), 0.25, places=3)

    def test_unicode_nfc_normalization(self):
        """Devanagari text with composite nuktas in NFD vs NFC must evaluate to CER = 0.0."""
        # \u0958 is precomposed क़ (QA), which decomposes in NFD to \u0915\u093c (KA + NUKTA)
        nfc_text = "\u0958\u0959\u095a\u095b\u095c\u095d\u095e"  # क़ख़ग़ज़ड़ढ़फ़
        nfd_text = unicodedata.normalize("NFD", nfc_text)

        # In raw bytes, decomposed string is longer and not equal
        self.assertNotEqual(nfc_text, nfd_text)
        self.assertGreater(len(nfd_text), len(nfc_text))

        # With NFC normalization enabled, CER is exactly 0.0
        cer = calculate_cer(nfc_text, nfd_text, nfc=True)
        self.assertEqual(cer, 0.0)

    def test_mock_ocr_engine_extraction(self):
        """MockOCREngine must return structured PageOCRResult with word tokens and confidences."""
        engine = MockOCREngine(error_rate=0.1, seed=42)
        blank_img = np.full((100, 100, 3), 255, dtype=np.uint8)

        res = engine.extract_page(
            image_input=blank_img,
            page_id="test_p1",
            document_id="test_doc",
            language=Language.ENGLISH,
            reference_text="This is a verified test document paragraph for mock OCR testing.",
        )

        self.assertEqual(res.page_id, "test_p1")
        self.assertGreater(len(res.words), 5)
        for w in res.words:
            self.assertGreaterEqual(w.confidence, 0.0)
            self.assertLessEqual(w.confidence, 100.0)
            self.assertEqual(len(w.bbox), 4)

    def test_tesseract_ocr_engine_interface(self):
        """TesseractOCREngine initializes cleanly and reports availability status."""
        engine = TesseractOCREngine()
        self.assertIsInstance(engine.is_available, bool)
        self.assertEqual(engine.name, "tesseract")

    def test_ocr_runner_single_page(self):
        """OCRRunner processes a benchmark page, outputs tokens, and computes CER/WER."""
        runner = OCRRunner()
        ocr_result, eval_summary = runner.process_single_page("doc_en_001_page_001")

        self.assertEqual(ocr_result.page_id, "doc_en_001_page_001")
        self.assertGreater(len(ocr_result.words), 20)
        self.assertEqual(eval_summary.page_id, "doc_en_001_page_001")
        self.assertGreater(eval_summary.cer, 0.0)
        self.assertGreater(eval_summary.wer, 0.0)

        # Verify JSON file written to disk
        expected_json = Path(f"data/ocr/doc_en_001_page_001.json")
        self.assertTrue(expected_json.exists())


if __name__ == "__main__":
    unittest.main()
