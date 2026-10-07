"""Unit tests for Pydantic data schemas and contract enforcement."""

import unittest
from src.core.schemas import (
    Language,
    DocumentVariant,
    QuestionType,
    DocumentMetadata,
    WordOCR,
    PageOCRResult,
    CorrectionSpan,
    PageCorrectionResult,
    Question,
    TextChunk,
    RetrievalItem,
    QuestionRetrievalResult,
    QuestionAnswerResult,
)


class TestSchemas(unittest.TestCase):
    def test_document_metadata_schema(self):
        doc = DocumentMetadata(
            document_id="doc_001",
            page_id="doc_001_page_001",
            language=Language.HINDI,
            image_path="data/images/doc_001_page_001.png",
            ground_truth_path="data/ground_truth/doc_001_page_001.txt",
        )
        self.assertEqual(doc.language, "hi")
        self.assertEqual(doc.page_number, 1)

    def test_word_ocr_confidence_bounds(self):
        word = WordOCR(text="भारत", confidence=96.2, bbox=[10, 20, 80, 50])
        self.assertEqual(word.confidence, 96.2)

        # Confidence out of bounds should fail
        with self.assertRaises(ValueError):
            WordOCR(text="test", confidence=105.0, bbox=[0, 0, 10, 10])
        with self.assertRaises(ValueError):
            WordOCR(text="test", confidence=-5.0, bbox=[0, 0, 10, 10])

    def test_page_ocr_result_schema(self):
        page = PageOCRResult(
            page_id="doc_001_page_001",
            document_id="doc_001",
            language=Language.HINDI,
            engine="tesseract",
            raw_text="भारत सरकार",
            words=[
                WordOCR(text="भारत", confidence=96.2, bbox=[10, 20, 80, 50]),
                WordOCR(text="सरकार", confidence=94.1, bbox=[90, 20, 170, 50]),
            ],
        )
        self.assertEqual(len(page.words), 2)
        self.assertEqual(page.words[0].text, "भारत")

    def test_correction_span_schema(self):
        span = CorrectionSpan(
            span_id="span_01",
            page_id="doc_001_page_001",
            start_word_idx=4,
            end_word_idx=5,
            original_text="सरकर",
            avg_confidence=42.0,
            min_confidence=42.0,
            context_before="ग्रामीण",
            context_after="के विकास",
            corrected_text="सरकार",
            changed=True,
        )
        self.assertTrue(span.changed)
        self.assertEqual(span.corrected_text, "सरकार")

    def test_question_schema(self):
        q = Question(
            question_id="q001",
            document_id="doc_001",
            page_id="doc_001_page_001",
            language=Language.HINDI,
            question="योजना का मुख्य उद्देश्य क्या है?",
            expected_answer="ग्रामीण विकास",
            source_page=1,
            question_type=QuestionType.FACTUAL,
        )
        self.assertEqual(q.language, Language.HINDI)
        self.assertEqual(q.question_type, "factual")

    def test_document_variants(self):
        variants = {v.value for v in DocumentVariant}
        self.assertIn("ground_truth", variants)
        self.assertIn("raw_ocr", variants)
        self.assertIn("corrected_ocr", variants)


if __name__ == "__main__":
    unittest.main()
