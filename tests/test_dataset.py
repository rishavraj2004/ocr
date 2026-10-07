"""Unit tests for dataset ingestion, management, and validation."""

import json
import tempfile
import unittest
from pathlib import Path

from src.core.schemas import Language, QuestionType
from src.dataset.manager import DatasetManager
from src.dataset.validator import DatasetValidator


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dm = DatasetManager()

    def test_benchmark_dataset_integrity(self):
        """The committed 10-page benchmark dataset must be 100% valid."""
        report = self.dm.validate()
        self.assertTrue(report.is_valid, f"Validation errors: {report.errors}")
        self.assertEqual(report.total_pages, 10)
        self.assertEqual(report.pages_by_language, {"en": 5, "hi": 5})
        self.assertGreaterEqual(report.total_questions, 30)
        self.assertEqual(len(report.errors), 0)

    def test_metadata_loading(self):
        """Metadata loading returns 10 valid DocumentMetadata entries."""
        metadata = self.dm.load_metadata()
        self.assertEqual(len(metadata), 10)

        # Check English and Hindi pages
        en_pages = self.dm.get_pages_by_language("en")
        hi_pages = self.dm.get_pages_by_language("hi")
        self.assertEqual(len(en_pages), 5)
        self.assertEqual(len(hi_pages), 5)

    def test_ground_truth_and_image_loading(self):
        """Ground truth text and image files can be loaded cleanly."""
        meta = self.dm.get_page_metadata("doc_en_001_page_001")
        self.assertIsNotNone(meta)

        gt_text = self.dm.load_ground_truth("doc_en_001_page_001")
        self.assertIn("300 gigawatts", gt_text)

        img = self.dm.load_image("doc_en_001_page_001")
        self.assertEqual(img.size, (1240, 1754))

        # Check Hindi ground truth
        hi_gt = self.dm.load_ground_truth("doc_hi_001_page_001")
        self.assertIn("८८ लाख", hi_gt)

    def test_questions_querying(self):
        """Questions can be queried globally or filtered by page/language."""
        all_q = self.dm.load_questions()
        self.assertGreaterEqual(len(all_q), 30)

        en_q = self.dm.load_questions(language="en")
        hi_q = self.dm.load_questions(language="hi")
        self.assertEqual(len(en_q) + len(hi_q), len(all_q))

        p_q = self.dm.load_questions(page_id="doc_en_001_page_001")
        self.assertEqual(len(p_q), 4)
        for q in p_q:
            self.assertEqual(q.page_id, "doc_en_001_page_001")

    def test_validator_detects_missing_image(self):
        """Validator must report error when an image file is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            meta_file = tmp / "meta.json"
            gt_file = tmp / "page.txt"
            gt_file.write_text("Sample text", encoding="utf-8")

            fake_meta = [{
                "document_id": "doc_x",
                "page_id": "doc_x_p1",
                "language": "en",
                "page_number": 1,
                "doc_type": "report",
                "layout_type": "single_column",
                "image_path": str(tmp / "nonexistent.png"),
                "ground_truth_path": str(gt_file),
            }]
            meta_file.write_text(json.dumps(fake_meta), encoding="utf-8")

            validator = DatasetValidator(
                metadata_path=meta_file,
                images_dir=tmp,
                ground_truth_dir=tmp,
                questions_path=tmp / "empty_q.json",
            )
            report = validator.validate()
            self.assertFalse(report.is_valid)
            self.assertTrue(any("Referenced image not found" in err for err in report.errors))

    def test_validator_detects_missing_ground_truth(self):
        """Validator must report error when ground truth text file is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            meta_file = tmp / "meta.json"
            img_file = tmp / "page.png"
            from PIL import Image
            Image.new("RGB", (100, 100), color="white").save(img_file)

            fake_meta = [{
                "document_id": "doc_x",
                "page_id": "doc_x_p1",
                "language": "en",
                "page_number": 1,
                "doc_type": "report",
                "layout_type": "single_column",
                "image_path": str(img_file),
                "ground_truth_path": str(tmp / "missing_gt.txt"),
            }]
            meta_file.write_text(json.dumps(fake_meta), encoding="utf-8")

            validator = DatasetValidator(
                metadata_path=meta_file,
                images_dir=tmp,
                ground_truth_dir=tmp,
                questions_path=tmp / "empty_q.json",
            )
            report = validator.validate()
            self.assertFalse(report.is_valid)
            self.assertTrue(any("Ground truth file not found" in err for err in report.errors))

    def test_validator_detects_orphan_question(self):
        """Validator must report error when a question references an unknown page_id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            meta_file = tmp / "meta.json"
            q_file = tmp / "questions.json"
            img_file = tmp / "page.png"
            gt_file = tmp / "page.txt"

            from PIL import Image
            Image.new("RGB", (100, 100), color="white").save(img_file)
            gt_file.write_text("Sample text", encoding="utf-8")

            fake_meta = [{
                "document_id": "doc_x",
                "page_id": "doc_x_p1",
                "language": "en",
                "page_number": 1,
                "doc_type": "report",
                "layout_type": "single_column",
                "image_path": str(img_file),
                "ground_truth_path": str(gt_file),
            }]
            meta_file.write_text(json.dumps(fake_meta), encoding="utf-8")

            fake_questions = [{
                "question_id": "q_orphan",
                "document_id": "doc_unknown",
                "page_id": "doc_unknown_p1",
                "language": "en",
                "question": "Where is this from?",
                "expected_answer": "Nowhere",
                "source_page": 1,
                "question_type": "factual",
            }]
            q_file.write_text(json.dumps(fake_questions), encoding="utf-8")

            validator = DatasetValidator(
                metadata_path=meta_file,
                images_dir=tmp,
                ground_truth_dir=tmp,
                questions_path=q_file,
            )
            report = validator.validate()
            self.assertFalse(report.is_valid)
            self.assertTrue(any("references unknown page_id" in err for err in report.errors))

    def test_validator_detects_language_mismatch(self):
        """Validator must report error when question language contradicts page language."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            meta_file = tmp / "meta.json"
            q_file = tmp / "questions.json"
            img_file = tmp / "page.png"
            gt_file = tmp / "page.txt"

            from PIL import Image
            Image.new("RGB", (100, 100), color="white").save(img_file)
            gt_file.write_text("Sample text", encoding="utf-8")

            fake_meta = [{
                "document_id": "doc_x",
                "page_id": "doc_x_p1",
                "language": "hi",  # Hindi page
                "page_number": 1,
                "doc_type": "report",
                "layout_type": "single_column",
                "image_path": str(img_file),
                "ground_truth_path": str(gt_file),
            }]
            meta_file.write_text(json.dumps(fake_meta), encoding="utf-8")

            fake_questions = [{
                "question_id": "q_mismatch",
                "document_id": "doc_x",
                "page_id": "doc_x_p1",
                "language": "en",  # English question against Hindi page
                "question": "What is this?",
                "expected_answer": "Mismatch",
                "source_page": 1,
                "question_type": "factual",
            }]
            q_file.write_text(json.dumps(fake_questions), encoding="utf-8")

            validator = DatasetValidator(
                metadata_path=meta_file,
                images_dir=tmp,
                ground_truth_dir=tmp,
                questions_path=q_file,
            )
            report = validator.validate()
            self.assertFalse(report.is_valid)
            self.assertTrue(any("does not match page language" in err for err in report.errors))


if __name__ == "__main__":
    unittest.main()
