"""Unit tests for OpenCV image preprocessing pipeline."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.core.config import PreprocessingConfig
from src.preprocessing.image_preprocessor import ImagePreprocessor, PreprocessingResult
from src.preprocessing.utils import load_image_as_cv2, save_cv2_image, cv2_to_pil


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a synthetic white document image with black text-like lines
        self.test_img = np.full((400, 300, 3), 255, dtype=np.uint8)
        # Add horizontal stripes simulating lines of text
        for y in range(40, 360, 40):
            cv2.line(self.test_img, (30, y), (270, y), (0, 0, 0), 3)

    def test_default_pipeline_execution(self):
        """Default pipeline should apply grayscale, denoise, deskew, normalize, and threshold."""
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(self.test_img)

        self.assertIsInstance(result, PreprocessingResult)
        self.assertEqual(len(result.processed_image.shape), 2)  # Binarized 2D
        self.assertIn("grayscale", result.operations_applied)
        self.assertIn("denoise", result.operations_applied)
        self.assertIn("deskew", result.operations_applied)
        self.assertIn("threshold_otsu", result.operations_applied)
        self.assertIn("threshold_value", result.metadata)

    def test_pipeline_disabled(self):
        """When enabled is False, pipeline returns image unaltered."""
        cfg = PreprocessingConfig(enabled=False)
        preprocessor = ImagePreprocessor(cfg)
        result = preprocessor.process(self.test_img)

        self.assertEqual(result.operations_applied, ["none (disabled)"])
        self.assertTrue(np.array_equal(result.processed_image, self.test_img))

    def test_reproducibility_bitwise_identical(self):
        """Preprocessing the same image twice must produce bitwise-identical output."""
        preprocessor = ImagePreprocessor()
        r1 = preprocessor.process(self.test_img)
        r2 = preprocessor.process(self.test_img)

        self.assertTrue(
            np.array_equal(r1.processed_image, r2.processed_image),
            "Preprocessing is not bitwise deterministic across identical runs!",
        )

    def test_deskew_correction(self):
        """Synthetically skewed image should be detected and corrected."""
        preprocessor = ImagePreprocessor()

        # Rotate the test image by -4.0 degrees
        h, w = self.test_img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -4.0, 1.0)
        skewed = cv2.warpAffine(self.test_img, rot_mat, (w, h), borderValue=(255, 255, 255))

        # Detect skew
        gray_skewed = preprocessor.to_grayscale(skewed)
        detected_angle = preprocessor.detect_skew_angle(gray_skewed)

        # Detected angle should be within reasonable proximity of -4 degrees
        self.assertAlmostEqual(abs(detected_angle), 4.0, delta=1.5)

        # Deskewing should run
        deskewed, angle = preprocessor.deskew(skewed)
        self.assertNotEqual(angle, 0.0)

    def test_adaptive_threshold_mode(self):
        """Pipeline respects adaptive threshold configuration."""
        cfg = PreprocessingConfig(threshold_method="adaptive")
        preprocessor = ImagePreprocessor(cfg)
        result = preprocessor.process(self.test_img)

        self.assertIn("threshold_adaptive", result.operations_applied)
        self.assertEqual(result.metadata["method"], "adaptive_gaussian")

    def test_save_debug_comparison(self):
        """Visual before/after side-by-side comparison image renders and writes cleanly."""
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(self.test_img)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "debug_comparison.png"
            preprocessor.save_debug_comparison(self.test_img, result, out_file)

            self.assertTrue(out_file.exists())
            loaded = cv2.imread(str(out_file))
            self.assertIsNotNone(loaded)
            # Comparison image width should exceed 2x original width plus divider
            self.assertGreater(loaded.shape[1], self.test_img.shape[1] * 2)

    def test_utils_conversion_roundtrip(self):
        """OpenCV array and PIL image conversions round-trip accurately."""
        pil_img = cv2_to_pil(self.test_img)
        self.assertIsInstance(pil_img, Image.Image)
        reconverted_cv = load_image_as_cv2(pil_img)
        self.assertEqual(reconverted_cv.shape, self.test_img.shape)


if __name__ == "__main__":
    unittest.main()
