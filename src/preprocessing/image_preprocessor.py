"""Configurable OpenCV image preprocessing pipeline for scanned document pages.

Applies deterministic transformations (grayscale, denoising, deskewing, thresholding,
and normalization) and logs exact parameters for experimental reproducibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import PreprocessingConfig
from src.core.logging import get_logger
from src.preprocessing.utils import load_image_as_cv2, save_cv2_image

logger = get_logger("preprocessing")


class PreprocessingResult:
    """Container holding preprocessed image, applied operations, and metadata."""

    def __init__(
        self,
        processed_image: np.ndarray,
        original_shape: Tuple[int, ...],
        operations_applied: List[str],
        metadata: Dict[str, Any],
        execution_time_sec: float,
    ):
        self.processed_image = processed_image
        self.original_shape = original_shape
        self.processed_shape = processed_image.shape
        self.operations_applied = operations_applied
        self.metadata = metadata
        self.execution_time_sec = execution_time_sec

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata excluding raw numpy pixel buffer."""
        return {
            "original_shape": list(self.original_shape),
            "processed_shape": list(self.processed_shape),
            "operations_applied": list(self.operations_applied),
            "execution_time_sec": self.execution_time_sec,
            "metadata": self.metadata,
        }


class ImagePreprocessor:
    """Modular, configurable OpenCV document image preprocessor."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert multi-channel image to single-channel 8-bit grayscale."""
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    def denoise(self, image: np.ndarray, h: float = 10.0) -> np.ndarray:
        """Apply Non-Local Means Denoising tailored for document text clarity."""
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(
                image, None, h=h, templateWindowSize=7, searchWindowSize=21
            )
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
        )

    def detect_skew_angle(self, gray_image: np.ndarray) -> float:
        """Estimate document skew angle in degrees using minimum bounding box of text."""
        # Invert: text should be white, background black
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Extract non-zero coordinates
        pts = np.column_stack(np.where(thresh > 0))
        if len(pts) < 100:
            return 0.0  # Not enough content to determine skew safely

        # cv2.minAreaRect returns: (center(x, y), size(width, height), angle of rotation)
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]

        # Handle OpenCV minAreaRect angle conventions
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        return float(angle)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around its center with white boundary padding."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_val = 255 if len(image.shape) == 2 else (255, 255, 255)

        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        return rotated

    def deskew(self, image: np.ndarray, min_angle_deg: float = 0.2) -> Tuple[np.ndarray, float]:
        """Detect and correct document skew if angle exceeds threshold."""
        gray = self.to_grayscale(image)
        angle = self.detect_skew_angle(gray)

        if abs(angle) >= min_angle_deg:
            logger.debug(f"Deskewing image by {angle:.2f} degrees")
            return self.rotate_image(image, angle), angle
        return image, 0.0

    def threshold(self, gray_image: np.ndarray, method: str = "otsu") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Binarize grayscale image using Otsu or Adaptive Gaussian thresholding."""
        if len(gray_image.shape) != 2:
            gray_image = self.to_grayscale(gray_image)

        if method.lower() == "adaptive":
            thresh = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=21,
                C=10,
            )
            return thresh, {"method": "adaptive_gaussian", "block_size": 21, "c": 10}
        else:
            thresh_val, thresh = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh, {"method": "otsu", "threshold_value": float(thresh_val)}

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Stretch intensity distribution across the dynamic range [0, 255]."""
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def process(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> PreprocessingResult:
        """Run the configured preprocessing pipeline on the input image."""
        start_time = time.perf_counter()
        img = load_image_as_cv2(image_input)
        original_shape = img.shape

        operations_applied: List[str] = []
        metadata: Dict[str, Any] = {}

        if not self.config.enabled:
            exec_time = time.perf_counter() - start_time
            return PreprocessingResult(
                processed_image=img,
                original_shape=original_shape,
                operations_applied=["none (disabled)"],
                metadata={"enabled": False},
                execution_time_sec=exec_time,
            )

        # 1. Grayscale
        if self.config.grayscale:
            img = self.to_grayscale(img)
            operations_applied.append("grayscale")

        # 2. Denoising
        if self.config.denoise:
            img = self.denoise(img, h=self.config.denoise_h)
            operations_applied.append("denoise")
            metadata["denoise_h"] = self.config.denoise_h

        # 3. Deskewing
        if self.config.deskew:
            # Requires grayscale for angle calculation
            deskew_input = self.to_grayscale(img) if len(img.shape) == 3 else img
            img, detected_angle = self.deskew(deskew_input)
            operations_applied.append("deskew")
            metadata["skew_angle_deg"] = detected_angle

        # 4. Normalization (applied before thresholding if both enabled)
        if self.config.normalize:
            img = self.normalize(img)
            operations_applied.append("normalize")

        # 5. Thresholding (Binarization)
        if self.config.threshold:
            gray_for_thresh = self.to_grayscale(img) if len(img.shape) == 3 else img
            img, thresh_meta = self.threshold(gray_for_thresh, method=self.config.threshold_method)
            operations_applied.append(f"threshold_{self.config.threshold_method}")
            metadata.update(thresh_meta)

        exec_time = time.perf_counter() - start_time

        logger.debug(
            f"Preprocessed image ({original_shape} -> {img.shape}) "
            f"in {exec_time:.3f}s: {operations_applied}"
        )

        return PreprocessingResult(
            processed_image=img,
            original_shape=original_shape,
            operations_applied=operations_applied,
            metadata=metadata,
            execution_time_sec=exec_time,
        )

    def create_comparison_image(
        self,
        original_input: Union[str, Path, np.ndarray, Image.Image],
        result: PreprocessingResult,
    ) -> np.ndarray:
        """Construct a visual side-by-side comparison for inspection and debugging."""
        orig = load_image_as_cv2(original_input)
        proc = result.processed_image

        # Match dimensions and color channels
        if len(orig.shape) == 2:
            orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = orig.copy()

        if len(proc.shape) == 2:
            proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        else:
            proc_bgr = proc.copy()

        # Resize proc to match orig height if deskew or crop changed dimensions
        if orig_bgr.shape[:2] != proc_bgr.shape[:2]:
            proc_bgr = cv2.resize(proc_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

        h, w = orig_bgr.shape[:2]

        # Top banner for annotations
        banner_h = 70
        banner = np.full((banner_h, w * 2 + 20, 3), 240, dtype=np.uint8)

        cv2.putText(
            banner,
            f"ORIGINAL SCAN: {w}x{h}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

        ops_str = " -> ".join(result.operations_applied)
        cv2.putText(
            banner,
            f"PREPROCESSED: {ops_str}",
            (w + 50, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 80, 20),
            2,
            cv2.LINE_AA,
        )

        # Divider between panels
        divider = np.full((h, 20, 3), 200, dtype=np.uint8)

        # Concatenate panels
        main_panels = np.hstack([orig_bgr, divider, proc_bgr])
        full_comparison = np.vstack([banner, main_panels])

        return full_comparison

    def save_debug_comparison(
        self,
        original_input: Union[str, Path, np.ndarray, Image.Image],
        result: PreprocessingResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Render and save side-by-side before/after comparison image to disk."""
        comparison_img = self.create_comparison_image(original_input, result)
        return save_cv2_image(comparison_img, output_path)
