"""Deterministic mock OCR engine for testing, simulation, and CI environments."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.core.schemas import PageOCRResult, WordOCR, Language
from src.ocr.base import BaseOCREngine


class MockOCREngine(BaseOCREngine):
    """Generates synthetic OCR output with configurable character error rate and calibrated confidences."""

    def __init__(
        self,
        error_rate: float = 0.08,
        seed: int = 42,
    ):
        super().__init__(name="mock_ocr", version="1.0.0-synthetic")
        self.error_rate = error_rate
        self.seed = seed
        self._rng = random.Random(seed)

    def _perturb_word(self, word: str, is_hindi: bool) -> tuple[str, float]:
        """Apply synthetic OCR noise and return (corrupted_word, confidence)."""
        if len(word) <= 1 or self._rng.random() > self.error_rate:
            # Uncorrupted word: high confidence
            conf = self._rng.uniform(88.0, 99.0)
            return word, round(conf, 1)

        # Corrupted word: low confidence
        conf = self._rng.uniform(32.0, 68.0)
        chars = list(word)
        idx = self._rng.randint(0, len(chars) - 1)

        if is_hindi:
            dev_substitutions = {
                "क": "फ", "व": "ब", "य": "थ", "र": "ट", "म": "भ",
                "न": "त", "ा": "ो", "ि": "ी", "ु": "ू"
            }
            chars[idx] = dev_substitutions.get(chars[idx], "र")
        else:
            latin_substitutions = {
                "e": "c", "c": "e", "l": "1", "1": "l", "o": "0",
                "0": "o", "m": "rn", "u": "v", "v": "u", "a": "o"
            }
            chars[idx] = latin_substitutions.get(chars[idx], "x")

        return "".join(chars), round(conf, 1)

    def extract_page(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        page_id: str,
        document_id: str,
        language: Union[str, Language],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        reference_text: Optional[str] = None,
    ) -> PageOCRResult:
        """Extract structured OCR tokens with synthetic noise."""
        start_time = time.perf_counter()
        lang_enum = Language(language) if isinstance(language, str) else language
        is_hindi = (lang_enum == Language.HINDI)

        # If reference text is provided, use it; otherwise read ground truth or fallback
        if reference_text is None:
            gt_candidate = Path(f"data/ground_truth/{page_id}.txt")
            if gt_candidate.exists():
                reference_text = gt_candidate.read_text(encoding="utf-8")
            else:
                reference_text = "Default synthetic OCR document text for testing."

        lines = reference_text.splitlines()
        words_out: List[WordOCR] = []
        raw_words_text: List[str] = []

        y_offset = 100
        line_num = 1

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("## "):
                continue

            tokens = line_str.split(" ")
            x_offset = 90
            word_num = 1

            for token in tokens:
                if not token:
                    continue

                perturbed_text, conf = self._perturb_word(token, is_hindi)
                w_width = len(token) * 15 + 10
                w_height = 25
                bbox = [x_offset, y_offset, w_width, w_height]

                w_obj = WordOCR(
                    text=perturbed_text,
                    confidence=conf,
                    bbox=bbox,
                    line_num=line_num,
                    word_num=word_num,
                )
                words_out.append(w_obj)
                raw_words_text.append(perturbed_text)

                x_offset += w_width + 12
                word_num += 1

            y_offset += 40
            line_num += 1

        exec_time = time.perf_counter() - start_time
        reconstructed_text = " ".join(raw_words_text)

        return PageOCRResult(
            page_id=page_id,
            document_id=document_id,
            language=lang_enum,
            engine=self.name,
            engine_version=self.version,
            preprocessing_config=preprocessing_config or {},
            raw_text=reconstructed_text,
            words=words_out,
            execution_time_sec=round(exec_time, 4),
        )
