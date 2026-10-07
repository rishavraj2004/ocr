"""Tesseract OCR engine integration with word-level confidence and bounding-box extraction."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import pytesseract

from src.core.config import OCRConfig
from src.core.logging import get_logger
from src.core.schemas import PageOCRResult, WordOCR, Language
from src.ocr.base import BaseOCREngine
from src.preprocessing.utils import load_image_as_cv2

logger = get_logger("ocr.tesseract")

# Standard Windows installation paths for Tesseract
STANDARD_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def resolve_tesseract_cmd(custom_cmd: Optional[str] = None) -> Optional[str]:
    """Locate the tesseract executable across config, PATH, and standard Windows directories."""
    if custom_cmd and Path(custom_cmd).exists():
        return custom_cmd

    # Check system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # Check standard Windows paths
    for p in STANDARD_WINDOWS_PATHS:
        if Path(p).exists():
            return p

    return None


class TesseractOCREngine(BaseOCREngine):
    """Concrete OCR engine utilizing Tesseract via pytesseract with full token-level telemetry."""

    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(name="tesseract", version="unknown")
        self.config = config or OCRConfig()

        cmd_path = resolve_tesseract_cmd(self.config.tesseract_cmd)
        if cmd_path:
            pytesseract.pytesseract.tesseract_cmd = cmd_path
            logger.info(f"Tesseract binary resolved at: {cmd_path}")
            try:
                self.version = str(pytesseract.get_tesseract_version())
            except Exception as e:
                logger.warning(f"Could not read Tesseract version: {e}")
        else:
            logger.warning(
                "Tesseract executable not detected on system. "
                "Ensure Tesseract is installed or specify path in config."
            )

        if self.config.tessdata_dir and Path(self.config.tessdata_dir).exists():
            os.environ["TESSDATA_PREFIX"] = str(Path(self.config.tessdata_dir).resolve())

    @property
    def is_available(self) -> bool:
        """Check if Tesseract binary is functional."""
        try:
            return resolve_tesseract_cmd(self.config.tesseract_cmd) is not None
        except Exception:
            return False

    def get_available_languages(self) -> List[str]:
        """Query installed tessdata models."""
        try:
            return pytesseract.get_languages()
        except Exception:
            return []

    def extract_page(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        page_id: str,
        document_id: str,
        language: Union[str, Language],
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> PageOCRResult:
        """Execute Tesseract OCR and extract text, confidences, and bounding boxes."""
        start_time = time.perf_counter()

        lang_enum = Language(language) if isinstance(language, str) else language
        tess_lang = self.config.lang_map.get(lang_enum.value, "eng")

        # Fallback check for Hindi model
        available_langs = self.get_available_languages()
        if available_langs and tess_lang not in available_langs:
            logger.warning(
                f"Tesseract language '{tess_lang}' not found in installed tessdata ({available_langs}). "
                "Falling back to 'eng'."
            )
            tess_lang = "eng"

        # Load image via preprocessing utils
        cv2_img = load_image_as_cv2(image_input)

        # Configure PSM and OEM flags
        custom_oem_psm = f"--oem {self.config.oem} --psm {self.config.psm}"

        try:
            data = pytesseract.image_to_data(
                cv2_img,
                lang=tess_lang,
                config=custom_oem_psm,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            logger.error(f"Pytesseract execution failed for page '{page_id}': {e}")
            raise RuntimeError(f"Tesseract OCR failed: {e}") from e

        n_boxes = len(data["text"])
        words_out: List[WordOCR] = []
        raw_words_text: List[str] = []

        for i in range(n_boxes):
            word_str = str(data["text"][i]).strip()
            conf_val = float(data["conf"][i])

            # Filter out non-word blocks (Tesseract flags empty layout blocks with conf -1)
            if conf_val < 0 or not word_str:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            w_obj = WordOCR(
                text=word_str,
                confidence=round(conf_val, 1),
                bbox=[x, y, w, h],
                line_num=int(data["line_num"][i]),
                word_num=int(data["word_num"][i]),
            )
            words_out.append(w_obj)
            raw_words_text.append(word_str)

        exec_time = time.perf_counter() - start_time
        reconstructed_text = " ".join(raw_words_text)

        logger.info(
            f"Extracted {len(words_out)} words for page '{page_id}' "
            f"in {exec_time:.3f}s using engine '{self.name}' v{self.version}"
        )

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
