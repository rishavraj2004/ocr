"""OCR engine abstraction, runner, and factory."""

from typing import Optional
from src.core.config import OCRConfig
from src.ocr.base import BaseOCREngine
from src.ocr.tesseract_ocr import TesseractOCREngine, resolve_tesseract_cmd
from src.ocr.mock_ocr import MockOCREngine
from src.ocr.runner import OCRRunner

__all__ = [
    "BaseOCREngine",
    "TesseractOCREngine",
    "MockOCREngine",
    "OCRRunner",
    "get_ocr_engine",
    "resolve_tesseract_cmd",
]


def get_ocr_engine(
    engine_name: Optional[str] = None,
    config: Optional[OCRConfig] = None,
) -> BaseOCREngine:
    """Instantiate and return requested OCR engine."""
    cfg = config or OCRConfig()
    name = (engine_name or cfg.engine).lower()

    if name == "tesseract":
        return TesseractOCREngine(cfg)
    elif name == "mock":
        return MockOCREngine()
    else:
        raise ValueError(f"Unsupported OCR engine requested: '{name}'. Supported: 'tesseract', 'mock'")
