"""Abstract base class for OCR engines.

Decouples the OCR backend (Tesseract, PaddleOCR, Mock) from downstream
confidence processing, selective correction, and RAG evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

from src.core.schemas import PageOCRResult, Language


class BaseOCREngine(ABC):
    """Abstract interface for all pluggable OCR engines."""

    def __init__(self, name: str, version: str = "unknown"):
        self.name = name
        self.version = version

    @abstractmethod
    def extract_page(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        page_id: str,
        document_id: str,
        language: Union[str, Language],
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> PageOCRResult:
        """Extract word-level text, confidence scores, and bounding boxes for a document page."""
        pass
