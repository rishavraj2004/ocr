"""Dataset manager providing uniform access to documents, images, ground truth, and questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from src.core.config import AppConfig, DatasetConfig, get_default_config
from src.core.schemas import DocumentMetadata, Question, Language
from src.core.logging import get_logger
from src.dataset.validator import DatasetValidator, ValidationReport

logger = get_logger("dataset.manager")


class DatasetManager:
    """Interface for querying and persisting dataset records."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        self.ds_config: DatasetConfig = self.config.dataset

        self.images_dir = Path(self.ds_config.images_dir)
        self.ground_truth_dir = Path(self.ds_config.ground_truth_dir)
        self.metadata_path = Path(self.ds_config.metadata_path)
        self.questions_dir = Path(self.ds_config.questions_dir)
        self.questions_file = self.questions_dir / "questions.json"

        self._metadata_cache: Optional[Dict[str, DocumentMetadata]] = None
        self._questions_cache: Optional[List[Question]] = None

    def load_metadata(self, force_reload: bool = False) -> List[DocumentMetadata]:
        """Load and cache all DocumentMetadata objects."""
        if self._metadata_cache is not None and not force_reload:
            return list(self._metadata_cache.values())

        if not self.metadata_path.exists():
            logger.warning(f"Metadata file does not exist at {self.metadata_path}")
            return []

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._metadata_cache = {
            item["page_id"]: DocumentMetadata.model_validate(item) for item in raw_data
        }
        return list(self._metadata_cache.values())

    def get_page_metadata(self, page_id: str) -> Optional[DocumentMetadata]:
        """Fetch metadata for a specific page ID."""
        if self._metadata_cache is None:
            self.load_metadata()
        return self._metadata_cache.get(page_id) if self._metadata_cache else None

    def get_pages_by_language(self, language: str) -> List[DocumentMetadata]:
        """Filter pages by language code ('en' or 'hi')."""
        all_pages = self.load_metadata()
        target_lang = Language(language)
        return [p for p in all_pages if p.language == target_lang]

    def load_ground_truth(self, page_id: str) -> str:
        """Read and return ground-truth text for a page."""
        meta = self.get_page_metadata(page_id)
        if meta is None:
            raise KeyError(f"Page ID '{page_id}' not found in metadata.")

        gt_path = Path(meta.ground_truth_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file missing at: {gt_path}")

        return gt_path.read_text(encoding="utf-8")

    def load_image(self, page_id: str) -> Image.Image:
        """Load PIL Image object for a given page, ensuring file handles are closed."""
        meta = self.get_page_metadata(page_id)
        if meta is None:
            raise KeyError(f"Page ID '{page_id}' not found in metadata.")

        img_path = Path(meta.image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Page image missing at: {img_path}")

        with Image.open(img_path) as img:
            return img.copy()

    def load_questions(
        self,
        page_id: Optional[str] = None,
        language: Optional[str] = None,
        force_reload: bool = False,
    ) -> List[Question]:
        """Load evaluation questions with optional filtering by page_id or language."""
        if self._questions_cache is None or force_reload:
            if not self.questions_file.exists():
                logger.warning(f"Questions file not found at {self.questions_file}")
                return []
            with open(self.questions_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self._questions_cache = [Question.model_validate(q) for q in raw_data]

        filtered = self._questions_cache
        if page_id is not None:
            filtered = [q for q in filtered if q.page_id == page_id]
        if language is not None:
            target_lang = Language(language)
            filtered = [q for q in filtered if q.language == target_lang]

        return filtered

    def save_metadata(self, metadata_list: List[DocumentMetadata]) -> None:
        """Persist metadata list to JSON."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.model_dump() for m in metadata_list]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._metadata_cache = {m.page_id: m for m in metadata_list}
        logger.info(f"Saved {len(metadata_list)} metadata entries to {self.metadata_path}")

    def save_questions(self, questions_list: List[Question]) -> None:
        """Persist evaluation questions to JSON."""
        self.questions_file.parent.mkdir(parents=True, exist_ok=True)
        data = [q.model_dump() for q in questions_list]
        with open(self.questions_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._questions_cache = list(questions_list)
        logger.info(f"Saved {len(questions_list)} questions to {self.questions_file}")

    def save_ground_truth(self, page_id: str, text: str, rel_path: Optional[str] = None) -> Path:
        """Write ground truth transcript text."""
        if rel_path is None:
            rel_path = f"data/ground_truth/{page_id}.txt"
        target_path = Path(rel_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text, encoding="utf-8")
        return target_path

    def validate(self) -> ValidationReport:
        """Execute validation across images, ground truth, metadata, and questions."""
        validator = DatasetValidator(
            metadata_path=self.metadata_path,
            images_dir=self.images_dir,
            ground_truth_dir=self.ground_truth_dir,
            questions_path=self.questions_file,
        )
        return validator.validate()
