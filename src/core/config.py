"""Hierarchical configuration system using Pydantic V2 and PyYAML.

Provides strongly typed configuration classes with default research values,
validation, and serialization to/from YAML files for experimental reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """File paths and dataset subset parameters."""
    images_dir: str = "data/images"
    ground_truth_dir: str = "data/ground_truth"
    ocr_dir: str = "data/ocr"
    corrected_dir: str = "data/corrected"
    questions_dir: str = "data/questions"
    metadata_path: str = "data/metadata/documents.json"
    languages: List[str] = ["en", "hi"]


class PreprocessingConfig(BaseModel):
    """Image preprocessing parameters applied before OCR."""
    enabled: bool = True
    grayscale: bool = True
    denoise: bool = True
    denoise_h: float = 10.0
    threshold: bool = True
    threshold_method: str = "otsu"  # "otsu" | "adaptive"
    deskew: bool = True
    normalize: bool = True


class OCRConfig(BaseModel):
    """OCR engine selection and execution parameters."""
    engine: str = "tesseract"  # "tesseract" | "paddleocr" | "mock"
    tesseract_cmd: Optional[str] = None
    tessdata_dir: Optional[str] = None
    psm: int = 3
    oem: int = 3
    lang_map: Dict[str, str] = Field(default_factory=lambda: {"en": "eng", "hi": "hin"})


class CorrectionConfig(BaseModel):
    """Confidence-guided selective correction configuration."""
    threshold: float = Field(default=70.0, ge=0.0, le=100.0)
    context_window_words: int = Field(default=5, ge=1)
    model_type: str = "mock"  # "openai" | "local" | "mock"
    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=150, ge=1)
    sweep_thresholds: List[float] = Field(
        default_factory=lambda: [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    )


class RAGConfig(BaseModel):
    """RAG pipeline configuration held strictly identical across all 3 variants."""
    chunk_size: int = Field(default=500, ge=50, description="Chunk size in units")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap in units")
    chunk_unit: str = "words"  # "words" | "tokens"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    distance_metric: str = "cosine"  # "cosine" | "l2"
    top_k: int = Field(default=5, ge=1)
    generator_type: str = "mock"  # "openai" | "local" | "mock"
    generator_model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    prompt_template: str = (
        "You are an assistant answering questions based strictly on the provided context.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely and accurately:"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be strictly less than chunk_size ({chunk_size})")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation metrics and text normalization rules."""
    ocr_metrics: List[str] = ["cer", "wer"]
    retrieval_k_values: List[int] = [1, 3, 5]
    answer_metrics: List[str] = ["exact_match", "f1"]
    normalize_unicode_nfc: bool = True
    ignore_case: bool = True
    ignore_punctuation: bool = True


class AppConfig(BaseModel):
    """Top-level configuration for the OCR-Aware RAG research prototype."""
    experiment_name: str = "ocr_multilingual_rag"
    seed: int = 42
    output_dir: str = "results"
    runs_dir: str = "experiments/runs"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    correction: CorrectionConfig = Field(default_factory=CorrectionConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def get_default_config() -> AppConfig:
    """Return an AppConfig instance with standard research defaults."""
    return AppConfig()


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to default config if none provided."""
    if config_path is None:
        default_candidate = Path("experiments/configs/default.yaml")
        if default_candidate.exists():
            config_path = default_candidate
        else:
            return get_default_config()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig.model_validate(data)


def save_config(config: AppConfig, path: str | Path) -> None:
    """Serialize and save an AppConfig instance to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
