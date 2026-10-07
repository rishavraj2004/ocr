"""Core module: configuration, logging, reproducibility, and schemas."""

from src.core.config import AppConfig, load_config, save_config, get_default_config
from src.core.logging import setup_logging, get_logger, close_logging_handlers
from src.core.reproducibility import (
    generate_run_id,
    set_seed,
    get_git_commit_hash,
    get_environment_info,
    ExperimentRunMetadata,
)
from src.core.schemas import (
    DocumentMetadata,
    WordOCR,
    PageOCRResult,
    CorrectionSpan,
    PageCorrectionResult,
    Question,
    DocumentVariant,
    TextChunk,
    RetrievalItem,
    QuestionRetrievalResult,
    QuestionAnswerResult,
)

__all__ = [
    "AppConfig",
    "load_config",
    "save_config",
    "get_default_config",
    "setup_logging",
    "get_logger",
    "generate_run_id",
    "set_seed",
    "get_git_commit_hash",
    "get_environment_info",
    "ExperimentRunMetadata",
    "DocumentMetadata",
    "WordOCR",
    "PageOCRResult",
    "CorrectionSpan",
    "PageCorrectionResult",
    "Question",
    "DocumentVariant",
    "TextChunk",
    "RetrievalItem",
    "QuestionRetrievalResult",
    "QuestionAnswerResult",
]
