"""Evaluation module: OCR metrics, retrieval metrics, answer accuracy, and statistical testing."""

from src.evaluation.ocr_metrics import (
    calculate_cer,
    calculate_wer,
    normalize_text_for_eval,
    evaluate_ocr_page,
    compute_aggregate_ocr_metrics,
    OCREvaluationAggregate,
)

__all__ = [
    "calculate_cer",
    "calculate_wer",
    "normalize_text_for_eval",
    "evaluate_ocr_page",
    "compute_aggregate_ocr_metrics",
    "OCREvaluationAggregate",
]
