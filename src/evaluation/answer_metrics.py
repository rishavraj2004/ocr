"""Question Answering evaluation metrics: Exact Match (EM) and Token-level F1.

Supports multilingual evaluation (English and Hindi) with Unicode NFC normalization,
Devanagari punctuation handling (danda, double danda), and token-level F1 computation.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from src.core.schemas import QuestionAnswerResult


# Additional Indic punctuation symbols
INDIC_PUNCTUATION = "।॥‘’“”"


def normalize_answer(text: str, language: str = "en") -> str:
    """Normalize answer string for fair, reproducible exact match and F1 scoring.

    Steps:
    1. Unicode NFC normalization.
    2. Lowercase (for Latin scripts).
    3. Strip standard ASCII punctuation and Indic punctuation (dandas, smart quotes).
    4. Strip English articles ('a', 'an', 'the') if English.
    5. Collapse whitespace.
    """
    if not text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip punctuation
    all_punctuation = string.punctuation + INDIC_PUNCTUATION
    # Replace punctuation characters with spaces to preserve word boundaries
    punc_pattern = re.compile(f"[{re.escape(all_punctuation)}]")
    text = punc_pattern.sub(" ", text)

    # 4. Remove articles (English)
    if language == "en":
        text = re.sub(r"\b(a|an|the)\b", " ", text)

    # 5. Fix whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(prediction: str, ground_truth: str, language: str = "en") -> float:
    """Compute binary Exact Match (EM) score (1.0 or 0.0)."""
    norm_pred = normalize_answer(prediction, language=language)
    norm_gold = normalize_answer(ground_truth, language=language)
    return 1.0 if norm_pred == norm_gold else 0.0


def compute_token_f1(prediction: str, ground_truth: str, language: str = "en") -> float:
    """Compute token-level F1 score in [0.0, 1.0]."""
    norm_pred = normalize_answer(prediction, language=language)
    norm_gold = normalize_answer(ground_truth, language=language)

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(f1)


def compute_answer_metrics(
    prediction: str,
    ground_truth: str,
    language: str = "en",
) -> Tuple[float, float]:
    """Compute (Exact Match, Token-level F1) for a predicted answer against ground truth."""
    em = compute_exact_match(prediction, ground_truth, language=language)
    f1 = compute_token_f1(prediction, ground_truth, language=language)
    return em, f1


def aggregate_answer_metrics(
    results: List[QuestionAnswerResult],
) -> Dict[str, float]:
    """Aggregate per-question QA metrics across an evaluation collection."""
    if not results:
        return {
            "mean_exact_match": 0.0,
            "mean_f1": 0.0,
            "total_questions": 0,
        }

    em_list = [r.exact_match for r in results]
    f1_list = [r.f1_score for r in results]

    return {
        "mean_exact_match": float(np.mean(em_list)),
        "mean_f1": float(np.mean(f1_list)),
        "total_questions": len(results),
    }
