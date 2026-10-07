"""Retrieval evaluation metrics: Recall@K and Mean Reciprocal Rank (MRR)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from src.core.schemas import QuestionRetrievalResult, RetrievalItem


def compute_retrieval_metrics_for_question(
    retrieved_items: List[RetrievalItem],
    target_page_id: str,
) -> Tuple[float, float, float, float]:
    """Compute Recall@1, Recall@3, Recall@5, and Reciprocal Rank for one question.

    Returns:
        (recall@1, recall@3, recall@5, mrr)
    """
    if not retrieved_items:
        return 0.0, 0.0, 0.0, 0.0

    r1 = 0.0
    r3 = 0.0
    r5 = 0.0
    reciprocal_rank = 0.0

    for item in retrieved_items:
        if item.page_id == target_page_id:
            rank = item.rank
            if rank == 1:
                r1 = 1.0
            if rank <= 3:
                r3 = 1.0
            if rank <= 5:
                r5 = 1.0

            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
            break  # First relevant match determines reciprocal rank

    return r1, r3, r5, reciprocal_rank


def aggregate_retrieval_metrics(
    results: List[QuestionRetrievalResult],
) -> Dict[str, float]:
    """Aggregate per-question retrieval scores across an evaluation collection."""
    if not results:
        return {
            "mean_recall_at_1": 0.0,
            "mean_recall_at_3": 0.0,
            "mean_recall_at_5": 0.0,
            "mean_mrr": 0.0,
        }

    r1_list = [r.recall_at_1 for r in results]
    r3_list = [r.recall_at_3 for r in results]
    r5_list = [r.recall_at_5 for r in results]
    mrr_list = [r.mrr for r in results]

    return {
        "mean_recall_at_1": float(np.mean(r1_list)),
        "mean_recall_at_3": float(np.mean(r3_list)),
        "mean_recall_at_5": float(np.mean(r5_list)),
        "mean_mrr": float(np.mean(mrr_list)),
    }
