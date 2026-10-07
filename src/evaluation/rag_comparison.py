"""Triangulation and comparative evaluation across Document Variants (A, B, C).

Computes degradation (Raw vs Ground Truth), recovery (Corrected vs Raw),
and recovery efficiency percentages across retrieval and QA metrics.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACKED_METRICS = [
    ("mean_recall_at_1", "Recall@1"),
    ("mean_recall_at_3", "Recall@3"),
    ("mean_recall_at_5", "Recall@5"),
    ("mean_mrr", "MRR"),
    ("mean_exact_match", "Exact Match (EM)"),
    ("mean_f1", "Token F1 Score"),
]


def _compute_metric_delta(a_val: float, b_val: float, c_val: float) -> Dict[str, Any]:
    """Calculate degradation (B - A), recovery (C - B), and recovery percentage."""
    degradation = b_val - a_val
    recovery = c_val - b_val

    # Recovery rate is the proportion of lost performance recovered
    # If there was a loss (degradation < 0), recovery_rate = (C - B) / (A - B)
    denominator = a_val - b_val
    if abs(denominator) > 1e-6:
        recovery_rate_pct = (recovery / denominator) * 100.0
    else:
        recovery_rate_pct = 0.0 if abs(recovery) < 1e-6 else 100.0

    return {
        "ground_truth": a_val,
        "raw_ocr": b_val,
        "corrected_ocr": c_val,
        "degradation": degradation,
        "recovery": recovery,
        "recovery_rate_pct": recovery_rate_pct,
    }


def compute_rag_triangulation(
    gt_report: Dict[str, Any],
    raw_report: Dict[str, Any],
    corr_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute complete 3-variant comparison (A vs B vs C) overall, by language, and by question type."""
    gt_overall = gt_report.get("overall", {})
    raw_overall = raw_report.get("overall", {})
    corr_overall = corr_report.get("overall", {})

    overall_metrics = {}
    for key, name in TRACKED_METRICS:
        a_val = float(gt_overall.get(key, 0.0))
        b_val = float(raw_overall.get(key, 0.0))
        c_val = float(corr_overall.get(key, 0.0))
        overall_metrics[name] = _compute_metric_delta(a_val, b_val, c_val)

    # By language
    gt_lang = gt_report.get("by_language", {})
    raw_lang = raw_report.get("by_language", {})
    corr_lang = corr_report.get("by_language", {})

    by_language = {}
    for lang in ["en", "hi"]:
        if lang in gt_lang and lang in raw_lang and lang in corr_lang:
            by_language[lang] = {}
            for key, name in TRACKED_METRICS:
                a_val = float(gt_lang[lang].get(key, 0.0))
                b_val = float(raw_lang[lang].get(key, 0.0))
                c_val = float(corr_lang[lang].get(key, 0.0))
                by_language[lang][name] = _compute_metric_delta(a_val, b_val, c_val)

    # By question type
    gt_types = gt_report.get("by_question_type", {})
    raw_types = raw_report.get("by_question_type", {})
    corr_types = corr_report.get("by_question_type", {})

    by_question_type = {}
    all_types = set(gt_types.keys()) | set(raw_types.keys()) | set(corr_types.keys())
    for qtype in sorted(all_types):
        if qtype in gt_types and qtype in raw_types and qtype in corr_types:
            by_question_type[qtype] = {}
            for key, name in TRACKED_METRICS:
                a_val = float(gt_types[qtype].get(key, 0.0))
                b_val = float(raw_types[qtype].get(key, 0.0))
                c_val = float(corr_types[qtype].get(key, 0.0))
                by_question_type[qtype][name] = _compute_metric_delta(a_val, b_val, c_val)

    return {
        "overall": overall_metrics,
        "by_language": by_language,
        "by_question_type": by_question_type,
    }


def format_triangulation_table(triangulation: Dict[str, Any]) -> str:
    """Format 3-variant comparison into an informative, publication-grade table."""
    lines = []
    header = (
        f"  {'Metric':<20} | {'GT (A)':<8} | {'Raw (B)':<8} | {'Corr (C)':<8} | "
        f"{'Degradation':<11} | {'Recovery':<10} | {'Recov %':<8}"
    )
    sep = "  " + "-" * (len(header) - 2)

    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    overall = triangulation.get("overall", {})
    for name, data in overall.items():
        lines.append(
            f"  {name:<20} | {data['ground_truth']:<8.4f} | {data['raw_ocr']:<8.4f} | {data['corrected_ocr']:<8.4f} | "
            f"{data['degradation']:+11.4f} | {data['recovery']:+10.4f} | {data['recovery_rate_pct']:+7.1f}%"
        )
    lines.append(sep)

    # Languages
    by_lang = triangulation.get("by_language", {})
    for lang, metrics in by_lang.items():
        lines.append(f"\n  --- Language Breakdown: [{lang.upper()}] ---")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for name, data in metrics.items():
            lines.append(
                f"  {name:<20} | {data['ground_truth']:<8.4f} | {data['raw_ocr']:<8.4f} | {data['corrected_ocr']:<8.4f} | "
                f"{data['degradation']:+11.4f} | {data['recovery']:+10.4f} | {data['recovery_rate_pct']:+7.1f}%"
            )
        lines.append(sep)

    # Question types
    by_type = triangulation.get("by_question_type", {})
    for qtype, metrics in by_type.items():
        lines.append(f"\n  --- Question Type Breakdown: [{qtype.upper()}] ---")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for name, data in metrics.items():
            lines.append(
                f"  {name:<20} | {data['ground_truth']:<8.4f} | {data['raw_ocr']:<8.4f} | {data['corrected_ocr']:<8.4f} | "
                f"{data['degradation']:+11.4f} | {data['recovery']:+10.4f} | {data['recovery_rate_pct']:+7.1f}%"
            )
        lines.append(sep)

    return "\n".join(lines)


def export_triangulation_summary(
    triangulation: Dict[str, Any],
    output_path: str = "results/rag/comparison_summary.json",
) -> None:
    """Save triangulation metrics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triangulation, f, indent=2, ensure_ascii=False)
    logger.info("Saved 3-variant comparison summary to %s", output_path)
