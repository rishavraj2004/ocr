"""Confidence Threshold Sweep Runner (Phase 8).

Systematically evaluates confidence thresholds tau in [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
measuring trade-offs between flagged token ratio, OCR error rates (CER/WER),
downstream retrieval efficacy, QA accuracy (EM/F1), and computational efficiency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.config import AppConfig, get_default_config, load_config
from src.core.schemas import (
    CorrectionSpan,
    DocumentMetadata,
    DocumentVariant,
    Language,
    PageCorrectionResult,
    PageOCRResult,
    WordOCR,
)
from src.correction.confidence_detector import ConfidenceDetector
from src.correction.corrector import BaseCorrector, MockCorrector, OpenAICorrector
from src.correction.runner import reconstruct_corrected_text
from src.correction.span_extractor import SpanExtractor
from src.dataset.manager import DatasetManager
from src.evaluation.ocr_metrics import compute_aggregate_ocr_metrics, evaluate_ocr_page
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ThresholdSweepRunner:
    """Coordinates multi-threshold sensitivity sweep across selective correction and RAG."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        self.dataset_manager = DatasetManager(self.config)
        self.ocr_dir = Path(self.config.dataset.ocr_dir)

        # Initialize corrector
        model_type = self.config.correction.model_type.lower()
        if model_type == "openai":
            self.corrector: BaseCorrector = OpenAICorrector(self.config.correction)
        else:
            self.corrector = MockCorrector()

    def correct_pages_for_threshold(
        self,
        threshold: float,
    ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
        """Run selective correction on all pages at a given threshold without clobbering disk artifacts.

        Returns:
            Tuple of:
              - page_texts: Dict mapping page_id -> (document_id, corrected_text)
              - ocr_stats: Aggregated statistics (flagged words, ratio, CER, WER, latency, tokens)
        """
        metadata_list = self.dataset_manager.load_metadata()
        detector = ConfidenceDetector(threshold=threshold)
        extractor = SpanExtractor(context_window_words=self.config.correction.context_window_words)

        page_texts: Dict[str, Tuple[str, str]] = {}
        page_results: List[PageCorrectionResult] = []
        corr_evals = []

        total_words_all = 0
        flagged_words_all = 0
        total_latency_all = 0.0
        total_prompt_tokens = 0
        total_comp_tokens = 0

        for meta in metadata_list:
            page_id = meta.page_id
            doc_id = meta.document_id

            ocr_path = self.ocr_dir / f"{page_id}.json"
            if not ocr_path.exists():
                raise FileNotFoundError(f"Missing raw OCR for {page_id}")

            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_dict = json.load(f)
            ocr_result = PageOCRResult.model_validate(ocr_dict)

            # Detect & extract spans
            raw_spans = detector.detect_low_confidence_spans(ocr_result.words)
            spans = extractor.extract_contextual_spans(page_id, ocr_result.words, raw_spans)

            # Query corrector
            page_latency = 0.0
            corrected_words_count = 0
            for s in spans:
                corr_text, changed, lat, p_tok, c_tok = self.corrector.correct_span(
                    s, ocr_result.language
                )
                s.corrected_text = corr_text
                s.changed = changed
                s.latency_sec = lat
                s.prompt_tokens = p_tok
                s.completion_tokens = c_tok

                page_latency += lat
                if p_tok:
                    total_prompt_tokens += p_tok
                if c_tok:
                    total_comp_tokens += c_tok
                corrected_words_count += (s.end_word_idx - s.start_word_idx)

            # Reconstruct corrected page text
            corrected_text = reconstruct_corrected_text(ocr_result.words, spans)
            page_texts[page_id] = (doc_id, corrected_text)

            total_words_all += len(ocr_result.words)
            flagged_words_all += corrected_words_count
            total_latency_all += page_latency

            # Evaluate OCR CER/WER
            gt_text = self.dataset_manager.load_ground_truth(page_id)
            eval_summary = evaluate_ocr_page(
                page_id=page_id,
                language=meta.language,
                variant=DocumentVariant.CORRECTED_OCR,
                ground_truth=gt_text,
                ocr_text=corrected_text,
                nfc=self.config.evaluation.normalize_unicode_nfc,
                ignore_case=self.config.evaluation.ignore_case,
                ignore_punctuation=self.config.evaluation.ignore_punctuation,
            )
            corr_evals.append(eval_summary)

        # Aggregate OCR metrics
        agg_en = compute_aggregate_ocr_metrics(corr_evals, "en", DocumentVariant.CORRECTED_OCR)
        agg_hi = compute_aggregate_ocr_metrics(corr_evals, "hi", DocumentVariant.CORRECTED_OCR)
        agg_all = compute_aggregate_ocr_metrics(corr_evals, "combined", DocumentVariant.CORRECTED_OCR)

        flagged_ratio = (flagged_words_all / total_words_all) if total_words_all > 0 else 0.0

        ocr_stats = {
            "threshold": threshold,
            "total_words": total_words_all,
            "flagged_words": flagged_words_all,
            "flagged_ratio": flagged_ratio,
            "correction_latency_sec": total_latency_all,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_comp_tokens,
            "mean_cer": agg_all.mean_cer,
            "mean_wer": agg_all.mean_wer,
            "en_cer": agg_en.mean_cer,
            "en_wer": agg_en.mean_wer,
            "hi_cer": agg_hi.mean_cer,
            "hi_wer": agg_hi.mean_wer,
        }

        return page_texts, ocr_stats

    def run_sweep(
        self,
        thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Execute threshold sweep across all specified thresholds."""
        sweep_thresholds = thresholds or self.config.correction.sweep_thresholds
        logger.info(
            "Starting Confidence Threshold Sweep over %d thresholds: %s",
            len(sweep_thresholds),
            sweep_thresholds,
        )

        # Load Ground Truth and Raw OCR baselines for recovery calculation
        gt_path = os.path.join(self.config.output_dir, "rag", "ground_truth_results.json")
        raw_path = os.path.join(self.config.output_dir, "rag", "raw_ocr_results.json")

        gt_em = 1.0
        gt_f1 = 1.0
        raw_em = 0.8684
        raw_f1 = 0.9224

        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                gt_em = gt_data["overall"].get("mean_exact_match", 1.0)
                gt_f1 = gt_data["overall"].get("mean_f1", 1.0)

        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                raw_em = raw_data["overall"].get("mean_exact_match", 0.8684)
                raw_f1 = raw_data["overall"].get("mean_f1", 0.9224)

        curve_points: List[Dict[str, Any]] = []

        t0_sweep = time.perf_counter()

        for tau in sweep_thresholds:
            logger.info("Evaluating threshold tau = %.1f...", tau)
            t0_tau = time.perf_counter()

            # 1. Selective correction
            page_texts, ocr_stats = self.correct_pages_for_threshold(tau)

            # 2. RAG Pipeline evaluation over in-memory corrected text
            rag_pipeline = RAGPipeline(
                variant=DocumentVariant.CORRECTED_OCR,
                config=self.config,
            )
            rag_pipeline.index_documents(page_texts=page_texts)
            rag_report = rag_pipeline.run_evaluation()

            rag_overall = rag_report["overall"]
            rag_lang = rag_report["by_language"]

            curr_em = rag_overall["mean_exact_match"]
            curr_f1 = rag_overall["mean_f1"]

            # Compute recovery rates
            denom_em = gt_em - raw_em
            denom_f1 = gt_f1 - raw_f1

            em_recovery_rate = ((curr_em - raw_em) / denom_em * 100.0) if denom_em > 1e-6 else 0.0
            f1_recovery_rate = ((curr_f1 - raw_f1) / denom_f1 * 100.0) if denom_f1 > 1e-6 else 0.0

            total_tau_time = time.perf_counter() - t0_tau

            point = {
                "threshold": tau,
                "flagged_words": ocr_stats["flagged_words"],
                "flagged_ratio": ocr_stats["flagged_ratio"],
                "correction_latency_sec": ocr_stats["correction_latency_sec"],
                "correction_tokens": ocr_stats["prompt_tokens"] + ocr_stats["completion_tokens"],
                "ocr": {
                    "cer": ocr_stats["mean_cer"],
                    "wer": ocr_stats["mean_wer"],
                    "en_cer": ocr_stats["en_cer"],
                    "en_wer": ocr_stats["en_wer"],
                    "hi_cer": ocr_stats["hi_cer"],
                    "hi_wer": ocr_stats["hi_wer"],
                },
                "retrieval": {
                    "recall_at_1": rag_overall["mean_recall_at_1"],
                    "recall_at_3": rag_overall["mean_recall_at_3"],
                    "recall_at_5": rag_overall["mean_recall_at_5"],
                    "mrr": rag_overall["mean_mrr"],
                },
                "qa": {
                    "exact_match": curr_em,
                    "f1_score": curr_f1,
                    "em_recovery_rate_pct": em_recovery_rate,
                    "f1_recovery_rate_pct": f1_recovery_rate,
                    "en_em": rag_lang["en"]["mean_exact_match"],
                    "en_f1": rag_lang["en"]["mean_f1"],
                    "hi_em": rag_lang["hi"]["mean_exact_match"],
                    "hi_f1": rag_lang["hi"]["mean_f1"],
                },
                "total_time_sec": total_tau_time,
            }
            curve_points.append(point)

        total_sweep_time = time.perf_counter() - t0_sweep

        # Determine optimal threshold tau*
        # Criterion: maximize F1 score, break ties with lowest flagged ratio
        best_point = max(
            curve_points,
            key=lambda p: (p["qa"]["f1_score"], p["qa"]["exact_match"], -p["flagged_ratio"]),
        )
        optimal_tau = best_point["threshold"]

        sweep_report = {
            "experiment_name": self.config.experiment_name,
            "thresholds": sweep_thresholds,
            "optimal_threshold": optimal_tau,
            "baselines": {
                "ground_truth": {"exact_match": gt_em, "f1_score": gt_f1},
                "raw_ocr": {"exact_match": raw_em, "f1_score": raw_f1},
            },
            "points": curve_points,
            "total_sweep_latency_sec": total_sweep_time,
        }

        # Export artifact to results/sweeps/threshold_sweep.json
        out_dir = os.path.join(self.config.output_dir, "sweeps")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "threshold_sweep.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sweep_report, f, indent=2, ensure_ascii=False)
        logger.info("Saved confidence threshold sweep report to %s", out_path)

        return sweep_report

    @staticmethod
    def format_sweep_table(sweep_report: Dict[str, Any]) -> str:
        """Render a publication-ready comparative sensitivity table across all thresholds."""
        lines = []
        header = (
            f"  {'tau':<5} | {'Flagged %':<10} | {'CER':<7} | {'WER':<7} | "
            f"{'Rec@5':<7} | {'MRR':<7} | {'EM':<7} | {'F1':<7} | "
            f"{'EM Recov%':<10} | {'F1 Recov%':<10} | {'Tokens':<7}"
        )
        sep = "  " + "-" * (len(header) - 2)

        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for p in sweep_report["points"]:
            tau_marker = f"{p['threshold']:.0f}"
            if p["threshold"] == sweep_report["optimal_threshold"]:
                tau_marker += " *"

            lines.append(
                f"  {tau_marker:<5} | {p['flagged_ratio'] * 100.0:>8.1f}%  | "
                f"{p['ocr']['cer']:<7.4f} | {p['ocr']['wer']:<7.4f} | "
                f"{p['retrieval']['recall_at_5']:<7.4f} | {p['retrieval']['mrr']:<7.4f} | "
                f"{p['qa']['exact_match']:<7.4f} | {p['qa']['f1_score']:<7.4f} | "
                f"{p['qa']['em_recovery_rate_pct']:>8.1f}%  | {p['qa']['f1_recovery_rate_pct']:>8.1f}%  | "
                f"{p['correction_tokens']:<7d}"
            )

        lines.append(sep)
        lines.append(f"  * Optimal operating threshold: tau* = {sweep_report['optimal_threshold']:.1f}")
        return "\n".join(lines)
