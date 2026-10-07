"""Full-Text Correction vs. Selective Correction Comparative Study (Phase 9).

Addresses Research Question 4:
"Whether selective correction is more efficient than correcting the entire OCR output,
and how they compare on cost, latency, error recovery, and hallucinations/over-correction."
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import AppConfig, get_default_config, load_config
from src.core.schemas import DocumentVariant, Language, PageEvaluationSummary, PageOCRResult
from src.correction.confidence_detector import ConfidenceDetector
from src.correction.corrector import BaseCorrector, MockCorrector, OpenAICorrector
from src.correction.full_text_corrector import FullPageCorrectionResult, FullTextCorrector
from src.correction.runner import reconstruct_corrected_text
from src.correction.span_extractor import SpanExtractor
from src.dataset.manager import DatasetManager
from src.evaluation.ocr_metrics import compute_aggregate_ocr_metrics, evaluate_ocr_page
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class FullVsSelectiveRunner:
    """Orchestrates head-to-head empirical comparison between Full-Text and Selective Correction."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        self.dataset_manager = DatasetManager(self.config)
        self.ocr_dir = Path(self.config.dataset.ocr_dir)
        self.full_corrector = FullTextCorrector(self.config.correction)

        model_type = self.config.correction.model_type.lower()
        if model_type == "openai":
            self.selective_corrector: BaseCorrector = OpenAICorrector(self.config.correction)
        else:
            self.selective_corrector = MockCorrector()

    def run_full_text_pipeline(self) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
        """Execute unconstrained full-text correction and evaluate across all pages."""
        metadata_list = self.dataset_manager.load_metadata()
        page_texts: Dict[str, Tuple[str, str]] = {}
        eval_summaries: List[PageEvaluationSummary] = []

        total_latency = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        total_over_corrections = 0
        total_words = 0

        for meta in metadata_list:
            page_id = meta.page_id
            doc_id = meta.document_id

            ocr_path = self.ocr_dir / f"{page_id}.json"
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_dict = json.load(f)
            raw_text = ocr_dict.get("raw_text") or ocr_dict.get("text", "")
            gt_text = self.dataset_manager.load_ground_truth(page_id)

            res = self.full_corrector.correct_page(
                page_id=page_id,
                raw_text=raw_text,
                language=meta.language,
                ground_truth=gt_text,
            )

            page_texts[page_id] = (doc_id, res.corrected_text)
            total_latency += res.latency_sec
            prompt_tokens += res.prompt_tokens
            completion_tokens += res.completion_tokens
            total_over_corrections += res.over_corrected_words
            total_words += len(raw_text.split())

            ev = evaluate_ocr_page(
                page_id=page_id,
                language=meta.language,
                variant=DocumentVariant.CORRECTED_OCR,
                ground_truth=gt_text,
                ocr_text=res.corrected_text,
                nfc=self.config.evaluation.normalize_unicode_nfc,
                ignore_case=self.config.evaluation.ignore_case,
                ignore_punctuation=self.config.evaluation.ignore_punctuation,
            )
            eval_summaries.append(ev)

        ocr_agg = compute_aggregate_ocr_metrics(eval_summaries, "combined", DocumentVariant.CORRECTED_OCR)

        stats = {
            "mode": "full_text",
            "total_words": total_words,
            "modified_words": total_words,  # Entire text processed
            "flagged_ratio": 1.0,
            "latency_sec": total_latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "over_corrected_words": total_over_corrections,
            "over_correction_rate": total_over_corrections / max(1, total_words),
            "cer": ocr_agg.mean_cer,
            "wer": ocr_agg.mean_wer,
        }
        return page_texts, stats

    def run_selective_pipeline(
        self,
        threshold: float = 70.0,
    ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Any]]:
        """Execute confidence-guided selective correction and evaluate across all pages."""
        metadata_list = self.dataset_manager.load_metadata()
        detector = ConfidenceDetector(threshold=threshold)
        extractor = SpanExtractor(context_window_words=self.config.correction.context_window_words)

        page_texts: Dict[str, Tuple[str, str]] = {}
        eval_summaries: List[PageEvaluationSummary] = []

        total_latency = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        total_over_corrections = 0
        total_words = 0
        flagged_words = 0

        for meta in metadata_list:
            page_id = meta.page_id
            doc_id = meta.document_id

            ocr_path = self.ocr_dir / f"{page_id}.json"
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_dict = json.load(f)
            ocr_result = PageOCRResult.model_validate(ocr_dict)
            gt_text = self.dataset_manager.load_ground_truth(page_id)
            gt_words = re.findall(r"\S+", gt_text)

            raw_spans = detector.detect_low_confidence_spans(ocr_result.words)
            spans = extractor.extract_contextual_spans(page_id, ocr_result.words, raw_spans)

            for s in spans:
                corr_text, changed, lat, p_tok, c_tok = self.selective_corrector.correct_span(
                    s, ocr_result.language
                )
                s.corrected_text = corr_text
                s.changed = changed
                s.latency_sec = lat
                s.prompt_tokens = p_tok
                s.completion_tokens = c_tok

                total_latency += lat
                if p_tok:
                    prompt_tokens += p_tok
                if c_tok:
                    completion_tokens += c_tok

                flagged_words += (s.end_word_idx - s.start_word_idx)

                # Over-correction check within span
                span_raw = s.original_text.split()
                span_corr = corr_text.split()
                for w_idx in range(min(len(span_raw), len(span_corr))):
                    global_idx = s.start_word_idx + w_idx
                    if global_idx < len(gt_words):
                        if span_raw[w_idx] == gt_words[global_idx] and span_corr[w_idx] != gt_words[global_idx]:
                            total_over_corrections += 1

            corrected_text = reconstruct_corrected_text(ocr_result.words, spans)
            page_texts[page_id] = (doc_id, corrected_text)
            total_words += len(ocr_result.words)

            ev = evaluate_ocr_page(
                page_id=page_id,
                language=meta.language,
                variant=DocumentVariant.CORRECTED_OCR,
                ground_truth=gt_text,
                ocr_text=corrected_text,
                nfc=self.config.evaluation.normalize_unicode_nfc,
                ignore_case=self.config.evaluation.ignore_case,
                ignore_punctuation=self.config.evaluation.ignore_punctuation,
            )
            eval_summaries.append(ev)

        ocr_agg = compute_aggregate_ocr_metrics(eval_summaries, "combined", DocumentVariant.CORRECTED_OCR)

        stats = {
            "mode": f"selective_tau_{threshold:.0f}",
            "threshold": threshold,
            "total_words": total_words,
            "modified_words": flagged_words,
            "flagged_ratio": flagged_words / max(1, total_words),
            "latency_sec": total_latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "over_corrected_words": total_over_corrections,
            "over_correction_rate": total_over_corrections / max(1, total_words),
            "cer": ocr_agg.mean_cer,
            "wer": ocr_agg.mean_wer,
        }
        return page_texts, stats

    def compare(
        self,
        selective_threshold: float = 70.0,
    ) -> Dict[str, Any]:
        """Execute head-to-head comparison across OCR, RAG retrieval, QA accuracy, and cost."""
        logger.info("Executing Full-Text Correction baseline...")
        full_texts, full_stats = self.run_full_text_pipeline()

        logger.info("Executing Selective Correction (tau=%.1f)...", selective_threshold)
        sel_texts, sel_stats = self.run_selective_pipeline(threshold=selective_threshold)

        # Run RAG evaluation on both variants
        logger.info("Evaluating RAG downstream performance for Full-Text Correction...")
        rag_full = RAGPipeline(variant=DocumentVariant.CORRECTED_OCR, config=self.config)
        rag_full.index_documents(page_texts=full_texts)
        report_full = rag_full.run_evaluation()

        logger.info("Evaluating RAG downstream performance for Selective Correction...")
        rag_sel = RAGPipeline(variant=DocumentVariant.CORRECTED_OCR, config=self.config)
        rag_sel.index_documents(page_texts=sel_texts)
        report_sel = rag_sel.run_evaluation()

        full_overall = report_full["overall"]
        sel_overall = report_sel["overall"]

        # Comparative Ratios
        token_savings_pct = (
            (1.0 - (sel_stats["total_tokens"] / max(1, full_stats["total_tokens"]))) * 100.0
        )
        token_reduction_factor = (
            full_stats["total_tokens"] / max(1, sel_stats["total_tokens"])
        )

        comparison = {
            "full_text": {
                **full_stats,
                "recall_at_5": full_overall["mean_recall_at_5"],
                "mrr": full_overall["mean_mrr"],
                "exact_match": full_overall["mean_exact_match"],
                "f1_score": full_overall["mean_f1"],
            },
            "selective": {
                **sel_stats,
                "recall_at_5": sel_overall["mean_recall_at_5"],
                "mrr": sel_overall["mean_mrr"],
                "exact_match": sel_overall["mean_exact_match"],
                "f1_score": sel_overall["mean_f1"],
            },
            "efficiency": {
                "token_savings_pct": token_savings_pct,
                "token_reduction_factor": token_reduction_factor,
                "over_correction_reduction_words": full_stats["over_corrected_words"] - sel_stats["over_corrected_words"],
                "qa_em_difference": sel_overall["mean_exact_match"] - full_overall["mean_exact_match"],
                "qa_f1_difference": sel_overall["mean_f1"] - full_overall["mean_f1"],
            },
        }

        # Export to results/sweeps/full_vs_selective.json
        out_dir = os.path.join(self.config.output_dir, "sweeps")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "full_vs_selective.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info("Saved Full vs Selective comparison to %s", out_file)

        return comparison

    @staticmethod
    def format_comparison_table(comparison: Dict[str, Any]) -> str:
        """Format a clear head-to-head comparison table for presentation."""
        ft = comparison["full_text"]
        sel = comparison["selective"]
        eff = comparison["efficiency"]

        lines = []
        header = f"  {'Dimension':<26} | {'Full-Text (Naive)':<18} | {'Selective (tau)':<16} | {'Advantage':<18}"
        sep = "  " + "-" * (len(header) - 2)

        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        rows = [
            ("Tokens Consumed (Total)", f"{ft['total_tokens']:,}", f"{sel['total_tokens']:,}", f"{eff['token_savings_pct']:+.1f}% ({eff['token_reduction_factor']:.1f}x cheaper)"),
            ("Prompt Tokens", f"{ft['prompt_tokens']:,}", f"{sel['prompt_tokens']:,}", f"{(1 - sel['prompt_tokens']/max(1,ft['prompt_tokens']))*100:+.1f}%"),
            ("Completion Tokens", f"{ft['completion_tokens']:,}", f"{sel['completion_tokens']:,}", f"{(1 - sel['completion_tokens']/max(1,ft['completion_tokens']))*100:+.1f}%"),
            ("Flagged Words Ratio", f"{ft['flagged_ratio']*100:.1f}%", f"{sel['flagged_ratio']*100:.1f}%", f"{(ft['flagged_ratio']-sel['flagged_ratio'])*100:+.1f}% focused"),
            ("Over-Corrected Words", f"{ft['over_corrected_words']}", f"{sel['over_corrected_words']}", f"{eff['over_correction_reduction_words']:+d} fewer corruptions"),
            ("Character Error Rate (CER)", f"{ft['cer']:.4f}", f"{sel['cer']:.4f}", f"{sel['cer']-ft['cer']:+.4f}"),
            ("Word Error Rate (WER)", f"{ft['wer']:.4f}", f"{sel['wer']:.4f}", f"{sel['wer']-ft['wer']:+.4f}"),
            ("Retrieval Recall@5", f"{ft['recall_at_5']:.4f}", f"{sel['recall_at_5']:.4f}", f"{sel['recall_at_5']-ft['recall_at_5']:+.4f}"),
            ("Retrieval MRR", f"{ft['mrr']:.4f}", f"{sel['mrr']:.4f}", f"{sel['mrr']-ft['mrr']:+.4f}"),
            ("QA Exact Match (EM)", f"{ft['exact_match']:.4f}", f"{sel['exact_match']:.4f}", f"{eff['qa_em_difference']:+.4f}"),
            ("QA Token F1 Score", f"{ft['f1_score']:.4f}", f"{sel['f1_score']:.4f}", f"{eff['qa_f1_difference']:+.4f}"),
        ]

        for dim, f_val, s_val, adv in rows:
            lines.append(f"  {dim:<26} | {f_val:<18} | {s_val:<16} | {adv:<18}")

        lines.append(sep)
        return "\n".join(lines)
