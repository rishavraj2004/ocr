"""Confidence-guided selective correction execution runner.

Coordinates span detection, context extraction, selective replacement,
audit logging, and comparative CER/WER evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.core.config import AppConfig, get_default_config, load_config
from src.core.logging import get_logger
from src.core.schemas import (
    CorrectionSpan,
    DocumentMetadata,
    DocumentVariant,
    Language,
    PageCorrectionResult,
    PageEvaluationSummary,
    PageOCRResult,
    WordOCR,
)
from src.correction.confidence_detector import ConfidenceDetector
from src.correction.corrector import BaseCorrector, MockCorrector, OpenAICorrector
from src.correction.span_extractor import SpanExtractor
from src.dataset.manager import DatasetManager
from src.evaluation.ocr_metrics import evaluate_ocr_page, compute_aggregate_ocr_metrics, OCREvaluationAggregate

logger = get_logger("correction.runner")


def reconstruct_corrected_text(
    words: List[WordOCR],
    corrected_spans: List[CorrectionSpan],
) -> str:
    """Assemble final page text by splicing corrected spans into high-confidence text."""
    if not corrected_spans:
        return " ".join(w.text for w in words)

    # Map start_word_idx to span for quick lookup
    span_map: Dict[int, CorrectionSpan] = {s.start_word_idx: s for s in corrected_spans}

    tokens: List[str] = []
    idx = 0
    total_words = len(words)

    while idx < total_words:
        if idx in span_map:
            span = span_map[idx]
            tokens.append(span.corrected_text)
            idx = span.end_word_idx  # Skip over replaced span words
        else:
            tokens.append(words[idx].text)
            idx += 1

    return " ".join(tokens)


class CorrectionRunner:
    """Executes confidence-guided selective correction across document collections."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        self.dataset_manager = DatasetManager(self.config)
        self.ocr_dir = Path(self.config.dataset.ocr_dir)
        self.corrected_dir = Path(self.config.dataset.corrected_dir)
        self.corrected_dir.mkdir(parents=True, exist_ok=True)

        # Select corrector
        model_type = self.config.correction.model_type.lower()
        if model_type == "openai":
            self.corrector: BaseCorrector = OpenAICorrector(self.config.correction)
        else:
            self.corrector = MockCorrector()

    def process_single_page(
        self,
        page_id: str,
        threshold: Optional[float] = None,
    ) -> Tuple[PageCorrectionResult, PageEvaluationSummary, PageEvaluationSummary]:
        """Perform selective correction for a single page and evaluate before vs after."""
        thresh = threshold if threshold is not None else self.config.correction.threshold
        meta = self.dataset_manager.get_page_metadata(page_id)
        if meta is None:
            raise KeyError(f"Page ID '{page_id}' not found in metadata.")

        # 1. Load Raw OCR artifact
        ocr_json_path = self.ocr_dir / f"{page_id}.json"
        if not ocr_json_path.exists():
            raise FileNotFoundError(f"Raw OCR artifact missing at: {ocr_json_path}. Run OCR first.")

        with open(ocr_json_path, "r", encoding="utf-8") as f:
            ocr_dict = json.load(f)
        ocr_result = PageOCRResult.model_validate(ocr_dict)

        # 2. Detect low-confidence spans
        detector = ConfidenceDetector(threshold=thresh)
        raw_spans = detector.detect_low_confidence_spans(ocr_result.words)

        # 3. Extract context
        extractor = SpanExtractor(context_window_words=self.config.correction.context_window_words)
        spans = extractor.extract_contextual_spans(page_id, ocr_result.words, raw_spans)

        # 4. Correct each span
        total_latency = 0.0
        corrected_words_count = 0

        for span in spans:
            corr_text, changed, lat, p_toks, c_toks = self.corrector.correct_span(
                span, ocr_result.language
            )
            span.corrected_text = corr_text
            span.changed = changed
            span.latency_sec = lat
            span.prompt_tokens = p_toks
            span.completion_tokens = c_toks
            total_latency += lat
            corrected_words_count += (span.end_word_idx - span.start_word_idx)

        # 5. Reconstruct full page text
        corrected_text = reconstruct_corrected_text(ocr_result.words, spans)

        correction_result = PageCorrectionResult(
            page_id=page_id,
            original_text=ocr_result.raw_text,
            corrected_text=corrected_text,
            spans=spans,
            threshold=thresh,
            total_words=len(ocr_result.words),
            corrected_words=corrected_words_count,
            total_latency_sec=round(total_latency, 4),
        )

        # 6. Save Corrected OCR JSON artifact
        corr_json_path = self.corrected_dir / f"{page_id}.json"
        with open(corr_json_path, "w", encoding="utf-8") as f:
            f.write(correction_result.model_dump_json(indent=2))

        # 7. Evaluate CER/WER Before (Raw OCR) and After (Corrected OCR)
        gt_text = self.dataset_manager.load_ground_truth(page_id)
        raw_eval = evaluate_ocr_page(
            page_id=page_id,
            language=meta.language,
            variant=DocumentVariant.RAW_OCR,
            ground_truth=gt_text,
            ocr_text=ocr_result.raw_text,
            nfc=self.config.evaluation.normalize_unicode_nfc,
            ignore_case=self.config.evaluation.ignore_case,
            ignore_punctuation=self.config.evaluation.ignore_punctuation,
        )

        corrected_eval = evaluate_ocr_page(
            page_id=page_id,
            language=meta.language,
            variant=DocumentVariant.CORRECTED_OCR,
            ground_truth=gt_text,
            ocr_text=corrected_text,
            nfc=self.config.evaluation.normalize_unicode_nfc,
            ignore_case=self.config.evaluation.ignore_case,
            ignore_punctuation=self.config.evaluation.ignore_punctuation,
        )

        return correction_result, raw_eval, corrected_eval

    def run_all(
        self,
        threshold: Optional[float] = None,
    ) -> Tuple[List[PageCorrectionResult], Dict[str, Any]]:
        """Run selective correction across all dataset pages and compute comparative metrics."""
        thresh = threshold if threshold is not None else self.config.correction.threshold
        pages_meta = self.dataset_manager.load_metadata()

        all_results: List[PageCorrectionResult] = []
        raw_evals: List[PageEvaluationSummary] = []
        corr_evals: List[PageEvaluationSummary] = []

        logger.info(
            f"Starting selective correction across {len(pages_meta)} pages "
            f"at confidence threshold tau={thresh}..."
        )

        for idx, meta in enumerate(pages_meta, 1):
            logger.info(f"[{idx}/{len(pages_meta)}] Correcting '{meta.page_id}' ({meta.language.value})...")
            corr_res, raw_ev, corr_ev = self.process_single_page(meta.page_id, threshold=thresh)
            all_results.append(corr_res)
            raw_evals.append(raw_ev)
            corr_evals.append(corr_ev)

        # Aggregate metrics
        en_raw = [e for e in raw_evals if e.language == Language.ENGLISH]
        en_corr = [e for e in corr_evals if e.language == Language.ENGLISH]
        hi_raw = [e for e in raw_evals if e.language == Language.HINDI]
        hi_corr = [e for e in corr_evals if e.language == Language.HINDI]

        aggregates = {
            "en_raw": compute_aggregate_ocr_metrics(en_raw, "en", DocumentVariant.RAW_OCR),
            "en_corrected": compute_aggregate_ocr_metrics(en_corr, "en", DocumentVariant.CORRECTED_OCR),
            "hi_raw": compute_aggregate_ocr_metrics(hi_raw, "hi", DocumentVariant.RAW_OCR),
            "hi_corrected": compute_aggregate_ocr_metrics(hi_corr, "hi", DocumentVariant.CORRECTED_OCR),
            "combined_raw": compute_aggregate_ocr_metrics(raw_evals, "combined", DocumentVariant.RAW_OCR),
            "combined_corrected": compute_aggregate_ocr_metrics(corr_evals, "combined", DocumentVariant.CORRECTED_OCR),
        }

        # Persist raw CSV records
        raw_dir = Path(self.config.output_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        for r_ev, c_ev, res in zip(raw_evals, corr_evals, all_results):
            pct_selected = (res.corrected_words / max(1, res.total_words)) * 100.0
            cer_diff = r_ev.cer - c_ev.cer
            wer_diff = r_ev.wer - c_ev.wer
            csv_rows.append({
                "page_id": res.page_id,
                "language": r_ev.language.value,
                "threshold": thresh,
                "total_words": res.total_words,
                "corrected_spans": len(res.spans),
                "corrected_words": res.corrected_words,
                "pct_words_selected": round(pct_selected, 2),
                "raw_cer": r_ev.cer,
                "corrected_cer": c_ev.cer,
                "cer_recovery": round(cer_diff, 4),
                "raw_wer": r_ev.wer,
                "corrected_wer": c_ev.wer,
                "wer_recovery": round(wer_diff, 4),
                "latency_sec": res.total_latency_sec,
            })
        df_csv = pd.DataFrame(csv_rows)
        df_csv.to_csv(raw_dir / "correction_results.csv", index=False, encoding="utf-8")

        # Persist processed summary
        proc_dir = Path(self.config.output_dir) / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = [
            {
                "language": "en",
                "threshold": thresh,
                "raw_mean_cer": round(aggregates["en_raw"].mean_cer, 4),
                "corrected_mean_cer": round(aggregates["en_corrected"].mean_cer, 4),
                "cer_improvement": round(aggregates["en_raw"].mean_cer - aggregates["en_corrected"].mean_cer, 4),
                "raw_mean_wer": round(aggregates["en_raw"].mean_wer, 4),
                "corrected_mean_wer": round(aggregates["en_corrected"].mean_wer, 4),
                "wer_improvement": round(aggregates["en_raw"].mean_wer - aggregates["en_corrected"].mean_wer, 4),
            },
            {
                "language": "hi",
                "threshold": thresh,
                "raw_mean_cer": round(aggregates["hi_raw"].mean_cer, 4),
                "corrected_mean_cer": round(aggregates["hi_corrected"].mean_cer, 4),
                "cer_improvement": round(aggregates["hi_raw"].mean_cer - aggregates["hi_corrected"].mean_cer, 4),
                "raw_mean_wer": round(aggregates["hi_raw"].mean_wer, 4),
                "corrected_mean_wer": round(aggregates["hi_corrected"].mean_wer, 4),
                "wer_improvement": round(aggregates["hi_raw"].mean_wer - aggregates["hi_corrected"].mean_wer, 4),
            },
            {
                "language": "combined",
                "threshold": thresh,
                "raw_mean_cer": round(aggregates["combined_raw"].mean_cer, 4),
                "corrected_mean_cer": round(aggregates["combined_corrected"].mean_cer, 4),
                "cer_improvement": round(aggregates["combined_raw"].mean_cer - aggregates["combined_corrected"].mean_cer, 4),
                "raw_mean_wer": round(aggregates["combined_raw"].mean_wer, 4),
                "corrected_mean_wer": round(aggregates["combined_corrected"].mean_wer, 4),
                "wer_improvement": round(aggregates["combined_raw"].mean_wer - aggregates["combined_corrected"].mean_wer, 4),
            },
        ]
        pd.DataFrame(summary_rows).to_csv(proc_dir / "correction_summary.csv", index=False, encoding="utf-8")

        return all_results, aggregates
