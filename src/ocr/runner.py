"""OCR execution pipeline and baseline evaluation runner.

Processes document images, coordinates preprocessing, extracts structured OCR tokens,
persists word-level confidence and bounding-box JSON artifacts, and evaluates CER/WER
against manually verified ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.core.config import AppConfig, get_default_config, load_config
from src.core.logging import get_logger
from src.core.schemas import DocumentMetadata, DocumentVariant, Language, PageOCRResult, PageEvaluationSummary
from src.dataset.manager import DatasetManager
from src.evaluation.ocr_metrics import evaluate_ocr_page, compute_aggregate_ocr_metrics, OCREvaluationAggregate
from src.ocr.base import BaseOCREngine
from src.ocr.tesseract_ocr import TesseractOCREngine
from src.ocr.mock_ocr import MockOCREngine
from src.preprocessing import ImagePreprocessor

logger = get_logger("ocr.runner")


class OCRRunner:
    """Executes OCR extraction and evaluation across document collections."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_default_config()
        self.dataset_manager = DatasetManager(self.config)
        self.preprocessor = ImagePreprocessor(self.config.preprocessing)
        self.ocr_dir = Path(self.config.dataset.ocr_dir)
        self.ocr_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate OCR engine
        engine_name = self.config.ocr.engine.lower()
        if engine_name == "tesseract":
            tess_engine = TesseractOCREngine(self.config.ocr)
            if tess_engine.is_available:
                self.engine = tess_engine
            else:
                logger.warning(
                    "Tesseract binary not found on system. "
                    "Falling back to MockOCREngine to preserve experimental continuity."
                )
                self.engine = MockOCREngine()
        else:
            self.engine = MockOCREngine()

    def process_single_page(self, page_id: str) -> Tuple[PageOCRResult, PageEvaluationSummary]:
        """Run preprocessing, OCR, persistence, and evaluation for a single page."""
        meta = self.dataset_manager.get_page_metadata(page_id)
        if meta is None:
            raise KeyError(f"Page ID '{page_id}' not found in metadata.")

        # 1. Load ground truth & image
        gt_text = self.dataset_manager.load_ground_truth(page_id)
        img = self.dataset_manager.load_image(page_id)

        # 2. Preprocessing
        prep_result = self.preprocessor.process(img)

        # 3. OCR extraction
        ocr_result = self.engine.extract_page(
            image_input=prep_result.processed_image,
            page_id=page_id,
            document_id=meta.document_id,
            language=meta.language,
            preprocessing_config=prep_result.to_dict(),
        )

        # 4. Save structured OCR JSON
        ocr_json_path = self.ocr_dir / f"{page_id}.json"
        with open(ocr_json_path, "w", encoding="utf-8") as f:
            f.write(ocr_result.model_dump_json(indent=2))

        # 5. Evaluate CER and WER against Ground Truth
        eval_summary = evaluate_ocr_page(
            page_id=page_id,
            language=meta.language,
            variant=DocumentVariant.RAW_OCR,
            ground_truth=gt_text,
            ocr_text=ocr_result.raw_text,
            nfc=self.config.evaluation.normalize_unicode_nfc,
            ignore_case=self.config.evaluation.ignore_case,
            ignore_punctuation=self.config.evaluation.ignore_punctuation,
        )

        return ocr_result, eval_summary

    def run_all(self) -> Tuple[List[PageOCRResult], Dict[str, OCREvaluationAggregate]]:
        """Execute OCR across all dataset pages and compute language-specific aggregates."""
        pages_meta = self.dataset_manager.load_metadata()
        all_ocr_results: List[PageOCRResult] = []
        eval_summaries: List[PageEvaluationSummary] = []

        logger.info(f"Starting OCR extraction across {len(pages_meta)} document pages using '{self.engine.name}'...")

        for idx, meta in enumerate(pages_meta, 1):
            logger.info(f"[{idx}/{len(pages_meta)}] Processing '{meta.page_id}' ({meta.language.value})...")
            ocr_res, eval_res = self.process_single_page(meta.page_id)
            all_ocr_results.append(ocr_res)
            eval_summaries.append(eval_res)

        # Separate summaries by language
        en_summaries = [s for s in eval_summaries if s.language == Language.ENGLISH]
        hi_summaries = [s for s in eval_summaries if s.language == Language.HINDI]

        aggregates: Dict[str, OCREvaluationAggregate] = {
            "en": compute_aggregate_ocr_metrics(en_summaries, "en", DocumentVariant.RAW_OCR),
            "hi": compute_aggregate_ocr_metrics(hi_summaries, "hi", DocumentVariant.RAW_OCR),
            "combined": compute_aggregate_ocr_metrics(eval_summaries, "combined", DocumentVariant.RAW_OCR),
        }

        # Persist raw CSV results
        raw_results_dir = Path(self.config.output_dir) / "raw"
        raw_results_dir.mkdir(parents=True, exist_ok=True)
        csv_records = [
            {
                "page_id": s.page_id,
                "language": s.language.value,
                "variant": s.variant.value,
                "cer": s.cer,
                "wer": s.wer,
                "engine": self.engine.name,
                "engine_version": self.engine.version,
            }
            for s in eval_summaries
        ]
        df_csv = pd.DataFrame(csv_records)
        csv_path = raw_results_dir / "ocr_results.csv"
        df_csv.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Persisted raw OCR evaluation metrics to: {csv_path}")

        # Persist processed summary
        proc_dir = Path(self.config.output_dir) / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        summary_records = [
            {
                "language": lang,
                "variant": agg.variant.value,
                "num_pages": agg.num_pages,
                "mean_cer": round(agg.mean_cer, 4),
                "std_cer": round(agg.std_cer, 4),
                "median_cer": round(agg.median_cer, 4),
                "mean_wer": round(agg.mean_wer, 4),
                "std_wer": round(agg.std_wer, 4),
                "median_wer": round(agg.median_wer, 4),
            }
            for lang, agg in aggregates.items()
        ]
        pd.DataFrame(summary_records).to_csv(proc_dir / "ocr_summary.csv", index=False, encoding="utf-8")

        return all_ocr_results, aggregates
