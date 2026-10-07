"""Master End-to-End Experiment Benchmark Runner (Phase 11).

Orchestrates the entire research evaluation in a single command with full
reproducibility tracking, metadata freezing, figure generation, and table export:

1. Environment & system reproducibility audit (seeds, git hash, library versions).
2. Dataset integrity and image preprocessing verification.
3. Raw OCR extraction & character/word error rate (CER/WER) evaluation.
4. Confidence-guided selective correction & recovery evaluation.
5. Triangulated RAG Evaluation:
   - Variant A: Ground Truth Oracle
   - Variant B: Raw OCR (noise propagation baseline)
   - Variant C: Corrected OCR (selective recovery)
6. Confidence Threshold Sensitivity Sweep (tau in [30, 90] to find tau* knee).
7. Head-to-head Efficiency Benchmark (Full-Text LLM vs. Selective LLM Correction).
8. Cross-Lingual Comparative Study (Latin English vs. Devanagari Hindi).
9. High-resolution publication figures (300 DPI PNG) and LaTeX booktabs tables.
10. Final freeze of run metadata and consolidated artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import AppConfig, load_config
from src.core.logging import get_logger, setup_logging
from src.core.reproducibility import (
    get_environment_info,
    get_git_commit_hash,
    get_package_versions,
    init_experiment_run,
)
from src.core.schemas import DocumentVariant
from src.dataset.manager import DatasetManager
from src.ocr.runner import OCRRunner
from src.correction.runner import CorrectionRunner
from src.rag.pipeline import RAGPipeline
from src.evaluation.rag_comparison import (
    compute_rag_triangulation,
    export_triangulation_summary,
    format_triangulation_table,
)
from src.experiments.threshold_sweep import ThresholdSweepRunner
from src.experiments.full_vs_selective import FullVsSelectiveRunner
from src.experiments.cross_lingual_analysis import CrossLingualAnalyzer
from src.experiments.export_figures import (
    generate_publication_figures,
    export_latex_and_markdown_tables,
)

logger = get_logger("master_runner")


class MasterExperimentRunner:
    """End-to-end master benchmark orchestrator for publication experiments."""

    def __init__(self, config: Optional[AppConfig] = None, config_path: str = "experiments/configs/default.yaml"):
        self.config_path = config_path
        self.config = config or load_config(config_path)

    def run_full_benchmark(
        self,
        run_id: Optional[str] = None,
        skip_sweep: bool = False,
        skip_figures: bool = False,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute complete end-to-end research experiment suite."""
        start_time = time.time()
        tau = threshold if threshold is not None else self.config.correction.threshold

        # 1. Initialize Experiment Run Directory & Freeze Metadata
        run_id, run_dir, meta = init_experiment_run(self.config, run_id=run_id)
        logger.info("=" * 80)
        logger.info("STARTING MASTER MULTILINGUAL OCR-AWARE RAG BENCHMARK")
        logger.info(f"Run ID: {run_id} | Working Dir: {run_dir}")
        logger.info(f"Config: {self.config_path} | Selective Threshold: tau={tau}")
        logger.info("=" * 80)

        # 2. Validate Dataset Integrity
        logger.info("\n--- STEP 1/8: VALIDATING BENCHMARK DATASET ---")
        dm = DatasetManager(self.config)
        val_report = dm.validate()
        if not val_report.is_valid:
            raise RuntimeError(f"Dataset validation failed: {val_report.errors}")
        pages = dm.load_metadata()
        questions = dm.load_questions()
        logger.info(f"Dataset verified: {len(pages)} pages, {len(questions)} questions.")

        # 3. Raw OCR Baseline
        logger.info("\n--- STEP 2/8: RUNNING RAW OCR EXTRACTION & CER/WER EVALUATION ---")
        ocr_runner = OCRRunner(self.config)
        ocr_results, ocr_aggregates = ocr_runner.run_all()
        raw_combined = ocr_aggregates.get("combined")
        logger.info(
            f"Raw OCR baseline: Combined CER = {raw_combined.mean_cer:.4f}, WER = {raw_combined.mean_wer:.4f}"
            if raw_combined else "Raw OCR completed."
        )

        # 4. Confidence-Guided Selective Correction
        logger.info(f"\n--- STEP 3/8: SELECTIVE OCR CORRECTION (tau={tau}) ---")
        corr_runner = CorrectionRunner(self.config)
        corr_results, corr_aggregates = corr_runner.run_all(threshold=tau)
        corr_combined = corr_aggregates.get("combined_corrected")
        if corr_combined and raw_combined:
            logger.info(
                f"Selective Correction: Combined CER = {corr_combined.mean_cer:.4f} (delta: {corr_combined.mean_cer - raw_combined.mean_cer:+.4f}), "
                f"WER = {corr_combined.mean_wer:.4f} (delta: {corr_combined.mean_wer - raw_combined.mean_wer:+.4f})"
            )

        # 5. Triangulated RAG Evaluation (Variants A, B, C)
        logger.info("\n--- STEP 4/8: RUNNING 3-VARIANT RAG EVALUATION ---")
        # Variant A: Ground Truth Oracle
        logger.info("Evaluating Variant A: Ground Truth...")
        pipe_gt = RAGPipeline(variant=DocumentVariant.GROUND_TRUTH, config=self.config)
        report_gt = pipe_gt.run_evaluation()

        # Variant B: Raw OCR
        logger.info("Evaluating Variant B: Raw OCR...")
        pipe_raw = RAGPipeline(variant=DocumentVariant.RAW_OCR, config=self.config)
        report_raw = pipe_raw.run_evaluation()

        # Variant C: Corrected OCR
        logger.info("Evaluating Variant C: Corrected OCR...")
        pipe_corr = RAGPipeline(variant=DocumentVariant.CORRECTED_OCR, config=self.config)
        report_corr = pipe_corr.run_evaluation()

        # Triangulate
        logger.info("Synthesizing 3-Variant Triangulation...")
        triangulation = compute_rag_triangulation(report_gt, report_raw, report_corr)
        tri_summary_path = os.path.join(self.config.output_dir, "rag", "comparison_summary.json")
        export_triangulation_summary(triangulation, output_path=tri_summary_path)

        # 6. Confidence Threshold Sweep
        sweep_report = None
        if not skip_sweep:
            logger.info("\n--- STEP 5/8: SENSITIVITY SWEEP (tau in [30, 90]) ---")
            sweep_runner = ThresholdSweepRunner(self.config)
            sweep_report = sweep_runner.run_sweep(thresholds=[30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
            logger.info(f"Sweep completed. Optimal knee: tau* = {sweep_report.get('optimal_threshold')}")
        else:
            logger.info("\n--- STEP 5/8: SKIPPING THRESHOLD SWEEP ---")

        # 7. Efficiency & Accuracy Comparison (Full-Text vs. Selective)
        logger.info("\n--- STEP 6/8: FULL-TEXT vs. SELECTIVE CORRECTION BENCHMARK ---")
        eff_runner = FullVsSelectiveRunner(self.config)
        eff_report = eff_runner.compare(selective_threshold=tau)
        logger.info(
            f"Efficiency: {eff_report['efficiency']['token_reduction_factor']:.1f}x cheaper "
            f"({eff_report['efficiency']['token_savings_pct']:.1f}% savings), "
            f"EM delta: {eff_report['efficiency']['qa_em_difference']:+.4f}"
        )

        # 8. Cross-Lingual Analysis
        logger.info("\n--- STEP 7/8: ENGLISH vs. HINDI CROSS-LINGUAL STUDY ---")
        cl_runner = CrossLingualAnalyzer(self.config)
        cl_report = cl_runner.run_analysis()
        logger.info("Cross-lingual study synthesized.")

        # 9. Figures and LaTeX/Markdown Tables
        figures = []
        tables = []
        if not skip_figures:
            logger.info("\n--- STEP 8/8: EXPORTING PUBLICATION FIGURES & TABLES ---")
            figures = generate_publication_figures(
                results_dir=self.config.output_dir,
                figures_dir=os.path.join(self.config.output_dir, "figures"),
            )
            tables = export_latex_and_markdown_tables(
                results_dir=self.config.output_dir,
                tables_dir=os.path.join(self.config.output_dir, "tables"),
            )
            logger.info(f"Generated {len(figures)} figures and {len(tables)} table files.")

        total_duration = time.time() - start_time

        # 10. Freeze Run Metadata
        meta_dict = meta.model_dump()
        meta_dict["status"] = "COMPLETED"
        meta_dict["completed_at"] = datetime.now().isoformat()
        meta_dict["total_duration_sec"] = total_duration
        meta_dict["artifacts"] = {
            "triangulation_summary": tri_summary_path,
            "figures": figures,
            "tables": tables,
        }
        if sweep_report:
            meta_dict["artifacts"]["sweep"] = os.path.join(self.config.output_dir, "sweeps", "threshold_sweep.json")
        meta_dict["artifacts"]["full_vs_selective"] = os.path.join(self.config.output_dir, "sweeps", "full_vs_selective.json")
        meta_dict["artifacts"]["cross_lingual"] = os.path.join(self.config.output_dir, "analysis", "cross_lingual_study.json")

        meta_file = Path(run_dir) / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        # Consolidated Master Run Result
        master_summary = {
            "run_id": run_id,
            "run_dir": run_dir,
            "total_duration_sec": total_duration,
            "triangulation": triangulation,
            "efficiency": eff_report.get("efficiency", {}),
            "figures": figures,
            "tables": tables,
        }
        if sweep_report:
            master_summary["optimal_threshold"] = sweep_report.get("optimal_threshold")

        logger.info("\n" + "=" * 80)
        logger.info(f"MASTER BENCHMARK SUCCESSFULLY COMPLETED IN {total_duration:.2f}s")
        logger.info(f"Run Artifacts Saved: {run_dir}")
        logger.info("=" * 80)

        return master_summary


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Master OCR-Aware Multilingual RAG Benchmark Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (auto-generated timestamp if not specified)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip confidence threshold parameter sweep",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure and table generation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override selective confidence threshold tau",
    )
    return parser.parse_args(args)


def main() -> int:
    setup_logging(log_level="INFO")
    args = parse_args()

    runner = MasterExperimentRunner(config_path=args.config)
    try:
        summary = runner.run_full_benchmark(
            run_id=args.run_id,
            skip_sweep=args.skip_sweep,
            skip_figures=args.skip_figures,
            threshold=args.threshold,
        )
        print("\n" + "=" * 90)
        print("MASTER BENCHMARK EXECUTION SUMMARY")
        print("=" * 90)
        print(f"  Run ID            : {summary['run_id']}")
        print(f"  Directory         : {summary['run_dir']}")
        print(f"  Duration          : {summary['total_duration_sec']:.2f} seconds")
        if "optimal_threshold" in summary:
            print(f"  Optimal Knee tau* : {summary['optimal_threshold']:.1f}")
        print(f"  Generated Figures : {len(summary['figures'])}")
        for fig in summary['figures']:
            print(f"    - {fig}")
        print(f"  Generated Tables  : {len(summary['tables'])}")
        for tab in summary['tables']:
            print(f"    - {tab}")
        print("=" * 90 + "\n")
        return 0
    except Exception as e:
        logger.exception(f"Master benchmark execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
