"""CLI entrypoint for OCR-Aware Multilingual RAG Research Framework.

Provides commands to inspect system environment, validate experimental configurations,
and initialize reproducible experiment runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.config import load_config, AppConfig
from src.core.logging import setup_logging, get_logger
from src.core.reproducibility import (
    get_environment_info,
    get_package_versions,
    get_git_commit_hash,
    init_experiment_run,
)

logger = get_logger("cli")


def check_environment() -> int:
    """Audit system dependencies and print diagnostic report."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT & SYSTEM AUDIT")
    logger.info("=" * 60)

    env_info = get_environment_info()
    for k, v in env_info.items():
        logger.info(f"  {k:25s}: {v}")

    commit, is_dirty = get_git_commit_hash()
    logger.info(f"  {'git_commit':25s}: {commit or 'None (not committed yet)'}")
    logger.info(f"  {'git_dirty':25s}: {is_dirty}")

    logger.info("-" * 60)
    logger.info("INSTALLED RESEARCH PACKAGES:")
    packages = get_package_versions()
    for pkg, ver in packages.items():
        logger.info(f"  {pkg:25s}: {ver}")

    logger.info("=" * 60)
    return 0


def validate_configuration(config_path: str) -> int:
    """Load and validate the specified experiment configuration."""
    logger.info(f"Validating configuration from: {config_path}")
    try:
        cfg = load_config(config_path)
        logger.info(f"Configuration is VALID. Experiment: '{cfg.experiment_name}'")
        logger.info(f"  Languages       : {cfg.dataset.languages}")
        logger.info(f"  OCR Engine      : {cfg.ocr.engine}")
        logger.info(f"  Confidence Thr. : {cfg.correction.threshold}")
        logger.info(f"  Chunk Size      : {cfg.rag.chunk_size} {cfg.rag.chunk_unit}")
        logger.info(f"  Embedding Model : {cfg.rag.embedding_model}")
        logger.info(f"  Retrieval Top-K : {cfg.rag.top_k}")
        return 0
    except Exception as e:
        logger.error(f"Configuration validation FAILED: {e}")
        return 1


def test_init_run(config_path: str) -> int:
    """Initialize a test experiment run to verify directory creation and reproducibility."""
    cfg = load_config(config_path)
    run_id, run_dir, metadata = init_experiment_run(cfg, run_id="phase0_verification_run")
    logger.info(f"Test run successfully created at: {run_dir}")
    logger.info(f"Run ID: {run_id}, status: {metadata.status}")
    return 0


def validate_dataset(config_path: str) -> int:
    """Validate consistency of dataset images, ground truth, metadata, and questions."""
    from src.dataset.manager import DatasetManager

    cfg = load_config(config_path)
    dm = DatasetManager(cfg)
    report = dm.validate()
    print(report.summary_str())
    return 0 if report.is_valid else 1


def generate_dataset() -> int:
    """Generate or re-synthesize the sample benchmark dataset."""
    from src.dataset.generator import generate_benchmark_dataset

    logger.info("Generating standard benchmark dataset (5 English + 5 Hindi pages)...")
    pages, questions = generate_benchmark_dataset()
    logger.info(f"Successfully generated {pages} pages and {questions} evaluation questions.")
    return 0


def preprocess_page(page_id: str, config_path: str) -> int:
    """Run preprocessing on a specific page and export debug before/after comparison."""
    from src.dataset.manager import DatasetManager
    from src.preprocessing.image_preprocessor import ImagePreprocessor

    cfg = load_config(config_path)
    dm = DatasetManager(cfg)
    meta = dm.get_page_metadata(page_id)
    if meta is None:
        logger.error(f"Page ID '{page_id}' not found in metadata.")
        return 1

    preprocessor = ImagePreprocessor(cfg.preprocessing)
    img_path = meta.image_path
    logger.info(f"Preprocessing page '{page_id}' from: {img_path}")
    result = preprocessor.process(img_path)

    logger.info("=" * 60)
    logger.info(f"PREPROCESSING COMPLETED: {page_id}")
    logger.info(f"  Operations Applied : {result.operations_applied}")
    logger.info(f"  Original Shape     : {result.original_shape}")
    logger.info(f"  Processed Shape    : {result.processed_shape}")
    logger.info(f"  Execution Time     : {result.execution_time_sec:.4f}s")
    for k, v in result.metadata.items():
        logger.info(f"  Metadata: {k:10s}: {v}")
    logger.info("=" * 60)

    out_debug = Path("results/figures/preprocessing_debug") / f"{page_id}_comparison.png"
    preprocessor.save_debug_comparison(img_path, result, out_debug)
    logger.info(f"Saved side-by-side debug comparison image to: {out_debug}")
    return 0


def run_ocr(config_path: str) -> int:
    """Execute OCR pipeline across dataset, persist tokens, and report CER/WER."""
    from src.ocr.runner import OCRRunner

    cfg = load_config(config_path)
    runner = OCRRunner(cfg)
    results, aggregates = runner.run_all()

    print("\n" + "=" * 60)
    print("OCR BASELINE EVALUATION RESULTS")
    print("=" * 60)
    for lang_key in ["en", "hi", "combined"]:
        if lang_key in aggregates:
            print("\n" + aggregates[lang_key].summary_table_str())

    return 0


def run_correction(config_path: str, threshold: Optional[float] = None) -> int:
    """Execute confidence-guided selective correction and evaluate recovery."""
    from src.correction.runner import CorrectionRunner

    cfg = load_config(config_path)
    runner = CorrectionRunner(cfg)
    results, aggregates = runner.run_all(threshold=threshold)

    thresh_val = threshold or cfg.correction.threshold
    print("\n" + "=" * 65)
    print(f"CONFIDENCE-GUIDED SELECTIVE CORRECTION RESULTS (tau={thresh_val})")
    print("=" * 65)

    for lang in ["en", "hi", "combined"]:
        raw_agg = aggregates.get(f"{lang}_raw")
        corr_agg = aggregates.get(f"{lang}_corrected")
        if raw_agg and corr_agg:
            cer_rec = raw_agg.mean_cer - corr_agg.mean_cer
            wer_rec = raw_agg.mean_wer - corr_agg.mean_wer
            print(f"\n--- {lang.upper()} EVALUATION SUMMARY ---")
            print(f"  Raw OCR Mean CER       : {raw_agg.mean_cer:.4f}  ->  Corrected CER : {corr_agg.mean_cer:.4f}  (Recovery: {cer_rec:+.4f})")
            print(f"  Raw OCR Mean WER       : {raw_agg.mean_wer:.4f}  ->  Corrected WER : {corr_agg.mean_wer:.4f}  (Recovery: {wer_rec:+.4f})")

    return 0


def run_rag(
    config_path: str,
    variant: str = "ground_truth",
    embedding_model: Optional[str] = None,
) -> int:
    """Run end-to-end RAG pipeline and evaluation for a specified document variant."""
    import json
    import os
    from src.core.schemas import DocumentVariant
    from src.rag.pipeline import RAGPipeline

    cfg = load_config(config_path)
    if embedding_model:
        cfg.rag.embedding_model = embedding_model

    doc_variant = DocumentVariant(variant)

    logger.info(f"Initializing RAG pipeline for variant: {doc_variant.value}")
    pipeline = RAGPipeline(variant=doc_variant, config=cfg)
    report = pipeline.run_evaluation()

    overall = report["overall"]
    by_lang = report["by_language"]
    by_type = report.get("by_question_type", {})

    print("\n" + "=" * 75)
    print(f"RAG EXPERIMENT EVALUATION: VARIANT {doc_variant.value.upper()}")
    print("=" * 75)
    print(f"  Pages Indexed          : {report['summary']['num_pages']}")
    print(f"  Questions Evaluated    : {report['summary']['num_questions']}")
    print(f"  Total Latency          : {overall['total_eval_latency_sec']:.2f} s")
    print("-" * 75)
    print("OVERALL RETRIEVAL & QA PERFORMANCE:")
    print(f"  Recall@1               : {overall['mean_recall_at_1']:.4f}")
    print(f"  Recall@3               : {overall['mean_recall_at_3']:.4f}")
    print(f"  Recall@5               : {overall['mean_recall_at_5']:.4f}")
    print(f"  MRR                    : {overall['mean_mrr']:.4f}")
    print(f"  Exact Match (EM)       : {overall['mean_exact_match']:.4f}")
    print(f"  Token F1 Score         : {overall['mean_f1']:.4f}")
    print("-" * 75)
    print("PER-LANGUAGE BREAKDOWN:")
    for lang, metrics in by_lang.items():
        print(f"  [{lang.upper()}] (N={metrics['num_questions']}):")
        print(f"    Recall@1: {metrics['mean_recall_at_1']:.4f} | Recall@5: {metrics['mean_recall_at_5']:.4f} | MRR: {metrics['mean_mrr']:.4f}")
        print(f"    EM: {metrics['mean_exact_match']:.4f}       | F1: {metrics['mean_f1']:.4f}")
    if by_type:
        print("-" * 75)
        print("PER-QUESTION-TYPE BREAKDOWN:")
        for qtype, metrics in by_type.items():
            print(f"  [{qtype.upper()}] (N={metrics['num_questions']}):")
            print(f"    Recall@1: {metrics['mean_recall_at_1']:.4f} | MRR: {metrics['mean_mrr']:.4f} | EM: {metrics['mean_exact_match']:.4f} | F1: {metrics['mean_f1']:.4f}")

    # If evaluating Raw OCR and Ground Truth results exist, print Comparative Delta Table
    if doc_variant == DocumentVariant.RAW_OCR:
        gt_path = os.path.join(cfg.output_dir, "rag", "ground_truth_results.json")
        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_report = json.load(f)
            gt_overall = gt_report["overall"]
            print("\n" + "=" * 75)
            print("COMPARATIVE STUDY: GROUND TRUTH (A) vs RAW OCR (B) DEGRADATION")
            print("=" * 75)
            print(f"  {'Metric':<22} | {'Ground Truth (A)':<16} | {'Raw OCR (B)':<14} | {'Degradation (Delta)':<18}")
            print("  " + "-" * 71)
            for m_key, m_name in [
                ("mean_recall_at_1", "Recall@1"),
                ("mean_recall_at_3", "Recall@3"),
                ("mean_recall_at_5", "Recall@5"),
                ("mean_mrr", "MRR"),
                ("mean_exact_match", "Exact Match (EM)"),
                ("mean_f1", "Token F1 Score"),
            ]:
                a_val = gt_overall.get(m_key, 0.0)
                b_val = overall.get(m_key, 0.0)
                delta = b_val - a_val
                print(f"  {m_name:<22} | {a_val:<16.4f} | {b_val:<14.4f} | {delta:+18.4f}")
            print("=" * 75 + "\n")

    # If evaluating Corrected OCR, produce full 3-variant Triangulation Table
    elif doc_variant == DocumentVariant.CORRECTED_OCR:
        from src.evaluation.rag_comparison import (
            compute_rag_triangulation,
            export_triangulation_summary,
            format_triangulation_table,
        )

        gt_path = os.path.join(cfg.output_dir, "rag", "ground_truth_results.json")
        raw_path = os.path.join(cfg.output_dir, "rag", "raw_ocr_results.json")

        if os.path.exists(gt_path) and os.path.exists(raw_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_report = json.load(f)
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_report = json.load(f)

            triangulation = compute_rag_triangulation(gt_report, raw_report, report)
            summary_path = os.path.join(cfg.output_dir, "rag", "comparison_summary.json")
            export_triangulation_summary(triangulation, output_path=summary_path)

            print("\n" + "=" * 90)
            print("3-VARIANT TRIANGULATION: A (Ground Truth) vs B (Raw OCR) vs C (Corrected OCR)")
            print("=" * 90)
            print(format_triangulation_table(triangulation))
            print("=" * 90 + "\n")

    print("=" * 75 + "\n")
    return 0


def compare_rag(config_path: str) -> int:
    """Load existing RAG evaluation results for A, B, C and display 3-variant triangulation table."""
    import json
    import os
    from src.evaluation.rag_comparison import (
        compute_rag_triangulation,
        export_triangulation_summary,
        format_triangulation_table,
    )

    cfg = load_config(config_path)
    rag_dir = os.path.join(cfg.output_dir, "rag")
    gt_path = os.path.join(rag_dir, "ground_truth_results.json")
    raw_path = os.path.join(rag_dir, "raw_ocr_results.json")
    corr_path = os.path.join(rag_dir, "corrected_ocr_results.json")

    missing = []
    if not os.path.exists(gt_path):
        missing.append("ground_truth (run: --run-rag --variant ground_truth)")
    if not os.path.exists(raw_path):
        missing.append("raw_ocr (run: --run-rag --variant raw_ocr)")
    if not os.path.exists(corr_path):
        missing.append("corrected_ocr (run: --run-rag --variant corrected_ocr)")

    if missing:
        logger.error(f"Cannot compare RAG variants. Missing artifacts:\n  - " + "\n  - ".join(missing))
        return 1

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_report = json.load(f)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_report = json.load(f)
    with open(corr_path, "r", encoding="utf-8") as f:
        corr_report = json.load(f)

    triangulation = compute_rag_triangulation(gt_report, raw_report, corr_report)
    summary_path = os.path.join(rag_dir, "comparison_summary.json")
    export_triangulation_summary(triangulation, output_path=summary_path)

    print("\n" + "=" * 90)
    print("3-VARIANT TRIANGULATION: A (Ground Truth) vs B (Raw OCR) vs C (Corrected OCR)")
    print("=" * 90)
    print(format_triangulation_table(triangulation))
    print("=" * 90 + "\n")
    return 0


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="OCR-Aware Multilingual RAG Research Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check and print system environment and installed package versions",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the specified configuration file",
    )
    parser.add_argument(
        "--init-run",
        action="store_true",
        help="Test initialize an experiment run directory and metadata freeze",
    )
    parser.add_argument(
        "--validate-dataset",
        action="store_true",
        help="Validate consistency of dataset images, ground truth, metadata, and questions",
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Synthesize the 10-page benchmark dataset (5 English, 5 Hindi)",
    )
    parser.add_argument(
        "--preprocess-page",
        type=str,
        metavar="PAGE_ID",
        help="Run image preprocessing on a specific page_id and save comparison image",
    )
    parser.add_argument(
        "--run-ocr",
        action="store_true",
        help="Execute OCR extraction across all pages, save tokens, and evaluate CER/WER",
    )
    parser.add_argument(
        "--run-correction",
        action="store_true",
        help="Execute confidence-guided selective correction and evaluate CER/WER delta",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold tau (default: 70.0)",
    )
    parser.add_argument(
        "--run-rag",
        action="store_true",
        help="Execute RAG pipeline and evaluation on specified document variant",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "raw_ocr", "corrected_ocr"],
        help="Document variant to evaluate: 'ground_truth', 'raw_ocr', or 'corrected_ocr'",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model name (e.g. 'mock' or 'BAAI/bge-m3')",
    )
    parser.add_argument(
        "--compare-rag",
        action="store_true",
        help="Display 3-variant comparison (A vs B vs C) from existing evaluation artifacts",
    )
    return parser.parse_args(args)


def main() -> int:
    setup_logging(log_level="INFO")
    args = parse_args()

    # If no flags passed, run default validation and environment check
    if not (args.check_env or args.validate_config or args.init_run or args.validate_dataset or args.generate_dataset or args.preprocess_page or args.run_ocr or args.run_correction or args.run_rag or args.compare_rag):
        logger.info("Running default Phase 0/1/2/3/4 sanity check...")
        val_status = validate_configuration(args.config)
        if val_status != 0:
            return val_status
        return check_environment()

    status = 0
    if args.generate_dataset:
        status = generate_dataset()
    if args.check_env and status == 0:
        status = check_environment()
    if args.validate_config and status == 0:
        status = validate_configuration(args.config)
    if args.validate_dataset and status == 0:
        status = validate_dataset(args.config)
    if args.preprocess_page and status == 0:
        status = preprocess_page(args.preprocess_page, args.config)
    if args.run_ocr and status == 0:
        status = run_ocr(args.config)
    if args.run_correction and status == 0:
        status = run_correction(args.config, threshold=args.threshold)
    if args.run_rag and status == 0:
        status = run_rag(args.config, variant=args.variant, embedding_model=args.embedding_model)
    if args.compare_rag and status == 0:
        status = compare_rag(args.config)
    if args.init_run and status == 0:
        status = test_init_run(args.config)

    return status


if __name__ == "__main__":
    sys.exit(main())

