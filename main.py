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
    return parser.parse_args(args)


def main() -> int:
    setup_logging(log_level="INFO")
    args = parse_args()

    # If no flags passed, run default validation and environment check
    if not (args.check_env or args.validate_config or args.init_run or args.validate_dataset or args.generate_dataset):
        logger.info("Running default Phase 0/1 sanity check...")
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
    if args.init_run and status == 0:
        status = test_init_run(args.config)

    return status


if __name__ == "__main__":
    sys.exit(main())
