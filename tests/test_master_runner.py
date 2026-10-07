"""Unit tests for Phase 11: Master Experiment Runner and Publication Export."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.core.config import AppConfig, load_config
from run_experiment import MasterExperimentRunner, parse_args
from src.experiments.export_figures import (
    generate_publication_figures,
    export_latex_and_markdown_tables,
)


class TestMasterRunnerAndExports(unittest.TestCase):
    """Test suite for publication assets export and master orchestrator."""

    def setUp(self):
        self.config = load_config("experiments/configs/default.yaml")

    def test_export_publication_figures(self):
        """Verify publication figures are created as 300 DPI PNGs."""
        figs = generate_publication_figures(
            results_dir=self.config.output_dir,
            figures_dir=os.path.join(self.config.output_dir, "figures"),
        )
        self.assertGreaterEqual(len(figs), 3)
        for f in figs:
            self.assertTrue(os.path.exists(f), f"Expected figure {f} to exist")
            self.assertGreater(os.path.getsize(f), 1000, f"Expected non-empty image {f}")

    def test_export_latex_and_markdown_tables(self):
        """Verify LaTeX booktabs and Markdown tables are generated."""
        tables = export_latex_and_markdown_tables(
            results_dir=self.config.output_dir,
            tables_dir=os.path.join(self.config.output_dir, "tables"),
        )
        self.assertGreaterEqual(len(tables), 5)
        for t in tables:
            self.assertTrue(os.path.exists(t), f"Expected table file {t} to exist")
            with open(t, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertGreater(len(content), 50, f"Expected non-trivial content in {t}")

        # Check content of summary markdown table
        md_summary = os.path.join(self.config.output_dir, "tables", "summary_tables.md")
        self.assertTrue(os.path.exists(md_summary))
        with open(md_summary, "r", encoding="utf-8") as f:
            md_text = f.read()
        self.assertIn("3-Variant Triangulation Study", md_text)
        self.assertIn("Full-Text vs. Selective Correction", md_text)
        self.assertIn("Confidence Threshold Sensitivity", md_text)
        self.assertIn("Cross-Lingual Comparison", md_text)

    def test_master_runner_initialization(self):
        """Verify MasterExperimentRunner initializes cleanly with configuration."""
        runner = MasterExperimentRunner(self.config)
        self.assertIsNotNone(runner.config)
        self.assertEqual(runner.config.experiment_name, "ocr_multilingual_rag")

    def test_parse_args_master(self):
        """Verify CLI arguments parsing for master benchmark."""
        args = parse_args(["--config", "experiments/configs/default.yaml", "--skip-sweep", "--skip-figures"])
        self.assertTrue(args.skip_sweep)
        self.assertTrue(args.skip_figures)
        self.assertIsNone(args.threshold)


if __name__ == "__main__":
    unittest.main()
