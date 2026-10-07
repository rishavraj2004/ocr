"""Unit tests for Phase 8: Confidence Threshold Sensitivity Sweep."""

from __future__ import annotations

import unittest
from src.core.config import get_default_config
from src.experiments.threshold_sweep import ThresholdSweepRunner


class TestThresholdSweep(unittest.TestCase):
    """Test suite for confidence threshold sensitivity sweep runner."""

    def test_sweep_runner_subset(self):
        cfg = get_default_config()
        cfg.rag.embedding_model = "mock"
        runner = ThresholdSweepRunner(cfg)

        test_thresholds = [30.0, 70.0]
        report = runner.run_sweep(thresholds=test_thresholds)

        self.assertEqual(len(report["points"]), 2)
        self.assertIn("optimal_threshold", report)
        self.assertIn(report["optimal_threshold"], test_thresholds)

        # Monotonicity check: tau=70.0 should flag >= words than tau=30.0
        pt30 = report["points"][0]
        pt70 = report["points"][1]
        self.assertEqual(pt30["threshold"], 30.0)
        self.assertEqual(pt70["threshold"], 70.0)
        self.assertGreaterEqual(pt70["flagged_words"], pt30["flagged_words"])

        table_str = runner.format_sweep_table(report)
        self.assertIn("30", table_str)
        self.assertIn("70", table_str)
        self.assertIn("Optimal operating threshold", table_str)


if __name__ == "__main__":
    unittest.main()
