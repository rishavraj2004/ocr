"""Unit tests for Phase 10: English vs. Hindi Cross-Lingual Comparative Study."""

from __future__ import annotations

import unittest
from src.core.config import get_default_config
from src.experiments.cross_lingual_analysis import CrossLingualAnalyzer


class TestCrossLingualAnalysis(unittest.TestCase):
    """Test suite for English vs Hindi cross-lingual analysis."""

    def test_cross_lingual_analyzer(self):
        cfg = get_default_config()
        analyzer = CrossLingualAnalyzer(cfg)

        report = analyzer.run_analysis()

        self.assertIn("script_profiles", report)
        self.assertIn("comparative_metrics", report)
        self.assertIn("question_type_breakdown", report)
        self.assertIn("key_insights", report)

        self.assertIn("en", report["script_profiles"])
        self.assertIn("hi", report["script_profiles"])
        self.assertEqual(report["script_profiles"]["en"]["script"], "Latin")
        self.assertEqual(report["script_profiles"]["hi"]["script"], "Devanagari")

        # Verify numerical breakdown
        self.assertIn("numerical", report["question_type_breakdown"]["en"])
        self.assertIn("numerical", report["question_type_breakdown"]["hi"])

        table_str = analyzer.format_cross_lingual_table(report)
        self.assertIn("English (Latin)", table_str)
        self.assertIn("Hindi (Devanagari)", table_str)
        self.assertIn("Numerical", table_str)


if __name__ == "__main__":
    unittest.main()
