"""Unit tests for Phase 9: Full-Text vs Selective Correction Comparative Study."""

from __future__ import annotations

import unittest
from src.core.config import get_default_config
from src.core.schemas import Language
from src.correction.full_text_corrector import FullTextCorrector
from src.experiments.full_vs_selective import FullVsSelectiveRunner


class TestEfficiencyStudy(unittest.TestCase):
    """Test suite for full-text naive correction vs confidence-guided selective correction."""

    def test_full_text_corrector_basic(self):
        corrector = FullTextCorrector()
        raw = "The solor enorgy transition reached 300 gigawatts in western states."
        gt = "The solar energy transition reached 300 gigawatts in western states."

        res = corrector.correct_page(
            page_id="test_p1",
            raw_text=raw,
            language=Language.ENGLISH,
            ground_truth=gt,
        )

        self.assertEqual(res.page_id, "test_p1")
        self.assertGreater(res.prompt_tokens, 0)
        self.assertGreater(res.completion_tokens, 0)
        self.assertEqual(res.total_tokens, res.prompt_tokens + res.completion_tokens)
        self.assertIn("solar", res.corrected_text)
        self.assertIn("energy", res.corrected_text)

    def test_full_vs_selective_comparison(self):
        cfg = get_default_config()
        cfg.rag.embedding_model = "mock"
        runner = FullVsSelectiveRunner(cfg)

        comp = runner.compare(selective_threshold=70.0)

        self.assertIn("full_text", comp)
        self.assertIn("selective", comp)
        self.assertIn("efficiency", comp)

        # Selective correction must consume fewer tokens than full-text correction
        self.assertLess(comp["selective"]["total_tokens"], comp["full_text"]["total_tokens"])
        self.assertGreater(comp["efficiency"]["token_savings_pct"], 50.0)
        self.assertGreater(comp["efficiency"]["token_reduction_factor"], 2.0)

        # Selective correction must touch fewer words
        self.assertLess(comp["selective"]["flagged_ratio"], comp["full_text"]["flagged_ratio"])

        table_str = runner.format_comparison_table(comp)
        self.assertIn("Tokens Consumed", table_str)
        self.assertIn("cheaper", table_str)


if __name__ == "__main__":
    unittest.main()
