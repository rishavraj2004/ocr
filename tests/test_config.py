"""Unit tests for configuration loading and validation."""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from src.core.config import (
    AppConfig,
    RAGConfig,
    load_config,
    save_config,
    get_default_config,
)


class TestConfig(unittest.TestCase):
    def test_default_config_validity(self):
        """Verify default configuration instantiates without errors and has sound defaults."""
        cfg = get_default_config()
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.experiment_name, "ocr_multilingual_rag")
        self.assertEqual(cfg.dataset.languages, ["en", "hi"])
        self.assertEqual(cfg.rag.chunk_size, 500)
        self.assertEqual(cfg.rag.chunk_overlap, 50)
        self.assertLess(cfg.rag.chunk_overlap, cfg.rag.chunk_size)

    def test_rag_chunk_overlap_validation(self):
        """Overlap must be strictly smaller than chunk size."""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=100, chunk_overlap=150)

    def test_save_and_load_roundtrip(self):
        """Verify configuration can be saved to YAML and reloaded identically."""
        cfg = get_default_config()
        cfg.experiment_name = "test_roundtrip"
        cfg.correction.threshold = 65.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_config.yaml"
            save_config(cfg, tmp_path)
            self.assertTrue(tmp_path.exists())

            reloaded = load_config(tmp_path)
            self.assertEqual(reloaded.experiment_name, "test_roundtrip")
            self.assertEqual(reloaded.correction.threshold, 65.0)

    def test_load_nonexistent_file_raises(self):
        """Attempting to load a nonexistent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_path/should_fail.yaml")

    def test_default_yaml_file_matches_model(self):
        """Ensure the committed default.yaml loads into AppConfig cleanly."""
        default_yaml = Path("experiments/configs/default.yaml")
        if default_yaml.exists():
            cfg = load_config(default_yaml)
            self.assertIsInstance(cfg, AppConfig)
            self.assertEqual(cfg.dataset.languages, ["en", "hi"])


if __name__ == "__main__":
    unittest.main()
