"""Unit tests for structured logging."""

import tempfile
import unittest
from pathlib import Path

from src.core.logging import setup_logging, get_logger, close_logging_handlers


class TestLogging(unittest.TestCase):
    def tearDown(self):
        close_logging_handlers()

    def test_logger_creation(self):
        """Verify get_logger returns a properly namespaced logger."""
        logger = get_logger("unit_test")
        self.assertEqual(logger.name, "ocr_rag.unit_test")

    def test_file_logging(self):
        """Verify log messages are written to a specified log file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test.log"
            logger = setup_logging(log_level="DEBUG", log_file=log_file)
            test_msg = "Unique experiment log verification token 12345"
            logger.info(test_msg)

            # Close handlers so Windows file lock is released
            close_logging_handlers()

            self.assertTrue(log_file.exists())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn(test_msg, content)


if __name__ == "__main__":
    unittest.main()
