"""Structured logging system for the research prototype.

Provides dual console and file logging with microsecond timestamps,
module identification, and log level controls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def close_logging_handlers() -> None:
    """Close and remove all handlers attached to the root and ocr_rag loggers."""
    for logger_name in ("", "ocr_rag"):
        l = logging.getLogger(logger_name)
        while l.handlers:
            h = l.handlers.pop()
            try:
                h.flush()
                h.close()
            except Exception:
                pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str | Path] = None,
    log_format: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure root logger for research console and optional run file logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Close and remove existing handlers to prevent resource leaks and locked files on Windows
    close_logging_handlers()

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=log_format, datefmt=DATE_FORMAT)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (if requested)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger("ocr_rag")


def get_logger(name: str) -> logging.Logger:
    """Obtain a namespaced child logger."""
    return logging.getLogger(f"ocr_rag.{name}")
