"""Experiment reproducibility, seed management, and metadata tracking.

Captures system environment, hardware architecture, git commit hash,
frozen experiment configs, and unique execution timestamps to guarantee
scientific traceability and auditability.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import platform
import random
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field

from src.core.config import AppConfig, save_config
from src.core.logging import setup_logging, get_logger

logger = get_logger("reproducibility")


class ExperimentRunMetadata(BaseModel):
    """Metadata recorded for every individual experiment run."""
    run_id: str = Field(..., description="Unique run identifier")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC start timestamp")
    git_commit: Optional[str] = Field(default=None, description="HEAD commit hash")
    git_dirty: bool = Field(default=False, description="Whether working tree had uncommitted changes")
    platform: Dict[str, Any] = Field(default_factory=dict, description="OS and hardware specs")
    python_version: str = Field(..., description="Python runtime version")
    package_versions: Dict[str, str] = Field(default_factory=dict, description="Key library versions")
    config_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Frozen config dump")
    status: str = Field(default="INITIALIZED", description="Run status")
    error_message: Optional[str] = Field(default=None)


def get_git_commit_hash() -> tuple[Optional[str], bool]:
    """Retrieve the current Git commit hash and whether the working directory is dirty."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_out) > 0
        return commit, is_dirty
    except Exception:
        return None, False


def get_environment_info() -> Dict[str, Any]:
    """Collect operating system and hardware details."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
    }


def get_package_versions() -> Dict[str, str]:
    """Inspect and return installed versions of key research dependencies."""
    key_packages = [
        "numpy",
        "pandas",
        "pydantic",
        "pyyaml",
        "opencv-python",
        "pillow",
        "pytesseract",
        "faiss-cpu",
        "sentence-transformers",
        "openai",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "jiwer",
    ]
    versions: Dict[str, str] = {}
    for pkg in key_packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
        except Exception as e:
            versions[pkg] = f"error: {str(e)}"
    return versions


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python standard library, NumPy, and PyTorch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_run_id(prefix: str = "run") -> str:
    """Generate a collision-resistant, chronologically sortable run ID."""
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}_{now_str}_{suffix}"


def init_experiment_run(
    config: AppConfig,
    run_id: Optional[str] = None,
) -> tuple[str, Path, ExperimentRunMetadata]:
    """Initialize an experiment run directory, freeze configuration, and set up logging."""
    if run_id is None:
        run_id = generate_run_id(config.experiment_name)

    run_dir = Path(config.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up run-specific log file
    log_file = run_dir / "experiment.log"
    setup_logging(log_level="INFO", log_file=log_file)

    # Set seed
    set_seed(config.seed)

    # Freeze config snapshot
    save_config(config, run_dir / "config.yaml")

    # Gather environment metadata
    commit, is_dirty = get_git_commit_hash()
    metadata = ExperimentRunMetadata(
        run_id=run_id,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=commit,
        git_dirty=is_dirty,
        platform=get_environment_info(),
        python_version=platform.python_version(),
        package_versions=get_package_versions(),
        config_snapshot=config.model_dump(),
        status="INITIALIZED",
    )

    # Save initial metadata
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    logger.info(f"Initialized experiment run '{run_id}' at: {run_dir}")
    return run_id, run_dir, metadata
