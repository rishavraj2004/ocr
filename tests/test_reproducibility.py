"""Unit tests for reproducibility tracking, seeding, and run initialization."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from src.core.logging import close_logging_handlers
from src.core.config import get_default_config
from src.core.reproducibility import (
    generate_run_id,
    set_seed,
    get_environment_info,
    get_package_versions,
    init_experiment_run,
)


class TestReproducibility(unittest.TestCase):
    def tearDown(self):
        close_logging_handlers()

    def test_run_id_generation(self):
        """Run IDs should have the given prefix and be unique."""
        r1 = generate_run_id("test")
        r2 = generate_run_id("test")
        self.assertTrue(r1.startswith("test_"))
        self.assertNotEqual(r1, r2)

    def test_set_seed_deterministic(self):
        """Setting seed should produce identical pseudorandom sequences."""
        set_seed(1234)
        sample_a = [np.random.rand() for _ in range(5)]
        set_seed(1234)
        sample_b = [np.random.rand() for _ in range(5)]
        self.assertEqual(sample_a, sample_b)

    def test_get_environment_info(self):
        """Environment info dictionary should contain OS, architecture, and python details."""
        info = get_environment_info()
        self.assertIn("os", info)
        self.assertIn("architecture", info)
        self.assertIn("python_implementation", info)

    def test_get_package_versions(self):
        """Package version audit should inspect key dependencies."""
        versions = get_package_versions()
        self.assertIn("numpy", versions)
        self.assertIn("pydantic", versions)
        self.assertIn("pyyaml", versions)

    def test_init_experiment_run(self):
        """Initializing a run must create directory, freeze config.yaml, and write metadata.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = get_default_config()
            cfg.runs_dir = tmp_dir
            run_id, run_path, meta = init_experiment_run(cfg, run_id="unit_test_run")

            self.assertEqual(run_id, "unit_test_run")
            self.assertTrue(run_path.exists())
            self.assertTrue((run_path / "config.yaml").exists())
            self.assertTrue((run_path / "metadata.json").exists())
            self.assertTrue((run_path / "experiment.log").exists())
            self.assertEqual(meta.status, "INITIALIZED")

            # Release file lock before temp directory cleanup
            close_logging_handlers()


if __name__ == "__main__":
    unittest.main()
