#!/usr/bin/env python3
"""Tests for performance harness repository-artifact filtering."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("validate_performance_harness_policy.py")


def load_validator():
    spec = importlib.util.spec_from_file_location("performance_policy", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtifactFilteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_validator()

    def test_ignored_development_environments_are_not_repository_artifacts(self):
        for path in [
            pathlib.Path("benchmarks/reporting/.venv/site-packages/figure.html"),
            pathlib.Path("benchmarks/target/criterion/report/index.html"),
            pathlib.Path("benchmarks/reporting/__pycache__/module.html"),
        ]:
            self.assertFalse(self.validator.is_forbidden_benchmark_artifact(path))

    def test_repository_reports_remain_forbidden(self):
        for path in [
            pathlib.Path("benchmarks/report.html"),
            pathlib.Path("benchmarks/criterion/report.json"),
            pathlib.Path("benchmarks/baseline"),
        ]:
            self.assertTrue(self.validator.is_forbidden_benchmark_artifact(path))


if __name__ == "__main__":
    unittest.main()
