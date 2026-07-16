#!/usr/bin/env python3
"""Contract tests for the committed performance-report generator."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("generate_performance_report.py")


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_performance_report", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURE = {
    "schema_version": 1,
    "snapshot": {
        "date": "2026-07-16",
        "processes": 5,
        "samples_per_process": 25,
        "thresholds": {
            "median_percent": 10.0,
            "median_absolute_ns": 1000.0,
            "confidence_excludes_percent": 5.0,
        },
    },
    "providers": [
        {
            "id": "cpu-baseline",
            "label": "CPU baseline",
            "device": "Example CPU",
            "scenarios": {
                "family/loss": {
                    "status": "reference_gap",
                    "library_process_median_ns": 1200.0,
                    "reference_process_median_ns": 1000.0,
                    "median_delta_ns": 200.0,
                    "median_percent": 20.0,
                },
                "family/tie": {
                    "status": "parity",
                    "library_process_median_ns": 1001.0,
                    "reference_process_median_ns": 1000.0,
                    "median_delta_ns": 1.0,
                    "median_percent": 0.1,
                },
                "family/win": {
                    "status": "library_faster",
                    "library_process_median_ns": 500.0,
                    "reference_process_median_ns": 1000.0,
                    "median_delta_ns": -500.0,
                    "median_percent": -50.0,
                },
            },
        },
        {
            "id": "cuda",
            "label": "CUDA",
            "device": "Example GPU",
            "scenarios": {
                "family/tie": {
                    "status": "parity",
                    "library_process_median_ns": 1000.0,
                    "reference_process_median_ns": 1000.0,
                    "median_delta_ns": 0.0,
                    "median_percent": 0.0,
                }
            },
        },
    ],
}


class PerformanceReportGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator()

    def test_report_is_exhaustive_and_links_source_data(self):
        report = self.generator.render_report(FIXTURE)
        self.assertIn("1 win", report)
        self.assertIn("Complete scenario matrix", report)
        self.assertIn("family/loss", report)
        self.assertIn("L +20.0% / +0.20 us", report)
        self.assertIn("benchmarks/data/performance-2026-07-16.json", report)

    def test_figures_are_deterministic_and_encode_every_outcome(self):
        outcomes = self.generator.render_outcomes_svg(FIXTURE)
        heatmap = self.generator.render_heatmap_svg(FIXTURE)
        self.assertEqual(outcomes, self.generator.render_outcomes_svg(FIXTURE))
        self.assertEqual(heatmap, self.generator.render_heatmap_svg(FIXTURE))
        for label in ["Win", "Tie", "Loss", "Skipped"]:
            self.assertIn(label, outcomes + heatmap)
        for scenario in ["family/loss", "family/tie", "family/win"]:
            self.assertIn(scenario, heatmap)
        self.assertIn("+20.0%", heatmap)
        self.assertIn("+0.20 us", heatmap)
        self.assertIn("role=\"img\"", outcomes)
        self.assertIn("role=\"img\"", heatmap)
        self.assertIn("data-render-fingerprint=\"", outcomes)
        self.assertTrue(self.generator.svg_is_current(outcomes, outcomes))
        self.assertFalse(
            self.generator.svg_is_current(
                outcomes.replace("data-render-fingerprint=\"", "data-stale=\"", 1),
                outcomes,
            )
        )


if __name__ == "__main__":
    unittest.main()
