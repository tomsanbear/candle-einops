#!/usr/bin/env python3
"""Contract tests for repeated-process library/reference gap summaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / ".github/scripts/summarize_reference_gaps.py"
SPEC = importlib.util.spec_from_file_location("summarize_reference_gaps", MODULE_PATH)
assert SPEC and SPEC.loader
summarize_reference_gaps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_reference_gaps)


def document(library_ns: float, reference_ns: float, profile: str = "release") -> dict:
    return {
        "schema_version": 2,
        "run": {
            "git_sha": "a" * 40,
            "build_profile": profile,
            "backend": "cpu",
            "cpu_implementation": "baseline",
            "device_index": None,
            "device_name": "CPU (baseline)",
        },
        "records": [
            {
                "scenario_id": "einsum/example",
                "workload": {"elements": 1024, "bytes": 4096},
                "sample_count": 25,
                "sampling_order_policy": "alternating_library_then_reference",
                "library": {"median_ns": library_ns},
                "reference": {"median_ns": reference_ns},
            }
        ],
        "skipped": [],
    }


class ReferenceGapSummaryTests(unittest.TestCase):
    def write_runs(
        self,
        root: Path,
        library_values: list[float],
        reference_ns: float,
        profile: str = "release",
    ) -> list[Path]:
        paths = []
        for index, library_ns in enumerate(library_values, 1):
            path = root / f"run-{index:02}.json"
            path.write_text(
                json.dumps(document(library_ns, reference_ns, profile)), encoding="utf-8"
            )
            paths.append(path)
        return paths

    def test_classifies_only_material_confident_reference_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_runs(
                Path(directory), [12_000, 12_100, 12_200, 12_300, 12_400], 10_000
            )
            report = summarize_reference_gaps.summarize_files(paths)

        self.assertEqual(report["process_count"], 5)
        self.assertEqual(report["summary"]["reference_gap"], 1)
        result = report["scenarios"][0]
        self.assertEqual(result["status"], "reference_gap")
        self.assertEqual(result["median_delta_ns"], 2_200)
        self.assertEqual(result["median_percent"], 22.0)
        self.assertGreater(result["percent_interval"]["lower_bound_percent"], 5.0)

    def test_large_percentage_below_one_microsecond_is_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_runs(Path(directory), [1_500] * 5, 1_000)
            report = summarize_reference_gaps.summarize_files(paths)

        self.assertEqual(report["scenarios"][0]["status"], "parity")

    def test_requires_five_optimized_independent_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            too_few = self.write_runs(root, [12_000] * 4, 10_000)
            with self.assertRaisesRegex(ValueError, "at least five"):
                summarize_reference_gaps.summarize_files(too_few)
            debug = self.write_runs(root, [12_000] * 5, 10_000, profile="debug")
            with self.assertRaisesRegex(ValueError, "release profile"):
                summarize_reference_gaps.summarize_files(debug)


if __name__ == "__main__":
    unittest.main()
