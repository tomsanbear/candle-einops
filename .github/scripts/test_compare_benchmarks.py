#!/usr/bin/env python3
"""Contract tests for advisory benchmark comparison reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / ".github/scripts/compare_benchmarks.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("compare_benchmarks", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_benchmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_benchmarks)


BASE_SHA = "1" * 40
HEAD_SHA = "2" * 40


def record(
    sha: str,
    median_ns: float,
    *,
    scenario_id: str = "einsum/binary/rank2",
    schema_version: int = 1,
    elements: int = 64,
    rust_version: object = "rustc 1.94.1",
    sample_count: int = 25,
    sampling_order_policy: str = "alternating_library_then_reference",
) -> dict[str, object]:
    estimate = {
        "median_ns": median_ns,
        "confidence_interval": {
            "confidence_level": 0.95,
            "lower_bound_ns": median_ns,
            "upper_bound_ns": median_ns,
        },
    }
    return {
        "schema_version": schema_version,
        "scenario_id": scenario_id,
        "tracked": True,
        "workload": {"elements": elements, "bytes": elements * 4, "flops": 128},
        "sample_count": sample_count,
        "library": estimate,
        "reference": estimate,
        "library_to_reference_ratio": 1.0,
        "sampling_order_policy": sampling_order_policy,
        "fingerprint": {
            "git_sha": sha,
            "rust_version": rust_version,
            "candle_version": "0.11.0",
            "os": "linux",
            "architecture": "x86_64",
            "backend": "cpu",
            "device": "cpu",
            "driver": None,
        },
    }


class AdvisoryComparisonTests(unittest.TestCase):
    def write_runs(
        self,
        root: Path,
        side: str,
        sha: str,
        medians: list[float],
        **changes: object,
    ) -> list[Path]:
        paths = []
        for index, median in enumerate(medians, 1):
            path = root / f"{side}-{index}.json"
            path.write_text(json.dumps([record(sha, median, **changes)]), encoding="utf-8")
            paths.append(path)
        return paths

    def test_report_marks_only_large_confident_movement_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(root, "base", BASE_SHA, [10_000] * 5)
            head = self.write_runs(root, "head", HEAD_SHA, [12_000] * 5)
            report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)

        self.assertTrue(report["advisory_only"])
        self.assertFalse(report["can_fail_required_ci"])
        self.assertEqual(report["process_pairs"], 5)
        result = report["scenarios"][0]
        self.assertEqual(result["status"], "advisory_regression")
        self.assertEqual(result["paired_median_delta_ns"], 2_000)
        self.assertEqual(result["paired_median_percent"], 20.0)
        self.assertEqual(result["required_repeated_observations"], 3)
        self.assertFalse(result["automatic_action"])

    def test_thresholds_require_ten_percent_one_microsecond_and_ci_beyond_five(self) -> None:
        classify = compare_benchmarks.classify_movement
        self.assertEqual(classify(12.0, 900.0, 11.0, 13.0), "no_advisory")
        self.assertEqual(classify(8.0, 2_000.0, 7.0, 9.0), "no_advisory")
        self.assertEqual(classify(12.0, 2_000.0, 4.0, 20.0), "no_advisory")
        self.assertEqual(classify(-12.0, -2_000.0, -20.0, -6.0), "advisory_improvement")

    def test_workload_schema_and_environment_mismatches_are_incomparable(self) -> None:
        cases = [
            ({"elements": 65}, "workload_mismatch"),
            ({"schema_version": 2}, "schema_mismatch"),
            ({"rust_version": "rustc 1.95.0"}, "fingerprint_mismatch"),
        ]
        for changes, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = self.write_runs(root, "base", BASE_SHA, [10_000] * 5)
                head = self.write_runs(root, "head", HEAD_SHA, [12_000] * 5, **changes)
                report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)
                self.assertEqual(report["scenarios"][0]["status"], "incomparable")
                self.assertEqual(report["scenarios"][0]["reason"], reason)

    def test_matching_unknown_schema_is_incomparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(
                root, "base", BASE_SHA, [10_000] * 5, schema_version=2
            )
            head = self.write_runs(
                root, "head", HEAD_SHA, [12_000] * 5, schema_version=2
            )
            report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)
        self.assertEqual(report["scenarios"][0]["status"], "incomparable")
        self.assertEqual(report["scenarios"][0]["reason"], "unsupported_schema")

    def test_sampling_methodology_mismatch_is_incomparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(root, "base", BASE_SHA, [10_000] * 5)
            head = self.write_runs(
                root,
                "head",
                HEAD_SHA,
                [12_000] * 5,
                sampling_order_policy="fixed_library_then_reference",
            )
            report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)
        self.assertEqual(report["scenarios"][0]["status"], "incomparable")
        self.assertEqual(
            report["scenarios"][0]["reason"], "sampling_policy_mismatch"
        )

    def test_sample_count_mismatch_is_incomparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(root, "base", BASE_SHA, [10_000] * 5)
            head = self.write_runs(
                root, "head", HEAD_SHA, [12_000] * 5, sample_count=999
            )
            report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)
        self.assertEqual(report["scenarios"][0]["status"], "incomparable")
        self.assertEqual(report["scenarios"][0]["reason"], "sample_count_mismatch")

    def test_malformed_fingerprint_is_incomparable_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(root, "base", BASE_SHA, [10_000] * 5)
            head = self.write_runs(
                root, "head", HEAD_SHA, [12_000] * 5, rust_version=[]
            )
            report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)
        self.assertEqual(report["scenarios"][0]["status"], "incomparable")
        self.assertEqual(report["scenarios"][0]["reason"], "fingerprint_mismatch")

    def test_missing_scenario_is_explicitly_incomparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(root, "base", BASE_SHA, [10_000] * 5)
            head = self.write_runs(
                root,
                "head",
                HEAD_SHA,
                [12_000] * 5,
                scenario_id="repeat/broadcast/consume",
            )
            report = compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)
        self.assertEqual(
            [(item["scenario_id"], item["reason"]) for item in report["scenarios"]],
            [
                ("einsum/binary/rank2", "missing_from_head"),
                ("repeat/broadcast/consume", "missing_from_base"),
            ],
        )

    def test_five_process_pairs_and_expected_shas_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.write_runs(root, "base", BASE_SHA, [10_000] * 4)
            head = self.write_runs(root, "head", HEAD_SHA, [12_000] * 4)
            with self.assertRaisesRegex(compare_benchmarks.ComparisonInputError, "at least five"):
                compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)

            base = self.write_runs(root, "base", "3" * 40, [10_000] * 5)
            head = self.write_runs(root, "head", HEAD_SHA, [12_000] * 5)
            with self.assertRaisesRegex(compare_benchmarks.ComparisonInputError, "expected base SHA"):
                compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)

    def test_empty_documents_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = []
            head = []
            for side, paths in (("base", base), ("head", head)):
                for index in range(1, 6):
                    path = root / f"{side}-{index}.json"
                    path.write_text("[]", encoding="utf-8")
                    paths.append(path)
            with self.assertRaisesRegex(
                compare_benchmarks.ComparisonInputError, "no benchmark scenarios"
            ):
                compare_benchmarks.compare_files(base, head, BASE_SHA, HEAD_SHA)


if __name__ == "__main__":
    unittest.main()
