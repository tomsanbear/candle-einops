#!/usr/bin/env python3
"""Contract tests for benchmark execution and GPU capture commands."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / ".github/scripts/run_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_benchmarks", MODULE_PATH)
assert SPEC and SPEC.loader
run_benchmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_benchmarks)


class CaptureCommandTests(unittest.TestCase):
    def test_measurement_commands_use_the_release_profile(self) -> None:
        for command in ("run", "probe", "capture", "gaps"):
            self.assertEqual(
                run_benchmarks.cargo_profile_arguments(command),
                ["--release"],
            )
        for command in ("compile", "smoke"):
            self.assertEqual(run_benchmarks.cargo_profile_arguments(command), [])

    def test_cuda_capture_uses_exact_profiler_api_range_without_cpu_sampling(self) -> None:
        command = run_benchmarks.nsys_capture_command(
            nsys=Path("/usr/local/bin/nsys"),
            binary=Path("/tmp/target/debug/harness"),
            output=Path("/tmp/captures/einsum-library"),
            harness_arguments=[
                "--filter",
                "einsum/binary/gemm",
                "--capture-operation",
                "library",
            ],
        )

        self.assertEqual(command[0], "/usr/local/bin/nsys")
        self.assertIn("--capture-range=cudaProfilerApi", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertIn("--sample=none", command)
        self.assertIn("--cpuctxsw=none", command)
        self.assertEqual(command[-4:], [
            "--filter",
            "einsum/binary/gemm",
            "--capture-operation",
            "library",
        ])

    def test_capture_requires_an_explicit_scenario_operation_and_gpu_backend(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--filter"):
            run_benchmarks.validate_capture_arguments(
                backend="metal", scenario_filter=None, operation="library"
            )
        with self.assertRaisesRegex(SystemExit, "--operation"):
            run_benchmarks.validate_capture_arguments(
                backend="cuda", scenario_filter="einsum", operation=None
            )
        with self.assertRaisesRegex(SystemExit, "GPU backend"):
            run_benchmarks.validate_capture_arguments(
                backend="cpu", scenario_filter="einsum", operation="library"
            )


if __name__ == "__main__":
    unittest.main()
