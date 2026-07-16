#!/usr/bin/env python3
"""Compile and run the isolated, locked performance harness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/Cargo.toml"
TARGET = ROOT / "target/benchmarks"


def feature_for(backend: str, cpu_implementation: str) -> str | None:
    if backend != "cpu":
        if cpu_implementation != "baseline":
            raise SystemExit("GPU backends require --cpu-implementation baseline")
        return backend
    return None if cpu_implementation == "baseline" else cpu_implementation


def cargo(*arguments: str, backend: str, cpu_implementation: str) -> None:
    subcommand, *subcommand_arguments = arguments
    command = [
        "cargo",
        "+1.94",
        subcommand,
        "--locked",
        "--manifest-path",
        str(MANIFEST),
    ]
    feature = feature_for(backend, cpu_implementation)
    if feature:
        command.extend(["--features", feature])
    command.extend(subcommand_arguments)
    environment = os.environ.copy()
    environment.setdefault("CARGO_TARGET_DIR", str(TARGET))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "smoke", "run", "probe"))
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument(
        "--cpu-implementation",
        choices=("baseline", "accelerate", "mkl"),
        default="baseline",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--filter")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    if args.device_index < 0:
        raise SystemExit("--device-index must be non-negative")
    feature_for(args.backend, args.cpu_implementation)
    if args.command == "compile":
        cargo(
            "bench",
            "--no-run",
            backend=args.backend,
            cpu_implementation=args.cpu_implementation,
        )
        return 0
    if args.command == "smoke":
        cargo("test", backend=args.backend, cpu_implementation=args.cpu_implementation)
        profile = args.backend if args.backend != "cpu" else f"cpu-{args.cpu_implementation}"
        smoke_output = args.output or TARGET / f"plumbing-smoke-{profile}.json"
        cargo(
            "run",
            "--bin",
            "harness",
            backend=args.backend,
            cpu_implementation=args.cpu_implementation,
            *(
                "--",
                "--include-plumbing",
                "--samples",
                "3",
                "--backend",
                args.backend,
                "--cpu-implementation",
                args.cpu_implementation,
                "--device-index",
                str(args.device_index),
                "--output",
                str(smoke_output),
            ),
        )
        return 0

    if args.command == "probe" and args.backend != "cpu":
        raise SystemExit("diagonal index preparation probes are CPU-only")

    harness_arguments = [
        "--samples",
        str(args.samples),
        "--backend",
        args.backend,
        "--cpu-implementation",
        args.cpu_implementation,
        "--device-index",
        str(args.device_index),
    ]
    if args.filter:
        harness_arguments.extend(["--filter", args.filter])
    if args.output:
        harness_arguments.extend(["--output", str(args.output)])
    binary = "diagonal_probe" if args.command == "probe" else "harness"
    cargo(
        "run",
        "--bin",
        binary,
        backend=args.backend,
        cpu_implementation=args.cpu_implementation,
        *("--", *harness_arguments),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
