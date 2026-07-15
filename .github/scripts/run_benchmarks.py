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


def cargo(*arguments: str, backend: str) -> None:
    subcommand, *subcommand_arguments = arguments
    command = [
        "cargo",
        "+1.94",
        subcommand,
        "--locked",
        "--manifest-path",
        str(MANIFEST),
    ]
    if backend != "cpu":
        command.extend(["--features", backend])
    command.extend(subcommand_arguments)
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(TARGET)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "smoke", "run", "probe"))
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--filter")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    if args.command == "compile":
        cargo("bench", "--no-run", backend=args.backend)
        return 0
    if args.command == "smoke":
        if args.backend != "cpu":
            raise SystemExit("foundation smoke measurements are CPU-only")
        cargo("test", backend="cpu")
        smoke_output = args.output or TARGET / "plumbing-smoke.json"
        cargo(
            "run",
            "--bin",
            "harness",
            backend="cpu",
            *(
                "--",
                "--include-plumbing",
                "--samples",
                "3",
                "--output",
                str(smoke_output),
            ),
        )
        return 0

    if args.command == "probe" and args.backend != "cpu":
        raise SystemExit("diagonal index preparation probes are CPU-only")

    harness_arguments = ["--samples", str(args.samples)]
    if args.filter:
        harness_arguments.extend(["--filter", args.filter])
    if args.output:
        harness_arguments.extend(["--output", str(args.output)])
    binary = "diagonal_probe" if args.command == "probe" else "harness"
    cargo("run", "--bin", binary, backend=args.backend, *("--", *harness_arguments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
