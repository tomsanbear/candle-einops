#!/usr/bin/env python3
"""Compile and run the isolated, locked performance harness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
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


def cargo(
    *arguments: str,
    backend: str,
    cpu_implementation: str,
    extra_environment: dict[str, str] | None = None,
) -> None:
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
    if extra_environment:
        environment.update(extra_environment)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def validate_capture_arguments(
    *, backend: str, scenario_filter: str | None, operation: str | None
) -> None:
    if backend == "cpu":
        raise SystemExit("capture requires a Metal or CUDA GPU backend")
    if not scenario_filter:
        raise SystemExit("capture requires an exact --filter")
    if operation is None:
        raise SystemExit("capture requires --operation library or reference")


def nsys_capture_command(
    *,
    nsys: Path,
    binary: Path,
    output: Path,
    harness_arguments: list[str],
) -> list[str]:
    return [
        str(nsys),
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "--output",
        str(output),
        str(binary),
        *harness_arguments,
    ]


def cargo_target_dir() -> Path:
    return Path(os.environ.get("CARGO_TARGET_DIR", TARGET))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("compile", "smoke", "run", "probe", "capture")
    )
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
    parser.add_argument("--operation", choices=("library", "reference"))
    parser.add_argument("--warmups", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    if args.device_index < 0:
        raise SystemExit("--device-index must be non-negative")
    if args.warmups < 1:
        raise SystemExit("--warmups must be positive")
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

    if args.command == "capture":
        validate_capture_arguments(
            backend=args.backend,
            scenario_filter=args.filter,
            operation=args.operation,
        )
        assert args.filter is not None
        assert args.operation is not None
        capture_name = "-".join(
            part for part in args.filter.replace("/", "-").split() if part
        )
        output = args.output or TARGET / "captures" / f"{capture_name}-{args.operation}"
        output.parent.mkdir(parents=True, exist_ok=True)
        harness_arguments = [
            "--backend",
            args.backend,
            "--cpu-implementation",
            args.cpu_implementation,
            "--device-index",
            str(args.device_index),
            "--filter",
            args.filter,
            "--capture-operation",
            args.operation,
            "--capture-warmups",
            str(args.warmups),
        ]
        if args.backend == "metal":
            metal_output = output.with_suffix(".gputrace")
            harness_arguments.extend(["--capture-output", str(metal_output)])
            cargo(
                "run",
                "--bin",
                "harness",
                backend=args.backend,
                cpu_implementation=args.cpu_implementation,
                extra_environment={"MTL_CAPTURE_ENABLED": "1"},
                *("--", *harness_arguments),
            )
            return 0

        nsys_path = shutil.which("nsys")
        if nsys_path is None:
            raise SystemExit("CUDA capture requires nsys on PATH")
        cargo(
            "build",
            "--bin",
            "harness",
            backend=args.backend,
            cpu_implementation=args.cpu_implementation,
        )
        command = nsys_capture_command(
            nsys=Path(nsys_path),
            binary=cargo_target_dir() / "debug/harness",
            output=output,
            harness_arguments=harness_arguments,
        )
        subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
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
