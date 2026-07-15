#!/usr/bin/env python3
"""Validate isolation and reproducibility of the performance harness."""

from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    benchmark_manifest = tomllib.loads(
        (ROOT / "benchmarks/Cargo.toml").read_text(encoding="utf-8")
    )
    wrapper = (ROOT / ".github/scripts/run_benchmarks.py").read_text(encoding="utf-8")
    harness = (ROOT / "benchmarks/src/lib.rs").read_text(encoding="utf-8")
    failures: list[str] = []

    if "benchmarks/" not in root_manifest["package"].get("exclude", []):
        failures.append("runtime package must explicitly exclude benchmarks/")
    if "benchmarks" not in root_manifest["workspace"].get("exclude", []):
        failures.append("root workspace must explicitly exclude benchmarks")
    if benchmark_manifest["package"].get("publish") is not False:
        failures.append("benchmark crate must set publish = false")
    if "workspace" not in benchmark_manifest:
        failures.append("benchmark crate must define its own workspace")
    if not (ROOT / "benchmarks/Cargo.lock").is_file():
        failures.append("benchmark crate must commit its own Cargo.lock")

    dependencies = benchmark_manifest.get("dependencies", {})
    required = {
        "candle-core": "=0.11.0",
        "criterion": "=0.7.0",
        "serde": "=1.0.228",
        "serde_json": "=1.0.149",
    }
    for dependency, version in required.items():
        configured = dependencies.get(dependency, {})
        if not isinstance(configured, dict) or configured.get("version") != version:
            failures.append(f"{dependency} must be pinned exactly to {version}")
    if dependencies.get("candle-einops", {}).get("path") != "..":
        failures.append("benchmark crate must use the repository root path dependency")

    for required_text in [
        '"+1.94"',
        '"--locked"',
        '"--manifest-path"',
        'ROOT / "target/benchmarks"',
    ]:
        if required_text not in wrapper:
            failures.append(f"benchmark wrapper must contain {required_text}")
    if "compile_error!" not in harness or 'feature = "cuda"' not in harness or 'feature = "metal"' not in harness:
        failures.append("benchmark crate must reject simultaneous Metal and CUDA features")

    forbidden = [
        path
        for path in (ROOT / "benchmarks").rglob("*")
        if path.is_file()
        and (path.name == "baseline" or "criterion" in path.parts or path.suffix == ".html")
    ]
    if forbidden:
        failures.append(f"benchmark reports or baselines must not be committed: {forbidden}")

    if failures:
        print("Performance harness policy validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Performance harness policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
