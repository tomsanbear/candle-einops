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
    comparison_script = ROOT / ".github/scripts/compare_benchmarks.py"
    comparison_workflow = ROOT / ".github/workflows/advisory-performance.yml"
    required_workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
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

    for required_command in [
        "python3 .github/scripts/run_benchmarks.py compile",
        "python3 .github/scripts/run_benchmarks.py smoke",
        "python3 .github/scripts/run_benchmarks.py compile --cpu-implementation mkl",
        "python3 .github/scripts/run_benchmarks.py smoke --cpu-implementation mkl",
        "python3 .github/scripts/run_benchmarks.py compile --cpu-implementation accelerate",
        "python3 .github/scripts/run_benchmarks.py smoke --cpu-implementation accelerate",
        "python3 .github/scripts/run_benchmarks.py compile --backend metal",
        "python3 .github/scripts/run_benchmarks.py compile --backend cuda",
        "python3 .github/scripts/test_run_benchmarks.py",
        "python3 .github/scripts/test_compare_benchmarks.py",
    ]:
        if required_command not in required_workflow:
            failures.append(f"required CI must run `{required_command}`")
    if "python3 .github/scripts/compare_benchmarks.py" in required_workflow:
        failures.append("required CI must not compare benchmark timings")
    if "nvidia/cuda:" not in required_workflow or "devel-ubuntu24.04" not in required_workflow:
        failures.append("required CUDA compile coverage must use a CUDA devel container")
    for package in ["intel-oneapi-mkl-core-devel", "intel-oneapi-openmp"]:
        if package not in required_workflow:
            failures.append(f"MKL smoke must install `{package}` from current oneAPI")
    if required_workflow.count("source /opt/intel/oneapi/setvars.sh") != 2:
        failures.append("MKL compile and smoke must initialize the oneAPI environment")
    if "continue-on-error: true" in required_workflow:
        failures.append("device profile smoke coverage must be blocking")

    if not comparison_script.is_file():
        failures.append("advisory benchmark comparison script is missing")
    if not comparison_workflow.is_file():
        failures.append("manual advisory benchmark comparison workflow is missing")
    else:
        workflow = comparison_workflow.read_text(encoding="utf-8")
        required_workflow_text = [
            "workflow_dispatch:",
            "base_sha:",
            "head_sha:",
            "persist-credentials: false",
            "git cat-file -t",
            "git worktree add --detach",
            "CARGO_TARGET_DIR",
            "status --porcelain --untracked-files=all",
            "seq 1 5",
            "pair % 2",
            "compare_benchmarks.py",
            "advisory-only",
            "retention-days:",
        ]
        for required_text in required_workflow_text:
            if required_text not in workflow:
                failures.append(
                    f"advisory comparison workflow must contain {required_text}"
                )
        forbidden_workflow_text = [
            "pull_request:",
            "schedule:",
            "git push",
            "gh pr comment",
            "gh issue create",
        ]
        for forbidden_text in forbidden_workflow_text:
            if forbidden_text in workflow:
                failures.append(
                    f"advisory comparison workflow must not contain {forbidden_text}"
                )

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
