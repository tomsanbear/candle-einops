#!/usr/bin/env python3
"""Validate the reproducible Python einops parity CI and artifact boundary."""

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "parity"
WORKFLOW = ROOT / ".github/workflows/ci.yml"
WRAPPER = ROOT / ".github/scripts/test_python_parity.py"
SETUP_UV = "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990"
UV_VERSION = "0.11.28"
PYTHON_VERSION = "3.12.10"
REPLAY_COMMAND = "python3 .github/scripts/test_python_parity.py"


def package_paths(package: str, failures: list[str]) -> set[str]:
    result = subprocess.run(
        ["cargo", "package", "--list", "--allow-dirty", "-p", package],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        failures.append(f"could not inspect {package} package: {result.stderr.strip()}")
        return set()
    return set(result.stdout.splitlines())


def main() -> int:
    failures: list[str] = []
    workflow = WORKFLOW.read_text(encoding="utf-8")
    pyproject = tomllib.loads((PARITY / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    artifact_policy = (ROOT / ".github/scripts/validate_artifact_policy.py").read_text(
        encoding="utf-8"
    )

    if (PARITY / ".python-version").read_text(encoding="utf-8").strip() != PYTHON_VERSION:
        failures.append(f"parity Python must be pinned to {PYTHON_VERSION}")
    if pyproject.get("tool", {}).get("uv", {}).get("required-version") != f"=={UV_VERSION}":
        failures.append(f"parity must require exactly uv {UV_VERSION}")
    expected_dependencies = {
        "einops==0.8.2",
        "numpy==2.5.1",
        "hypothesis==6.156.6",
    }
    dependencies = set(pyproject["project"]["dependencies"])
    if dependencies != expected_dependencies:
        failures.append("parity Python dependencies must remain exactly pinned")
    lockfile = (PARITY / "uv.lock").read_text(encoding="utf-8")
    for marker in ["einops==0.8.2", "numpy==2.5.1", "hypothesis==6.156.6"]:
        name, version = marker.split("==")
        if f'name = "{name}"' not in lockfile or f'version = "{version}"' not in lockfile:
            failures.append(f"uv.lock must contain {marker}")

    if not WRAPPER.is_file():
        failures.append("one supported local/CI parity wrapper is missing")
        wrapper = ""
    else:
        wrapper = WRAPPER.read_text(encoding="utf-8")
    for marker in [
        "--frozen",
        "--exact",
        "--managed-python",
        "--no-build",
        "unittest",
        "cargo",
        "--locked",
        "parity/runner/Cargo.toml",
    ]:
        if marker not in wrapper:
            failures.append(f"parity wrapper must enforce {marker}")

    for marker in [
        "python-parity:",
        "Python einops parity",
        SETUP_UV,
        f'version: "{UV_VERSION}"',
        f'python-version: "{PYTHON_VERSION}"',
        "enable-cache: true",
        "cache-dependency-glob: parity/uv.lock",
        REPLAY_COMMAND,
    ]:
        if marker not in workflow:
            failures.append(f"CI parity job must include {marker}")
    if workflow.count(REPLAY_COMMAND) != 1:
        failures.append("parity wrapper must run exactly once in its isolated CI job")

    excludes = set(runtime_manifest["package"].get("exclude", []))
    if "parity/" not in excludes:
        failures.append("the runtime package must explicitly exclude parity/")
    for package in ["candle-einops", "candle-einops-macros"]:
        leaked = sorted(
            path
            for path in package_paths(package, failures)
            if path.startswith("parity/")
            or path.endswith((".py", "uv.lock", ".python-version", "pyproject.toml"))
        )
        if leaked:
            failures.append(f"{package} artifact contains parity/Python files: {leaked[:3]}")
    for marker in ["candle-einops", "candle-einops-macros", "FORBIDDEN_ARTIFACT_PATHS"]:
        if marker not in artifact_policy:
            failures.append(f"artifact policy must enforce parity isolation with {marker}")

    runner = (PARITY / "runner/src/lib.rs").read_text(encoding="utf-8")
    for marker in ["--frozen", "--no-sync", "--managed-python", "--no-build"]:
        if marker not in runner:
            failures.append(f"Rust parity runner must launch the oracle with {marker}")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / ".github/RELEASE_CHECKLIST.md"]
    )
    if not (PARITY / "README.md").is_file():
        failures.append("parity contract and replay documentation is missing")
    elif REPLAY_COMMAND not in (PARITY / "README.md").read_text(encoding="utf-8"):
        failures.append("parity documentation must give the one-command replay")
    if documentation.count(REPLAY_COMMAND) < 3:
        failures.append("README, changelog, and release checklist must document parity replay")

    if failures:
        print("Python parity policy validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Python parity policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
