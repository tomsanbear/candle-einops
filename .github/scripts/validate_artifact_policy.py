#!/usr/bin/env python3
"""Validate that CI exercises documentation and publishable crate contents."""

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
RUNTIME_MANIFEST = ROOT / "Cargo.toml"
MACRO_MANIFEST = ROOT / "candle-einops-macros/Cargo.toml"
FIXTURE_HARNESS = ROOT / "candle-einops-macros/tests/dependency_names.rs"
FORBIDDEN_ARTIFACT_PATHS = (
    "benchmarks/",
    "parity/",
    ".venv/",
)
FORBIDDEN_ARTIFACT_NAMES = {
    ".python-version",
    "pyproject.toml",
    "uv.lock",
}


def package_paths(package: str, failures: list[str]) -> set[str]:
    package_list = subprocess.run(
        ["cargo", "package", "--list", "--allow-dirty", "-p", package],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if package_list.returncode != 0:
        failures.append(f"could not inspect {package} package: {package_list.stderr.strip()}")
        return set()
    return set(package_list.stdout.splitlines())


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runtime_manifest = tomllib.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    macro_manifest = MACRO_MANIFEST.read_text(encoding="utf-8")
    fixture_harness = FIXTURE_HARNESS.read_text(encoding="utf-8")
    failures: list[str] = []

    if "cargo test --doc --workspace" not in workflow:
        failures.append("CI must run workspace doctests")
    if "RUSTDOCFLAGS: -D warnings" not in workflow:
        failures.append("CI doctests must deny rustdoc warnings")
    if "python3 .github/scripts/test_published_artifacts.py" not in workflow:
        failures.append("CI must unpack and test both published artifacts")

    if "parity/" not in runtime_manifest["package"].get("exclude", []):
        failures.append("runtime package must explicitly exclude parity/")
    if "benchmarks/" not in runtime_manifest["package"].get("exclude", []):
        failures.append("runtime package must explicitly exclude benchmarks/")
    if "benchmarks" not in runtime_manifest["workspace"].get("exclude", []):
        failures.append("root workspace must explicitly exclude benchmarks")

    packages = {
        package: package_paths(package, failures)
        for package in ["candle-einops", "candle-einops-macros"]
    }
    for package, packaged in packages.items():
        leaked = sorted(
            path
            for path in packaged
            if path.startswith(FORBIDDEN_ARTIFACT_PATHS)
            or Path(path).name in FORBIDDEN_ARTIFACT_NAMES
            or Path(path).suffix == ".py"
        )
        if leaked:
            failures.append(
                f"{package} artifact contains benchmark or Python/parity material: {leaked}"
            )

    packaged = packages["candle-einops-macros"]
    if packaged:
        harness = "tests/dependency_names.rs" in packaged
        fixtures = any(path.startswith("tests/fixtures/") for path in packaged)
        if harness and not fixtures:
            failures.append(
                "macro package includes the dependency fixture harness without its nested fixtures"
            )
        if not harness and "tests/dependency_names.rs" not in macro_manifest:
            failures.append("macro fixture harness exclusion must be explicit in Cargo.toml")

    if '.args(["run", "--quiet", "--manifest-path"])' not in fixture_harness:
        failures.append("downstream fixtures must execute rather than only compile")
    for fixture in ["normal", "renamed", "hygiene"]:
        if f'"{fixture}"' not in fixture_harness:
            failures.append(f"downstream fixture harness must execute `{fixture}`")

    if failures:
        print("Artifact policy validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Artifact policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
