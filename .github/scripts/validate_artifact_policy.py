#!/usr/bin/env python3
"""Validate that CI exercises documentation and publishable crate contents."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
MACRO_MANIFEST = ROOT / "candle-einops-macros/Cargo.toml"


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    macro_manifest = MACRO_MANIFEST.read_text(encoding="utf-8")
    failures: list[str] = []

    if "cargo test --doc --workspace" not in workflow:
        failures.append("CI must run workspace doctests")
    if "RUSTDOCFLAGS: -D warnings" not in workflow:
        failures.append("CI doctests must deny rustdoc warnings")
    if "python3 .github/scripts/test_published_artifacts.py" not in workflow:
        failures.append("CI must unpack and test both published artifacts")

    package_list = subprocess.run(
        ["cargo", "package", "--list", "-p", "candle-einops-macros"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if package_list.returncode != 0:
        failures.append(f"could not inspect macro package: {package_list.stderr.strip()}")
    else:
        packaged = set(package_list.stdout.splitlines())
        harness = "tests/dependency_names.rs" in packaged
        fixtures = any(path.startswith("tests/fixtures/") for path in packaged)
        if harness and not fixtures:
            failures.append(
                "macro package includes the dependency fixture harness without its nested fixtures"
            )
        if not harness and "tests/dependency_names.rs" not in macro_manifest:
            failures.append("macro fixture harness exclusion must be explicit in Cargo.toml")

    if failures:
        print("Artifact policy validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Artifact policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
