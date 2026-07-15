#!/usr/bin/env python3
"""Validate that einsum is documented, packaged, and tested as a 0.2 API."""

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MANIFEST = ROOT / "Cargo.toml"
MACRO_MANIFEST = ROOT / "candle-einops-macros/Cargo.toml"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    runtime = tomllib.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    macros = tomllib.loads(MACRO_MANIFEST.read_text(encoding="utf-8"))
    readme = read("README.md")
    changelog = read("CHANGELOG.md")
    contract = read("docs/einsum-contract.md")
    lib_docs = read("src/lib.rs")
    macro_docs = read("candle-einops-macros/src/lib.rs")
    workflow = read(".github/workflows/ci.yml")
    artifact_runner = read(".github/scripts/test_published_artifacts.py")
    failures: list[str] = []

    versions = {runtime["package"]["version"], macros["package"]["version"]}
    if versions != {"0.2.0"}:
        failures.append("runtime and macro crates must both be versioned 0.2.0")
    macro_requirement = runtime["dependencies"]["candle-einops-macros"]["version"]
    if macro_requirement != "=0.2.0":
        failures.append("the private macro/runtime ABI requires an exact =0.2.0 dependency")

    forbidden_history = [
        "no einsum in the root",
        "uncompilable probe",
        "oracle-only branch",
        "Subsequent implementation tickets",
    ]
    public_text = "\n".join([readme, changelog, contract, lib_docs])
    for phrase in forbidden_history:
        if phrase.lower() in public_text.lower():
            failures.append(f"public documentation retains historical unsupported language: {phrase!r}")

    required_contract_terms = {
        "unary": "unary",
        "binary": "binary",
        "ellipsis": "ellipsis",
        "diagonal": "diagonal",
        "n-ary": "n-ary",
    }
    for document_name, document in [("README", readme), ("einsum contract", contract)]:
        lowered = document.lower()
        for feature, marker in required_contract_terms.items():
            if marker not in lowered:
                failures.append(f"{document_name} must describe the supported {feature} contract")

    if "arbitrary-arity" not in lib_docs:
        failures.append("crate rustdoc must present einsum as arbitrary-arity")
    if "exactly the same version" not in macro_docs:
        failures.append("macro rustdoc must explain its private runtime ABI version coupling")

    equations = [
        "rows columns -> columns rows",
        "row inner, inner column -> row column",
        ".. feature -> feature",
        "index index -> index",
        "row inner, inner column, column -> row",
    ]
    documented = readme + contract + lib_docs
    for equation in equations:
        if equation not in documented:
            failures.append(f"public docs need a runnable supported example for {equation!r}")

    if "einsum!" not in changelog or "validate_einsum_release.py" not in changelog:
        failures.append("0.2.0 changelog/release checklist must cover einsum and its release validator")
    if not (ROOT / ".github/RELEASE_CHECKLIST.md").is_file():
        failures.append("a repository release checklist must describe the complete 0.2.0 gate")
    if "python3 .github/scripts/validate_einsum_release.py" not in workflow:
        failures.append("CI must run the einsum release-completeness validator")

    property_source = ROOT / "candle-einops-macros/src/einsum/properties.rs"
    if not property_source.is_file():
        failures.append("einsum parser/IR needs bounded deterministic never-unwind properties")
    else:
        properties = property_source.read_text(encoding="utf-8")
        for marker in ["catch_unwind", "DeterministicRandom", "REGRESSION_SEEDS", "minimize_unwind"]:
            if marker not in properties:
                failures.append(f"einsum parser/IR properties must include {marker}")

    for fixture in ["normal", "renamed", "keyword-alias"]:
        fixture_source = read(f"candle-einops-macros/tests/fixtures/{fixture}/src/main.rs")
        if "einsum!(" not in fixture_source:
            failures.append(f"the {fixture} downstream fixture must execute einsum")

    for marker in ["test_downstream_consumers", "normal-consumer", "renamed-consumer", "keyword-consumer"]:
        if marker not in artifact_runner:
            failures.append(f"unpacked artifacts must execute downstream package case {marker}")

    if "einsum" not in runtime["package"]["description"].lower():
        failures.append("runtime package description must advertise einsum support")
    if macros["package"].get("documentation") != "https://docs.rs/candle-einops":
        failures.append("macro package metadata must direct users to runtime documentation")

    for license_name in ["LICENSE-APACHE", "LICENSE-MIT"]:
        if not (ROOT / "candle-einops-macros" / license_name).is_file():
            failures.append(f"the published macro crate must carry {license_name}")

    package_list = subprocess.run(
        ["cargo", "package", "--list", "--allow-dirty", "-p", "candle-einops-macros"],
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
        for license_name in ["LICENSE-APACHE", "LICENSE-MIT"]:
            if license_name not in packaged:
                failures.append(f"macro artifact must include {license_name}")

    if failures:
        print("Einsum release validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Einsum release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
