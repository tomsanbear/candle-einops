#!/usr/bin/env python3
"""Validate security and portability invariants in the CI configuration."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
WORKFLOWS = ROOT / ".github/workflows"
DENY_CONFIG = ROOT / "deny.toml"
MSRV_POLICY = ROOT / ".github/MSRV_POLICY.md"
DEPENDABOT = ROOT / ".github/dependabot.yml"


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    failures: list[str] = []

    for workflow_path in sorted(WORKFLOWS.glob("*.yml")):
        workflow_text = workflow_path.read_text(encoding="utf-8")
        if not re.search(r"(?m)^permissions:\n  contents: read$", workflow_text):
            failures.append(
                f"{workflow_path.name} must set top-level `permissions: contents: read`"
            )
        uses = re.findall(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)", workflow_text)
        mutable = [
            value
            for value in uses
            if not value.startswith("./") and not re.search(r"@[0-9a-f]{40}$", value)
        ]
        if mutable:
            failures.append(
                f"{workflow_path.name} action references must use full commit SHAs: "
                + ", ".join(mutable)
            )
        checkout_count = sum(value.startswith("actions/checkout@") for value in uses)
        credential_opt_outs = len(
            re.findall(r"persist-credentials:\s*false", workflow_text)
        )
        if checkout_count == 0 or credential_opt_outs != checkout_count:
            failures.append(
                f"{workflow_path.name} checkout steps must set persist-credentials false"
            )

    if not DENY_CONFIG.is_file():
        failures.append("dependency policy config `deny.toml` is missing")
    if "cargo deny check" not in workflow and "cargo-deny-action@" not in workflow:
        failures.append("workflow must run `cargo deny check`")

    dependabot = DEPENDABOT.read_text(encoding="utf-8")
    if "package-ecosystem: github-actions" not in dependabot:
        failures.append("Dependabot must continue updating pinned actions")

    if not re.search(r"runs-on:\s*macos-", workflow):
        failures.append("workflow must include a representative macOS job")

    if not MSRV_POLICY.is_file():
        failures.append("MSRV and lock-resolution policy is undocumented")
    else:
        policy = MSRV_POLICY.read_text(encoding="utf-8")
        for required in ["Rust 1.94", "resolver 3", "Cargo.lock", "unlocked"]:
            if required not in policy:
                failures.append(f"MSRV policy must explain `{required}`")

    if failures:
        print("CI policy validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("CI policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
