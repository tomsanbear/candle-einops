#!/usr/bin/env python3
"""Validate security and portability invariants in the CI configuration."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
DENY_CONFIG = ROOT / ".github/deny.toml"
MSRV_POLICY = ROOT / ".github/MSRV_POLICY.md"


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    failures: list[str] = []

    if not re.search(r"(?m)^permissions:\n  contents: read$", workflow):
        failures.append("workflow must set top-level `permissions: contents: read`")

    uses = re.findall(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)", workflow)
    mutable = [value for value in uses if not re.search(r"@[0-9a-f]{40}$", value)]
    if mutable:
        failures.append(f"action references must use full commit SHAs: {', '.join(mutable)}")

    checkout_count = sum(value.startswith("actions/checkout@") for value in uses)
    credential_opt_outs = len(re.findall(r"persist-credentials:\s*false", workflow))
    if checkout_count == 0 or credential_opt_outs != checkout_count:
        failures.append("every checkout step must set `persist-credentials: false`")

    if not DENY_CONFIG.is_file():
        failures.append("dependency policy config `.github/deny.toml` is missing")
    if "cargo deny check" not in workflow:
        failures.append("workflow must run `cargo deny check`")

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
