#!/usr/bin/env python3
"""Run the only supported local and CI Python einops parity workflow."""

from pathlib import Path
import argparse
import os
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "parity"
RUNNER_MANIFEST = PARITY / "runner/Cargo.toml"


def run(arguments: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(arguments)}", flush=True)
    subprocess.run(arguments, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="deterministic Rust property seed")
    parser.add_argument("--cases", type=int, help="bounded cases per property")
    parser.add_argument("--max-elements", type=int, help="maximum tensor elements")
    parser.add_argument("--replay-file", type=Path, help="replay one minimized JSON request")
    arguments = parser.parse_args()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_MANAGED_PYTHON": "1",
            "UV_NO_BUILD": "1",
        }
    )
    overrides = {
        "CANDLE_EINOPS_PARITY_SEED": arguments.seed,
        "CANDLE_EINOPS_PARITY_CASES": arguments.cases,
        "CANDLE_EINOPS_PARITY_MAX_ELEMENTS": arguments.max_elements,
    }
    env.update({name: str(value) for name, value in overrides.items() if value is not None})

    run(
        [
            "uv",
            "run",
            "--project",
            "parity",
            "--frozen",
            "--exact",
            "--managed-python",
            "--no-build",
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            "parity/tests",
            "-v",
        ],
        env,
    )

    cargo = ["cargo"]
    if arguments.replay_file is None:
        cargo.extend(
            [
                "test",
                "--locked",
                "--manifest-path",
                "parity/runner/Cargo.toml",
                "--all-targets",
                "--all-features",
            ]
        )
    else:
        cargo.extend(
            [
                "run",
                "--locked",
                "--manifest-path",
                "parity/runner/Cargo.toml",
                "--",
                "--replay-file",
                str(arguments.replay_file),
            ]
        )
    run(cargo, env)


if __name__ == "__main__":
    main()
