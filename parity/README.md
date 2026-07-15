# Python einops parity

This repository checks the public Rust `einops!` behavior against a locked
NumPy-backed Python einops oracle. The environment is fixed to uv 0.11.28,
Python 3.12.10, einops 0.8.2, NumPy 2.5.1, and Hypothesis 6.156.6 by
`pyproject.toml`, `.python-version`, and `uv.lock`.

Run the complete protocol and Rust property suite from the repository root:

```console
python3 .github/scripts/test_python_parity.py
```

This wrapper is the supported local and CI entrypoint. It performs a frozen,
exact sync using only managed Python and binary distributions, runs the bounded
Python protocol tests, then runs the parity runner with its committed Cargo
lockfile. Ordinary `cargo test` and published crates never require Python.

## Supported parity contract

The corpus compares shapes, acceptance, and flattened values for:

- rearrangement permutations, composition, decomposition, singleton squeeze,
  zero through three ellipsis captures, and non-contiguous inputs;
- repeats at leading, trailing, grouped, and ellipsis positions, including zero
  lengths; and
- `sum`, `mean`, `min`, `max`, and `prod` reductions over individual,
  consecutive, grouped, all, and ellipsis axes.

The two syntaxes express equivalent operations differently. Rust `..` maps to
Python `...`; Rust braced sizes such as `{copies}` map to Python keyword axis
lengths; Rust inline reductions such as `a sum(b) -> a` map to Python
`reduce(value, "a b -> a", "sum")`; and Rust group annotations such as `c:2`
map to Python keyword lengths. Each stable pattern id records this translation.

Rearrange and repeat results, plus min/max reductions, compare exactly. Other
floating reductions use a documented f32 epsilon scaled by reduction length.
Empty mean is excluded because NumPy returns NaNs with a warning while Candle's
backend policy differs. Einsum, device placement, gradients, and compile-time
diagnostic text are covered by Rust-native suites rather than this parity lane.

## Bounded runs and replay

Defaults are 64 cases per property, seed `6840158760032202765`, and at most 512
tensor elements. Override them without bypassing the locked environment:

```console
python3 .github/scripts/test_python_parity.py --seed 42 --cases 17 --max-elements 128
```

Property failures print their deterministic seed. Rearrange failures also save
the minimized JSON request under `parity/regressions/`. Replay it with:

```console
python3 .github/scripts/test_python_parity.py --replay-file parity/regressions/rearrange-last-failure.jsonl
```

Promote a minimized failure into the explicit edge corpus before deleting the
temporary replay file.

## Dependency updates

Pins change only in a dedicated review. Update the requested dependency and
lockfile with the pinned uv binary, for example:

```console
uv lock --project parity --upgrade-package einops==NEW_VERSION
```

Update the exact declaration in `parity/pyproject.toml`, review the `uv.lock`
diff and upstream release notes, then run the supported wrapper and package
policy validators. Never upgrade dependencies during CI.
