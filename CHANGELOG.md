# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-16

### Changed

- Upgraded `candle-core` from 0.6 to 0.11.
- Migrated both crates to Rust 2024 with a minimum supported Rust version of
  1.94.
- Changed `einops!` and backend transformations to return
  `candle_core::Result`, preserving Candle error context instead of panicking.
- Made procedural-macro expansions work when the runtime dependency is renamed.

### Added

- Added the supported `einsum!` API: unary permutation and reduction, binary
  broadcast and GEMM-lowered contraction, right-aligned ellipsis handling,
  repeated-label diagonals and traces, and deterministic arbitrary-arity
  greedy contraction planning.
- Defined einsum dtype, device, gradient, scalar, zero-axis, evaluation-order,
  and error contracts, with rename-safe and keyword-alias expansion coverage.
- Added bounded deterministic parser/IR never-unwind properties and compile-fail
  diagnostics for malformed einsum equations.
- Added a reproducible Python einops parity lane for rearrange, repeat, and
  reduction semantics, with deterministic minimized replay and published-crate
  isolation.
- Added an isolated, locked performance-measurement harness with paired direct
  Candle references and a versioned machine-readable result contract.
- Added a reproducible cross-provider performance report with normalized source
  data and generated CPU, Accelerate, Metal, and CUDA figures.
- Added product reductions with `prod(...)`.
- Added compile-time diagnostics for malformed expressions and invalid axis
  relationships.
- Added independent value-based coverage for rearrange, reduce, repeat,
  composition, decomposition, squeeze, ellipsis, and owned inputs.

### Fixed

- Preserved standalone braced axes during composition.
- Applied reductions attached to inferred axes, including axes after an
  ellipsis.
- Rejected duplicate axes, unmatched ellipses, empty groups, and ambiguous
  axis sizes rather than generating invalid code or panicking in the macro.

### Migration

- Change `candle-core` and `candle-einops` dependency requirements to `0.11`
  and `0.2`, respectively.
- Add `?` or explicit error handling to every `einops!` call.
- Update custom `Backend` implementations so all transformations except
  `shape` return `candle_core::Result<Self::Output>`.
- `einsum!` is now supported in 0.2.0. Add `?` or explicitly handle its Candle
  result just as for `einops!`.
- Depend on the runtime crate rather than the implementation crate. The hidden
  macro/runtime ABI is coupled at exactly 0.2.0.

### Release plan

Both workspace crates are versioned at 0.2.0. After the full CI and package
gates pass, publish in dependency order:

1. Publish `candle-einops-macros` 0.2.0.
2. Wait until that version is available from the crates.io index.
3. Publish `candle-einops` 0.2.0, whose manifest requires the macro crate at
   that same version.

Before either publish, verify a clean checkout with:

```console
cargo +1.94.0 test --workspace --all-targets --all-features
cargo +stable test --workspace --all-targets --all-features
cargo +stable fmt --all -- --check
cargo +stable clippy --workspace --all-targets --all-features -- -D warnings \
  -A dead-code -A clippy::excessive-precision -A clippy::identity-op \
  -A clippy::map-flatten
RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --all-features --no-deps
RUSTDOCFLAGS="-D warnings" cargo +stable test --doc --workspace --all-features
python3 .github/scripts/validate_ci_policy.py
python3 .github/scripts/validate_artifact_policy.py
python3 .github/scripts/validate_einsum_release.py
python3 .github/scripts/validate_python_parity_policy.py
python3 .github/scripts/validate_performance_harness_policy.py
python3 .github/scripts/test_validate_performance_harness_policy.py
uv sync --frozen --project benchmarks/reporting
uv run --frozen --project benchmarks/reporting \
  python .github/scripts/test_generate_performance_report.py
uv run --frozen --project benchmarks/reporting \
  python .github/scripts/generate_performance_report.py --check
actionlint .github/workflows/ci.yml
cargo deny --all-features check
python3 .github/scripts/test_python_parity.py
python3 .github/scripts/test_published_artifacts.py
```

The artifact gate packages and unpacks both crates in dependency order, runs
their tests and doctests with warnings denied, and executes normal, renamed,
and keyword-alias downstream consumers without publishing to a registry.

Publishing, tagging, and creating a GitHub release remain deliberate manual
steps and are not performed by the release-readiness work.

## [0.1.2] - 2024-07-09

Last release before the Candle 0.11 modernization.
