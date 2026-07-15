---
id: performance-harness-foundation
title: Establish a high-signal performance measurement contract
status: in-progress
priority: p0
dependencies: []
related: []
scopes: [benchmarks, tooling, docs]
shared_scopes: [ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: ci-release
lease_expires_at: 1784142275
---
# Establish a high-signal performance measurement contract

## Required outcome

Create a standalone, unpublished `benchmarks/` crate with its own workspace and
committed lockfile. It path-depends on this repository, remains isolated from
normal workspace features and published artifacts, and exposes one supported
wrapper command for benchmark compilation, smoke checks, filtered execution,
and versioned JSON output.

## Measurement design

The harness owns methodology, not a broad scenario matrix. A scenario declares
an immutable id, deterministic setup, library operation, direct-Candle
reference, correctness check outside timing, synchronization boundary, and
meaningful work units. Timed samples retain/black-box outputs and synchronize
devices before and after measurement. JSON records absolute median and
confidence information, library/direct ratio, workload, git SHA, Rust/Candle
versions, OS/architecture, device/driver identity, and backend.

Criterion output is secondary. Downstream automation consumes only the stable
JSON schema. CPU is mandatory; optional Metal/CUDA selection is mutually
exclusive. The wrapper always uses `--locked` and a shared ignored target
directory instead of creating another multi-gigabyte nested target.

## Red-first work

First commit contract tests and a compile-only target referring to the absent
scenario/measurement API. Record the expected compile failure, then implement
the minimum plumbing-only smoke fixture. Do not invent a performance regression
to manufacture a red test.

## Acceptance

- Root package/workspace exclusions and artifact policy prove benchmark sources,
  reports, manifests, locks, and targets never enter either published crate.
- Tests prove setup/correctness are outside timing and synchronization surrounds samples.
- Malformed or incomplete metadata fails deterministically.
- Documentation gives exact locked commands, filter semantics, and profiler hooks.
- `cargo bench --no-run` and a short correctness smoke pass on the MSRV.

## Non-goals

- No committed universal timing baseline or hosted-runner timing gate.
- No dtype/shape/backend Cartesian matrix, Python performance comparison, gradients, or error cases.
- No claim of exact kernel, enqueue, or cross-device allocation counts.

