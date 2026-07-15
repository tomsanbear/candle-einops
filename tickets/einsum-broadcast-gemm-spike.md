---
id: einsum-broadcast-gemm-spike
title: Select a broadcast-aware GEMM materialization strategy
status: in-progress
priority: p0
dependencies: [performance-harness-foundation, einsum-binary-fastpaths]
related: [einsum-broadcast-gemm-lowering]
scopes: []
shared_scopes: [tests, benchmarks, docs, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: ci-release
lease_expires_at: 1784144284
---
# Select a broadcast-aware GEMM materialization strategy

## Decision question

Current stride-zero broadcasts are reshaped and therefore copied before GEMM;
Candle’s own broadcast matmul also concretizes expanded matrices. Choose a
portable strategy that balances expanded-copy bytes, number of submissions,
backend capability, autograd, and small/large batch behavior.

## Spike work

Prototype eager expanded GEMM, direct no-expansion cases, and at least one
batch/slice or equivalent strategy using public Candle APIs. Use exactly four
representative cases: left broadcast, right broadcast, both broadcast, and a
transposed/layout-hostile operand. Record copy bytes, peak temporary elements,
synchronized latency, and diagnostic submissions where observable.

## Red-first evidence

Commit strategy-selection and structural materialization contracts against the
absent candidates. Preserve a case demonstrating the current expanded copy.

## Acceptance

- Written backend-aware decision and crossover rationale.
- Selected design preserves dtype/device/gradient semantics and covers zero/singleton dimensions.
- Refine the conditional implementation ticket with explicit fast/fallback rules.

## Result

NO-GO for a new portable broadcast lowering. CPU slicing removed expanded
operand copies but was slower in all three broadcast cases, and raw stride-zero
batch `matmul` produced incorrect values. Preserve direct `matmul` only for
no-expansion layouts and eager materialization for broadcast, zero, unsupported,
and backend-specific fallback cases. Full evidence and reopening criteria are in
`benchmarks/broadcast-gemm-spike.md`.

## Non-goals

No Candle fork, universal strategy selected from one CPU, or claim of exact GPU enqueue counts.
