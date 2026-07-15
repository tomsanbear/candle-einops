---
id: einsum-nary-layout-aware-planner
title: Implement the selected layout-aware n-ary planner
status: todo
priority: p1
dependencies: [einsum-nary-cost-model-spike]
related: [einsum-nary-cost-model-spike]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement the selected layout-aware n-ary planner

## Entry condition

The spike selected a bounded hybrid for calibrated CPU backends. Implement the
exact model only for arity three or four when current greedy estimates at least
100,000 FLOPs. Retain current greedy for arity above four, smaller work,
overflow/model errors, unsupported operand metadata, planner-budget overruns,
and CUDA/Metal until each backend has synchronized crossover data. The CPU
budget is 150 us on the frozen arity-four corpus.

## Red-first contract

Freeze pair-selection assertions from every spike counterexample, deterministic
ties, maximum test arity runtime, forward and gradient parity under changed
association, zero-K estimates, and broadcast/layout costs before implementation.
Use checked `u128` terms and stable original-operand member bitmasks for the
lexicographic tie-break. The arity-five/six exact implementation is oracle-only
and must not be reachable from production planning.

The score owns FLOPs, copied bytes, the sum of pair-output elements, peak-live
elements, and submissions with the CPU weights frozen in
`benchmarks/nary-cost-model-spike.md`. Carry sufficient intermediate shape and
layout metadata to estimate every candidate pair without allocating tensors.
Model broadcast materialization as full eager expansion plus one GEMM
submission; do not assume stride-zero batched GEMM support.

## Benchmark ownership

Reuse exactly the four whole-network spike scenarios. No new chain-length or
shape sweep. Compare path, intermediate/peak estimates, planner overhead, and
wall time against the current greedy planner.

## Acceptance

- Never worse than current estimates on the frozen corpus and materially improves
  the identified pathological cases.
- Carries canonical intermediate layout when safe, avoiding permutation followed
  by inverse canonicalization.
- Planner overhead remains negligible relative to its target contractions.
- Falls back deterministically to current greedy on every boundary above and
  preserves current zero-output, error, and dtype behavior.
- Changed associations preserve forward values and input gradients within the
  spike's mixed tolerance `0.002 * max(abs(reference), 1)`; bitwise
  floating-point equality is not required.

## Non-goals

No exact global optimization at arbitrary arity or unstable nondeterministic tuning.
