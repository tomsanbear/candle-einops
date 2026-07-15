---
id: einsum-nary-layout-aware-planner
title: Implement the selected layout-aware n-ary planner
status: todo
priority: p1
dependencies: [einsum-nary-cost-model-spike]
related: []
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement the selected layout-aware n-ary planner

## Entry condition

Implement only the model selected by the spike, within its planner-time budget.

## Red-first contract

Freeze pair-selection assertions from every spike counterexample, deterministic
ties, maximum test arity runtime, forward and gradient parity under changed
association, zero-K estimates, and broadcast/layout costs before implementation.

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

## Non-goals

No exact global optimization at arbitrary arity or unstable nondeterministic tuning.

