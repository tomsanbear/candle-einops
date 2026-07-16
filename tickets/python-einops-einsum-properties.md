---
id: python-einops-einsum-properties
title: Add randomized Python parity for einsum
status: done
priority: p0
dependencies: [python-einops-parity-ci]
related: [einsum-contract-oracles]
scopes: [parity]
shared_scopes: [ticketing]
tags: [python-einops-parity]
---
# Add randomized Python parity for einsum

## Required outcome

Extend the locked oracle protocol and standalone Rust runner to compare
`einsum!` against the public `einops.einsum` API with proptest-generated
bounded shapes and finite values.

## Red-first work

Commit protocol and randomized semantic tests while einsum oracle/dispatch is
missing, observe the targeted failure, then implement the smallest bridge and
literal macro matrix needed to make them green.

## Acceptance

- Unary permutation/reduction, binary elementwise/outer/matmul/batched
  contraction, ellipsis broadcasting/reduction, repeated-label diagonal/trace,
  and three/four-operand paths are represented.
- Scalars, singleton dimensions, zero-sized non-contracting dimensions, and
  shared invalid shape/rank behavior are covered where both APIs define them.
- Shapes are exact and finite f32 results use explicit operation-scaled tolerances.
- Fixed bounded CI generation and minimized full-request replay are preserved.
- The supported wrapper and isolated CI job discover and run the new suite.

