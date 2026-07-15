---
id: einsum-diagonal-slice
title: Add repeated-label diagonal einsum
status: todo
priority: p1
dependencies: [einsum-binary-gemm-slice]
related: []
scopes: [runtime, macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [einsum-implementation]
---
# Add repeated-label diagonal einsum

## Vertical outcome

Support repeated labels within an operand for diagonal extraction and trace, including labels repeated more than twice.

## TDD

Add red forward/gradient cases for matrix diagonal, trace, batched diagonals, higher multiplicity, unequal repeated dimensions, scalars, and zero-sized axes.

## Green

Validate equal repeated extents, permute occurrences adjacent, compute checked diagonal indices on the input device, select via differentiable Candle operations, and collapse duplicate axes before normal contraction.

## Acceptance

- Forward values and gradients match host/direct-Candle oracles.
- Repeated dimensions are never broadcast against themselves.
- Index/stride arithmetic is checked and errors never unwind.
