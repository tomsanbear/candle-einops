---
id: einsum-binary-gemm-slice
title: Implement binary GEMM-lowered einsum
status: todo
priority: p0
dependencies: [einsum-unary-explicit-slice]
related: []
scopes: [runtime, macros, tests, docs]
shared_scopes: [ticketing]
paths: []
tags: [einsum-implementation]
---
# Implement binary GEMM-lowered einsum

## Vertical outcome

Support two operands for dot, outer product, Hadamard/broadcast multiplication, matrix-vector, matrix multiplication, batched contraction, scalars, and zero dimensions.

## Lowering

Classify pair labels into batch/shared-output, left-free, contracted, and right-free groups. Pre-reduce safe private axes, permute to canonical order, checked-flatten to B/M/K and B/K/N, apply Candle matmul/broadcasting, reshape, and permute to explicit output.

## TDD

Turn the binary red oracle corpus green incrementally. Add runtime failures for rank, named-axis broadcast conflicts, dtype/device mismatch, and checked shape overflow, plus CPU gradient comparisons for dot/matmul/broadcast.

## Acceptance

- Values and gradients match independent/direct Candle oracles with dtype-appropriate tolerances.
- Operand expressions evaluate once and all errors retain einsum context.
- No naive full outer-product intermediate is used for eligible contractions.
