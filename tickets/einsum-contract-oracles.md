---
id: einsum-contract-oracles
title: Freeze einsum semantics and independent oracles
status: done
priority: p0
dependencies: [edge-shape-dtype-properties]
related: []
scopes: [tests, docs]
shared_scopes: [ticketing]
paths: [docs/einsum-contract.md]
tags: [einsum-implementation]
---
# Freeze einsum semantics and independent oracles

## Required outcome

Define the public equation contract and an implementation-independent reference before production code exists.

## Contract

- Syntax is whitespace-delimited named axes with a mandatory explicit arrow: `"batch i k, batch k j -> batch i j"`.
- Operand expressions evaluate exactly once, left to right; calls return `candle_core::Result<Tensor>`.
- Omitted labels reduce, retained shared labels broadcast when equal or one, output labels are unique and originate in an input.
- Scalars and zero-sized axes are valid. Repeated input labels and `..` are reserved for their dedicated slices.

## Red-first work

Add public-facing tests that cannot compile because `einsum!` does not yet exist, plus a small host `Vec<f64>` interpreter and checked-in expected cases for unary transpose/reduction, dot, outer, Hadamard broadcast, matvec, matmul, and batched contraction.

## Acceptance

- Oracles do not call Candle contraction/reshape/matmul helpers.
- Cases include invalid ranks/shapes and operand-evaluated-once behavior.
- The red contract is stable enough for subsequent slices to turn green without changing expected semantics.
