---
id: einsum-zero-k-autograd
title: Preserve autograd through zero-length contractions
status: todo
priority: p0
dependencies: [performance-harness-foundation, einsum-device-dtype-gradient-matrix]
related: []
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2]
---
# Preserve autograd through zero-length contractions

## Required outcome

Fix the correctness gap inherited from Candle’s K=0 matmul shortcut: forward
zeros must retain valid autograd dependencies on every operand without
submitting a backend GEMM.

## Red-first contract

- Variable operands `(B,M,0)` and `(B,0,N)` produce exact `(B,M,N)` zeros.
- Backward returns present, correctly shaped zero gradients for both operands.
- Cover unbatched, singleton-broadcast, scalar-adjacent, and zero-output-axis variants.
- Validate incompatible contracted/batch dimensions before taking the shortcut.
- CPU and available accelerator smoke tests compare a graph-preserving direct construction.

## High-signal benchmark

Own one K=0 family with three output sizes. Report synchronized zero-fill cost
and structural confirmation that no GEMM occurs; do not compare against nonzero
GEMM because that answers a different question.

## Acceptance

- Forward contract is unchanged and gradients are present zeros for every operand.
- Work scales with output elements, not hypothetical contraction FLOPs.
- No production dtype/device support is reduced.

## Risks and non-goals

The graph-preserving construction must not add unsupported arithmetic or
surprising graph depth. Optimizing nonzero contractions or changing Candle
upstream is out of scope.

