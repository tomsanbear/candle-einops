---
id: einsum-binary-fastpaths
title: Add non-contraction and canonical GEMM fast paths
status: in-progress
priority: p0
dependencies: [performance-harness-foundation]
related: [einsum-broadcast-gemm-spike]
scopes: [runtime, macros]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: behavior-tests
lease_expires_at: 1784143072
---
# Add non-contraction and canonical GEMM fast paths

## Required outcome

Classify known binary cases before the general flattened GEMM path:

- no contracted labels: scalar, elementwise/Hadamard, and outer products use
  broadcast multiplication;
- canonical rank-2 and rank-3 contractions call direct Candle matmul;
- identity broadcasts, reshapes, and output permutations are skipped;
- the current general fallback remains for multi-axis/layout-changing contractions.

## Red-first contract

- CPU BF16 and U8/U32/I64 parity with Candle multiplication for elementwise,
  outer, and scalar-vector equations currently rejected by matmul.
- F16/F32/F64 value and gradient parity for all fast paths.
- Structural dispatch tests prove non-contractions avoid matmul and canonical
  GEMMs avoid the five wrapper view/graph operations.
- Cover singleton dimensions, non-contiguous inputs, empty free axes, and rank-2/3 fallback boundaries.

## High-signal benchmark

Own four mechanisms only: Hadamard, outer, rank-2 GEMM, and rank-3 batched GEMM.
Measure one overhead-sensitive and one throughput size per mechanism against
the exact direct Candle operation. General broadcast expansion is deliberately
left to its spike.

## Acceptance

- Demonstrate meaningful improvement on fast paths and no material regression on fallback GEMM.
- New dtype acceptance is documented and parity-tested.
- Operand evaluation order, errors, and gradients remain stable.

## Non-goals

No n-ary ordering change, general broadcast strategy, or contraction dtype expansion.

