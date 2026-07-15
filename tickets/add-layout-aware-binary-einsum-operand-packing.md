---
id: add-layout-aware-binary-einsum-operand-packing
title: Add layout-aware binary einsum operand packing
status: todo
priority: p0
dependencies: [correct-canonical-broadcast-gemm-lowering]
related: [fused-permute-compose-layout]
scopes: [runtime, macros]
shared_scopes: [tests, benchmarks, ticketing]
paths: []
tags: [kernel-enqueue-hardening]
---
# Add layout-aware binary einsum operand packing

## Goal

Avoid copy submissions when general binary einsum can present canonical B/M/K
and B/K/N operands to Candle through safe layout-preserving views.

## Work

- First add structural tests that expose current storage-changing operand packs.
- Broaden direct `matmul` recognition to arbitrary exact batch ranks where
  operand axes already match Candle's contract.
- Reuse bounded whole-group layout recovery for multi-axis free/contracted
  groups before falling back byte-for-byte to eager `broadcast_as + reshape`.
- Never use stride-zero direct batched matmul; preserve the correctness ticket's guard.
- Benchmark construction/materialization and synchronized consumption against
  the existing eager path using a small, mechanism-distinct scenario set.

## Acceptance

- Eligible exact-batch and recoverable-layout cases retain input storage until GEMM.
- Ineligible/broadcast cases take the existing eager fallback.
- Values, gradients, zero/singleton extents, dtypes, and backend errors remain stable.
- Evidence demonstrates a copy/submission reduction without a consumption regression.
