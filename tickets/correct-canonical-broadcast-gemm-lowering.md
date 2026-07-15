---
id: correct-canonical-broadcast-gemm-lowering
title: Correct canonical broadcast GEMM lowering
status: in-progress
priority: p0
dependencies: []
related: [einsum-broadcast-gemm-spike]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
paths: []
tags: [kernel-enqueue-hardening]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784156512
---
# Correct canonical broadcast GEMM lowering

## Goal

Prevent canonical batched einsum from passing stride-zero batch views into
Candle 0.11 `matmul`, whose CPU implementation is value-incorrect for the
repository's frozen large broadcast case.

## Work

- Add a red production regression using the frozen 32x32 singleton-batch case.
- Keep exact-shape canonical operands on one direct `matmul` submission.
- Route operands requiring batch expansion through eager materialization and
  one batched GEMM; preserve output order, gradients, errors, and zero handling.
- Record structural coverage for exact versus expanded dispatch.

## Acceptance

- The regression fails before the fix and exactly matches the eager oracle after it.
- Exact canonical GEMM still dispatches directly without materialization.
- Expanded left, right, and cross-batch cases match values and gradients.
- Focused and workspace tests pass.
