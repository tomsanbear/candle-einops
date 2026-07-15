---
id: einsum-broadcast-gemm-lowering
title: Implement selected broadcast-aware GEMM lowering
status: todo
priority: p0
dependencies: [einsum-broadcast-gemm-spike]
related: []
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement selected broadcast-aware GEMM lowering

## Entry condition

Proceed only after the spike selects a portable strategy and explicit crossover/fallback rules.

## Red-first contract

Add structural tests for left/right/both batch broadcasting and transposed
operands showing the selected materialization behavior. Cover values, gradients,
zero/singleton extents, mixed ellipsis capture ranks, dtype/device errors, and
fallback behavior before implementation.

## Benchmark ownership

Reuse the spike’s exact four cases and candidate metrics. Add no new shape grid;
the decision is whether the chosen strategy improves its frozen target cases
without materially regressing the fallback.

## Acceptance

- Avoids the measured full expanded copy where the strategy promises it.
- Preserves all Python parity, gradient, and accelerator smoke tests.
- Documents backend-specific crossover or fallback behavior without hidden transfers.

## Non-goals

No n-ary planner changes or broad Candle matmul rewrite.

