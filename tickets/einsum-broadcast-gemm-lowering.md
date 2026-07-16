---
id: einsum-broadcast-gemm-lowering
title: Implement selected broadcast-aware GEMM lowering
status: closed
priority: p0
dependencies: [einsum-broadcast-gemm-spike]
related: [einsum-broadcast-gemm-spike]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
closed_reason: wontdo
closed_note: Spike found no portable value-safe performance win; reopen with accelerator crossover evidence.
---
# Implement selected broadcast-aware GEMM lowering

## Entry condition

Not met. `einsum-broadcast-gemm-spike` records a no-go: the portable slice
prototype regresses all frozen broadcast cases, while stride-zero direct matmul
is value-incorrect on Candle 0.11 CPU.

## Disposition

Do not implement production lowering. Existing binary fast paths continue to
own direct `matmul` when neither operand needs batch expansion and Candle accepts
the layout. General broadcast keeps eager materialization, including zero
extents, unsupported layouts/dtypes, and backend failures.

Reopen only when synchronized CUDA/Metal evidence establishes an explicit
backend/shape/memory crossover, or an upstream value-tested Candle primitive
avoids concretization. Any reopened implementation must retain eager fallback
and add value/gradient tests ahead of runtime changes.

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
