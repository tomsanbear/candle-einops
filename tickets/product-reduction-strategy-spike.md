---
id: product-reduction-strategy-spike
title: Select a portable product reduction strategy
status: todo
priority: p1
dependencies: [performance-harness-foundation]
related: [homogeneous-reduction-fusion, native-product-reduction]
scopes: []
shared_scopes: [tests, benchmarks, docs, ticketing]
tags: [performance-0.2]
---
# Select a portable product reduction strategy

## Decision question

Candle 0.11 has no public product `ReduceOp`; the current implementation launches
K-1 narrow/squeeze/multiply steps. Determine whether to pursue an upstream
Candle primitive, a portable local custom operation, or retain the fallback.

## Spike work

- Freeze the existing curve for `[256, K]` at K=8, 64, and 512, plus one
  two-axis case. Three scale points reveal complexity without a noisy sweep.
- Profile allocations/launches externally where available and record synchronized latency.
- Inventory CPU/CUDA/Metal dtype, empty identity, autograd, and packaging requirements.
- Prototype at least one plausible adapter in spike/test code; do not commit a
  CPU-only production fast path as if it were portable.

## Red-first evidence

Register the candidate adapter contract before it exists and record the compile
failure. Existing correctness parity is not a substitute for the missing strategy.

## Acceptance

- A written go/no-go decision includes measured curves, maintenance cost,
  supported dtype/device/gradient matrix, and upstream feasibility.
- The follow-up ticket is refined to the selected API or closed/superseded if no
  portable win exists.

## Non-goals

No `exp(sum(log))`, dependency fork without approval, or universal claim from a
CPU-only prototype.

