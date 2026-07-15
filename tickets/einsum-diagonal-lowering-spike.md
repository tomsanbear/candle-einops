---
id: einsum-diagonal-lowering-spike
title: Select an efficient repeated-label diagonal strategy
status: in-progress
priority: p1
dependencies: [performance-harness-foundation]
related: [einsum-diagonal-fastpath]
scopes: []
shared_scopes: [tests, benchmarks, docs, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: ci-release
lease_expires_at: 1784143089
---
# Select an efficient repeated-label diagonal strategy

## Decision question

Choose between direct strided gather, reusable device indices, a portable custom
operation, or an upstream Candle primitive without copying a full permuted dense
input merely to return its diagonal.

## Spike work

Characterize `i i -> i`, `i i i -> i`, and the layout-hostile `i j i j -> i j`.
Benchmark only simple diagonal and interleaved diagonal at three scaling points;
the triple-repeat case is correctness/design evidence, not another redundant
timing family. Record input/copy/output elements, index preparation, latency,
and available profiler evidence. Prototype at least one gradient-capable alternative.

## Red-first evidence

Structural tests demonstrate current permute-plus-reshape materialization and
per-call index construction before candidate APIs exist.

## Acceptance

- Written decision covers gradients, zero extents, multiple repeated labels,
  device lifetime/caching, and CPU/CUDA/Metal feasibility.
- Refine the conditional implementation ticket with exact ownership and fallback.

## Result

GO for one original-layout flat gather only when the original operand is
contiguous and repeated-axis adjacency lowering would materialize a dense copy.
The current path remains the adjacent, non-contiguous, overflow, and unsupported
fallback. Do not introduce a cross-call/device-global cache. Full evidence and
the CPU measurements are recorded in
`benchmarks/diagonal-lowering-spike.md`.

## Non-goals

No production runtime change in the spike and no general gather optimization.
