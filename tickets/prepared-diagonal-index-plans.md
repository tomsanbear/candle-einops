---
id: prepared-diagonal-index-plans
title: Add reusable CPU and CUDA diagonal index plans
status: todo
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [einsum-diagonal-fastpath, spike-reusable-diagonal-index-plans]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, einsum]
---
## Evidence

Five optimized 25-sample processes cleared Metal. CUDA remained 29% to 63% / 2.8 to 6.2 microseconds behind cached-index reference in all six cases. CPU baseline and Accelerate retained one material interleaved n16 gap around 1.7 microseconds.

## Work

- Design a caller-owned prepared plan keyed by equation, shape, index dtype, and device identity; do not add an unbounded global cache.
- Keep the one-shot macro API correct while exposing an explicit reusable path for repeated shapes.
- Measure preparation separately from steady-state extraction with the existing six diagonal scenarios.

## Acceptance

- Red-first tests cover plan reuse, wrong shape/device rejection, zero extents, overflow, values, and gradients.
- Prepared steady-state extraction is no slower than cached reference by the protocol threshold on CPU and CUDA.
- One-shot behavior, Metal behavior, and error compatibility remain unchanged.
