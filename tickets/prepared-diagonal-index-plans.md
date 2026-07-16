---
id: prepared-diagonal-index-plans
title: Add reusable prepared diagonal index plans
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

Per-call diagonal extraction remains 1.16x to 3.88x slower than the cached-index reference because the library rebuilds and uploads indices for every invocation.

## Work

- Design a caller-owned prepared plan keyed by equation, shape, index dtype, and device identity; do not add an unbounded global cache.
- Keep the one-shot macro API correct while exposing an explicit reusable path for repeated shapes.
- Measure preparation separately from steady-state extraction with the existing six diagonal scenarios.

## Acceptance

- Red-first tests cover plan reuse, wrong shape/device rejection, zero extents, overflow, values, and gradients.
- Prepared steady-state extraction is no slower than the cached reference by the protocol threshold on CPU, Metal, and CUDA.
- One-shot behavior and error compatibility remain unchanged.
