---
id: collapsed-extrema-provider-regression
title: Eliminate collapsed extrema provider regressions
status: todo
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: [fuse-collapsible-multi-axis-extrema-reductions]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, reductions]
---
## Evidence

The collapsed contiguous extrema candidate measured 1.40x to 1.90x slower than sequential direct Candle across CPU, Accelerate, Metal, and CUDA despite issuing one public reduction.

## Work

- Confirm the regression in the existing six-case extrema matrix using optimized runs.
- Determine whether collapse materialization, kernel shape, or provider reduction implementation dominates.
- Tighten selection to retain one-call collapse only where it is actually faster.

## Acceptance

- Tests first freeze contiguous leading/trailing eligibility, strided fallback, values, gradients, dtypes, and empty-axis errors.
- Every provider uses the faster route within the protocol threshold.
- Structural call-count claims remain honest and are not treated as performance wins by themselves.
