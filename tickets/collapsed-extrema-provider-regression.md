---
id: collapsed-extrema-provider-regression
title: Eliminate Metal and CUDA collapsed extrema regressions
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

Five optimized 25-sample processes cleared CPU baseline and Accelerate. Metal contiguous extrema remained 37% to 53% / 66 to 88 microseconds behind reference. CUDA retained gaps across all six cases: roughly 10% to 142% / 1.7 to 15.8 microseconds.

## Work

- Freeze the existing six-case extrema matrix by provider and route.
- Determine whether collapsed extent, required materialization, or provider reduction implementation dominates.
- Use sequential reduction on provider/layout cells where collapse does not clear the materiality threshold.

## Acceptance

- Red-first tests freeze contiguous leading/trailing eligibility, strided fallback, values, gradients, dtypes, and empty-axis errors.
- Metal and CUDA use the faster route within the protocol threshold.
- CPU keeps its optimized collapsed route and structural call-count claims remain honest.
