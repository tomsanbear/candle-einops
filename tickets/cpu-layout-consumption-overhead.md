---
id: cpu-layout-consumption-overhead
title: Materialize CPU rank-two compositions eagerly
status: done
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: [fused-permute-compose-layout, identity-reshape-elision]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, cpu]
---
## Resolution

Recovered rank-two transpose views are valuable while lazy, but Candle CPU materializes those views more slowly than its direct permute-and-reshape path. Because the tensor API cannot observe a future contiguous call, non-empty rank-two CPU composition now materializes eagerly. Higher-rank CPU compositions and all GPU compositions retain storage-sharing recovery.

## Acceptance evidence

- Red-first layout tests require eager CPU rank-two layout while preserving higher-rank views, values, gradients, offsets, zero extents, exhaustive bounded permutations, and deterministic errors.
- Five optimized 25-sample processes show zero reference gaps for both layout families on CPU baseline and Accelerate.
- Metal and CUDA repeated-process validation show zero reference gaps; their view-only construction paths remain skipped from GPU timing and their consumed paths retain the recovered route.
- CPU c-ab consume moved from about +42% / +52 to 54 microseconds to parity; post-reduction consume moved from +24% to 26% / +9 to 17 microseconds to parity.
