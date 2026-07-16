---
id: cpu-layout-consumption-overhead
title: Avoid CPU consumed-layout regressions
status: todo
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: [fused-permute-compose-layout, identity-reshape-elision]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, cpu]
---
## Evidence

Five optimized 25-sample processes found two material CPU gaps while Metal and CUDA cleared the threshold. Baseline and Accelerate c-ab consumed composition remained about 42% to 43% / 52 to 54 microseconds behind direct Candle. Post-reduction consumed composition remained 24% to 26% / 9 to 17 microseconds behind.

## Work

- Add red-first route tests for the two affected consumed layouts and their construction controls.
- Attribute whether the CPU gap comes from deferred materialization, an extra copy, or composition ordering.
- Choose a CPU-specific lowering boundary only for affected layouts while preserving GPU copy avoidance.

## Acceptance

- Values, gradients, storage-sharing construction behavior, and non-contiguous cases remain correct.
- Both CPU baseline and Accelerate consumed cases are within the protocol threshold.
- Metal and CUDA remain within threshold and construction-mode copy avoidance remains intact.
