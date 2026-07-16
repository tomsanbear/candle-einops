---
id: cuda-view-consumption-overhead
title: Avoid CUDA view consumption regressions
status: done
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [fused-permute-compose-layout, identity-reshape-elision]
scopes: [runtime, macros, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, cuda]
---
## Evidence

CUDA consumed permute/composition paths measured 1.42x to 1.51x slower than the direct Candle reference, while CPU was neutral and Metal was near parity. Sub-microsecond identity-view differences are controls, not independent regressions.

## Work

- Reproduce the two existing consumed layout fixtures under the optimized protocol.
- Capture whether deferred materialization adds an extra CUDA copy, command, or synchronization.
- Add a device/layout-aware selection boundary rather than a CUDA-wide disable.

## Acceptance

- Tests first freeze storage sharing, values, gradients, and exact fallback selection.
- Material CUDA consumption is no slower than reference by the protocol threshold.
- Construction-mode copy avoidance and CPU/Metal behavior remain intact.
