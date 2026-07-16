---
id: provider-aware-reduction-selection
title: Calibrate reduction fusion by provider and layout
status: todo
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [homogeneous-reduction-fusion]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, reductions]
---
## Evidence

Fused sum/mean was near parity on CPU, but directional results included Metal contiguous reductions up to 1.16x slower and CUDA cases around 1.08x to 1.13x slower.

## Work

- Re-measure exactly the existing four-scenario layout matrix by provider.
- Model public reduction count, required layout materialization, and fixed dispatch cost.
- Select fusion only for provider/layout combinations that clear the materiality threshold.

## Acceptance

- Red-first route tests cover each provider class without changing values, gradients, dtype errors, or mixed-operation ordering.
- No supported provider/layout cell is materially slower than its direct Candle reference.
- Eligible cases retain the minimum reduction count.
