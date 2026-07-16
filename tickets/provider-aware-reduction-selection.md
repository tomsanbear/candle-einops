---
id: provider-aware-reduction-selection
title: Preserve physical fused-reduction axis order
status: done
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [homogeneous-reduction-fusion]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, reductions]
---
## Resolution

The fused planner emitted dimensions in descending logical order. Candle Metal uses the supplied order when constructing its physical reduction layout, so contiguous trailing dimensions arrived as 3,2 and missed the contiguous fast-reduce kernel reached by direct Candle with 2,3.

Sum and mean runs now preserve ascending physical stride order while extrema keeps descending execution order where dimension removal requires it.

## Acceptance evidence

- Red-first planner assertions freeze ascending fused sum/mean axes and mixed-run ordering.
- Values, gradients, ellipsis, boundary shapes, dtype behavior, mixed operations, and extrema fallback tests pass.
- Five optimized 25-sample processes report all four reduction cells as parity on CPU baseline, Accelerate, Metal, and CUDA.
- Metal contiguous trailing moved from roughly +16% / +30 microseconds to 0% / below 1 microsecond.
