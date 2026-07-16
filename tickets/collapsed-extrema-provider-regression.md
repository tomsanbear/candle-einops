---
id: collapsed-extrema-provider-regression
title: Validate collapsed extrema provider performance
status: done
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: [fuse-collapsible-multi-axis-extrema-reductions]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, reductions]
---
## Resolution

The benchmark had inverted semantics: it timed the old sequential route as library and the implemented collapsed route as reference. A red-first contract now requires the selected einops route to be library and sequential direct Candle to be reference.

Five corrected optimized 25-sample processes show no reference gap. The selected route is 28% to 42% faster on Metal contiguous cases and 55% faster on CUDA contiguous-trailing cases; all remaining Metal/CUDA cells are parity. CPU was already within threshold.

## Acceptance evidence

- The benchmark contract names selected versus sequential routes explicitly.
- Values, gradients, dtypes, empty-axis errors, contiguous collapse, and strided fallback tests pass.
- No provider is materially slower than its sequential direct Candle reference.
