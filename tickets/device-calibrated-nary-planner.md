---
id: device-calibrated-nary-planner
title: Calibrate and cache CPU nary greedy execution
status: done
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [einsum-nary-layout-aware-planner]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, einsum]
---
## Resolution

The bounded exact planner improves the abstract FLOP and peak-memory score but its canonical execution route was slower on every frozen CPU network. Production therefore keeps the exact planner as a deterministic bounded analysis tool while calibrated execution uses streaming greedy.

A bounded 16-entry thread-local cache now stores only the stable greedy member sequence, keyed by operand axes, dimensions, layouts, strides, and output. Warm replay skips repeated pair search but still uses the same general binary lowering; no tensors, device allocations, or unbounded global state are cached.

## Acceptance evidence

- Red-first selection tests freeze the Calibration fallback across balanced, broadcast, threshold, overflow, dtype, backend, layout, and arity boundaries.
- Red-first execution tests prove first-run planning then cached replay with identical member sequence, forward values, all gradients, and deterministic ordering.
- Exact planner analysis, overflow, zero-K, broadcast materialization, and planner budget tests remain green.
- Five optimized 25-sample processes report all four networks as parity on CPU baseline and Accelerate, with medians now 0% to 2% faster and 0.17 to 0.33 microseconds faster than reference.
- Metal and CUDA continue down their existing greedy backend route.
