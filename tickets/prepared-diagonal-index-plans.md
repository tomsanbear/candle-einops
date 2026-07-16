---
id: prepared-diagonal-index-plans
title: Add caller-owned prepared diagonal plans
status: done
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [einsum-diagonal-fastpath, spike-reusable-diagonal-index-plans]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, einsum]
---
## Resolution

Added public PreparedDiagonalPlan for caller-owned, device-bound diagonal indices. Numeric axis ids encode the input-side equation, repeated ids select diagonals, and first-occurrence order defines the extraction output. Convenience constructors cover repeated and interleaved forms.

Plans store one u32 index tensor, require the exact prepared shape, contiguous input, and same device, and expose no global cache. One-shot einsum behavior and non-contiguous fallback remain unchanged.

## Acceptance evidence

- Red-first tests cover reuse across inputs, wrong shape, non-contiguous input, unequal repeated extents, missing repeats, u32/usize overflow, zero extents, exact values, and gradients.
- The benchmark now invokes the public plan as the library path with preparation outside timing.
- Existing benchmark tests prove index storage reuse and shape/device binding.
- Five optimized 25-sample processes report all six scenarios as parity on CPU baseline, Accelerate, Metal, and CUDA.
- README documents prepared versus one-shot usage and ownership boundaries.
