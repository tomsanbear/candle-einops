---
id: device-calibrated-nary-planner
title: Calibrate n-ary planning by device
status: todo
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [einsum-nary-layout-aware-planner]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, einsum]
---
## Evidence

The selected n-ary strategy is workload/provider sensitive: layout-hostile cases were 1.48x slower on baseline CPU, 2.16x on Accelerate, and 1.54x on Metal, while balanced-tree selection also regressed 1.54x on Accelerate. CUDA often preferred different paths.

## Work

- Reproduce only the existing four whole-network fixtures under the optimized protocol.
- Calibrate cost weights and crossover eligibility by provider class without runtime autotuning.
- Preserve deterministic bounded planning and the existing greedy fallback.

## Acceptance

- Red-first path assertions cover every provider/workload crossover and deterministic ties.
- Selected execution is never materially slower than greedy/direct reference on the frozen matrix.
- Forward values, gradients, overflow fallback, and planner budget remain unchanged.
