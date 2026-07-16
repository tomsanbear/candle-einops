---
id: device-calibrated-nary-planner
title: Calibrate n-ary planning for CPU providers
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

Five optimized 25-sample processes cleared Metal and CUDA. Baseline CPU retained 13% to 101% / 4.7 to 13.8 microsecond gaps across the frozen n-ary matrix. Accelerate retained 48% to 116% / 6.2 to 13.3 microsecond gaps except broadcast-heavy, which cleared the threshold.

## Work

- Freeze the four whole-network fixtures by CPU implementation.
- Calibrate cost weights and crossover eligibility for baseline and Accelerate without runtime autotuning.
- Preserve deterministic bounded planning and the existing greedy fallback; do not alter Metal or CUDA selection without evidence.

## Acceptance

- Red-first path assertions cover CPU workload crossovers and deterministic ties.
- Selected CPU execution is never materially slower than greedy/direct reference on the frozen matrix.
- Forward values, gradients, overflow fallback, planner budget, Metal, and CUDA remain unchanged.
