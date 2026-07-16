---
id: cpu-leading-extrema-overhead
title: Optimize CPU leading-axis extrema
status: todo
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: []
scopes: [runtime, benchmarks, tests]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
---
The corrected optimized five-process CPU baseline matrix reports spike/extrema/contiguous-leading/min and max at roughly +58% and +19 us versus sequential direct Candle. Add focused red tests for both operations, compare axis ordering/layout materialization routes, implement a provider-aware or generally superior lowering, and require parity on CPU baseline and Accelerate with regression checks on Metal and CUDA.
