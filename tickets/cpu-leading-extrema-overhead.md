---
id: cpu-leading-extrema-overhead
title: Optimize CPU leading-axis extrema
status: in-progress
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: []
scopes: [runtime, benchmarks, tests]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784224815
---
The corrected optimized five-process CPU baseline matrix reports spike/extrema/contiguous-leading/min and max at roughly +58% and +19 us versus sequential direct Candle. Add focused red tests for both operations, compare axis ordering/layout materialization routes, implement a provider-aware or generally superior lowering, and require parity on CPU baseline and Accelerate with regression checks on Metal and CUDA.
