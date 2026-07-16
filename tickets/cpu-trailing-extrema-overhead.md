---
id: cpu-trailing-extrema-overhead
title: Optimize CPU trailing-axis extrema
status: done
priority: p0
dependencies: [cpu-leading-extrema-overhead]
related: []
scopes: [runtime, benchmarks, tests]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
---
The post-fix complete five-process CPU baseline matrix reports spike/extrema/contiguous-trailing/min at +19.92% and +5.79 us; earlier evidence was statistically inconclusive. Extend the CPU provider decision to trailing homogeneous extrema, starting with a red backend route test, and require the complete min/max extrema family to reach parity without changing Metal or CUDA fusion.
