---
id: cpu-product-association-overhead
title: Optimize CPU product association overhead
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
lease_expires_at: 1784224948
---
The optimized five-process CPU baseline matrix reports product/sequential-vs-balanced/k-64 at +13.95% and +2.46 us versus the balanced direct-Candle reference. Add a focused red benchmark/contract, identify whether library product lowering is choosing a suboptimal reduction tree or paying avoidable dispatch overhead, implement the smallest semantics-preserving correction, and require repeated-process parity on baseline and Accelerate while checking Metal and CUDA for regressions.
