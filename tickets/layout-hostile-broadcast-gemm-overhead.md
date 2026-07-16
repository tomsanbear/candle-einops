---
id: layout-hostile-broadcast-gemm-overhead
title: Optimize layout-hostile broadcast GEMM
status: done
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: []
scopes: [runtime, benchmarks, tests]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
---
The optimized five-process CPU baseline matrix reports spike/broadcast-gemm/layout-hostile at +144.12% and +18.38 us versus direct Candle. Treat the existing spike as design evidence: add a focused red test, inspect packing/transposition and temporary allocation costs, then promote only a lowering that is correct across dtype, shape, gradient, and supported devices and reaches repeated-process parity.
