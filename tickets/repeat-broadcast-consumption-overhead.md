---
id: repeat-broadcast-consumption-overhead
title: Optimize consumed repeat broadcast
status: closed
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: []
scopes: [runtime, benchmarks, tests]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
closed_reason: wontdo
closed_note: Preserve the zero-copy repeat contract; document eager Candle repeat for known immediate materialization.
---
The optimized five-process CPU baseline matrix reports repeat/broadcast/single-axis/consume at +13.77% and +479.88 us versus direct Candle. Add a focused red test that includes the downstream consumer, separate view construction from materialization/consumer cost, and implement a device-safe lowering that reaches parity without regressing zero-copy GPU view paths.
