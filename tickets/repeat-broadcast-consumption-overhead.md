---
id: repeat-broadcast-consumption-overhead
title: Optimize consumed repeat broadcast
status: blocked
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: []
scopes: [runtime, benchmarks, tests]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784225093
---
The optimized five-process CPU baseline matrix reports repeat/broadcast/single-axis/consume at +13.77% and +479.88 us versus direct Candle. Add a focused red test that includes the downstream consumer, separate view construction from materialization/consumer cost, and implement a device-safe lowering that reaches parity without regressing zero-copy GPU view paths.
