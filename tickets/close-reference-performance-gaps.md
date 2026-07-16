---
id: close-reference-performance-gaps
title: Close material performance gaps to direct Candle
status: todo
priority: p0
dependencies: [optimized-provider-performance-protocol, cuda-simple-einsum-dispatch-overhead, cuda-view-consumption-overhead, provider-aware-reduction-selection, collapsed-extrema-provider-regression, prepared-diagonal-index-plans, device-calibrated-nary-planner, cpu-zero-k-fixed-overhead, cpu-layout-consumption-overhead]
related: []
scopes: [ticketing]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
---
Complete the measurement-first remediation of every material scenario where candle-einops is slower than its direct Candle reference. Finish only after optimized CPU, Accelerate, Metal, and CUDA evidence shows parity or an explicitly documented non-promoted route for every confirmed gap.
