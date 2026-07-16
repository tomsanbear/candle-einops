---
id: benchmark-device-support
title: Support every Candle device and CPU provider honestly
status: todo
priority: p1
dependencies: [performance-harness-foundation, repeat-broadcast-view-lowering, homogeneous-reduction-fusion, einsum-binary-fastpaths, validate-metal-device-support, validate-cuda-device-support]
related: [advisory-performance-comparison-ci]
scopes: [benchmarks, tooling]
shared_scopes: [docs, ticketing]
tags: [performance-0.2, device-support]
---
Integrate fail-closed benchmark and accelerator-test support for Candle CPU, Metal, and CUDA devices plus baseline, Accelerate, and MKL CPU profiles. Complete only after schema, capability, diagnostic, capture, CI, Metal, and CUDA validation tickets are done.
