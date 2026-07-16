---
id: validate-cuda-device-support
title: Validate CUDA support on balthasar
status: done
priority: p1
dependencies: [benchmark-device-diagnostics, benchmark-gpu-capture-command, benchmark-device-profile-ci]
related: []
scopes: [benchmarks]
shared_scopes: [docs, ticketing]
paths: []
tags: [device-support]
---
Stream the committed branch to balthasar, run fail-closed RTX 4070 smoke and supported scenarios, validate CUDA identity and diagnostics, and produce Nsight Systems evidence.
