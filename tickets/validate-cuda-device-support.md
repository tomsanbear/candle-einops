---
id: validate-cuda-device-support
title: Validate CUDA support on balthasar
status: in-progress
priority: p1
dependencies: [benchmark-device-diagnostics, benchmark-gpu-capture-command, benchmark-device-profile-ci]
related: []
scopes: [benchmarks]
shared_scopes: [docs, ticketing]
paths: []
tags: [device-support]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784220117
---
Stream the committed branch to balthasar, run fail-closed RTX 4070 smoke and supported scenarios, validate CUDA identity and diagnostics, and produce Nsight Systems evidence.
