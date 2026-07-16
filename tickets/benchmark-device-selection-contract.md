---
id: benchmark-device-selection-contract
title: Make benchmark device selection fail closed
status: done
priority: p0
dependencies: []
related: []
scopes: [benchmarks]
shared_scopes: [tooling]
paths: []
tags: [device-support]
---
Add red-first contracts and real device/profile construction for CPU baseline, Accelerate, MKL, Metal, and CUDA. Pass backend, CPU implementation, and device index through the wrapper; reject feature mismatches and unavailable devices without output.
