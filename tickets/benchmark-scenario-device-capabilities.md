---
id: benchmark-scenario-device-capabilities
title: Declare benchmark scenario device capabilities
status: in-progress
priority: p1
dependencies: [benchmark-result-schema-v2]
related: []
scopes: [benchmarks]
shared_scopes: []
paths: []
tags: [device-support]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784217286
---
Add typed CPU, Metal, and CUDA support metadata. Report unsupported selected scenarios with reasons, exclude view-only accelerator timings, and fail when no selected scenario is runnable.
