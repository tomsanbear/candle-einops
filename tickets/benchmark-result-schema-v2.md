---
id: benchmark-result-schema-v2
title: Introduce benchmark result schema v2
status: todo
priority: p0
dependencies: [benchmark-device-selection-contract]
related: []
scopes: [benchmarks]
shared_scopes: [tooling]
paths: []
tags: [device-support]
---
Replace the record array with a versioned run document containing execution identity, records, explicit skips, and validated availability envelopes. Keep legacy v1 readable by comparison tooling but intentionally incomparable with v2.
