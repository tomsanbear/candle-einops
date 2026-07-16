---
id: benchmark-result-schema-v2
title: Introduce benchmark result schema v2
status: in-progress
priority: p0
dependencies: [benchmark-device-selection-contract]
related: []
scopes: [benchmarks]
shared_scopes: [tooling]
paths: []
tags: [device-support]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784217081
---
Replace the record array with a versioned run document containing execution identity, records, explicit skips, and validated availability envelopes. Keep legacy v1 readable by comparison tooling but intentionally incomparable with v2.
