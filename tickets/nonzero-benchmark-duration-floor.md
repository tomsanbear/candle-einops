---
id: nonzero-benchmark-duration-floor
title: Represent sub-clock benchmark durations
status: in-progress
priority: p2
dependencies: [optimized-provider-performance-protocol]
related: []
scopes: [benchmarks, tooling]
shared_scopes: []
paths: []
tags: [performance-gap, hardening]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784224613
---
## Goal

Make optimized all-scenario reports represent operations below the host clock resolution without producing schema-invalid zero medians.

## Work

- Add a tested elapsed-time helper that clamps equal or reversed clock observations to the explicit 1 ns resolution floor.
- Use it only for host-clock operation samples; do not change device timing or materiality thresholds.
- Re-run the full optimized CPU gap command that exposed the failure.

## Acceptance

- Red-first tests require positive elapsed samples for equal and reversed clock observations while preserving exact positive durations.
- Full optimized gap collection completes and sub-microsecond controls remain parity.
- Schema-v2 validation and comparator contracts remain unchanged.
