---
id: fuse-collapsible-multi-axis-extrema-reductions
title: Fuse collapsible multi-axis extrema reductions
status: done
priority: p1
dependencies: [spike-multi-axis-extrema-reduction-lowering]
related: [homogeneous-reduction-fusion]
scopes: [runtime]
shared_scopes: [tests, ticketing]
paths: []
tags: [kernel-enqueue-hardening]
---
## Goal
Lower adjacent homogeneous min/max runs through one public Candle reduction only when their axes collapse as a storage-sharing view.

## Acceptance
- Tests first cover contiguous leading/trailing runs, strided fallback, values, gradients, dtypes, and empty-axis errors.
- Eligible runs issue one public reduction; ineligible layouts retain sequential lowering without a new copy.
- Full workspace tests and lint pass.
