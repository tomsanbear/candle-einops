---
id: spike-multi-axis-extrema-reduction-lowering
title: Spike multi-axis extrema reduction lowering
status: todo
priority: p2
dependencies: [performance-harness-foundation]
related: [homogeneous-reduction-fusion]
scopes: [benchmarks]
shared_scopes: [runtime, tests, ticketing]
paths: []
tags: [kernel-enqueue-hardening, spike]
---
# Spike multi-axis extrema reduction lowering

## Goal

Decide whether safely collapsing compatible axes can reduce N sequential
min/max calls to one public Candle reduction without paying a worse copy cost.

## Work

- Freeze contiguous trailing, contiguous non-trailing, and strided cases for
  both min and max; do not duplicate sum/mean measurements.
- Compare sequential production lowering with permute/reshape/single-reduction
  candidates, recording public operation counts and materialized elements.
- Cover values, gradients where Candle supports them, empty-axis errors, and dtypes.

## Acceptance

- Record a GO/NO-GO boundary with synchronized timings and structural evidence.
- A GO creates/refines an implementation slice; a NO-GO leaves production unchanged.
- Fused mean and product are explicitly out of scope because they need upstream
  Candle support or backend-specific kernels.
