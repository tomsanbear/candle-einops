---
id: cpu-zero-k-fixed-overhead
title: Reduce CPU zero-K contraction overhead
status: todo
priority: p2
dependencies: [optimized-provider-performance-protocol]
related: [einsum-zero-k-autograd]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, cpu]
---
## Evidence

All three zero-K output sizes measured roughly 1.35x to 1.45x slower than reference on baseline and Accelerate CPU, an absolute gap around 1.3 microseconds for the smallest case. Metal and CUDA were at parity.

## Work

- Confirm the fixed CPU overhead with optimized independent processes.
- Attribute validation, graph-preserving anchor construction, and allocation costs without weakening autograd.
- Remove redundant work or use a CPU-specific equivalent graph construction only if gradients remain exact.

## Acceptance

- Red-first tests preserve both autograd edges, dtype/device validation order, non-contiguous inputs, and empty batch/free axes.
- CPU is within the materiality threshold for all three existing sizes.
- Metal and CUDA do not regress.
