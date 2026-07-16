---
id: cuda-simple-einsum-dispatch-overhead
title: Remove CUDA simple einsum dispatch overhead
status: todo
priority: p0
dependencies: [optimized-provider-performance-protocol]
related: [einsum-binary-fastpaths]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, cuda]
---
## Evidence

Directional RTX 4070 ratios against direct Candle were 1.34x for Hadamard, 1.23x for outer product, 1.31x for rank-2 GEMM, and 1.49x for batched GEMM. CPU and Metal were generally near parity.

## Work

- Reproduce in optimized independent-process measurements and use one exact-operation Nsight capture per distinct lowering.
- Separate macro/planner cost, tensor preparation, synchronization, and enqueue count without adding redundant size sweeps.
- Hoist or bypass work that direct canonical binary paths do not require.

## Acceptance

- Red-first tests freeze the canonical fast-path route and prevent extra allocations or public tensor operations.
- Every materially regressed CUDA throughput scenario reaches parity within the protocol threshold or gains a documented device-aware bypass that does.
- CPU and Metal remain within the same threshold.
