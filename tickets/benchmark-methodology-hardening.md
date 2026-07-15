---
id: benchmark-methodology-hardening
title: Harden benchmark inputs and timing methodology
status: in-progress
priority: p0
dependencies: [performance-harness-foundation]
related: [einsum-binary-fastpaths, einsum-broadcast-gemm-spike, homogeneous-reduction-fusion, product-reduction-strategy-spike, einsum-diagonal-lowering-spike, repeat-broadcast-view-lowering, einsum-nary-cost-model-spike]
scopes: [benchmarks]
shared_scopes: [tests, ticketing]
paths: []
tags: [performance-0.2]
claimed_from: todo
assignee: python-oracle-design
lease_expires_at: 1784146416
---
## Required outcome

Make performance correctness inputs and timing methodology discriminating without adding scenario families.

## Red-first contract

- A benchmark contract test rejects all-constant binary and broadcast-GEMM operands where axis or operand swaps would otherwise pass.
- Binary GEMM dimensions include non-square M, K, and N values and deterministic nonuniform operands.
- A timing contract proves Criterion does not nest the JSON harness clock/synchronization wrapper.
- Paired JSON sampling alternates library/reference order deterministically to avoid a fixed first-run bias.

## Acceptance

- Direct Candle references still use identical input tensors and untimed correctness remains outside samples.
- Criterion measures the operation directly with its own timing model; device synchronization remains correct for asynchronous backends.
- JSON results remain reproducible and record the sampling order as an additive v1 field with a legacy default; the environment fingerprint is unchanged.
- Existing product, diagonal, binary, reduction, repeat, broadcast, n-ary, and future scenario registrations remain intact.
- The binary same-ID workload discontinuity from non-square inputs is documented so old and new medians are not compared as identical workloads.
- No new benchmark scenarios or hosted-CI timing threshold.

## Evidence

Independent review of the binary fast-path benchmarks found square all-ones operands could conceal transposes or operand swaps, and Criterion currently includes the internal clock wrapper in each iteration.
