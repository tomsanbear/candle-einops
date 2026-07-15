---
id: product-reduction-strategy-spike
title: Select a portable product reduction strategy
status: in-progress
priority: p1
dependencies: [performance-harness-foundation]
related: [homogeneous-reduction-fusion, native-product-reduction]
scopes: []
shared_scopes: [tests, benchmarks, docs, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: python-oracle-design
lease_expires_at: 1784143073
---
# Select a portable product reduction strategy

## Decision question

Candle 0.11 has no public product `ReduceOp`; the current implementation launches
K-1 narrow/squeeze/multiply steps. Determine whether to pursue an upstream
Candle primitive, a portable local custom operation, or retain the fallback.

## Spike work

- Freeze the existing curve for `[256, K]` at K=8, 64, and 512, plus one
  two-axis case. Three scale points reveal complexity without a noisy sweep.
- Profile allocations/launches externally where available and record synchronized latency.
- Inventory CPU/CUDA/Metal dtype, empty identity, autograd, and packaging requirements.
- Prototype at least one plausible adapter in spike/test code; do not commit a
  CPU-only production fast path as if it were portable.

## Red-first evidence

Register the candidate adapter contract before it exists and record the compile
failure. Existing correctness parity is not a substitute for the missing strategy.

## Acceptance

- A written go/no-go decision includes measured curves, maintenance cost,
  supported dtype/device/gradient matrix, and upstream feasibility.
- The follow-up ticket is refined to the selected API or closed/superseded if no
  portable win exists.

## Non-goals

No `exp(sum(log))`, dependency fork without approval, or universal claim from a
CPU-only prototype.

## Decision: no-go for a local native reduction in 0.2

Retain the existing sequential Candle multiplication fallback. The portable
balanced-tree prototype preserves the product contract, but it still submits
K-1 multiplication operations, retains more live intermediates, and is slower
at every measured scale. A genuinely native path would require three device
kernels plus a zero-aware backward implementation, which is disproportionate
maintenance for this adapter crate. Revisit only after Candle exposes a public,
portable product reduction primitive.

### Measured evidence

Measurements used Rust 1.94.1, Candle 0.11.0, macOS arm64 CPU, 25 synchronized
samples per run, and three complete runs. Values below are the median of each
run's median. `balanced` is the benchmark-only portable candidate.

| Shape | Current sequential | Balanced candidate | Candidate change |
| --- | ---: | ---: | ---: |
| `[256, 8]` | 27.459 us | 29.291 us | 6.7% slower |
| `[256, 64]` | 243.958 us | 260.083 us | 6.6% slower |
| `[256, 512]` | 2.020 ms | 2.144 ms | 6.1% slower |
| `[256, 8, 8]`, reduce two axes | 183.125 us | 257.375 us | 40.5% slower |

For one axis, both implementations enqueue exactly K-1 elementwise
multiplications. The balanced candidate only reduces dependency depth; Candle's
ordered device stream cannot turn that into fewer submissions. For the two-axis
case, the current implementation needs 14 multiplications (7 for each axis),
whereas flattening and balancing 64 factors needs 63. Narrow and squeeze are
storage-sharing views, but every multiplication allocates an output. The
balanced tree also retains a wider frontier of intermediates.

### Portability and maintenance inventory

| Requirement | Existing fallback | Local `CustomOp1` | Upstream Candle `ReduceOp::Prod` |
| --- | --- | --- | --- |
| CPU/CUDA/Metal | Candle `mul` on each supported device | Three implementations required | Three Candle backends required |
| Dtypes | Inherits Candle multiplication | Explicit storage/kernel dispatch | Explicit backend dispatch |
| Arbitrary layouts | Narrow/squeeze preserve layouts | Must implement strides and offsets | Backend reduction machinery can own it |
| Empty identity | Explicit ones tensor | Must define in every backend | Must be added to reduction contract |
| Autograd and zeros | Multiplication graph is naturally zero-aware | Custom backward must handle zero multiplicity without `result / input` | Candle backprop must add product logic |
| Packaging | No kernels or features | Ships/maintains CUDA and Metal kernels | Requires a future Candle release |

Candle 0.11's public `CustomOp1` requires `cpu_fwd`; CUDA and Metal default to
runtime errors unless separately implemented, and backward defaults to
unsupported. Candle's internal `ReduceOp` currently contains only sum, min,
max, argmin, and argmax. An upstream primitive is technically the right layer,
but it cannot improve this release without an unapproved fork or a future
Candle upgrade.

### Follow-on disposition

`native-product-reduction` is closed as wont-fix for 0.2. Its entry condition
required a portable go decision, which this spike did not produce. The frozen
benchmark scenarios remain as evidence and can be reused if Candle adds native
product reduction later.
