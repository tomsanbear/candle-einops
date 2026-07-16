---
id: optimized-provider-performance-protocol
title: Establish optimized provider performance protocol
status: todo
priority: p0
dependencies: []
related: []
scopes: [benchmarks, tooling]
shared_scopes: []
paths: []
tags: [performance-gap, performance-0.2]
---
## Goal

Replace debug validation timings with an optimized, repeatable protocol that can decide whether a provider-specific gap is material.

## Work

- Make JSON `run` and capture preparation use an optimized Cargo profile and record that profile in run metadata.
- Add a focused regression command that runs five independent processes with deterministic alternating order.
- Re-measure only the named gap scenarios on CPU baseline, Accelerate, Metal, and CUDA where supported.
- Classify a regression only when it exceeds both 10% and 1 microsecond and its deterministic interval excludes parity.

## Acceptance

- Tests fail first when the wrapper uses a debug binary or omits profile metadata.
- Results from different profiles are incomparable.
- The protocol emits a compact provider/scenario gap report without parsing proprietary GPU traces.
- Existing advisory CI remains non-blocking.
