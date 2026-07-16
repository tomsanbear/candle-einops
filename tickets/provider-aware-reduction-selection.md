---
id: provider-aware-reduction-selection
title: Calibrate Metal contiguous reduction fusion
status: in-progress
priority: p1
dependencies: [optimized-provider-performance-protocol]
related: [homogeneous-reduction-fusion]
scopes: [runtime, benchmarks]
shared_scopes: []
paths: []
tags: [performance-gap, reductions]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784223076
---
## Evidence

Five optimized 25-sample processes cleared CPU baseline, Accelerate, and CUDA. Metal contiguous trailing sum remained 16% / 30.2 microseconds behind its sequential direct Candle reference; mean remained 17% / 30.1 microseconds behind. Strided non-adjacent cases were not material gaps.

## Work

- Freeze route tests for contiguous trailing and strided non-adjacent sum/mean on Metal versus other providers.
- Attribute the Metal gap to collapsed extent, dispatch cost, or layout preparation using the existing high-signal matrix.
- Select fusion only for provider/layout combinations that clear the materiality threshold.

## Acceptance

- Red-first route tests cover provider and layout without changing values, gradients, dtype errors, or mixed-operation ordering.
- Metal contiguous trailing sum and mean are no longer materially slower than direct Candle.
- CPU and CUDA retain their optimized behavior and eligible cases retain the minimum reduction count.
