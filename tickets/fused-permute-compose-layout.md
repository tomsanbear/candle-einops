---
id: fused-permute-compose-layout
title: Implement selected permute and composition fusion
status: todo
priority: p2
dependencies: [permute-compose-layout-spike]
related: []
scopes: [runtime, macros]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement selected permute and composition fusion

## Entry condition — satisfied, conditional GO

The spike measured an 852x–1231x CPU construction improvement for `c (a b)`
and `n (h w) c` at 98,304 elements. Immediate contiguous consumption was
neutral (0.997x–1.005x), so this optimization avoids work only when downstream
operations accept the view or materialization is delayed. Implement the narrow
public-operation plan; do not construct arbitrary layouts.

## Red-first contract

A recording custom backend expects one default-compatible fused call rather
than separate transpose/reshape calls. The default implementation must perform
the current expanded transpose then reshape, with the associated-output bound
needed by today's generated sequence. The Candle Tensor implementation may
specialize only when an ordering of whole requested groups makes the actual
layout C-contiguous under Candle's exact predicate; it then performs public
`permute -> reshape -> permute` views. Exhaustive bounded shapes compare output
index order; eligible Tensor cases remain storage-sharing, while ineligible
cases still execute the existing path. Include offsets, zero/singleton axes,
identity-reshape boundaries, invalid metadata ordering, and gradients.

Group members must retain requested logical order; only whole groups may move.
Restrict fusion to a pure permutation-plus-composition boundary and do not cross
repeat or reduction operations. Use checked products and deterministic plan
selection. If classification is ineligible, execute the existing expanded
transpose then reshape byte-for-byte. Propagate errors from a selected plan;
do not retry after partial execution.

## Benchmark ownership

Reuse only the spike corpus and compare identical construct/consume filters.
Do not add a general rearrange syntax matrix.

## Acceptance

- Eligible cases avoid the measured materialization.
- The default Backend implementation preserves compatibility for third parties.
- Ineligible layouts retain current correct behavior and all expansion/hygiene tests pass.
- CPU is required; CUDA/Metal feature builds establish backend-neutral API
  feasibility where available but make no GPU timing claim.

## Non-goals

No assertion that every permute/reshape can be a view, no unsafe/private layout
construction, and no unconditional `contiguous`.
