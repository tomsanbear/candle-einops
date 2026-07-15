---
id: homogeneous-reduction-fusion
title: Fuse homogeneous sum and mean reduction runs
status: in-progress
priority: p0
dependencies: [performance-harness-foundation]
related: [product-reduction-strategy-spike]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: behavior-tests
lease_expires_at: 1784147867
---
# Fuse homogeneous sum and mean reduction runs

## Hypothesis

The backend can preserve descending mixed-operation semantics while batching
adjacent homogeneous Sum or Mean runs into Candle multi-axis reductions. This
should turn N sum reductions into one backend reduction and N means into one
multi-axis reduction plus scaling.

## Red-first contract

- A pure planner test groups three sum axes into one run.
- Sum/max/sum remains three ordered runs; identical operations only group within
  adjacent runs and never cross another operation.
- A test recording backend-facing execution count expects one call for multi-sum
  and one for multi-mean.
- Values, gradients, non-contiguous inputs, empty/singleton axes, ellipsis, and
  current dtype support match direct Candle and Python parity within tolerances.

## High-signal benchmark

Own exactly two layouts on one representative 4-D tensor: contiguous trailing
multi-axis reduction and strided/non-adjacent reduction. Report sum and mean for
each. These four measurements distinguish kernel fusion and layout sensitivity;
single-axis and min/max measurements are redundant and excluded.

## Acceptance

- Homogeneous runs use the minimum Candle reductions described above.
- Mixed operation ordering and results remain unchanged.
- Floating accumulation differences remain within the existing explicit parity tolerances.
- Integer mean or unsupported dtype behavior conservatively falls back if required.

## Non-goals

No mixed-operation reordering, min/max batching, or product implementation.

