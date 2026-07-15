---
id: python-einops-rearrange-properties
title: Add randomized Python parity for rearrange
status: todo
priority: p0
dependencies: [python-einops-oracle-harness]
related: []
scopes: [tests]
shared_scopes: [ticketing]
tags: [python-einops-parity]
---
# Add randomized Python parity for rearrange

## Required outcome

Use proptest-generated shapes, values, compositions, decompositions,
permutations, singleton removal, and ellipsis cases. Batch those cases through
the Python oracle and compare Candle output shapes and values.

## Red-first work

Commit the randomized test matrix while its oracle bridge or an intentionally
unsupported case is red, preserve the reproducing seed, then make the suite green.

## Acceptance

- Every macro pattern is a compile-time literal and is identified in failures.
- Random cases are bounded, deterministic in CI, and replayable locally.
- Shape and exact finite-value parity covers zero, singleton, contiguous, and non-contiguous cases.

