---
id: python-einops-repeat-reduce-properties
title: Add randomized Python parity for repeat and reduce
status: todo
priority: p0
dependencies: [python-einops-oracle-harness]
related: []
scopes: [tests]
shared_scopes: [ticketing]
tags: [python-einops-parity]
---
# Add randomized Python parity for repeat and reduce

## Required outcome

Use proptest-generated tensors and axis sizes to compare repeat plus sum,
mean, min, max, and product reductions against Python einops through the
locked batch oracle.

## Red-first work

Commit the randomized contract tests red first, retain any discovered
counterexample/seed, then correct production or test bridge behavior until green.

## Acceptance

- Reduction tolerances are explicit by dtype and operation.
- Zero-length behavior is tested only where both public contracts define it;
  differences are documented rather than silently normalized.
- Repeat covers new axes, grouped axes, ellipsis, and runtime axis lengths.

