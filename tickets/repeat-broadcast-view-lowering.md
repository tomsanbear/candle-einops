---
id: repeat-broadcast-view-lowering
title: Lower inserted repeat axes as broadcast views
status: in-progress
priority: p0
dependencies: [performance-harness-foundation]
related: []
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: behavior-tests
lease_expires_at: 1784148686
---
# Lower inserted repeat axes as broadcast views

## Hypothesis

For axes introduced by `einops!`, one singleton-expanding reshape followed by
`broadcast_as` can replace the current per-axis unsqueeze and eager Candle
`repeat` concatenation. Pure repeat construction should become allocation-free
until a consumer requires materialization.

## Red-first contract

- Insert one middle axis of length four and assert exact values/shape plus a
  non-contiguous broadcast layout; the current eager result is contiguous.
- Cover multiple leading/middle/trailing axes with lengths 0, 1, and N.
- Check gradients, including zero-length output, and non-contiguous input.
- Check repeat followed by composition so deferred materialization remains correct.

## High-signal benchmark

Own only two scenario families in the shared harness: one large single-axis
repeat and one two-axis repeat. Each has `construct` mode and `consume` mode.
The first isolates eliminated concatenation; the second detects intermediate
growth. Do not sweep syntax, dtypes, or every axis location.

## Acceptance

- Length greater than one constructs a zero-stride view without cat intermediates.
- Values, shapes, Python parity, and gradients pass on CPU and available accelerators.
- Consume-mode evidence shows no material regression when a later operation must copy.
- The changed contiguity/aliasing contract is documented.

## Risks and non-goals

Downstream kernels vary in zero-stride support and callers may have assumed a
contiguous result. Returned repeats are not required to be contiguous; arbitrary
tiling of existing axes is out of scope.

