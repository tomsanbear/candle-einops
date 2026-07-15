---
id: identity-reshape-elision
title: Elide identity reshape nodes and hidden copies
status: todo
priority: p1
dependencies: [performance-harness-foundation]
related: [permute-compose-layout-spike]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2]
---
# Elide identity reshape nodes and hidden copies

## Hypothesis

When requested dimensions equal current dimensions, the Tensor backend can
return a shallow clone before calling Candle reshape. This removes a graph node
and prevents an identity reshape from copying non-contiguous storage.

## Red-first contract

- Permute a tensor, request its identical shape through `Backend::reshape`, and
  require values/gradients plus preserved non-contiguous layout; current Candle
  reshape copies and becomes contiguous.
- Cover contiguous, offset/narrowed, zero, and singleton shapes.
- Exercise public singleton grouping patterns and a recording custom backend.

## High-signal benchmark

Own one contiguous control and one non-contiguous identity reshape at the same
representative size, in construct and consume modes. More shape variants would
measure the same mechanism and are excluded.

## Acceptance

- Identical shape performs no data copy and no new reshape operation where observable.
- Shape-changing reshape behavior is untouched.
- Document that identity reshape no longer provides accidental contiguity.

## Non-goals

No static macro normalization, general non-contiguous reshape, or layout fusion.

