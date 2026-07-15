---
id: einsum-diagonal-fastpath
title: Implement selected repeated-label diagonal fast path
status: todo
priority: p1
dependencies: [einsum-diagonal-lowering-spike]
related: []
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement selected repeated-label diagonal fast path

## Entry condition

Proceed only after the spike records a portable, gradient-capable go decision.

## Red-first contract

Cover interleaved `i j i j`, higher repetition, multiple distinct repeated
labels, non-contiguous inputs, zero extents, unequal-extent rejection, and
forward/gradient parity against explicit indexing before production changes.

## Benchmark ownership

Reuse the spike’s simple/interleaved cases and scaling points. The implementation
must improve interleaved extraction without materially regressing simple diagonal.

## Acceptance

- Avoids full permuted-input materialization where selected by the design.
- Avoids rebuilding/uploading equivalent indices when the design safely permits.
- Preserves dtype/device, errors, autograd, Python parity, and deterministic behavior.

## Non-goals

No device-global unbounded cache or unrelated indexing API changes.

