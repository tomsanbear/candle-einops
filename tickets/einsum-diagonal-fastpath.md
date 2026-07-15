---
id: einsum-diagonal-fastpath
title: Implement selected repeated-label diagonal fast path
status: todo
priority: p1
dependencies: [einsum-diagonal-lowering-spike]
related: [einsum-diagonal-lowering-spike]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement selected repeated-label diagonal fast path

## Entry condition

Satisfied by `einsum-diagonal-lowering-spike`: implement the portable,
gradient-capable original-layout flat gather selected in
`benchmarks/diagonal-lowering-spike.md`.

## Runtime ownership and selection

Own the change in `normalize_repeated_axes` and a private helper in
`src/einsum.rs`. Validate all repeated extents first. For a contiguous original
operand, compute original row-major offsets satisfying every repeated label and
use one `index_select` plus reshape when the existing adjacency permutation
would make its flattening materialize. Preserve unique axes in first-appearance
order.

Keep the current sequential lowering for adjacent repeated axes, non-contiguous
inputs, offset/index overflow, and backend/index combinations rejected by
Candle. Build one device-local index tensor per selected invocation; do not add
a cross-call or device-global cache. A future caller-owned compiled plan may
cache by equation, shape, index dtype, and device, but that lifetime is outside
this ticket.

## Red-first contract

Cover interleaved `i j i j`, higher repetition, multiple distinct repeated
labels, non-contiguous inputs, zero extents, unequal-extent rejection, and
forward/gradient parity against explicit indexing before production changes.
Assert the selection boundary so adjacent and fallback cases retain the current
path.

## Benchmark ownership

Reuse the spike’s simple/interleaved cases and scaling points. The implementation
must improve interleaved extraction without materially regressing simple diagonal.

## Acceptance

- Avoids full permuted-input materialization where selected by the design.
- Avoids rebuilding/uploading equivalent indices when the design safely permits.
- Preserves dtype/device, errors, autograd, Python parity, and deterministic behavior.

## Non-goals

No device-global unbounded cache or unrelated indexing API changes.
