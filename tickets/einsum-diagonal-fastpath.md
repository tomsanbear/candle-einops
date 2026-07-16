---
id: einsum-diagonal-fastpath
title: Implement selected repeated-label diagonal fast path
status: done
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

Selection follows simulated layout behavior, not syntax alone: leading `i i`
stays sequential when its current flatten is a view, while contiguous
`batch i i` may select the gather when moving the repeated axes would
materialize. Keep the current sequential lowering for non-materializing cases,
non-contiguous inputs, offset/index overflow, and explicitly unsupported
backend/index combinations. Unexpected allocation or device execution errors
remain errors rather than being swallowed as fallback. Build one device-local
combined index tensor per selected invocation; do not add
a cross-call or device-global cache. A future caller-owned compiled plan may
cache by equation, shape, index dtype, and device, but that lifetime is outside
this ticket.

## Red-first contract

Cover interleaved `i j i j`, higher repetition, multiple distinct repeated
labels, non-contiguous inputs, zero extents, unequal-extent rejection, and
forward/gradient parity against explicit indexing before production changes.
Assert the simulated materialization boundary, including leading adjacent and
batched-adjacent cases, so non-materializing and fallback cases retain the
current path.

## Benchmark ownership

Reuse the spike’s simple/interleaved cases and scaling points. The implementation
must improve interleaved extraction without materially regressing simple diagonal.

## Acceptance

- Avoids full permuted-input materialization where selected by the design.
- Builds/uploads at most one combined index tensor per selected invocation,
  including multiple repeated-label groups; cross-call reuse is out of scope.
- Preserves dtype/device, errors, autograd, Python parity, and deterministic behavior.

## Non-goals

No device-global unbounded cache or unrelated indexing API changes.

## Result

- Repeated extents are validated before planning. A pure layout simulation
  selects one original-flat gather only for contiguous operands whose existing
  sequential permutation/reshape would materialize; adjacent, non-contiguous,
  arithmetic-overflow, and Candle index-sentinel cases remain sequential.
- Selected invocations build one local U32 offset tensor, flatten the unchanged
  operand, perform one differentiable `index_select`, and reshape unique axes in
  first-appearance order. Allocation and device execution errors propagate.
- Forward, dtype/device, error, zero-extent, and gradient contracts cover
  adjacent, batched, interleaved, triple, and multiple repeated-label groups.
- The six spike scenarios and scaling points remain stable. Same-machine
  release measurements improved interleaved cases by 44.5–83.2%; sequential
  simple cases remained within 3.6–9.8% and 42 ns of the exact pre-change
  commit. The cached reference is explicitly labeled as a floor.
- Independent review found the reserved `u32::MAX` sentinel boundary; the fix
  and exact equality regression test were re-reviewed with no findings.
