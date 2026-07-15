---
id: native-product-reduction
title: Implement the selected native product reduction
status: closed
priority: p1
dependencies: [product-reduction-strategy-spike, homogeneous-reduction-fusion]
related: []
scopes: [runtime, tooling]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
closed_reason: wontdo
closed_note: "Spike found no portable win for 0.2: balanced Candle operations are slower and a native path belongs upstream in Candle."
---
# Implement the selected native product reduction

## Entry condition

Proceed only if `product-reduction-strategy-spike` records a portable go
decision. Otherwise close this ticket with the spike evidence rather than
shipping a weak or backend-specific optimization.

## Red-first contract

Specify product across one/consecutive/all/ellipsis/grouped axes; negatives,
signed zero, zeros, empty identity one, non-contiguous layouts, gradients, and
every supported dtype/device. A structural execution test must demonstrate the
spike-approved O(1), or explicitly bounded, reduction submissions per run.

## Benchmark ownership

Reuse the spike’s exact inputs and filters. Add no new shape sweep. The only
decision is before/after latency, peak temporary behavior, and submissions for
the same K=8/64/512 and two-axis cases.

## Acceptance

- Implements the selected portable strategy and preserves the complete product contract.
- Demonstrates a material win without silently reducing dtype/device support.
- Integrates homogeneous runs with the reduction planner where applicable.

## Risks and non-goals

Numerical order, empty identities, custom kernels, and autograd are high risk.
Changing product semantics or maintaining an unapproved Candle fork is excluded.

