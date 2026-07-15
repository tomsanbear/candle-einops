---
id: spike-zero-size-contraction-submission-reduction
title: Spike zero-size contraction submission reduction
status: todo
priority: p2
dependencies: [einsum-zero-k-autograd]
related: [einsum-zero-k-autograd]
scopes: [benchmarks]
shared_scopes: [runtime, tests, ticketing]
paths: []
tags: [kernel-enqueue-hardening, spike]
---
# Spike zero-size contraction submission reduction

## Goal

Determine whether zero-sized contractions can preserve both autograd edges with
fewer public Candle operations than the current two reductions plus addition.

## Work

- Compare the current two-anchor construction with flatten/cat/sum and any other
  public-operation candidate that preserves both gradients.
- Use three output sizes but one mechanism family; record operations, temporary
  elements, synchronized latency, dtype/device support, and backward behavior.
- Preserve validation ordering and exact zero output semantics.

## Acceptance

- Record a GO/NO-GO decision; implement only a candidate that is structurally
  smaller and does not materially regress end-to-end latency or memory.
- Both operand gradients remain present with exact original shapes.
- No custom kernel is introduced.
