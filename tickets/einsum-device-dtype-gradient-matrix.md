---
id: einsum-device-dtype-gradient-matrix
title: Harden einsum dtypes devices and gradients
status: todo
priority: p1
dependencies: [einsum-nary-greedy-planner]
related: []
scopes: [runtime, tests]
shared_scopes: [ticketing]
paths: []
tags: [einsum-implementation]
---
# Harden einsum dtypes devices and gradients

## Vertical outcome

Establish the supported dtype/device/autograd contract across the complete einsum semantics.

## TDD

Add red or coverage-first matrices for BF16/F16/F32/F64, supported integer behavior, NaN/infinity, broadcast gradients, diagonal gradients, zero-sensitive contractions, and representative accelerator execution when available.

## Green

Fix only reproduced divergences using tracked public Candle operations; never silently cast or transfer devices. Unsupported combinations return contextual Candle errors.

## Acceptance

- CPU finite-difference/analytic gradients cover unary, binary, diagonal, ellipsis, and n-ary cases.
- Optional Metal/CUDA forward smoke tests compare results after transfer to CPU.
- Same-device/same-dtype requirements are documented and enforced.
