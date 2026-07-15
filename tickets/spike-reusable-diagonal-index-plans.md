---
id: spike-reusable-diagonal-index-plans
title: Spike reusable diagonal index plans
status: in-progress
priority: p2
dependencies: [einsum-diagonal-fastpath]
related: [einsum-diagonal-lowering-spike]
scopes: [benchmarks]
shared_scopes: [runtime, tests, ticketing]
paths: []
tags: [kernel-enqueue-hardening, spike]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784159260
---
# Spike reusable diagonal index plans

## Goal

Determine whether a caller-owned prepared diagonal plan materially reduces
repeated index construction/device-transfer overhead without introducing a
global cache or backend-specific kernel.

## Work

- Compare current per-call index creation with one explicitly prepared,
  device-bound index tensor for simple and interleaved diagonals.
- Separate preparation from steady-state execution and record index bytes,
  gather calls, and synchronized consumption.
- Sketch the minimum safe ownership/API contract only if evidence is positive.

## Acceptance

- Record a GO/NO-GO decision with break-even reuse count and device/shape boundaries.
- Values and gradients match current lowering.
- No global cache, implicit lifetime, custom operation, or public API is added by the spike.
