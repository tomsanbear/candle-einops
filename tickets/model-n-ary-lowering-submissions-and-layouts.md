---
id: model-n-ary-lowering-submissions-and-layouts
title: Model n-ary lowering submissions and layouts
status: in-progress
priority: p1
dependencies: [add-layout-aware-binary-einsum-operand-packing]
related: [einsum-nary-layout-aware-planner]
scopes: [runtime]
shared_scopes: [tests, benchmarks, ticketing]
paths: []
tags: [kernel-enqueue-hardening]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784157644
---
# Model n-ary lowering submissions and layouts

## Goal

Make n-ary contraction selection account for the actual Candle operation graph
rather than treating every pair as one contiguous contraction submission.

## Work

- Add model tests for pair-local pre-reductions, eager broadcast/layout copies,
  multiplication versus GEMM, and the resulting intermediate layout.
- Share a pure lowering classifier between execution and cost estimation so the
  model cannot silently drift from production.
- Recalibrate only the existing four high-signal fixtures; keep accelerator
  enablement disabled without hardware evidence.

## Acceptance

- Estimated submissions and copies match structural execution traces.
- Existing selected contraction orders remain or change for an explained reason.
- Forward/gradient parity and the planner host-time budget remain green.
- Documentation labels counts as public-operation estimates, not profiler counters.
