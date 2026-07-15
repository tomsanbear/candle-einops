---
id: permute-compose-layout-spike
title: Determine viable permute and composition layout fusion
status: in-progress
priority: p1
dependencies: [performance-harness-foundation, identity-reshape-elision]
related: [fused-permute-compose-layout]
scopes: []
shared_scopes: [tests, benchmarks, docs, ticketing]
tags: [performance-0.2]
claimed_from: todo
assignee: behavior-tests
lease_expires_at: 1784149770
---
# Determine viable permute and composition layout fusion

## Decision question

Which generated permute-plus-composition patterns are truly stride-collapsible
through Candle’s public APIs, and does a default-compatible fused Backend method
justify the added API/codegen complexity?

## Spike work

Build a pure stride-collapse classifier and a compact corpus: one naturally
viewable grouping, `a b c -> c (a b)`, a channel-first/channel-last flatten,
and offset/non-contiguous variants with zero/singleton extents. Compare logical
values, strides, materialization, gradients, and current residual copy cost.

## Red-first evidence

Commit the decision-table tests against the absent classifier. A green spike
classifies viewable versus copy-required cases and prototypes public API
feasibility without unsafe production layout construction.

## Acceptance

- Written go/no-go decision with measured upside, API feasibility, flatten-order
  proof obligations, and a refined implementation ticket.
- Explicitly identify patterns where copying is unavoidable or public Candle APIs
  cannot construct the required safe view.

## Non-goals

No unconditional `contiguous`, unsafe private-layout construction, or production
claim based only on matching shapes.

