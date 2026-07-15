---
id: fused-permute-compose-layout
title: Implement selected permute and composition fusion
status: todo
priority: p2
dependencies: [permute-compose-layout-spike]
related: []
scopes: [runtime, macros]
shared_scopes: [tests, benchmarks, ticketing]
tags: [performance-0.2, conditional]
---
# Implement selected permute and composition fusion

## Entry condition

Proceed only on a measured go decision from `permute-compose-layout-spike`.

## Red-first contract

A recording custom backend expects one default-compatible fused call rather
than separate transpose/reshape calls. Exhaustive bounded shapes compare output
index order; eligible Tensor cases remain storage-sharing, while ineligible
cases still materialize correctly. Include offsets, zero/singleton axes, and gradients.

## Benchmark ownership

Reuse only the spike corpus and compare identical construct/consume filters.
Do not add a general rearrange syntax matrix.

## Acceptance

- Eligible cases avoid the measured materialization.
- The default Backend implementation preserves compatibility for third parties.
- Ineligible layouts retain current correct behavior and all expansion/hygiene tests pass.

## Non-goals

No assertion that every permute/reshape can be a view and no bypass of Candle safety.

