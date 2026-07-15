---
id: gradient-differential-coverage
title: Add differential gradient coverage
status: todo
priority: p1
dependencies: [runtime-rank-shape-validation, decomposition-arithmetic-safety]
related: []
scopes: [tests]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Add differential gradient coverage

## Vertical outcome

Einops transformations preserve Candle autograd behavior relative to equivalent primitive operations.

## Red

Write direct-Candle gradient oracles for transpose, compose/decompose, repeat, sum, mean, and product, including zero-sensitive product cases. Confirm the custom product path's current behavior before adjusting production code, if needed.

## Green

Fix only demonstrated gradient divergences; otherwise retain the red-first tests as coverage.

## Acceptance

- Forward values and input gradients match independent Candle primitives.
- Product covers no zero, one zero, and multiple zeros.
- Tests run on CPU under Rust 1.94 and stable.
