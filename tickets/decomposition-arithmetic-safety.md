---
id: decomposition-arithmetic-safety
title: Make decomposition arithmetic checked and fallible
status: todo
priority: p0
dependencies: []
related: []
scopes: [macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Make decomposition arithmetic checked and fallible

## Vertical outcome

Decomposition factors cannot trigger division by zero or integer overflow at macro-expansion or runtime.

## Red

Add regressions for zero factors, non-divisible inferred factors, overflowing literal products, and overflowing runtime products. Capture the current panic or incorrect acceptance before implementation.

## Green

Use checked compile-time multiplication with `syn::Error`, and checked/nonzero/divisible runtime arithmetic that returns Candle errors.

## Acceptance

- Each regression is observed red before its corresponding implementation.
- Invalid arithmetic produces a compile diagnostic or runtime `Err`, never a procedural-macro/runtime panic.
- Valid decomposition values and shapes remain unchanged.
