---
id: independent-behavior-tests
title: Expand independent behavior tests
status: todo
priority: p1
dependencies: [candle-011-baseline]
related: []
scopes: [tests]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
---
# Expand independent behavior tests

## Goal

Validate behavior against explicit expected tensors rather than other macro expressions.

## Gap

Many integration tests compare one einops pattern with another, allowing correlated implementation bugs to survive.

## Work

Add value-based coverage for reductions, reshaping, repeating, squeezing, ellipses, owned and borrowed inputs, and representative dtypes.

## Acceptance

Every supported operation has at least one independent value oracle and edge cases are covered without relying on a second macro expansion.
