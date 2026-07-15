---
id: product-reduction
title: Implement product reduction end to end
status: todo
priority: p0
dependencies: [candle-011-baseline]
related: []
scopes: [runtime, macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
---
# Implement product reduction end to end

## Goal

Make the already-parsed prod operation a supported public reduction.

## Gap

The parser and token generator advertise Operation::Prod, but the public enum and Candle backend do not implement it.

## Work

Add the public operation variant, implement Candle product reduction, and cover single-axis, multi-axis, ellipsis, and dtype behavior.

## Acceptance

prod expressions compile and return independently verified values; unsupported dtypes return or document the same behavior as Candle.
