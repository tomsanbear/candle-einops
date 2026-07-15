---
id: edge-shape-dtype-properties
title: Add edge-shape and dtype property coverage
status: todo
priority: p1
dependencies: [runtime-rank-shape-validation, decomposition-arithmetic-safety]
related: []
scopes: [tests]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Add edge-shape and dtype property coverage

## Vertical outcome

Fixed macro patterns are exercised across generated dimensions, values, dtypes, scalar/empty/singleton shapes, and non-contiguous inputs.

## Red

Introduce independent host/Candle oracles and properties for inverse permutations, compose/decompose round trips, repeat indexing, ellipsis equivalence, and backend metadata rejection. Record any failing seeds before fixes.

## Green

Fix only reproduced defects and promote minimized seeds to explicit regression tests.

## Acceptance

- Properties are deterministic and bounded for CI.
- Invalid metadata returns `Err` without unwinding.
- Supported dtype results match oracles; unsupported combinations return Candle errors.
