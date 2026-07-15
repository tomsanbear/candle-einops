---
id: einsum-ellipsis-slice
title: Add einsum ellipsis broadcasting
status: todo
priority: p1
dependencies: [einsum-binary-gemm-slice]
related: []
scopes: [runtime, macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [einsum-implementation]
---
# Add einsum ellipsis broadcasting

## Vertical outcome

Support at most one `..` per operand/output with variable-rank, right-aligned broadcasting and optional ellipsis reduction.

## TDD

Add red oracle cases for zero/one/many captured axes, operands with different ellipsis ranks, retained and omitted output ellipses, scalars, broadcast conflicts, and duplicate/misplaced ellipses.

## Green

Expand runtime ellipses into ordered synthetic labels, right-align dimensions, fill absent leading dimensions with one, then feed the existing resolver/lowering pipeline.

## Acceptance

- Ellipsis cases match the independent reference interpreter.
- Rank/broadcast errors name the operand and observed ellipsis dimensions.
- Existing explicit-label contractions remain unchanged.
