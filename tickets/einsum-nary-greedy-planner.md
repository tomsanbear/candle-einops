---
id: einsum-nary-greedy-planner
title: Add arbitrary-arity greedy einsum planning
status: done
priority: p1
dependencies: [einsum-ellipsis-slice, einsum-diagonal-slice]
related: []
scopes: [runtime, macros, tests, docs]
shared_scopes: [ticketing]
paths: [docs/einsum-contract.md]
tags: [einsum-implementation]
---
# Add arbitrary-arity greedy einsum planning

## Vertical outcome

Support three or more operands with deterministic, shape-aware contraction order and bounded intermediates.

## TDD

Add red three/four-operand oracle cases, live-label/safe-reduction cases, equal-cost tie cases, zero dimensions, checked cost overflow, and a pathological equation where left-to-right creates a much larger intermediate.

## Green

At each step evaluate every pair using checked `u128` peak-element and FLOP estimates; select lexicographically by peak size, FLOPs, then stable operand order. Contract only labels absent from the final output and all remaining operands.

## Acceptance

- Arbitrary arity matches the reference interpreter.
- Planning is deterministic and never drops a live label.
- Tests assert the planner avoids the known pathological intermediate.
