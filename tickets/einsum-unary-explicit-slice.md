---
id: einsum-unary-explicit-slice
title: Implement unary explicit-output einsum
status: done
priority: p0
dependencies: [einsum-contract-oracles, parser-never-unwinds-properties, published-artifact-verification]
related: []
scopes: [runtime, macros, tests, docs]
shared_scopes: [ticketing]
paths: []
tags: [einsum-implementation]
---
# Implement unary explicit-output einsum

## Vertical outcome

Ship a working `einsum!` for one operand with explicit output, unique labels, permutation, reduction, scalar output, and zero-sized dimensions.

## Architecture

Create a dedicated equation IR rather than reusing the unary einops positional pipeline. The macro interns labels, binds the operand once, and calls a doc-hidden rename-safe runtime ABI. Runtime validates rank and uses public Candle permute/reshape/sum operations with checked arithmetic.

## TDD

Start from the red contract/oracle cases and compile-fail fixtures for missing/multiple arrows, operand-count mismatch, invalid tokens, duplicate output, unknown output, and not-yet-supported repeated labels/ellipsis.

## Acceptance

- Unary transpose, reduction, identity, scalar, and empty-axis cases match the independent oracle.
- All invalid inputs produce `syn::Error` or Candle `Err`, never unwind.
- The hidden ABI remains private and the macro works through a renamed dependency.
