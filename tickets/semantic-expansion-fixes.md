---
id: semantic-expansion-fixes
title: Fix semantic macro expansion defects
status: todo
priority: p0
dependencies: [dsl-axis-invariants]
related: []
scopes: [macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
---
# Fix semantic macro expansion defects

## Goal

Generate correct tensor transformations for braced axes and derived reductions.

## Gap

Standalone braced RHS axes can disappear from composition shapes, and reductions attached to inferred dimensions are discarded.

## Work

Correct composition tracking and preserve derived-axis reduction metadata through parsing and token generation.

## Acceptance

Independent regression tests reproduce both prior failures and verify correct output shapes and values without panics.
