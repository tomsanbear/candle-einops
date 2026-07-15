---
id: candle-style-errors
title: Align runtime error handling with Candle
status: todo
priority: p1
dependencies: [product-reduction, rename-safe-expansions]
related: []
scopes: [runtime, macros, tests, docs]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
---
# Align runtime error handling with Candle

## Goal

Adopt a deliberate, documented failure model consistent with Candle Result APIs.

## Gap

Every backend operation unwraps Candle errors, while the public Backend trait and macro expose an infallible Tensor API.

## Work

Evaluate compatibility options, document the selected API, implement it with an appropriate semver plan, and test invalid shapes and axes.

## Acceptance

The chosen behavior is explicit and tested; invalid runtime operations no longer cause undocumented panics; migration guidance exists for any breaking API change.
