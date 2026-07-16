---
id: panic-free-parser
title: Make macro parsing panic-free
status: done
priority: p0
dependencies: [macro-diagnostic-tests]
related: []
scopes: [macros]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
---
# Make macro parsing panic-free

## Goal

Ensure invalid user expressions produce precise compiler diagnostics rather than procedural-macro panics.

## Gap

Parser paths use expect and unwrap for missing RHS sizes, unmatched ellipses, and assumed grammar states.

## Work

Replace user-triggerable panic paths with span-aware syn::Error results and clean up diagnostic wording.

## Acceptance

Known panic reproductions compile-fail cleanly, no user-controlled parser path relies on expect or unwrap, and UI tests lock the messages.
