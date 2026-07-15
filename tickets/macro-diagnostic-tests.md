---
id: macro-diagnostic-tests
title: Add macro diagnostic test infrastructure
status: in-progress
priority: p0
dependencies: [candle-011-baseline]
related: []
scopes: [macros]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784123401
---
# Add macro diagnostic test infrastructure

## Goal

Provide compile-pass and compile-fail coverage for the procedural macro grammar.

## Gap

The macro crate has no parser or UI test suite, so panic paths and poor diagnostics are untested.

## Work

Add a trybuild-style harness and representative valid and invalid fixtures, including current panic reproductions.

## Acceptance

The macro crate runs UI tests in the workspace suite, expected diagnostics are checked in, and fixtures cover missing sizes, unmatched ellipses, and malformed groups.
