---
id: rename-safe-expansions
title: Make generated crate paths rename-safe
status: in-progress
priority: p1
dependencies: [semantic-expansion-fixes]
related: []
scopes: [macros]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
claimed_from: todo
assignee: codex-rename-safe
lease_expires_at: 1784124556
---
# Make generated crate paths rename-safe

## Goal

Allow downstream consumers to rename the candle-einops dependency normally.

## Gap

Generated code hard-codes ::candle_einops and fails when the dependency has a Cargo alias.

## Work

Resolve the consumer-visible crate path safely and add a fixture that imports the library under another name.

## Acceptance

Normal and renamed dependency fixtures both compile, with no hard-coded public path assumptions left in generated code.
