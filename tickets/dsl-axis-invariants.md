---
id: dsl-axis-invariants
title: Enforce DSL axis invariants
status: todo
priority: p0
dependencies: [panic-free-parser]
related: []
scopes: [macros]
shared_scopes: []
paths: []
tags: [candle-0.11-modernization]
---
# Enforce DSL axis invariants

## Goal

Reject structurally invalid einops expressions during macro expansion.

## Gap

Count-only RHS validation lets duplicate axes conceal missing ones; empty groups and meaningless top-level size annotations are accepted.

## Work

Track consumed axes explicitly and validate duplicates, omissions, ellipses, empty groups, and shape annotations.

## Acceptance

Each invalid class has a compile-fail fixture, diagnostics identify the offending axis or group, and valid ellipsis patterns remain compatible.
