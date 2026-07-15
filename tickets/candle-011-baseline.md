---
id: candle-011-baseline
title: Establish the Candle 0.11 baseline
status: todo
priority: p0
dependencies: []
related: []
scopes: [tooling, macros]
shared_scopes: []
paths: []
tags: [candle-0.11-modernization]
---
# Establish the Candle 0.11 baseline

## Goal

Build and test the full repository against Candle 0.11 from a coherent Cargo workspace.

## Gap

The root manifest pins Candle 0.6, fresh dependency resolution no longer builds, and the proc-macro crate is not a workspace member.

## Work

Create the workspace, upgrade candle-core to 0.11, align the local macro dependency version, and record an explicit supported Rust baseline.

## Acceptance

Workspace tests pass against Candle 0.11, metadata includes both crates, and the chosen Rust version is documented in both manifests.

## Refs

Audit finding: Candle 0.11.0 passed all 11 existing tests without source changes.
