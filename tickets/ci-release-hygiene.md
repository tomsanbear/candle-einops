---
id: ci-release-hygiene
title: Modernize CI and release hygiene
status: in-progress
priority: p1
dependencies: [candle-011-baseline]
related: []
scopes: [tooling, docs]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
claimed_from: todo
assignee: codex-ci
lease_expires_at: 1784123226
---
# Modernize CI and release hygiene

## Goal

Make the repository continuously verifiable and ready to publish both crates safely.

## Gap

CI uses archived actions, omits workspace-wide targets and docs/package checks, and the declared licenses have no repository files.

## Work

Use maintained toolchain setup and direct Cargo commands; test workspace targets, clippy, docs, packages, stable and MSRV; add license files and release-order checks.

## Acceptance

CI covers both crates on every change, package smoke checks pass, required license files exist, and supported Rust versions are enforced.
