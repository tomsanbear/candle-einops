---
id: release-readiness
title: Complete documentation and release readiness
status: in-progress
priority: p1
dependencies: [candle-style-errors, ci-release-hygiene, independent-behavior-tests]
related: []
scopes: [docs, runtime, macros, tooling]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
claimed_from: todo
assignee: codex-release
lease_expires_at: 1784125609
---
# Complete documentation and release readiness

## Goal

Finish the modernization with accurate, tested user guidance and publishable artifacts.

## Gap

README and rustdoc examples are incomplete or invalid, compatibility and failure behavior are undocumented, and there is no changelog or release checklist.

## Work

Repair runnable examples, document supported syntax and Candle compatibility, add migration and release notes, and perform final workspace/package verification.

## Acceptance

All doctests and release checks pass, documentation matches the final API, and both crates have an explicit publish order and version plan.
