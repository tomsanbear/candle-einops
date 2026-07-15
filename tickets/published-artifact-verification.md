---
id: published-artifact-verification
title: Verify doctests and packaged artifacts in CI
status: in-progress
priority: p1
dependencies: [macro-expansion-hygiene]
related: []
scopes: [tooling, macros]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
claimed_from: todo
assignee: codex-artifacts
lease_expires_at: 1784130087
---
# Verify doctests and packaged artifacts in CI

## Vertical outcome

CI tests the same crate contents and runnable documentation that users receive from crates.io.

## Red

Add a packaged-source test that exposes the macro crate's missing nested fixtures, and a CI doctest step that would catch the prior rustdoc crate-path regression.

## Green

Make fixtures package-safe or exclude/re-home the harness deliberately, then unpack and test both `.crate` artifacts in dependency order. Execute dependency fixtures rather than only checking them.

## Acceptance

- Warning-denied doctests run in CI.
- Packaged sources have self-contained tests or an explicitly tested exclusion policy.
- Both normal and renamed fixture assertions execute successfully.
