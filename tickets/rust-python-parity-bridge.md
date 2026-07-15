---
id: rust-python-parity-bridge
title: Build the reusable Rust to Python parity bridge
status: done
priority: p0
dependencies: [python-einops-oracle-harness]
related: []
scopes: [parity]
shared_scopes: [ticketing]
tags: [python-einops-parity]
---
# Build the reusable Rust to Python parity bridge

## Required outcome

Add an unpublished standalone Rust test runner under `parity/` with proptest,
JSON serialization, a persistent JSONL Python subprocess, stable pattern IDs,
and complete minimized-case replay. Keep it outside the published workspace and
ordinary `cargo test` path.

## Red-first work

Specify protocol/version validation, ordered request matching, child cleanup,
success/error normalization, batching, and replay behavior in tests. Observe
them fail before implementing the bridge.

## Acceptance

- The Python subprocess starts once per test process and flushes each response.
- Stable IDs dispatch to compile-time literal macro patterns in later slices.
- Proptest failures print and persist a directly replayable JSON request.
- The runner has bounded cases/elements and deterministic CI seed overrides.

