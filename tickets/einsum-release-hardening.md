---
id: einsum-release-hardening
title: Complete einsum documentation packaging and release gates
status: done
priority: p0
dependencies: [einsum-device-dtype-gradient-matrix]
related: []
scopes: [runtime, macros, tests, docs, tooling]
shared_scopes: [ticketing]
paths: []
tags: [einsum-implementation]
---
# Complete einsum documentation packaging and release gates

## Required outcome

Declare einsum complete only when explicit labels, arbitrary arity, broadcasting, diagonals, ellipsis, scalars/zero dimensions, efficient pair lowering, deterministic planning, diagnostics, gradients, and packaging all ship together.

## Work

Document syntax and migration for the supported API. Settle exact macro/runtime
private-ABI version coupling, add renamed-dependency and packaged-artifact
execution, bounded parser/property fuzzing, examples, changelog, and release
checklist updates.

## TDD

Begin with packaged/downstream examples and documentation tests that fail until the complete feature surface is exported and self-contained.

## Acceptance

- Rust 1.94/stable tests, strict clippy, warning-denied docs/doctests, dependency policy, macOS representative tests, and packaged-artifact execution pass.
- Compile/runtime diagnostics cover every unsupported or invalid equation class without panics.
- `einsum!` is publicly documented as a supported feature, not future work.
