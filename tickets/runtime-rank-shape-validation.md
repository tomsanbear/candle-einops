---
id: runtime-rank-shape-validation
title: Validate generated runtime ranks and shape indices
status: done
priority: p0
dependencies: []
related: []
scopes: [macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Validate generated runtime ranks and shape indices

## Vertical outcome

Malformed runtime ranks return a Candle error instead of panicking in generated indexing or ellipsis arithmetic.

## Red

Add focused runtime regressions for an ellipsis that captures too few axes and generated accesses beyond the input rank. Prove the current expansion unwinds with `catch_unwind` before changing implementation.

## Green

Validate the minimum rank and every generated shape access before arithmetic. Return contextual `candle_core::Error` values through the existing `Result` closure.

## Acceptance

- The committed regressions fail against the pre-fix implementation and pass after the fix.
- Rank-deficient calls return `Err` and never unwind.
- Existing ellipsis and composition behavior remains green on Rust 1.94 and stable.
