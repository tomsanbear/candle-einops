---
id: parser-never-unwinds-properties
title: Property-test parser and token planning against unwinds
status: done
priority: p1
dependencies: [grouped-rhs-error-propagation, decomposition-arithmetic-safety, macro-axis-invariant-hardening, macro-expansion-hygiene]
related: []
scopes: [macros]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Property-test parser and token planning against unwinds

## Vertical outcome

Broad malformed and grammar-generated input establishes that parsing/token planning returns success or `syn::Error`, never an unwind or hang.

## Red

Seed property tests with every prior crash/diagnostic case and verify at least one seed exposes the pre-hardening failure history.

## Green

Add bounded arbitrary UTF-8 and grammar-aware generation around parser/token planning with `catch_unwind` invariants and deterministic regression seeds.

## Acceptance

- Tests are deterministic, bounded, and practical in normal CI.
- Failures persist their minimal input as a normal regression case.
- The parser/token planner never unwinds across the generated corpus.
