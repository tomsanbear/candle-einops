---
id: macro-axis-invariant-hardening
title: Enforce remaining axis and reduction invariants
status: done
priority: p1
dependencies: [grouped-rhs-error-propagation]
related: []
scopes: [macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Enforce remaining axis and reduction invariants

## Vertical outcome

Reduced LHS axes and newly introduced RHS axes obey the same uniqueness and size rules as ordinary axes.

## Red

Add compile-fail cases for duplicate reduced names/ellipses, duplicate new RHS axes, and top-level reduction size annotations that are currently ignored.

## Green

Validate all LHS identities before reduction filtering, track newly introduced RHS identities, and reject unsupported reduction size annotations.

## Acceptance

- Every rule has a red UI fixture followed by a stable green diagnostic.
- Valid mixed reductions and repetitions still compile and preserve values.
