---
id: macro-expansion-hygiene
title: Harden generated identifiers and absolute paths
status: done
priority: p1
dependencies: []
related: []
scopes: [macros]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Harden generated identifiers and absolute paths

## Vertical outcome

Caller bindings and locally shadowed prelude names cannot capture or break generated expansion code.

## Red

Add downstream fixtures that use braced expressions named `input`, `input_shape`, and `input_ignored_len`, and shadow `Vec` and `std`. Observe the pre-fix compile failure or wrong capture.

## Green

Generate private mixed-site identifiers and fully qualify standard-library paths. Cover dependency aliases that require raw identifiers if Cargo permits them.

## Acceptance

- Adversarial downstream fixtures compile and execute their assertions.
- Ordinary and renamed dependency fixtures stay green.
- No unqualified generated `Vec` or relative `std` paths remain.
