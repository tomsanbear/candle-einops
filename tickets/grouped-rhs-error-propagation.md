---
id: grouped-rhs-error-propagation
title: Preserve grouped RHS parser errors
status: done
priority: p0
dependencies: []
related: []
scopes: [macros]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
---
# Preserve grouped RHS parser errors

## Vertical outcome

The first invalid entry in a grouped RHS expression is reported even when later entries are valid.

## Red

Add compile-fail fixtures with invalid or duplicate middle entries in grouped RHS expressions and record the current erroneous acceptance/diagnostic.

## Green

Replace the overwriting fold with short-circuiting parsing that preserves the first `syn::Error`.

## Acceptance

- The new UI fixtures demonstrably fail before the parser fix.
- Middle-entry errors produce stable, local diagnostics after the fix.
- Valid grouped composition/repetition tests remain green.
