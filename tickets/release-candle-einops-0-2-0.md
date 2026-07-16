---
id: release-candle-einops-0-2-0
title: Release candle-einops 0.2.0
status: in-progress
priority: p0
dependencies: [publish-reproducible-performance-report]
related: []
scopes: [docs, tooling, macros, runtime]
shared_scopes: [ticketing]
paths: []
tags: [release-0.2.0]
claimed_from: todo
assignee: codex-root
lease_expires_at: 1784227445
---
Prepare, merge, publish, and verify candle-einops 0.2.0 and candle-einops-macros 0.2.0. Update the release checklist for the reporting environment, validate the exact branch and merged main artifacts, push and merge through a fully green GitHub PR, publish the macro crate first and wait for its index entry, publish the root crate second, then create the v0.2.0 tag and GitHub release and record downstream/docs verification. Crates.io publication remains an explicit irreversible confirmation gate.
