---
id: ci-supply-chain-platform-hardening
title: Harden CI permissions dependencies and platform coverage
status: in-progress
priority: p1
dependencies: []
related: []
scopes: [tooling]
shared_scopes: [ticketing]
paths: []
tags: [hardening-0.2]
claimed_from: todo
assignee: codex-ci-hardening
lease_expires_at: 1784126863
---
# Harden CI permissions dependencies and platform coverage

## Vertical outcome

CI uses least privilege, immutable action references, an explicit dependency policy, and at least one non-Linux platform.

## Red

Add configuration validation that demonstrates current mutable action references, broad default permissions, and absent dependency-policy/platform gates.

## Green

Set read-only permissions, disable persisted checkout credentials, pin actions by commit, add a configured dependency/license/advisory check, and add a cost-conscious macOS job. Document the lockfile/MSRV resolution policy.

## Acceptance

- Workflow/config lint passes.
- Dependabot continues updating pinned actions.
- Dependency exceptions are narrow, reasoned, and time-bounded where supported.
- MSRV remains explicitly exercised and macOS runs a focused representative suite.
