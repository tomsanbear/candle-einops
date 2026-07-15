---
id: python-einops-parity-ci
title: Gate CI on reproducible Python einops parity
status: done
priority: p1
dependencies: [python-einops-rearrange-properties, python-einops-repeat-reduce-properties]
related: []
scopes: [tooling, docs, parity]
shared_scopes: [ticketing]
tags: [python-einops-parity]
---
# Gate CI on reproducible Python einops parity

## Required outcome

Add a pinned, isolated CI parity job using the repository's `uv` lockfile and
document local execution, seed replay, and dependency-update procedure.

## Red-first work

Add a policy validator which fails while the parity job, frozen environment,
or replay documentation is absent. Make policy and workflow green afterward.

## Acceptance

- CI uses a pinned uv action/tool version and `uv sync --frozen`.
- The parity job runs separately from ordinary Rust tests and has a bounded timeout.
- Published Rust artifacts do not include or depend on the Python environment.
- Contributor documentation provides one-command execution and seed replay.

