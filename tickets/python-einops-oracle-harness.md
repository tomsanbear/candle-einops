---
id: python-einops-oracle-harness
title: Establish a locked Python einops oracle harness
status: done
priority: p0
dependencies: []
related: []
scopes: [parity]
shared_scopes: [ticketing]
tags: [python-einops-parity]
---
# Establish a locked Python einops oracle harness

## Required outcome

Create a dedicated `uv` project under `parity/` which installs and locks
Python einops, NumPy, and Hypothesis. Provide a deterministic batched protocol
that accepts explicit input shapes, values, patterns, operations, and axis
sizes and returns either normalized shape/value results or normalized errors.

## Red-first work

Add protocol/contract tests before the implementation and capture their
failure with the missing harness. Then implement the smallest generic oracle
needed by the Rust parity slices.

## Acceptance

- The environment is reproducible from a committed `pyproject.toml` and `uv.lock`.
- Oracle execution calls the public Python einops API rather than reimplementing it.
- Request ordering and responses are deterministic and replayable by seed/case id.
- Scalars, zero-sized axes, finite floating values, and normalized failures are covered.

## Completion evidence

- Red: `python3 -m unittest discover -s parity/tests -v` failed because the
  specified `oracle` module did not exist.
- Green: `uv lock --project parity --check`, `uv sync --project parity
  --frozen`, and the six-test unittest/Hypothesis suite pass against the locked
  Python 3.12.10 environment.
- Scope guard passes for the complete `parity/` slice.
