---
id: advisory-performance-comparison-ci
title: Add reproducible advisory performance comparisons
status: done
priority: p2
dependencies: [performance-harness-foundation, repeat-broadcast-view-lowering, homogeneous-reduction-fusion, einsum-binary-fastpaths]
related: [benchmark-device-support]
scopes: [tooling]
shared_scopes: [benchmarks, docs, ticketing]
tags: [performance-0.2]
---
# Add reproducible advisory performance comparisons

## Required outcome

Keep required PR CI structural and deterministic while providing a manual
base/head comparison workflow for evidence-driven performance review.

## Design

Required CI compiles the locked standalone harness and runs only correctness,
schema, synchronization, and materialization smoke contracts. A manual workflow
accepts exact base/head SHAs, checks them out in isolated worktrees, alternates
execution order on the same runner, collects at least five independent process
samples, and uploads versioned JSON plus secondary Criterion artifacts.

The job summary marks a result advisory only when the paired median movement is
both greater than 10% and greater than 1 microsecond and its confidence interval
excludes 5%. Fingerprint mismatch yields `incomparable`, not failure. A repeated
observation across three runs is required before filing a regression.

## Red-first work

Add a policy validator that initially fails because the compile/smoke gate,
manual inputs, alternating comparison, fingerprint checks, and advisory-only
semantics are absent. Make policy and workflow green afterward.

## Acceptance

- No timing result can fail required PR CI or block a release.
- Checkouts are read-only with pinned actions and no push/comment/issue mutation.
- Artifacts expire; results are not committed to a benchmark-data branch.
- Promotion to a real timing gate requires a pinned runner, at least 30 baseline
  runs, stable variance, and a separate explicit policy ticket.

## Non-goals

- No scheduled hosted-runner trend job, badge, PR bot comment, or automatic issue.
- No GPU threshold; GPU observability remains opt-in and diagnostic.

## Result

- Required CI compiles and smoke-tests the locked standalone harness without
  comparing timings.
- A pinned, manual-only workflow collects five alternating process pairs from
  exact commit objects in clean detached worktrees, with Cargo output stored
  outside each worktree and expiring JSON/Criterion artifacts uploaded for
  inspection.
- The versioned comparison report matches scenario IDs and declares schema,
  workload, sample-count, sampling-policy, and environment differences
  incomparable. Its paired thresholds remain advisory-only and require three
  independent observations before follow-up.
- Contract tests and policy validators cover report behavior, workflow
  isolation, immutable action references, and the absence of mutation or
  timing gates.
