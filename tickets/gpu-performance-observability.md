---
id: gpu-performance-observability
title: Add honest optional GPU performance observability
status: todo
priority: p2
dependencies: [performance-harness-foundation, repeat-broadcast-view-lowering, homogeneous-reduction-fusion, einsum-binary-fastpaths]
related: [advisory-performance-comparison-ci]
scopes: [benchmarks]
shared_scopes: [docs, ticketing]
tags: [performance-0.2]
---
# Add honest optional GPU performance observability

## Required outcome

Extend the standalone harness with mutually exclusive optional Metal and CUDA
execution without exposing backend features on the published crate or normal
workspace test matrix.

## Measurement contract

Both backends report synchronized host end-to-end time. CUDA may additionally
report stream-event device elapsed time and free/total memory snapshots using
public Candle/cudarc APIs. Metal may report current allocated size around a
fresh-device warmed sample. Memory values are labeled diagnostic because pools,
rounding, and other work make them unsuitable as exact allocation counts.

JSON fields for unavailable kernel/command counts are explicitly null with a
reason. Documentation provides opt-in Nsight Systems/CUPTI and Metal capture
recipes rather than parsing proprietary traces in this repository.

## Red-first work

Contract tests first reject simultaneous backends, unavailable devices, missing
synchronization, and fabricated counter values. Compile optional feature paths
where the environment permits; unavailable hardware is a clear skip before any
partial result is emitted.

## Acceptance

- View-only scenarios are excluded from GPU timing because they enqueue no useful work.
- Device identity, driver/runtime, feature set, and synchronization mode are recorded.
- Captures and reports are ignored and excluded from packages.
- CPU-only users and normal `--all-features` workspace CI remain unaffected.

## Non-goals

- No Candle fork, private command-counter access, trace parser, GPU CI requirement,
  cross-device ranking, or memory/timing gate.
