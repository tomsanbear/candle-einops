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

First make backend selection real and fail-closed: the wrapper passes an
explicit backend and device index to the harness, which constructs
`Device::new_metal` or `Device::new_cuda` directly. Never use an availability
helper that silently falls back to CPU, and never emit a partial record when an
explicitly requested device is unavailable.

Both backends report synchronized host end-to-end time. CUDA may additionally
report stream-event device elapsed time and free/total memory snapshots using
public Candle/cudarc APIs. Metal may report current allocated size around a
fresh-device warmed sample. Memory values are labeled diagnostic because pools,
rounding, and other work make them unsuitable as exact allocation counts.

JSON uses validated metric envelopes: available or diagnostic metrics carry a
value and source; unavailable metrics carry null plus a non-empty reason.
Kernel, command-buffer, allocation, and enqueue counts remain unavailable with
an `external profiler required` reason. Precise Metal driver version is likewise
nullable with a reason; do not fill unknown metadata with a placeholder.
Documentation provides opt-in Nsight Systems/CUPTI and Metal capture recipes
rather than parsing proprietary traces in this repository.

## Red-first work

Contract tests first expose the current feature-only false-GPU path, then reject
simultaneous or mismatched backends, unavailable devices, missing
synchronization, dishonest metric envelopes, and fabricated counter values.
Compile optional feature paths where the environment permits. Automated
hardware tests may skip before emission; an explicit CLI GPU request fails with
no stdout record or output file when unavailable.

## Acceptance

- View-only scenarios are excluded from GPU timing because they enqueue no useful work.
- Scenario exclusion is explicit capability metadata, not name matching; CPU
  storage-inspection scenarios are marked GPU-unsupported.
- Device identity, driver/runtime, feature set, and synchronization mode are recorded.
- Captures and reports are ignored and excluded from packages.
- CPU-only users and normal `--all-features` workspace CI remain unaffected.

## Non-goals

- No Candle fork, private command-counter access, trace parser, GPU CI requirement,
  cross-device ranking, or memory/timing gate.
- GPU traces are optional escalation evidence only when one filtered warmed
  scenario has an unresolved kernel/command-structure decision. They are not
  required for this ticket and never populate harness counters.
