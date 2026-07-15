---
id: extend-permute-compose-fusion-across-runtime-shapes
title: Extend permute-compose fusion across runtime shapes
status: todo
priority: p1
dependencies: [add-layout-aware-binary-einsum-operand-packing]
related: [fused-permute-compose-layout]
scopes: [runtime, macros]
shared_scopes: [tests, benchmarks, ticketing]
paths: []
tags: [kernel-enqueue-hardening]
---
# Extend permute-compose fusion across runtime shapes

## Goal

Extend the proven permute-and-compose copy avoidance beyond the current
fully-static, transformation-isolated macro slice.

## Work

- Add red codegen and storage tests for runtime ellipsis/group metadata and for
  permute-plus-compose after a homogeneous reduction.
- Generate checked runtime permutation/group metadata and refresh shape data at
  the exact transformation boundary.
- Select `Backend::permute_and_compose` only when semantic ordering is unchanged;
  retain the current operation sequence otherwise.
- Benchmark only cases where the old path demonstrably materializes.

## Acceptance

- Selected runtime/post-reduction cases avoid the historical layout copy.
- Third-party `Backend` implementations retain the default operation contract.
- Ellipsis, zero/singleton extents, gradients, and error ordering are covered.
- No redundant benchmark families are added.
