---
id: einsum-nary-cost-model-spike
title: Design a layout-aware n-ary contraction cost model
status: done
priority: p1
dependencies: [performance-harness-foundation, einsum-binary-fastpaths, einsum-broadcast-gemm-spike]
related: [einsum-broadcast-gemm-lowering, einsum-nary-layout-aware-planner]
scopes: []
shared_scopes: [tests, benchmarks, docs, ticketing]
tags: [performance-0.2]
---
# Design a layout-aware n-ary contraction cost model

## Decision question

After pair execution is optimized, choose weighted greedy, bounded lookahead,
or exact planning below an arity threshold. The current immediate-output-first
model ignores copy bytes, broadcast expansion, layouts, peak live memory,
backend behavior, and downstream cost.

## Spike work

Build a bounded exhaustive-order oracle for arity 3–6 and preserve counterexamples
where current greedy loses. Own four whole-network scenarios only: linear chain,
balanced tree, broadcast-heavy network, and layout-hostile transposed network.
Record FLOPs, intermediate/output elements, estimated copy bytes, peak live
elements, planner time, and synchronized wall time. Pair microbenchmarks remain
owned by binary tickets.

## Red-first evidence

Decision tests initially target an absent cost-model interface and frozen
counterexamples where output size conflicts with FLOPs/copy/peak memory.

## Acceptance

- Written model choice, deterministic tie-breaking, planner-time budget, backend
  assumptions, and numerical reassociation policy.
- Refined implementation expectations include zero-K and broadcast-aware estimates.

## Non-goals

No unbounded optimal solver, arbitrary-arity exhaustive search, or compile-time
planning for runtime ellipsis shapes.

## Result

GO with a bounded hybrid planner on calibrated CPU backends. Use exact planning
only for arity three or four when the current greedy estimate is at least
100,000 FLOPs; otherwise retain current greedy. The exhaustive arity-three
through arity-six implementation remains a test oracle. See
`benchmarks/nary-cost-model-spike.md` for the checked model, stable tie-break,
175 us CPU budget, broadcast-GEMM no-go treatment, backend caveats, numerical
policy, and synchronized measurements.
