---
id: benchmark-methodology-hardening
title: Harden benchmark inputs and timing methodology
status: todo
priority: p0
dependencies: [performance-harness-foundation]
related: [einsum-binary-fastpaths]
scopes: [benchmarks]
shared_scopes: [tests, ticketing]
paths: []
tags: [performance-0.2]
---
## Required outcome\n\nMake performance correctness inputs and timing methodology discriminating without adding scenario families.\n\n## Red-first contract\n\n- A benchmark contract test rejects all-constant operands for mechanisms where axis or operand swaps would otherwise pass.\n- Binary GEMM dimensions include non-square M, K, and N values and deterministic nonuniform operands.\n- A timing contract proves Criterion does not nest the JSON harness clock/synchronization wrapper.\n- Paired JSON sampling alternates or randomizes library/reference order deterministically to avoid a fixed first-run bias while preserving the schema.\n\n## Acceptance\n\n- Direct Candle references still use identical input tensors and untimed correctness remains outside samples.\n- Criterion measures the operation directly with its own timing model; device synchronization remains correct for asynchronous backends.\n- JSON results remain reproducible and record the sampling order policy.\n- Existing product, diagonal, binary, and future scenario registrations remain intact.\n- No new benchmark scenarios or hosted-CI timing threshold.\n\n## Evidence\n\nIndependent review of the binary fast-path benchmarks found square all-ones operands could conceal transposes or operand swaps, and Criterion currently includes the internal clock wrapper in each iteration.
