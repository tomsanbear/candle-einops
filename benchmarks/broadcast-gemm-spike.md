# Broadcast-aware GEMM strategy decision

Decision: **NO-GO** for replacing eager broadcast materialization with a
portable slice strategy in production. Keep eager materialization as the
broadcast fallback. Continue to use direct `matmul` only when neither operand
needs batch expansion and Candle accepts its layout; that class is already
owned by the binary fast paths.

## Frozen evidence

The repository harness ran exactly four cases with 501 synchronized samples on
Candle 0.11.0, Rust 1.94.1, macOS/aarch64 CPU at commit `0ab2aed`. Times are
medians in milliseconds. The pair is eager expanded GEMM versus the selected
prototype (slice for broadcast, direct for no-expansion).

| case | eager | candidate | eager / candidate | result |
| --- | ---: | ---: | ---: | --- |
| left broadcast, batch 32 | 3.934 | 4.110 | 0.957x | slice 4.5% slower |
| right broadcast, batch 32 | 3.879 | 4.041 | 0.960x | slice 4.2% slower |
| both broadcast, 8x8 batches | 7.790 | 8.147 | 0.956x | slice 4.6% slower |
| transposed, no expansion, batch 16 | 3.613 | 1.760 | 2.052x | direct 2.05x faster |

Bootstrap 95% intervals do not overlap for any pair. Results are advisory for
this CPU fingerprint and are not universal thresholds.

The structural contract records expanded operand copy bytes, peak additional
temporary elements (required final output excluded), and modeled GEMM
submissions. Submission counts describe public GEMM calls, not exact GPU
enqueue or kernel counts.

| case | eager copy | eager peak temp | eager GEMMs | candidate copy | candidate peak temp | candidate GEMMs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| left broadcast | 128 KiB | 32,768 | 1 | 0 | 32,768 | 32 |
| right broadcast | 128 KiB | 32,768 | 1 | 0 | 32,768 | 32 |
| both broadcast | 512 KiB | 131,072 | 1 | 0 | 65,536 | 64 |
| transposed, no expansion | 64 KiB | 16,384 | 1 | 0 | 0 | 1 |

CPU storage identity proves `broadcast_as(...).reshape(B, M, K)` allocates the
full expanded operand. The slice prototype uses `narrow`, one rank-2 `matmul`
per output batch matrix, and `stack`; it preserves values and gradients without
operand expansion. Direct transposed `matmul` preserves the view and avoids the
layout copy.

A more tempting candidate is invalid: passing a stride-zero batch view directly
to Candle 0.11 CPU `matmul` succeeds but produces incorrect tail values (the
frozen probe first differs at output element 2,016 for the left-broadcast case).
The benchmark-only direct helper therefore rejects every shape requiring
expansion instead of treating backend acceptance as correctness.

## Crossover and backend assessment

Eager expansion pays O(expanded operand elements) memory traffic but submits one
batched GEMM. Slicing removes that traffic while paying O(output batch matrices)
public operations and retaining slice outputs until `stack`. The cost curves
can cross only when copy pressure dominates dispatch and stacking overhead; the
three frozen CPU cases remain on the eager side even at 32 and 64 matrices.
Singleton/no-broadcast batches take direct GEMM, and zero batch or matrix extents
stay on the existing eager/zero-output behavior.

- **CPU:** slice is value/gradient correct but slower in all broadcast cases.
  Raw stride-zero batched GEMM is value-incorrect and cannot be a fallback.
- **CUDA:** Candle lowers matmul through strided batched GEMM and structurally
  accepts some zero batch strides, but no hardware correctness, synchronization,
  allocation, or submission evidence was collected. Slice would multiply host
  submissions by the batch count.
- **Metal:** Candle passes tensor strides to its GEMM kernel, but again there is
  no device evidence that broadcast strides are correct or faster. Slice has the
  same multi-submission risk.

All prototypes use public differentiable Candle operations and perform no host
transfer. Dtype and device support remains whatever Candle `matmul` provides.
The selected production behavior is therefore conservative: direct only with
no batch expansion; otherwise eager expansion. Preserve current errors and use
the existing path for zero dimensions, unsupported dtypes/layouts, and all
backend-specific failures.

## Reopening criteria

Reopen broadcast lowering only with synchronized accelerator evidence showing a
specific backend/shape crossover, or after an upstream Candle primitive offers
value-tested broadcast GEMM without concretization. Any implementation must
encode an explicit backend/shape/memory threshold, retain eager fallback, and
cover values and gradients before changing production.

## Reproduction

```console
python3 .github/scripts/run_benchmarks.py run --filter spike/broadcast-gemm --samples 501 --output target/benchmarks/broadcast-gemm-spike-501.json
```

The generated JSON remains under the ignored `target/benchmarks` directory and
is not committed.
