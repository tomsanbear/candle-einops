# Repeated-label diagonal lowering decision

Decision: **GO** for a narrow, portable fast path that gathers directly from a
contiguous operand's original flat layout when the current repeated-axis
permutation would force a dense copy. Keep the existing lowering for adjacent
diagonals and as the fallback. Do not add a device-global cache or a custom
kernel.

## Evidence

The repository wrapper ran both probes with 1,001 samples on Candle 0.11.0,
Rust 1.94.1, macOS/aarch64 CPU at commit `ddbcf8b`. Times are medians in
microseconds. `cached` measures only flat `index_select`; `index prep` measures
host offset construction plus `Tensor::from_vec` on the device. Their sum is a
conservative, separately sampled estimate for an uncached direct gather, not a
paired measurement.

| case | input / current copy / output / index elements | current | cached | index prep | uncached estimate | current / estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `i i`, n=16 | 256 / 0 / 16 / 16 | 4.625 | 2.417 | 1.208 | 3.625 | 1.28x |
| `i i`, n=64 | 4,096 / 0 / 64 / 64 | 6.209 | 3.542 | 3.542 | 7.084 | 0.88x |
| `i i`, n=256 | 65,536 / 0 / 256 / 256 | 16.375 | 9.958 | 12.667 | 22.625 | 0.72x |
| `i j i j`, n=4 | 256 / 256 / 16 / 16 | 22.750 | 2.167 | 1.375 | 3.542 | 6.42x |
| `i j i j`, n=8 | 4,096 / 4,096 / 64 / 64 | 105.917 | 3.792 | 2.708 | 6.500 | 16.30x |
| `i j i j`, n=16 | 65,536 / 65,536 / 256 / 256 | 779.916 | 10.208 | 7.792 | 18.000 | 43.33x |

The structural probe compares CPU storage identity. For `i j i j`, the current
permutation is a view, but flattening the now non-contiguous repeated axes
allocates and copies every input element. The direct candidate flattens the
unchanged contiguous input as a view and gathers only output elements. A second
probe confirms the current index tensor owns fresh storage on every call.

Correctness covers `i i -> i`, `i i i -> i`, and `i j i j -> i j`, including a
zero extent. Autograd gradients from the direct candidate exactly match the
current lowering. The triple-repeat case is intentionally not another timing
family. No GPU profiler data was collected; CPU storage identity and Candle's
backend implementations are the available structural evidence.

## Strategy assessment

- **Selected: one original-layout flat gather.** Compute contiguous row-major
  offsets for all repeated-label constraints, create one index tensor, perform
  differentiable `index_select`, and reshape to unique axes in first-appearance
  order. Combining multiple repeated labels into one gather avoids sequential
  intermediate diagonals and the full permuted-input copy.
- **Reusable device indices:** useful only when a caller-owned compiled plan has
  a lifetime and device identity. The current public API has no such owner.
  Cross-call caching would require invalidation, bounded capacity, and separate
  entries by equation, shape, index dtype, and device; the initial fast path
  must therefore build once per invocation and must not introduce a global
  cache. The measured uncached estimate is already decisive for interleaved
  layouts.
- **Current permute/reshape/index-select:** retain it for adjacent repeated axes,
  where reshape is a view and the uncached direct alternative regresses two of
  three scales, and for all unsupported cases.
- **Custom operation:** rejected. Candle `CustomOp1` requires separate CPU,
  CUDA, and Metal forwards plus an explicit backward implementation. That is
  disproportionate when the built-in differentiable operation is available.
- **Upstream primitive:** Candle 0.11.0 exposes no public tensor diagonal
  primitive. `strided_index` is a host layout iterator, not a differentiable
  device operation.

## Portability and boundaries

Candle 0.11.0 implements `index_select` for CPU, CUDA, and Metal, and records a
core autograd operation for its backward pass. Metal requires contiguous index
storage, which `Tensor::from_vec` supplies. This establishes API feasibility,
not equal performance across devices; each backend still needs correctness and
gradient tests, and device-specific measurements where hardware is available.

The fast path should require a contiguous original operand and offsets
representable by Candle's accepted index dtype. Non-contiguous operands,
overflow, or an unsupported backend/index combination fall back to the current
lowering without changing errors. Empty index tensors preserve zero extents.
Repeated-label extent validation remains ahead of either path. Index tensors
are local to an invocation and therefore cannot outlive or mismatch their
device.

## Reproduction

```console
python3 .github/scripts/run_benchmarks.py run --filter spike/diagonal --samples 1001 --output target/benchmarks/diagonal-spike-1001.json
python3 .github/scripts/run_benchmarks.py probe --filter spike/diagonal --samples 1001 --output target/benchmarks/diagonal-index-1001.json
```

Generated JSON stays under the ignored `target/benchmarks` directory and is not
committed.
