# Performance harness

This standalone, unpublished crate owns measurement methodology. It deliberately
contains an untracked plumbing scenario, product-reduction and diagonal-lowering
spikes, plus paired binary einsum fast-path scenarios for Hadamard, outer,
rank-2 GEMM, and rank-3 batched GEMM. Homogeneous reduction fusion owns four
paired sum/mean scenarios across contiguous trailing and strided non-adjacent
multi-axis layouts. Repeat broadcast lowering owns one large single-axis family
and one two-axis family, each split into view-construction and
materializing-consumption modes.
The zero-length contraction family compares the historical two-anchor graph
with a concatenate-and-sum candidate at three output sizes, records public
operation and temporary-element counts, and verifies both autograd edges. Its
GO decision and CPU evidence are recorded in `zero-k-submission-spike.md`.
Identity reshape elision owns one contiguous control and one non-contiguous
input at the same size, each measured as view construction and explicit
contiguous consumption.
The permute-plus-composition layout spike owns only `c (a b)` and
`n (h w) c` candidates at equal element counts. Each compares the current
permute-then-reshape copy with a spike-only public-operation prototype that
reshapes contiguous groups before permuting them, in construction and explicit
contiguous-consumption modes. The construction result exposes copy avoidance;
the consumption result shows when the same copy is merely deferred.
The extended-composition matrix applies the same two-mode distinction only to
the newly eligible runtime-ellipsis and post-reduction paths; it intentionally
does not add redundant shape-size variants.
The extrema spike compares sequential two-axis min/max lowering with a single
collapsed reduction across adjacent trailing, adjacent leading, and strided
trailing layouts; it does not duplicate sum/mean or add shape-size variants.

Use the repository wrapper for every supported operation:

```console
python3 .github/scripts/run_benchmarks.py compile
python3 .github/scripts/run_benchmarks.py smoke
python3 .github/scripts/run_benchmarks.py run --filter rearrange/view-permute --output target/benchmarks/result.json
python3 .github/scripts/run_benchmarks.py probe --filter spike/diagonal --output target/benchmarks/index-preparation.json
python3 .github/scripts/run_benchmarks.py capture --backend metal --filter reshape/identity/non-contiguous/consume --operation library
python3 .github/scripts/run_benchmarks.py capture --backend cuda --filter reshape/identity/non-contiguous/consume --operation reference
```

The wrapper pins Rust 1.94, uses this crate's committed lockfile, and shares the
root ignored `target/benchmarks` directory. Select `--backend cpu|metal|cuda`,
`--cpu-implementation baseline|accelerate|mkl`, and `--device-index` explicitly
when the defaults are not appropriate. `run` selects tracked scenarios by
substring. `smoke` additionally opts into the untracked plumbing fixture.

`capture` requires a GPU backend, a filter that resolves to exactly one
supported scenario, and `--operation library|reference`. It performs three
synchronized warmups by default (`--warmups` changes this), then captures one
synchronized operation. Metal capture uses Candle's `MTLCaptureManager` path
and writes an Xcode `.gputrace` bundle. CUDA capture builds the locked harness
and invokes Nsight Systems with the exact `cudaProfilerApi` range, CPU sampling
disabled, and no proprietary-report parsing. For CUDA, `--output` is the report
prefix to which Nsight Systems adds `.nsys-rep`; `nsys` must be on `PATH`.

`probe` is a CPU-only companion for the repeated-label diagonal spike. It
isolates host index construction plus device upload and records the input,
materialized-copy, output, and index element counts beside its timing. This
separates reusable setup cost from the paired current/cached operation timing.

Each scenario has an immutable id, deterministic setup, a library operation, a
direct Candle reference, an untimed correctness check, and elements/bytes with
optional FLOPs. A sample synchronizes immediately before the clock starts and
after its black-boxed output is produced. The output remains alive through the
second synchronization.

Paired JSON samples alternate deterministically between library-first and
reference-first execution. Each record's `sampling_order_policy` field records
that policy. The advisory comparator still reads legacy v1 arrays, treating a
missing policy as fixed library-first order, but it never compares v1 and v2
results as if they were equivalent. Criterion invokes the operations directly
under its own timer and synchronizes after every output, rather than nesting
the JSON harness clock.

The versioned JSON document is the automation contract. Schema v2 owns one run
identity, paired library/reference records, explicit unsupported-scenario
entries, a deterministic 95% bootstrap interval, ratios, work units, and
per-record device diagnostics. Every diagnostic is either a value with its
source or an explicit unavailable reason. Metal reports physical name,
registry ID, and current allocation; CUDA reports physical name and UUID,
installed NVIDIA driver, CUDA runtime, and free/total/derived allocation.
Synchronized host timing remains the primary timing. Device-event, kernel,
command-buffer, and enqueue counts remain unavailable in JSON until a native
capture supplies them. Criterion output and native GPU traces are secondary
and must not be parsed by automation or committed.

## Advisory base/head comparisons

The manual `Advisory performance comparison` workflow accepts exact base and
head commit SHAs plus an optional scenario-id filter. It creates detached
worktrees on one runner, alternates base/head order across five independent
harness processes, and keeps all Cargo build output outside those worktrees so
their complete Git status remains clean. It uploads the primary versioned JSON
records, a versioned comparison report, its job summary, and secondary
Criterion output. Artifacts expire after 14 days and are never written to a
benchmark-data branch.

The comparison matches scenario IDs and rejects changed work units, sample
counts, sampling order, unsupported or mismatched schemas, and environment
fingerprints that differ in anything other than the expected Git SHA. Such
results are `incomparable`, not failures.
For comparable scenarios, a movement is highlighted only when the paired
process median exceeds both 10% and 1 microsecond and its deterministic 95%
interval lies beyond 5%. Every result is advisory: it cannot fail required CI,
block a release, file an issue, or comment on a pull request. File a regression
only after the same observation appears in three independent workflow runs.

The required pull-request job only runs `compile` and `smoke`; it never compares
timings. Promoting timing to a gate requires a separate policy ticket, a pinned
runner, at least 30 baseline runs, and demonstrated stable variance. There is no
universal baseline, scheduled hosted-runner trend, or GPU threshold.

For local inspection, collect at least five files for each exact commit and run:

```console
python3 .github/scripts/compare_benchmarks.py \
  --base-sha 1111111111111111111111111111111111111111 \
  --head-sha 2222222222222222222222222222222222222222 \
  --base target/benchmarks/base/*.json \
  --head target/benchmarks/head/*.json \
  --output target/benchmarks/comparison-v1.json
```

The binary outer-product and GEMM scenarios changed from square, constant
operands to deterministic nonuniform operands with non-square M/K/N dimensions.
Their stable ids preserve registration continuity, but results produced before
this methodology change are different workloads and must not be compared with
new medians unless the recorded work units also match.

CPU is the default backend. Metal, CUDA, Accelerate, and MKL are mutually
exclusive feature builds. View-only construction scenarios are retained for
CPU analysis but explicitly skipped on accelerators because they enqueue no GPU
work; their materializing consumers remain supported. Use the capture command
instead of wrapping ad hoc binaries so warmup and exact-operation boundaries
stay consistent. There is no universal timing baseline or CI regression gate.
