# N-ary contraction cost-model spike

## Decision

**GO with a bounded hybrid planner on calibrated CPU backends.** Keep the
current immediate-output greedy planner for small contractions and for
uncalibrated backends. For arity three or four, use the exact planner only when
the current greedy plan estimates at least 100,000 FLOPs. The exhaustive
arity-five and arity-six implementation is a test oracle, not a runtime path.

This boundary keeps the exact search away from cases where its roughly 0.1 ms
host cost can erase the execution benefit. It selects the exact plan for the
balanced and broadcast-heavy fixtures and retains current greedy for the
linear and layout-hostile fixtures.

## Model contract

All arithmetic is checked `u128`; overflow is an error and must fall back to
the current planner in production. A pair estimate records public-operation
estimates:

- work: the input traversal for each pair-local pre-reduction plus the
  multiply/GEMM axis product (stored in the historical `flops` field);
- output elements: the product of retained-axis extents;
- copy bytes: the full expanded operand size only when the production layout
  classifier selects eager broadcast or packing materialization;
- submissions: one per pair-local reduction plus the multiply/GEMM operation,
  or the three public operations used by a zero-size contraction anchor; and
- peak live elements: all live inputs plus the newly allocated pair output.

The network score uses the sum of pair-output elements (including the final
pair), reports final-output elements separately, sums FLOPs/copy bytes/
submissions, and takes the maximum peak-live value. The calibrated CPU weights
are:

| Term | Weight |
| --- | ---: |
| FLOP | 1 |
| copied byte | 1 |
| pair-output element | 2 |
| peak-live element | 2 |
| submission | 1,024 |

The exact oracle enumerates all pair sequences for arity three through six.
It minimizes total weighted score, then breaks equal scores
lexicographically by the sequence of stable original-operand member bitmasks.
The current-model comparator minimizes immediate pair-output elements, then
pair FLOPs, then the production planner's stable ordinal and live-index key.

The benchmark oracle and runtime planner now call the same pure production
lowering classifier. The broadcast fixture therefore models a full 128 KiB
eager expansion and one GEMM submission, while the rank-two transposed control
correctly records no copy because whole-group packing recovers a view. Pair
outputs are contiguous except for the graph-preserving zero-size broadcast,
which is explicitly unsupported as a subsequent exact-planner layout.

Zero-length contracted axes contribute zero FLOPs while output/intermediate
allocation and peak-live terms remain visible. A production planner must
preserve the existing zero-output behavior and Candle error and dtype
semantics.

## Frozen CPU results

Measured on Darwin 24.6.0 arm64 with Rust 1.94.1 and Candle 0.11. Planner
medians use 1,001 samples. Whole-network medians and 95% confidence intervals
use 501 synchronized samples.

| Fixture | Current score | Exact score | Current FLOPs | Exact FLOPs | Current peak | Exact peak | Greedy planner | Exact planner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear chain | 25,322 | 17,447 | 16,500 | 9,375 | 2,075 | 1,875 | 72.875 us | 397.875 us |
| balanced tree | 214,208 | 88,064 | 139,328 | 18,432 | 18,432 | 16,640 | 72.583 us | 396.834 us |
| broadcast-heavy | 752,832 | 517,952 | 505,280 | 291,840 | 31,424 | 25,504 | 92.708 us | 503.666 us |
| layout-hostile | 25,322 | 17,447 | 16,500 | 9,375 | 2,075 | 1,875 | 74.125 us | 402.042 us |

| Fixture | Current wall time (95% CI) | Selected wall time (95% CI) | Current / selected |
| --- | ---: | ---: | ---: |
| linear chain | 166.666 us (166.416–167.000) | 241.167 us (240.750–241.666) | 0.6911x |
| balanced tree | 586.625 us (584.750–587.708) | 288.958 us (288.250–289.875) | 2.0301x |
| broadcast-heavy | 3,012.750 us (3,008.792–3,017.375) | 2,740.750 us (2,737.459–2,746.208) | 1.0992x |
| layout-hostile | 168.625 us (168.333–168.833) | 245.417 us (245.167–245.791) | 0.6871x |

The selected-path benchmark executes the production selector and executor,
while the current side executes the frozen greedy plan. Planner probes call a
benchmark-feature-gated production seam and report its full preparation and
selector p95, including the metadata snapshot, greedy threshold pass, and exact
search when selected. Repeated exact signatures use a bounded thread-local
member-sequence cache, but rebuild all three step estimates through the shared
classifier on every hit. Each record reports `budget_us: 175` and `budget_met`,
and the probe exits unsuccessfully if any fixture exceeds that budget.
Copy-byte and submission values are public-operation estimates; they are not
profiler counters or claims about backend kernel fusion.

The production-seam debug probe on Darwin arm64 with Rust 1.94.1 measured
1,001-sample p95 values of 79.709 us (linear), 34.167 us (balanced), 45.083 us
(broadcast-heavy), and 84.791 us (layout-hostile). Every record reported
`budget_met: true` against `budget_us: 175`.

## Backend and numerical policy

The threshold and weights are CPU calibration data, not universal constants.
CUDA and Metal must retain current greedy until synchronized device
measurements establish their own crossover: host planning is proportionally
more expensive for fast device contractions, and copy/submission costs differ
by backend. The combined selector p95 budget for the frozen arity-four CPU
cases is 175 us. It is a benchmark acceptance boundary, never a wall-clock
cutoff in production; deterministic structural eligibility alone selects the
runtime path.

Floating-point reassociation need not be bitwise identical. Forward values and
input gradients must match the current association within the benchmark's
mixed tolerance `0.002 * max(abs(reference), 1)`. Production tests must cover changed-path forward and
gradient parity before enabling the planner. Integer contraction support is
outside this spike and must not be inferred from reassociation tests.

## Reproduction

```sh
CARGO_TARGET_DIR=target/benchmarks cargo +1.94 run --locked --manifest-path benchmarks/Cargo.toml --bin nary_cost_probe -- --samples 1001 --output target/benchmarks/nary-cost-planner-1001.json
python3 .github/scripts/run_benchmarks.py run --filter spike/nary-cost --samples 501 --output target/benchmarks/nary-cost-final-wall-precomputed-501.json
```

Generated JSON remains under `target/` and is not a repository artifact.
