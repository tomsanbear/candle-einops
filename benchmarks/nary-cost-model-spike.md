# N-ary contraction cost-model spike

## Decision

**GO with a bounded hybrid planner on calibrated CPU backends.** Keep the
current immediate-output greedy planner for small contractions and for
uncalibrated backends. For arity at most four, use the exact planner only when
the current greedy plan estimates at least 100,000 FLOPs. The exhaustive
arity-five and arity-six implementation is a test oracle, not a runtime path.

This boundary keeps the exact search away from cases where its roughly 0.1 ms
host cost can erase the execution benefit. It selects the exact plan for the
balanced and broadcast-heavy fixtures and retains current greedy for the
linear and layout-hostile fixtures.

## Model contract

All arithmetic is checked `u128`; overflow is an error and must fall back to
the current planner in production. A pair estimate records:

- FLOPs: the product of every axis extent in the union of the two operands;
- output elements: the product of retained-axis extents;
- copy bytes: the full expanded operand size for an eager broadcast
  materialization plus the full operand size for a conservative non-contiguous
  layout materialization;
- submissions: one contraction submission; and
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
pair FLOPs, then the same stable member masks.

The copy estimate is deliberately conservative. A direct binary fast path can
sometimes consume a transposed layout without copying, but the general n-ary
path may canonicalize it. The broadcast fixture follows the broadcast-GEMM
spike's no-go result: it models a full 128 KiB eager expansion and one GEMM
submission, not per-slice submissions and not an unsupported stride-zero
batched GEMM.

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
| linear chain | 25,322 | 17,447 | 16,500 | 9,375 | 2,075 | 1,875 | 18.625 us | 112.958 us |
| balanced tree | 214,208 | 88,064 | 139,328 | 18,432 | 18,432 | 16,640 | 18.500 us | 112.042 us |
| broadcast-heavy | 752,832 | 517,952 | 505,280 | 291,840 | 31,424 | 25,504 | 24.292 us | 137.958 us |
| layout-hostile | 29,522 | 21,647 | 16,500 | 9,375 | 2,075 | 1,875 | 18.541 us | 111.958 us |

| Fixture | Current wall time (95% CI) | Selected wall time (95% CI) | Current / selected |
| --- | ---: | ---: | ---: |
| linear chain | 142.958 us (142.791–143.083) | 142.958 us (142.792–143.125) | 1.0000x |
| balanced tree | 566.708 us (565.417–568.125) | 333.625 us (332.250–334.625) | 1.6986x |
| broadcast-heavy | 3,003.458 us (2,998.959–3,009.292) | 2,811.958 us (2,804.250–2,817.459) | 1.0681x |
| layout-hostile | 233.167 us (232.583–234.041) | 232.583 us (232.083–233.292) | 1.0025x |

The selected-path benchmark excludes planner-decision timing from both sides;
planner cost is reported separately above. This isolates execution-path value
and makes the runtime threshold an explicit implementation concern.

## Backend and numerical policy

The threshold and weights are CPU calibration data, not universal constants.
CUDA and Metal must retain current greedy until synchronized device
measurements establish their own crossover: host planning is proportionally
more expensive for fast device contractions, and copy/submission costs differ
by backend. The planner-time budget for the frozen arity-four CPU cases is
150 us; exceeding it falls back to current greedy.

Floating-point reassociation need not be bitwise identical. Forward values and
input gradients must match the current association within the benchmark's
0.2% relative tolerance. Production tests must cover changed-path forward and
gradient parity before enabling the planner. Integer contraction support is
outside this spike and must not be inferred from reassociation tests.

## Reproduction

```sh
CARGO_TARGET_DIR=target/benchmarks cargo +1.94 run --locked --manifest-path benchmarks/Cargo.toml --bin nary_cost_probe -- --samples 1001 --output target/benchmarks/nary-cost-planner-1001.json
python3 .github/scripts/run_benchmarks.py run --filter spike/nary-cost --samples 501 --output target/benchmarks/nary-cost-final-wall-501.json
```

Generated JSON remains under `target/` and is not a repository artifact.
