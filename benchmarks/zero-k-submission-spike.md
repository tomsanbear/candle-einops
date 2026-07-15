# Zero-size contraction submission spike

## Decision

**GO** for the public-operation `flatten -> cat -> sum -> broadcast` lowering.
It preserves both autograd edges while replacing two zero-anchor reductions and
their addition with one concatenation and one reduction. The broadcast remains
a view in both lowerings.

The candidate is deliberately expressed only with Candle tensor operations; it
does not add a custom kernel or backend-specific path.

## Structural comparison

| Lowering | Public data operations | Materialized temporary elements |
| --- | ---: | ---: |
| two empty-slice reductions + add | 3 | 2 scalar anchors |
| concatenate empty operands + sum | 2 | 0 |

`flatten_all` and the final `broadcast_as` are views for these contiguous
zero-length operands. Concatenating two empty tensors produces no data-bearing
temporary, while `sum_all` provides the single scalar zero anchor.

## CPU timing evidence

The standalone harness ran 1,001 paired, synchronized samples on Apple arm64
with Rust 1.94.1 and Candle 0.11.0. The candidate is the reference column.

| Output | Current median | Candidate median | Candidate/current |
| --- | ---: | ---: | ---: |
| `1 x 1` | 6.750 us | 3.583 us | 0.531 |
| `64 x 64` | 6.875 us | 3.666 us | 0.533 |
| `512 x 512` | 6.833 us | 3.625 us | 0.530 |

Output size does not materially affect either path because the result is a
broadcast view. The candidate removes about 47% of CPU dispatch latency in this
matrix and is structurally smaller, so it satisfies the implementation bar.
These measurements are CPU evidence only; no CUDA or Metal enqueue claim is
made without a suitable host and trace.

## Correctness boundary

Dedicated tests cover all three output sizes, exact zero values, both gradient
edges with their original zero-length shapes, supported dtype parity, and the
mixed-dtype error boundary. The replacement occurs after the existing equation,
rank, dimension, device, and dtype validations, so validation ordering is
unchanged.
