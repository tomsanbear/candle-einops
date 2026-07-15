# Multi-axis extrema lowering spike

## Decision

**GO only when adjacent min/max axes collapse as a storage-sharing contiguous
view.** Keep sequential lowering when the collapse would materialize input
elements. This is a structural boundary, not a timing threshold.

The candidate reshapes the adjacent reduced run into one axis and issues one
public Candle `min` or `max`. Production currently issues one reduction for
each axis. Sum and mean already have their own fusion path; product and mean
through extrema-style collapsing are out of scope.

## Frozen matrix

All cases use 98,304 deterministic `f32` elements and 501 alternating paired
CPU samples on Darwin arm64, Rust 1.94.1, and Candle 0.11. Values match exactly.
Unique-extrema gradient tests, `f64`, `u32`, and empty-axis errors also match.

| Layout / operation | Sequential median | Collapsed median | Sequential / collapsed | Sequential submissions | Collapsed submissions | Candidate copied elements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous trailing min | 1.251 ms | 0.806 ms | 1.551x | 2 | 1 | 0 |
| contiguous trailing max | 1.288 ms | 0.846 ms | 1.524x | 2 | 1 | 0 |
| contiguous leading min | 0.855 ms | 0.806 ms | 1.062x | 2 | 1 | 0 |
| contiguous leading max | 0.933 ms | 0.828 ms | 1.126x | 2 | 1 | 0 |
| strided trailing min | 2.376 ms | 2.193 ms | 1.083x | 2 | 1 | 98,304 |
| strided trailing max | 2.289 ms | 2.112 ms | 1.084x | 2 | 1 | 98,304 |

The strided result is a NO-GO despite its local CPU timing: it trades one
reduction submission for a full-input copy, and that trade is shape-, dtype-,
and backend-dependent. The runtime implementation must prove the collapsed
reshape shares storage before selecting it and otherwise preserve the existing
sequential path.

These submission and copy counts describe public operations and materialized
elements. They are not profiler kernel counters and make no GPU claim.
