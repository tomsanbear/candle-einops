![candle-einops](https://github.com/tomsanbear/candle-einops/workflows/CI/badge.svg)
[![crates](https://img.shields.io/crates/v/candle-einops)](https://crates.io/crates/candle-einops)
[![docs](https://img.shields.io/docsrs/candle-einops)](https://docs.rs/candle-einops)

# candle-einops

`candle-einops` provides compile-time tensor rearrange, reduce, repeat, and
Einstein summation expressions for
[Candle](https://github.com/huggingface/candle). It is based on
the original Rust [einops](https://github.com/VasanthakumarV/einops) macro and
the Python [einops](https://github.com/arogozhnikov/einops) notation.

Version 0.2 targets Candle 0.11, uses Rust 2024, and requires Rust 1.94 or newer.

```toml
[dependencies]
candle-core = "0.11"
candle-einops = "0.2"
```

## Example

`einops!` returns `candle_core::Result<Tensor>`, so callers can propagate Candle
shape, axis, dtype, and device errors with `?`.

```rust
use candle_core::{Device, Result, Tensor};
use candle_einops::einops;

fn main() -> Result<()> {
    let input = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 4))?;
    let output = einops!("batch height width -> width batch height", &input)?;

    assert_eq!(output.dims(), &[4, 2, 3]);
    Ok(())
}
```

## Expression guide

The left side of `->` describes the input axes and the right side describes the
output axes. Transformations can be combined in one expression.

| Operation | Example | Shape change |
| --- | --- | --- |
| Transpose | `h w c -> c h w` | `(28, 28, 3)` to `(3, 28, 28)` |
| Compose | `b h w c -> (b h) w c` | `(10, 28, 28, 3)` to `(280, 28, 3)` |
| Decompose | `(b1:2 b2) h w c -> b1 b2 h w c` | `(10, 28, 28, 3)` to `(2, 5, 28, 28, 3)` |
| Reduce | `mean(b) h w c -> h w c` | `(10, 28, 28, 3)` to `(28, 28, 3)` |
| Repeat | `h w c -> h copy:5 w c` | `(28, 28, 3)` to `(28, 5, 28, 3)` |
| Squeeze | `1 h w c -> h w c` | `(1, 28, 28, 3)` to `(28, 28, 3)` |

Supported reductions are `min`, `max`, `sum`, `mean`, and `prod`. A reduction
can cover consecutive axes, as in `batch sum(row column) -> batch`. Use `..` to
preserve or reduce a runtime number of axes.

Axis sizes may be literals (`copy:5`) or Rust expressions in braces. For
example, with `let copies = 5`, `h w -> h {copies} w` inserts an axis of that
length. New named axes and decomposed groups require an explicit size whenever
it cannot be inferred.

Invalid expressions are reported by the procedural macro at compile time.
Tensor-dependent failures are returned as Candle errors at runtime.

## Einsum guide

`einsum!` accepts an explicit-output equation followed by one tensor expression
per comma-separated input list. It returns `candle_core::Result<Tensor>` and
evaluates every operand once, from left to right. Labels are
whitespace-delimited, exactly one `->` is required, and labels omitted from the
output are summed.

The supported contract includes:

- Unary permutation and reduction: `"rows columns -> columns rows"`.
- Binary broadcasting, outer products, and GEMM-lowered contraction:
  `"row inner, inner column -> row column"`.
- A single ellipsis (`..`) per axis list for right-aligned variable-rank
  broadcasting or reduction: `".. feature -> feature"`.
- Repeated labels within an operand for diagonal extraction and traces:
  `"index index -> index"` and `"index index ->"`.
- Arbitrary n-ary equations with deterministic, shape-aware greedy planning:
  `"row inner, inner column, column -> row"`.

```rust
use candle_core::{Device, Result, Tensor};
use candle_einops::einsum;

fn main() -> Result<()> {
    let left = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let right = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
    let product = einsum!("row inner, inner column -> row column", &left, &right)?;
    assert_eq!(product.to_vec2::<f32>()?, [[22., 28.], [49., 64.]]);

    let weights = Tensor::new(&[1f32, 1.], &Device::Cpu)?;
    let projected = einsum!(
        "row inner, inner column, column -> row",
        &left,
        &right,
        &weights,
    )?;
    assert_eq!(projected.to_vec1::<f32>()?, [50., 113.]);
    Ok(())
}
```

Retained labels shared by operands broadcast when their extents are equal or
one. Repeated occurrences of a label in one operand must have equal extents.
Scalars and zero-sized axes are supported. Einsum never casts or moves tensors:
multi-operand inputs must have the same dtype and device, and unsupported
Candle operations return contextual errors.

Axes introduced by `einops!` repeat patterns are returned as broadcast views.
These tensors can be non-contiguous and share storage with the input; operations
that require contiguous storage may materialize them when consumed.

## Migrating from 0.1

Version 0.2 contains four compatibility changes:

- Candle is upgraded from 0.6 to 0.11.
- `einops!` now returns `candle_core::Result<Tensor>` instead of panicking on a
  backend error. Add `?` or handle the result explicitly.
- Custom `Backend` implementations must return `candle_core::Result` from
  `reshape`, `transpose`, `reduce_axes`, and `add_axes`. `shape` remains
  infallible.
- `einsum!` is now a supported public API for unary, binary, ellipsis,
  diagonal, and arbitrary n-ary equations.

Dependency renaming is supported. For example,
`tensor-ops = { package = "candle-einops", version = "0.2" }` can be imported
with `use tensor_ops::{einops, einsum};`.

Because generated expansions call a private runtime ABI, direct users of the
implementation crate must keep `candle-einops-macros` at exactly the same
version as `candle-einops`. Applications should normally depend only on
`candle-einops`, which enforces this pairing.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes and publish
order.

## Scope

The crate implements rearrange, reduce, and repeat operations plus
arbitrary-arity explicit-label `einsum!`. Einsum supports permutation,
reduction, outer products, elementwise broadcasting, and GEMM-lowered
contractions.
Ellipses provide right-aligned variable-rank broadcasting and optional
reduction. Repeated labels within an input extract a diagonal before the
remaining contraction, including batched diagonals and traces. Multi-operand
equations use deterministic, shape-aware greedy contraction planning.

## Development parity

Contributors can run the locked Python einops semantic oracle and bounded Rust
property suite with one command:

```console
python3 .github/scripts/test_python_parity.py
```

This check is mandatory in CI but opt-in locally; ordinary Rust tests do not
install or invoke Python. See [parity/README.md](parity/README.md) for supported
operations, syntax translations, tolerances, deterministic replay, and the
deliberate dependency-update process.

## Development performance measurements

The locked, unpublished performance harness is isolated from the normal Cargo
workspace and published crates. Compile it or run its correctness-only plumbing
smoke with the supported wrapper:

```console
python3 .github/scripts/run_benchmarks.py compile
python3 .github/scripts/run_benchmarks.py smoke
python3 .github/scripts/run_benchmarks.py capture --backend metal --filter reshape/identity/non-contiguous/consume --operation library
```

Timing results are advisory rather than a CI gate. See
[benchmarks/README.md](benchmarks/README.md) for device profiles, filtering,
schema v2 diagnostics, the CPU/Accelerate/MKL and Metal/CUDA support matrix,
exact-operation GPU capture, required host libraries, and measurement
boundaries.

Licensed under either Apache-2.0 or MIT, at your option.
