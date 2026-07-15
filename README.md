![candle-einops](https://github.com/tomsanbear/candle-einops/workflows/CI/badge.svg)
[![crates](https://img.shields.io/crates/v/candle-einops)](https://crates.io/crates/candle-einops)
[![docs](https://img.shields.io/docsrs/candle-einops)](https://docs.rs/candle-einops)

# candle-einops

`candle-einops` provides compile-time tensor rearrange, reduce, and repeat
expressions for [Candle](https://github.com/huggingface/candle). It is based on
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

## Migrating from 0.1

Version 0.2 contains three compatibility changes:

- Candle is upgraded from 0.6 to 0.11.
- `einops!` now returns `candle_core::Result<Tensor>` instead of panicking on a
  backend error. Add `?` or handle the result explicitly.
- Custom `Backend` implementations must return `candle_core::Result` from
  `reshape`, `transpose`, `reduce_axes`, and `add_axes`. `shape` remains
  infallible.

Dependency renaming is supported. For example,
`tensor-ops = { package = "candle-einops", version = "0.2" }` can be imported
with `use tensor_ops::einops;`.

See [CHANGELOG.md](CHANGELOG.md) for the complete release notes and publish
order.

## Scope

The crate currently implements rearrange, reduce, and repeat operations. It
does not implement `einsum`; the historical `AddEinsumSupport` branch is an
incomplete Candle 0.4-era prototype and is not part of the 0.2 release.

Licensed under either Apache-2.0 or MIT, at your option.
