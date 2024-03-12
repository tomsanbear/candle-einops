![candle-einops](https://github.com/tomsanbear/candle-einops/workflows/CI/badge.svg)
[![crates](https://img.shields.io/crates/v/candle-einops)](https://crates.io/crates/candle-einops)
[![docs](https://img.shields.io/docsrs/candle-einops)](https://docs.rs/candle-einops)

# candle-einops

This library is a fork of [einops](https://github.com/VasanthakumarV/einops) intended to bring support for einops to [Candle](https://github.com/huggingface/candle). Thank you @VasanthakumarV for such a fantastic macro based library to build off. The original library was implemented with TCH as the backing library and was based on the [einops](https://github.com/arogozhnikov/einops) python library.

For the most part everything from the original library has remained and only the device/dtype bindings have been altered. I do have plans to port `einsum` functionality to this library in the future.

Difference from the python version:

- All code generated at compile time, avoiding the need for caching
- One common api for rearrange, reduce and repeat operations
- Shape and reduction operations can be directly specified in the expression

## Getting Started

__Transpose__

Permute/Transpose dimensions, left side of `->` is the original state, right of `->` describes the end state

```rust
// (28, 28, 3) becomes (3, 28, 28)
let output = einops!("h w c -> c h w", &input);
```

__Composition__

Combine dimensions by putting them inside a parenthesis on the right of `->`

```rust
// (10, 28, 28, 3) becomes (280, 28, 3)
let output = einops!("b h w c -> (b h) w c", &input);
```

__Transpose + Composition__

Transpose a tensor, followed by a composing two dimensions into one, in one single expression

```rust
// (10, 28, 28, 3) becomes (28, 280, 3)
let output = einops!("b h w c -> h (b w) c", &input);
```

__Decomposition__

Split a dimension into two, by specifying the details inside parenthesis on the left,
specify the shape of the new dimensions like so `b1:2`, `b1` is a new dimension with shape 2

```rust
// (10, 28, 28, 3) becomes (2, 5, 28, 28, 3)
let output = einops!("(b1:2 b2) h w c -> b1 b2 h w c", &input);
```

New axis can also be specified from variables or fields (struct and enum) using curly braces

```rust
let b1 = 2;
let output = einops!("({b1} b2) h w c -> {b1} b2 h w c", &input);
```

__Decomposition + Transpose + Composition__

We can perform all operations discussed so far in a single expression

```rust
// (10, 28, 28, 3) becomes (56, 140 3)
let output = einops!("b h (w w2:2) c -> (h w2) (b w) c", &input);
```

__Reduce__

We can reduce axes using operations like, `sum`, `min`, `max`, and `mean`.
if the same operations has to be performed on multiple continuous axes we can do `sum(a b c)`

```rust
// (10, 28, 28, 3) becomes (28, 28, 3)
let output = einops!("mean(b) h w c -> h w c", &input);
```

__Decomposition + Reduce + Transpose + Composition__

Single expression for combining all functionalities discussed

```rust
// (10, 28, 28, 3) becomes (14, 140, 3)
let output = einops!("b (h max(h2:2)) (w max(w2:2)) c -> h (b w) c", &input);
```

__Repeat__

We can repeat axes by specify it on the right side of `->`, it can named, or it can simply be a number

```rust
// (28, 28, 3) becomes (28, 5, 28, 3)
let output = einops!("h w c -> h repeat:5 w c", &input);
```

Repeating axis's shape can be from a variables or a field (struct, enum)

```rust
let repeat = 5;
let output = einops!("h w c -> h {repeat} w c", &input);
```

__Squeeze__

Squeeze axes of shape 1

```rust
// (1, 28, 28, 3) becomes (28, 28, 3)
let output = einops!("1 h w c -> h w c")
```
