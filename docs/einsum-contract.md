# Einsum contract

This document records the supported 0.2.0 contract for the public `einsum!`
macro. Equations may contain any positive number of operands. Each input and
output axis list may contain one `..` for right-aligned variable-rank
broadcasting.

## Equation and evaluation contract

- Equations use whitespace-delimited named axes and exactly one explicit `->`,
  for example `"batch i k, batch k j -> batch i j"`.
- The number of comma-separated input axis lists equals the number of operand
  expressions, including an empty axis list for a scalar.
- Every operand expression is evaluated exactly once, from left to right.
- The macro returns `candle_core::Result<Tensor>`.
- Labels omitted from the output are reduced. A retained label shared by
  operands broadcasts when its extents are equal or one.
- A label repeated within one operand extracts that operand's diagonal before
  broadcasting or contraction. All repeated occurrences must have exactly
  equal extents; they never broadcast against each other.
- Multi-operand equations greedily select the pair with the smallest retained
  intermediate, then the fewest estimated FLOPs, then the earliest original
  operand order. A label is reduced only after it is absent from both the
  explicit output and every remaining operand.
- Output labels are unique and each originates in an input.
- Scalars and zero-sized axes are valid.

## Supported forms

Unary equations permute and reduce one input, for example
`"rows columns -> columns rows"`. Binary equations support outer products,
retained-label broadcasting, and contractions such as
`"row inner, inner column -> row column"`.

An ellipsis captures a runtime number of axes. Captures align from the right;
retaining `..` broadcasts them and omitting it reduces them. For example,
`".. feature -> feature"` reduces every captured leading axis.

Repeating a label within one input selects its diagonal before other work.
`"index index -> index"` returns a diagonal and `"index index ->"` returns a
trace. Repeated extents must be equal and do not broadcast within an operand.

N-ary equations contain three or more operands, such as
`"row inner, inner column, column -> row"`. Planning is deterministic and
shape-aware. It minimizes retained intermediate size, then estimated work, then
original operand order. This planning choice affects performance, not results.

All forms return `candle_core::Result<Tensor>`. Operands are evaluated once in
source order. Tensor-dependent rank, extent, dtype, and device failures retain
Candle error context; malformed equations and operand-count mismatches are
compile-time diagnostics.

## Version coupling

Macro expansions call a doc-hidden runtime surface in `candle-einops`. That is
a private ABI rather than a semver-stable public implementation API, so the
runtime depends on `candle-einops-macros` with an exact version requirement.
Applications should depend on the runtime crate and use its macro re-export.
