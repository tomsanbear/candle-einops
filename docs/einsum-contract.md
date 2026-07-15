# Einsum contract

This document records the contract for the public `einsum!` macro. Unary and
binary explicit-label equations are implemented; larger operand counts remain
reserved for a later slice. Each input and output axis list may contain one
`..` for right-aligned variable-rank broadcasting.

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
- Output labels are unique and each originates in an input.
- Scalars and zero-sized axes are valid.

## Red-first evidence

On 2026-07-15, a temporary integration test imported
`candle_einops::einsum` and invoked
`einsum!("i k, k j -> i j", &lhs, &rhs)`. Running
`cargo +1.94.0 test --test einsum_api_red` failed with Rust error `E0432`:
`no einsum in the root`. The uncompilable probe was then removed so this
oracle-only branch remains green. Subsequent implementation tickets can use
the contract and host corpus in `tests/einsum_oracle.rs` to turn public cases
green without redefining their semantics.
