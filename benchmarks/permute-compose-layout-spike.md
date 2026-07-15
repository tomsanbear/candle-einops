# Permute-plus-composition layout decision

## Decision

**Conditional GO** for the narrow implementation in
`fused-permute-compose-layout`; **NO-GO** for arbitrary-stride tensor
construction or a claim that all locally collapsible layouts are public views.

The viable operation is not a custom layout. For a pure permutation followed
by composition, keep each requested output group internally ordered, find an
ordering of whole groups that makes the actual input C-contiguous, reshape the
groups in that order, then permute the grouped axes into requested output
order. For contiguous `[a,b,c]`, `c (a b)` becomes reshape `[a*b,c]` followed
by transpose; NCHW to `n (h w) c` becomes reshape `[n,c,h*w]` followed by a
group transpose. Candle's public `permute` and contiguous `reshape` operations
preserve storage, offsets, and autograd.

The spike prototype lives only in the benchmark crate. No production runtime
or macro path is changed by this ticket.

## CPU measurements

Twenty alternating samples on 98,304 `f32` elements produced:

| scenario | current median | public-view prototype | current/prototype |
| --- | ---: | ---: | ---: |
| `c (a b)` construct | 8.4903125 ms | 6.896 us | 1231.1938x |
| `c (a b)` consume contiguous | 8.425062 ms | 8.3851665 ms | 1.00476x |
| `n (h w) c` construct | 8.505646 ms | 9.9795 us | 852.3118x |
| `n (h w) c` consume contiguous | 8.448917 ms | 8.473417 ms | 0.99711x |

Reproduction command:

```console
python3 .github/scripts/run_benchmarks.py run --filter layout/permute-compose --samples 20 --output target/benchmarks/permute-compose-spike.json
```

The wrapper's `run` operation used its current default Cargo development
profile. The fingerprint was commit `0bead7692eb5fa1b4367edd40d4886a1ebc936c0`,
Rust `1.94.1`, Candle `0.11.0`, macOS/aarch64, CPU device/backend. The JSON
artifact is intentionally ignored; the durable table above records its paired
medians and ratios. These values are decision evidence, not a release-mode or
cross-machine performance guarantee.

Construction avoids the current eager copy by three orders of magnitude. An
immediate contiguous consumer removes that advantage: the prototype defers the
same copy and is effectively neutral. The implementation is therefore useful
when downstream operations accept the resulting strided view or when
materialization can be delayed or avoided; it is not an end-to-end speedup for
an unconditional contiguous consumer.

## Correctness and materialization evidence

The bounded corpus compares exact logical values against today's
permute-then-reshape route and covers `c (a b)`, NCHW to `n (h w) c`, an
already-viewable adjacent grouping, a non-collapsible `(a c)` grouping,
non-zero offsets, singleton and zero extents, invalid metadata, and an exact
identity owned by identity-reshape elision. Candidate tensors retain the input
storage address and offset, expose the predicted strides, and produce the same
input gradient as the current route. Copy-required cases retain today's
materialized contiguous output.

Flatten-order proof obligation: axes inside every output group must remain in
the requested order. Only whole groups may move before reshape. If a group is
`(a b)`, `b` remains its fastest-varying logical axis; a plan that reverses or
interleaves group members is invalid even if its element count matches.

## Public API and backend feasibility

Candle 0.11 publicly exposes `Layout` inspection, but `Tensor::from_storage`
always installs contiguous strides. There is no safe public constructor that
attaches an arbitrary layout to aliased storage while retaining autograd. A
locally affine layout such as `[c,a*b]` with strides `[1,c]` is therefore not,
by itself, an implementation route.

The selected prototype uses only shared Tensor metadata operations. CPU is
measured. CUDA and Metal should use the same public permutation/reshape graph
and shared backprop definitions, so eligibility is backend-neutral; this is an
inference, not a GPU performance claim. The implementation ticket must compile
the feature backends where available and retain CPU value/gradient coverage;
GPU timing remains outside this ticket.

## Exact eligibility and fallback rules

1. Apply only to a pure permutation-plus-composition boundary. Do not cross a
   repeat, reduction, or other operation.
2. Preserve existing validation order and checked group products.
3. Leave exact-dimension identity reshapes to the integrated shallow-clone
   elision; do not create a fused node for them.
4. Require non-empty requested groups to partition every input axis exactly
   once, with members in required logical flatten order.
5. Search deterministically over orderings of whole groups. Concatenate each
   ordering and apply Candle's exact C-contiguity predicate to the actual dims
   and strides: scanning right-to-left, require `stride == accumulated` only
   for extents greater than one, then checked-multiply the accumulated extent.
   This exact rule governs offsets, singleton axes, and zero extents.
6. For an eligible ordering, use public `permute -> reshape -> permute` views.
   Never construct a Tensor from `Layout::new` or private storage internals.
7. If no whole-group ordering is eligible, use the existing expanded
   `transpose` followed by `reshape` byte-for-byte. Do not add an unconditional
   `contiguous` call and do not retry a selected plan after a backend error.
