# Performance reporting environment

This locked `uv` project owns development-only dependencies used to render the
versioned performance report. It is separate from the Python einops oracle in
`parity/` and is never part of either published Rust crate.

```console
uv sync --frozen --project benchmarks/reporting
uv run --frozen --project benchmarks/reporting \
  python .github/scripts/test_generate_performance_report.py
uv run --frozen --project benchmarks/reporting \
  python .github/scripts/generate_performance_report.py --check
```

Matplotlib and Seaborn render deterministic, accessible SVGs; Pandas builds the
provider outcome frames. To refresh the snapshot, first collect all four
optimized `gaps` summaries and follow the import command in
[`docs/performance.md`](../../docs/performance.md#data-and-reproduction).
