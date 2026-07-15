use candle_einops_benchmarks::{
    criterion_plumbing_benchmark, criterion_product_benchmarks, diagonal_spike,
};
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    criterion_plumbing_benchmark,
    criterion_product_benchmarks,
    diagonal_spike::criterion_benchmarks
);
criterion_main!(benches);
