use candle_einops_benchmarks::criterion_plumbing_benchmark;
use criterion::{criterion_group, criterion_main};

criterion_group!(benches, criterion_plumbing_benchmark);
criterion_main!(benches);
