use candle_einops_benchmarks::{
    broadcast_gemm_spike, criterion_binary_fast_paths, criterion_plumbing_benchmark,
    criterion_product_benchmarks, criterion_reduction_fusion, criterion_repeat_broadcast,
    diagonal_spike, nary_cost_model_spike,
};
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    criterion_plumbing_benchmark,
    criterion_product_benchmarks,
    diagonal_spike::criterion_benchmarks,
    criterion_binary_fast_paths,
    criterion_reduction_fusion,
    criterion_repeat_broadcast,
    broadcast_gemm_spike::criterion_benchmarks,
    nary_cost_model_spike::criterion_benchmarks
);
criterion_main!(benches);
