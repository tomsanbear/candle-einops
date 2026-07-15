use candle_einops_benchmarks::{
    binary_operand_packing, broadcast_gemm_spike, criterion_binary_fast_paths,
    criterion_identity_reshape, criterion_plumbing_benchmark, criterion_product_benchmarks,
    criterion_reduction_fusion, criterion_repeat_broadcast, criterion_zero_k, diagonal_spike,
    extended_compose, extrema_spike, nary_cost_model_spike, permute_compose_layout_spike,
};
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    criterion_plumbing_benchmark,
    criterion_product_benchmarks,
    diagonal_spike::criterion_benchmarks,
    criterion_binary_fast_paths,
    binary_operand_packing::criterion_benchmarks,
    criterion_zero_k,
    criterion_reduction_fusion,
    criterion_repeat_broadcast,
    criterion_identity_reshape,
    permute_compose_layout_spike::criterion_benchmarks,
    extended_compose::criterion_benchmarks,
    extrema_spike::criterion_benchmarks,
    broadcast_gemm_spike::criterion_benchmarks,
    nary_cost_model_spike::criterion_benchmarks
);
criterion_main!(benches);
