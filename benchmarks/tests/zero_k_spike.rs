use candle_core::Result;
use candle_einops_benchmarks::zero_k_scenarios;

#[test]
fn cat_candidate_reduces_public_operations_without_data_temporaries() -> Result<()> {
    for scenario in zero_k_scenarios() {
        let metrics = scenario.structural_metrics();
        assert_eq!(metrics.current_public_operations, 3);
        assert_eq!(metrics.candidate_public_operations, 2);
        assert_eq!(metrics.current_temporary_elements, 2);
        assert_eq!(metrics.candidate_temporary_elements, 0);
        assert_eq!(metrics.gemm_submissions, 0);
    }
    Ok(())
}
