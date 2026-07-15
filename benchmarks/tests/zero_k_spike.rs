use candle_core::{DType, Device, Result, Tensor, Var};
use candle_einops_benchmarks::{zero_k_cat_candidate, zero_k_scenarios};

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

#[test]
fn cat_candidate_preserves_both_gradient_edges_for_all_output_sizes() -> Result<()> {
    let device = Device::Cpu;
    for (rows, columns) in [(1, 1), (64, 64), (512, 512)] {
        let left = Var::zeros((rows, 0), DType::F32, &device)?;
        let right = Var::zeros((0, columns), DType::F32, &device)?;
        let output = zero_k_cat_candidate(
            left.as_tensor(),
            right.as_tensor(),
            &[rows, columns],
        )?;
        assert_eq!(output.dims(), [rows, columns]);
        assert!(
            output
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&value| value == 0.)
        );
        let gradients = output.sum_all()?.backward()?;
        assert_eq!(gradients.get(left.as_tensor()).unwrap().dims(), [rows, 0]);
        assert_eq!(
            gradients.get(right.as_tensor()).unwrap().dims(),
            [0, columns]
        );
    }
    Ok(())
}

#[test]
fn cat_candidate_matches_current_dtype_and_validation_boundaries() -> Result<()> {
    let device = Device::Cpu;
    for dtype in [DType::F32, DType::F64, DType::BF16, DType::U8, DType::U32, DType::I64] {
        let left = Tensor::zeros((2, 0), dtype, &device)?;
        let right = Tensor::zeros((0, 3), dtype, &device)?;
        let candidate = zero_k_cat_candidate(&left, &right, &[2, 3]);
        let current = left
            .unsqueeze(0)?
            .narrow(0, 0, 0)?
            .sum_all()?
            .add(&right.unsqueeze(0)?.narrow(0, 0, 0)?.sum_all()?)?
            .broadcast_as((2, 3));
        assert_eq!(candidate.is_ok(), current.is_ok(), "{dtype:?}");
        if let (Ok(candidate), Ok(current)) = (candidate, current) {
            assert_eq!(candidate.dtype(), current.dtype());
            assert_eq!(
                candidate.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                current.to_dtype(DType::F32)?.to_vec2::<f32>()?
            );
        }
    }
    let left = Tensor::zeros((2, 0), DType::F32, &device)?;
    let different_dtype = Tensor::zeros((0, 3), DType::F64, &device)?;
    assert!(zero_k_cat_candidate(&left, &different_dtype, &[2, 3]).is_err());
    Ok(())
}
