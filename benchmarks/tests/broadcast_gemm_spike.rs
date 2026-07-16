use candle_core::{DType, Device, Result, Tensor, Var};
use candle_einops_benchmarks::Scenario;
use candle_einops_benchmarks::broadcast_gemm_spike::{
    StructuralMetrics, broadcast_scenarios, direct_broadcast_gemm, eager_expansion_probe,
    selected_broadcast_gemm, sliced_broadcast_gemm,
};

fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

#[test]
fn frozen_cases_report_copy_peak_and_submission_metrics() -> Result<()> {
    let metrics = broadcast_scenarios()
        .iter()
        .map(|scenario| scenario.structural_metrics(&Device::Cpu))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        metrics,
        vec![
            StructuralMetrics {
                eager_copy_bytes: 131_072,
                eager_peak_temporary_elements: 32_768,
                eager_gemm_submissions: 1,
                selected_copy_bytes: 0,
                selected_peak_temporary_elements: 32_768,
                selected_gemm_submissions: 32,
            },
            StructuralMetrics {
                eager_copy_bytes: 131_072,
                eager_peak_temporary_elements: 32_768,
                eager_gemm_submissions: 1,
                selected_copy_bytes: 0,
                selected_peak_temporary_elements: 32_768,
                selected_gemm_submissions: 32,
            },
            StructuralMetrics {
                eager_copy_bytes: 524_288,
                eager_peak_temporary_elements: 131_072,
                eager_gemm_submissions: 1,
                selected_copy_bytes: 0,
                selected_peak_temporary_elements: 65_536,
                selected_gemm_submissions: 64,
            },
            StructuralMetrics {
                eager_copy_bytes: 65_536,
                eager_peak_temporary_elements: 16_384,
                eager_gemm_submissions: 1,
                selected_copy_bytes: 0,
                selected_peak_temporary_elements: 0,
                selected_gemm_submissions: 1,
            },
        ]
    );
    Ok(())
}

#[test]
fn eager_strategy_materializes_broadcast_operands() -> Result<()> {
    let left = Tensor::ones((1, 8, 8), DType::F32, &Device::Cpu)?;
    let right = Tensor::ones((4, 8, 8), DType::F32, &Device::Cpu)?;
    let left_probe = eager_expansion_probe(&left, &right)?;
    assert_eq!(left_probe.left_copy_elements, 4 * 8 * 8);
    assert_eq!(left_probe.right_copy_elements, 0);
    assert_eq!(left_probe.peak_temporary_elements, 4 * 8 * 8);

    let left = Tensor::ones((8, 1, 8, 8), DType::F32, &Device::Cpu)?;
    let right = Tensor::ones((1, 8, 8, 8), DType::F32, &Device::Cpu)?;
    let both_probe = eager_expansion_probe(&left, &right)?;
    assert_eq!(both_probe.left_copy_elements, 8 * 8 * 8 * 8);
    assert_eq!(both_probe.right_copy_elements, 8 * 8 * 8 * 8);
    assert_eq!(both_probe.peak_temporary_elements, 2 * 8 * 8 * 8 * 8);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn eager_strategy_reports_materialization_on_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    let left = Tensor::ones((1, 8, 8), DType::F32, &device)?;
    let right = Tensor::ones((4, 8, 8), DType::F32, &device)?;
    let probe = eager_expansion_probe(&left, &right)?;

    assert_eq!(probe.left_copy_elements, 4 * 8 * 8);
    assert_eq!(probe.right_copy_elements, 0);
    assert_eq!(probe.output.dims(), &[4, 8, 8]);
    device.synchronize()?;
    Ok(())
}

#[test]
fn direct_and_slice_candidates_cover_their_structural_boundaries() -> Result<()> {
    let left = Tensor::ones((4, 8, 8), DType::F32, &Device::Cpu)?;
    let right = Tensor::ones((4, 8, 8), DType::F32, &Device::Cpu)?;
    assert_eq!(direct_broadcast_gemm(&left, &right)?.dims(), &[4, 8, 8]);

    let broadcast_left = Tensor::ones((1, 8, 8), DType::F32, &Device::Cpu)?;
    assert!(direct_broadcast_gemm(&broadcast_left, &right).is_err());
    assert_eq!(
        sliced_broadcast_gemm(&broadcast_left, &right)?.dims(),
        &[4, 8, 8]
    );

    let cross_left = Tensor::ones((4, 1, 8, 8), DType::F32, &Device::Cpu)?;
    let cross_right = Tensor::ones((1, 4, 8, 8), DType::F32, &Device::Cpu)?;
    assert!(direct_broadcast_gemm(&cross_left, &cross_right).is_err());
    assert_eq!(
        sliced_broadcast_gemm(&cross_left, &cross_right)?.dims(),
        &[4, 4, 8, 8]
    );
    Ok(())
}

#[test]
fn stride_zero_backend_behavior_does_not_change_the_portable_candidate() -> Result<()> {
    let left = Tensor::ones((1, 32, 32), DType::F32, &Device::Cpu)?;
    let right = Tensor::ones((32, 32, 32), DType::F32, &Device::Cpu)?;
    let _stride_zero = left.broadcast_as((32, 32, 32))?.matmul(&right)?;
    let eager = eager_expansion_probe(&left, &right)?.output;
    let selected = selected_broadcast_gemm(&left, &right)?;

    #[cfg(feature = "accelerate")]
    assert_eq!(
        flat(&_stride_zero)?,
        flat(&eager)?,
        "Accelerate should preserve stride-zero batched matmul values"
    );
    #[cfg(not(any(feature = "accelerate", feature = "mkl")))]
    assert_ne!(
        flat(&_stride_zero)?,
        flat(&eager)?,
        "the baseline Candle 0.11 CPU backend should expose the known stride-zero boundary"
    );
    assert_eq!(flat(&selected)?, flat(&eager)?);
    Ok(())
}

#[test]
fn selected_candidate_matches_eager_values_and_gradients() -> Result<()> {
    for scenario in broadcast_scenarios() {
        let inputs = scenario.setup(&Device::Cpu)?;
        let eager = scenario.run_library(&inputs)?;
        let selected = scenario.run_reference(&inputs)?;
        scenario.check(&eager, &selected)?;
    }

    let values = (0..64).map(|value| value as f32 / 64.).collect::<Vec<_>>();
    let eager_left = Var::from_vec(values.clone(), (1, 8, 8), &Device::Cpu)?;
    let selected_left = Var::from_vec(values, (1, 8, 8), &Device::Cpu)?;
    let right = Tensor::ones((4, 8, 8), DType::F32, &Device::Cpu)?;
    let eager = eager_expansion_probe(eager_left.as_tensor(), &right)?.output;
    let selected = selected_broadcast_gemm(selected_left.as_tensor(), &right)?;
    let eager_gradients = eager.sum_all()?.backward()?;
    let selected_gradients = selected.sum_all()?.backward()?;
    assert_eq!(
        flat(eager_gradients.get(eager_left.as_tensor()).unwrap())?,
        flat(selected_gradients.get(selected_left.as_tensor()).unwrap())?
    );
    Ok(())
}

#[test]
fn selected_candidate_preserves_zero_and_singleton_batches() -> Result<()> {
    let singleton_left = Tensor::ones((1, 4, 8), DType::F32, &Device::Cpu)?;
    let singleton_right = Tensor::ones((1, 8, 4), DType::F32, &Device::Cpu)?;
    assert_eq!(
        selected_broadcast_gemm(&singleton_left, &singleton_right)?.dims(),
        &[1, 4, 4]
    );

    let empty_left = Tensor::zeros((0, 4, 8), DType::F32, &Device::Cpu)?;
    let empty_right = Tensor::zeros((1, 8, 4), DType::F32, &Device::Cpu)?;
    assert_eq!(
        selected_broadcast_gemm(&empty_left, &empty_right)?.dims(),
        &[0, 4, 4]
    );
    Ok(())
}
