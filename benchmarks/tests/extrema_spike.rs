use candle_core::{Device, Result, Tensor, Var};
use candle_einops_benchmarks::extrema_spike::{
    ExtremaKind, ExtremaLayout, collapsed_candidate, fixture, sequential, structural_metrics,
};

#[test]
fn candidate_matrix_has_one_reduction_and_only_strided_copy_pressure() -> Result<()> {
    for layout in [
        ExtremaLayout::ContiguousTrailing,
        ExtremaLayout::ContiguousLeading,
        ExtremaLayout::StridedTrailing,
    ] {
        let metrics = structural_metrics(layout);
        assert_eq!(metrics.current_submissions, 2);
        assert_eq!(metrics.candidate_submissions, 1);
        assert_eq!(
            metrics.candidate_materialized_elements > 0,
            layout == ExtremaLayout::StridedTrailing
        );
        let input = fixture(layout, &Device::Cpu)?;
        for kind in [ExtremaKind::Min, ExtremaKind::Max] {
            let current = sequential(&input, layout, kind)?;
            let candidate = collapsed_candidate(&input, layout, kind)?;
            assert_eq!(current.dims(), candidate.dims());
            assert_eq!(
                current.flatten_all()?.to_vec1::<f32>()?,
                candidate.flatten_all()?.to_vec1::<f32>()?
            );
        }
    }
    Ok(())
}

fn close(left: &Tensor, right: &Tensor) -> Result<()> {
    let left = left.flatten_all()?.to_vec1::<f32>()?;
    let right = right.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(left.len(), right.len());
    for (&left, &right) in left.iter().zip(&right) {
        assert!((left - right).abs() <= 1e-5 * right.abs().max(1.));
    }
    Ok(())
}

#[test]
fn candidate_preserves_supported_gradients() -> Result<()> {
    const ELEMENTS: usize = 32 * 24 * 16 * 8;
    let device = Device::Cpu;
    for layout in [
        ExtremaLayout::ContiguousTrailing,
        ExtremaLayout::ContiguousLeading,
        ExtremaLayout::StridedTrailing,
    ] {
        for kind in [ExtremaKind::Min, ExtremaKind::Max] {
            let values = (0..ELEMENTS).map(|value| value as f32).collect::<Vec<_>>();
            let current_var = Var::from_vec(values.clone(), (32, 16, 24, 8), &device)?;
            let candidate_var = Var::from_vec(values, (32, 16, 24, 8), &device)?;
            let arrange = |tensor: &Tensor| match layout {
                ExtremaLayout::StridedTrailing => tensor.permute([0, 2, 1, 3]),
                _ => tensor.reshape((32, 24, 16, 8)),
            };
            let current_input = arrange(current_var.as_tensor())?;
            let candidate_input = arrange(candidate_var.as_tensor())?;
            let current = sequential(&current_input, layout, kind)?;
            let candidate = collapsed_candidate(&candidate_input, layout, kind)?;
            close(&current, &candidate)?;
            let weights = Tensor::arange(1f32, (current.elem_count() + 1) as f32, &device)?
                .reshape(current.dims())?;
            let current_gradients = current.mul(&weights)?.sum_all()?.backward()?;
            let candidate_gradients = candidate.mul(&weights)?.sum_all()?.backward()?;
            close(
                current_gradients.get(current_var.as_tensor()).unwrap(),
                candidate_gradients.get(candidate_var.as_tensor()).unwrap(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn candidate_preserves_f64_u32_and_empty_axis_errors() -> Result<()> {
    const ELEMENTS: usize = 32 * 24 * 16 * 8;
    let device = Device::Cpu;
    let f64_input = Tensor::from_vec(
        (0..ELEMENTS).map(|value| value as f64).collect::<Vec<_>>(),
        (32, 24, 16, 8),
        &device,
    )?;
    let u32_input = Tensor::from_vec(
        (0..ELEMENTS).map(|value| value as u32).collect::<Vec<_>>(),
        (32, 24, 16, 8),
        &device,
    )?;
    for kind in [ExtremaKind::Min, ExtremaKind::Max] {
        let current = sequential(&f64_input, ExtremaLayout::ContiguousTrailing, kind)?;
        let candidate = collapsed_candidate(
            &f64_input,
            ExtremaLayout::ContiguousTrailing,
            kind,
        )?;
        assert_eq!(
            current.flatten_all()?.to_vec1::<f64>()?,
            candidate.flatten_all()?.to_vec1::<f64>()?
        );
        let current = sequential(&u32_input, ExtremaLayout::ContiguousLeading, kind)?;
        let candidate =
            collapsed_candidate(&u32_input, ExtremaLayout::ContiguousLeading, kind)?;
        assert_eq!(
            current.flatten_all()?.to_vec1::<u32>()?,
            candidate.flatten_all()?.to_vec1::<u32>()?
        );
    }

    let empty = Tensor::zeros((32, 24, 0, 8), candle_core::DType::F32, &device)?;
    for kind in [ExtremaKind::Min, ExtremaKind::Max] {
        assert!(sequential(&empty, ExtremaLayout::ContiguousTrailing, kind).is_err());
        assert!(collapsed_candidate(&empty, ExtremaLayout::ContiguousTrailing, kind).is_err());
    }
    Ok(())
}
