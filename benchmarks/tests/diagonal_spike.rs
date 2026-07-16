use candle_core::{Device, Result, Tensor, Var};
use candle_einops::einsum;
use candle_einops_benchmarks::diagonal_spike::{
    break_even_reuses, build_interleaved_indices, build_repeated_indices, cached_flat_gather,
    cpu_storage_id, probe_current_interleaved_lowering, PreparedDiagonalPlan,
};

fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

#[test]
fn measured_break_even_reuse_counts_are_ceilings_and_reject_no_win() {
    assert_eq!(break_even_reuses(1_250, 4_875, 2_416), Some(1));
    assert_eq!(break_even_reuses(3_542, 6_500, 3_542), Some(2));
    assert_eq!(break_even_reuses(12_708, 16_458, 9_958), Some(2));
    assert_eq!(break_even_reuses(1_375, 7_292, 2_125), Some(1));
    assert_eq!(break_even_reuses(2_667, 13_542, 3_667), Some(1));
    assert_eq!(break_even_reuses(7_875, 38_500, 9_959), Some(1));
    assert_eq!(break_even_reuses(10, 5, 5), None);
    assert_eq!(break_even_reuses(10, 4, 5), None);
}

#[test]
fn prepared_plan_is_explicitly_shape_and_device_bound_and_reusable() -> Result<()> {
    let device = Device::Cpu;
    let plan = PreparedDiagonalPlan::interleaved(4, 3, &device)?;
    let index_storage = cpu_storage_id(plan.indices())?;
    let first = Tensor::arange(0f32, 144., &device)?.reshape((4, 3, 4, 3))?;
    let second = Tensor::arange(1f32, 145., &device)?.reshape((4, 3, 4, 3))?;
    assert_eq!(
        flat(&plan.execute(&first)?)?,
        flat(&einsum!("i j i j -> i j", &first)?)?
    );
    assert_eq!(
        flat(&plan.execute(&second)?)?,
        flat(&einsum!("i j i j -> i j", &second)?)?
    );
    assert_eq!(cpu_storage_id(plan.indices())?, index_storage);

    let wrong_shape = Tensor::zeros((4, 4), candle_core::DType::F32, &device)?;
    assert!(plan.execute(&wrong_shape).is_err());
    let non_contiguous = first.permute([1, 0, 3, 2])?;
    assert!(plan.execute(&non_contiguous).is_err());

    let repeated = PreparedDiagonalPlan::repeated(8, 3, &device)?;
    let cube = Tensor::arange(0f32, 512., &device)?.reshape((8, 8, 8))?;
    assert_eq!(
        flat(&repeated.execute(&cube)?)?,
        flat(&einsum!("i i i -> i", &cube)?)?
    );
    Ok(())
}

#[test]
fn current_interleaved_lowering_materializes_the_full_input() -> Result<()> {
    let input = Tensor::arange(0f32, 144f32, &Device::Cpu)?.reshape((4, 3, 4, 3))?;
    let probe = probe_current_interleaved_lowering(&input, 4, 3)?;

    assert_eq!(cpu_storage_id(&input)?, cpu_storage_id(&probe.adjacent)?);
    assert!(!probe.adjacent.is_contiguous());
    assert!(probe.flattened.is_contiguous());
    assert_ne!(cpu_storage_id(&input)?, cpu_storage_id(&probe.flattened)?);
    assert_eq!(probe.flattened.elem_count(), input.elem_count());
    assert_eq!(probe.copied_elements, input.elem_count());
    Ok(())
}

#[test]
fn current_index_construction_allocates_fresh_storage_per_call() -> Result<()> {
    let first = build_interleaved_indices(4, 3, &Device::Cpu)?;
    let second = build_interleaved_indices(4, 3, &Device::Cpu)?;
    assert_ne!(cpu_storage_id(&first)?, cpu_storage_id(&second)?);
    assert_eq!(first.elem_count(), 12);
    Ok(())
}

#[test]
fn cached_flat_gather_matches_simple_triple_and_interleaved_diagonals() -> Result<()> {
    let matrix = Tensor::arange(0f32, 64f32, &Device::Cpu)?.reshape((8, 8))?;
    let matrix_indices = build_repeated_indices(8, 2, &Device::Cpu)?;
    let matrix_candidate = cached_flat_gather(&matrix, &matrix_indices, &[8])?;
    assert_eq!(
        flat(&matrix_candidate)?,
        flat(&einsum!("i i -> i", &matrix)?)?
    );

    let cube = Tensor::arange(0f32, 512f32, &Device::Cpu)?.reshape((8, 8, 8))?;
    let cube_indices = build_repeated_indices(8, 3, &Device::Cpu)?;
    let cube_candidate = cached_flat_gather(&cube, &cube_indices, &[8])?;
    assert_eq!(
        flat(&cube_candidate)?,
        flat(&einsum!("i i i -> i", &cube)?)?
    );

    let interleaved = Tensor::arange(0f32, 144f32, &Device::Cpu)?.reshape((4, 3, 4, 3))?;
    let interleaved_indices = build_interleaved_indices(4, 3, &Device::Cpu)?;
    let interleaved_candidate = cached_flat_gather(&interleaved, &interleaved_indices, &[4, 3])?;
    assert_eq!(
        flat(&interleaved_candidate)?,
        flat(&einsum!("i j i j -> i j", &interleaved)?)?
    );
    Ok(())
}

#[test]
fn cached_flat_gather_is_gradient_capable_and_has_an_explicit_fallback_boundary() -> Result<()> {
    let data = (0..144).map(|value| value as f32).collect::<Vec<_>>();
    let candidate_input = Var::from_vec(data.clone(), (4, 3, 4, 3), &Device::Cpu)?;
    let current_input = Var::from_vec(data, (4, 3, 4, 3), &Device::Cpu)?;
    let indices = build_interleaved_indices(4, 3, &Device::Cpu)?;
    let candidate = cached_flat_gather(candidate_input.as_tensor(), &indices, &[4, 3])?;
    let current = einsum!("i j i j -> i j", current_input.as_tensor())?;
    let candidate_gradients = candidate.sum_all()?.backward()?;
    let current_gradients = current.sum_all()?.backward()?;
    assert_eq!(
        flat(
            candidate_gradients
                .get(candidate_input.as_tensor())
                .unwrap()
        )?,
        flat(current_gradients.get(current_input.as_tensor()).unwrap())?
    );

    let non_contiguous = candidate_input.as_tensor().permute((1, 0, 3, 2))?;
    let error = cached_flat_gather(&non_contiguous, &indices, &[4, 3])
        .expect_err("the portable fast path must require contiguous storage");
    assert!(error.to_string().contains("contiguous"));

    let empty = Tensor::zeros((0, 0), candle_core::DType::F32, &Device::Cpu)?;
    let empty_indices = build_repeated_indices(0, 2, &Device::Cpu)?;
    assert_eq!(
        cached_flat_gather(&empty, &empty_indices, &[0])?.dims(),
        &[0]
    );
    Ok(())
}
