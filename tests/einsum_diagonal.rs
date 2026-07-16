use candle_core::{DType, Device, Result, Tensor, Var};
use candle_einops::{PreparedDiagonalPlan, einsum};

fn flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn assert_close(actual: &Tensor, expected: &Tensor, context: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{context}: shape");
    for (index, (&actual, &expected)) in flat_f32(actual)?
        .iter()
        .zip(&flat_f32(expected)?)
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-5 * (1. + expected.abs()),
            "{context}[{index}]: actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

#[test]
fn prepared_diagonal_plan_reuses_device_indices_with_strict_boundaries() -> Result<()> {
    let device = Device::Cpu;
    let plan = PreparedDiagonalPlan::new(&[4, 3, 4, 3], &[0, 1, 0, 1], &device)?;
    for start in [0f32, 1.] {
        let input = Tensor::arange(start, start + 144., &device)?.reshape((4, 3, 4, 3))?;
        assert_close(
            &plan.execute(&input)?,
            &einsum!("i j i j -> i j", &input)?,
            "prepared interleaved diagonal",
        )?;
    }
    assert_eq!(plan.input_shape(), &[4, 3, 4, 3]);
    assert_eq!(plan.output_shape(), &[4, 3]);
    assert_eq!(plan.index_dtype(), DType::U32);

    assert!(plan.execute(&Tensor::zeros((4, 4), DType::F32, &device)?).is_err());
    let noncontiguous = Tensor::zeros((3, 4, 3, 4), DType::F32, &device)?
        .permute((1, 0, 3, 2))?;
    assert!(plan.execute(&noncontiguous).is_err());
    assert!(PreparedDiagonalPlan::new(&[2, 3], &[0, 0], &device).is_err());
    assert!(PreparedDiagonalPlan::new(&[2, 3], &[0, 1], &device).is_err());
    assert!(
        PreparedDiagonalPlan::new(&[usize::MAX, usize::MAX], &[0, 0], &device).is_err()
    );
    Ok(())
}

#[test]
fn prepared_diagonal_plan_preserves_zero_extents_and_gradients() -> Result<()> {
    let device = Device::Cpu;
    let empty_plan = PreparedDiagonalPlan::new(&[2, 0, 0], &[0, 1, 1], &device)?;
    let empty = Var::from_vec(Vec::<f32>::new(), (2, 0, 0), &device)?;
    let empty_output = empty_plan.execute(empty.as_tensor())?;
    assert_eq!(empty_output.dims(), &[2, 0]);
    assert_eq!(
        empty_output
            .sum_all()?
            .backward()?
            .get(empty.as_tensor())
            .expect("prepared empty gather keeps the input edge")
            .dims(),
        &[2, 0, 0]
    );

    let plan = PreparedDiagonalPlan::new(&[3, 3], &[0, 0], &device)?;
    let input = Var::from_vec((0..9).map(|value| value as f32).collect(), (3, 3), &device)?;
    let output = plan.execute(input.as_tensor())?;
    let gradients = output.sum_all()?.backward()?;
    assert_eq!(
        flat_f32(
            gradients
                .get(input.as_tensor())
                .expect("prepared diagonal keeps the input edge")
        )?,
        vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]
    );
    Ok(())
}

#[test]
fn extracts_matrix_diagonal_trace_and_higher_multiplicity() -> Result<()> {
    let matrix = Tensor::arange(0f32, 9f32, &Device::Cpu)?.reshape((3, 3))?;
    assert_eq!(
        einsum!("i i -> i", &matrix)?.to_vec1::<f32>()?,
        [0., 4., 8.]
    );
    assert_eq!(einsum!("i i ->", &matrix)?.to_scalar::<f32>()?, 12.);

    let cube = Tensor::arange(0f32, 27f32, &Device::Cpu)?.reshape((3, 3, 3))?;
    assert_eq!(
        einsum!("i i i -> i", &cube)?.to_vec1::<f32>()?,
        [0., 13., 26.]
    );
    Ok(())
}

#[test]
fn extracts_batched_ellipsis_diagonals_and_traces() -> Result<()> {
    let batched = Tensor::arange(0f32, 18f32, &Device::Cpu)?.reshape((2, 3, 3))?;
    assert_eq!(
        einsum!(".. i i -> .. i", &batched)?.to_vec2::<f32>()?,
        [[0., 4., 8.], [9., 13., 17.]]
    );
    assert_eq!(
        einsum!(".. i i -> ..", &batched)?.to_vec1::<f32>()?,
        [12., 39.]
    );
    Ok(())
}

#[test]
fn diagonal_labels_participate_in_binary_output_and_contraction() -> Result<()> {
    let matrix = Tensor::arange(0f32, 9f32, &Device::Cpu)?.reshape((3, 3))?;
    let vector = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    assert_eq!(
        einsum!("i i, i -> i", &matrix, &vector)?.to_vec1::<f32>()?,
        [0., 8., 24.]
    );
    assert_eq!(
        einsum!("i i, i ->", &matrix, &vector)?.to_scalar::<f32>()?,
        32.
    );
    Ok(())
}

#[test]
fn diagonal_handles_scalars_zero_axes_and_unequal_extent_errors() -> Result<()> {
    let scalar = Tensor::new(7f32, &Device::Cpu)?;
    assert_eq!(einsum!(".. -> ..", &scalar)?.to_scalar::<f32>()?, 7.);

    let empty = Tensor::zeros((0, 0), DType::F32, &Device::Cpu)?;
    assert_eq!(einsum!("i i -> i", &empty)?.dims(), &[0]);

    let unequal = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    let error = einsum!("i i -> i", &unequal).expect_err("unequal repeated axes must fail");
    assert!(
        error
            .to_string()
            .contains("einsum operand 0 repeated label `i`")
    );
    assert!(error.to_string().contains("unequal extents 2 and 3"));
    Ok(())
}

#[test]
fn diagonal_cpu_gradients_match_direct_candle_index_selection() -> Result<()> {
    let macro_input = Var::from_vec((0..9).map(|v| v as f32).collect(), (3, 3), &Device::Cpu)?;
    let direct_input = Var::from_vec((0..9).map(|v| v as f32).collect(), (3, 3), &Device::Cpu)?;
    let indices = Tensor::new(&[0u32, 4, 8], &Device::Cpu)?;

    let macro_output = einsum!("i i -> i", macro_input.as_tensor())?;
    let direct_output = direct_input.flatten_all()?.index_select(&indices, 0)?;
    assert_close(&macro_output, &direct_output, "diagonal forward")?;
    let macro_gradients = macro_output.sum_all()?.backward()?;
    let direct_gradients = direct_output.sum_all()?.backward()?;
    assert_close(
        macro_gradients
            .get(macro_input.as_tensor())
            .expect("macro gradient"),
        direct_gradients
            .get(direct_input.as_tensor())
            .expect("direct gradient"),
        "diagonal gradient",
    )
}

#[test]
fn interleaved_triple_and_multiple_groups_match_one_flat_selection() -> Result<()> {
    let interleaved = Tensor::arange(0f32, 36f32, &Device::Cpu)?.reshape((2, 3, 2, 3))?;
    let interleaved_indices = Tensor::new(&[0u32, 7, 14, 21, 28, 35], &Device::Cpu)?;
    let interleaved_expected = interleaved
        .flatten_all()?
        .index_select(&interleaved_indices, 0)?
        .reshape((2, 3))?;
    assert_close(
        &einsum!("i j i j -> i j", &interleaved)?,
        &interleaved_expected,
        "interleaved diagonal",
    )?;

    let triple = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((2, 3, 2, 2))?;
    let triple_indices = Tensor::new(&[0u32, 4, 8, 15, 19, 23], &Device::Cpu)?;
    let triple_expected = triple
        .flatten_all()?
        .index_select(&triple_indices, 0)?
        .reshape((2, 3))?;
    assert_close(
        &einsum!("i j i i -> i j", &triple)?,
        &triple_expected,
        "triple diagonal",
    )?;

    let multiple = Tensor::arange(0f32, 72f32, &Device::Cpu)?.reshape((2, 3, 2, 2, 3))?;
    let mut offsets = Vec::new();
    for i in 0..2u32 {
        for j in 0..3u32 {
            for k in 0..2u32 {
                offsets.push(i * 42 + j * 13 + k * 3);
            }
        }
    }
    let multiple_indices = Tensor::from_vec(offsets, 12, &Device::Cpu)?;
    let multiple_expected = multiple
        .flatten_all()?
        .index_select(&multiple_indices, 0)?
        .reshape((2, 3, 2))?;
    assert_close(
        &einsum!("i j i k j -> i j k", &multiple)?,
        &multiple_expected,
        "multiple repeated groups",
    )
}

#[test]
fn selected_diagonal_preserves_f32_f64_u32_and_noncontiguous_values() -> Result<()> {
    for dtype in [DType::F32, DType::F64, DType::U32] {
        let input = Tensor::arange(0u32, 36u32, &Device::Cpu)?
            .reshape((2, 3, 2, 3))?
            .to_dtype(dtype)?;
        let expected = input
            .flatten_all()?
            .index_select(&Tensor::new(&[0u32, 7, 14, 21, 28, 35], &Device::Cpu)?, 0)?
            .reshape((2, 3))?;
        let actual = einsum!("i j i j -> i j", &input)?;
        assert_eq!(actual.dtype(), dtype);
        assert!(actual.device().same_device(input.device()));
        assert_eq!(flat_f32(&actual)?, flat_f32(&expected)?);
    }

    let noncontiguous = Tensor::arange(0f32, 36f32, &Device::Cpu)?
        .reshape((3, 2, 3, 2))?
        .permute((1, 0, 3, 2))?;
    assert!(!noncontiguous.is_contiguous());
    let expected = noncontiguous
        .flatten_all()?
        .index_select(&Tensor::new(&[0u32, 7, 14, 21, 28, 35], &Device::Cpu)?, 0)?
        .reshape((2, 3))?;
    assert_close(
        &einsum!("i j i j -> i j", &noncontiguous)?,
        &expected,
        "noncontiguous fallback",
    )?;
    Ok(())
}

#[test]
fn interleaved_f32_gradient_matches_one_flat_selection() -> Result<()> {
    let values = (0..36).map(|value| value as f32 / 7.).collect::<Vec<_>>();
    let macro_input = Var::from_vec(values.clone(), (2, 3, 2, 3), &Device::Cpu)?;
    let direct_input = Var::from_vec(values, (2, 3, 2, 3), &Device::Cpu)?;
    let indices = Tensor::new(&[0u32, 7, 14, 21, 28, 35], &Device::Cpu)?;
    let macro_output = einsum!("i j i j -> i j", macro_input.as_tensor())?;
    let direct_output = direct_input
        .flatten_all()?
        .index_select(&indices, 0)?
        .reshape((2, 3))?;
    let macro_gradients = macro_output.sum_all()?.backward()?;
    let direct_gradients = direct_output.sum_all()?.backward()?;
    assert_close(
        macro_gradients
            .get(macro_input.as_tensor())
            .expect("macro gradient"),
        direct_gradients
            .get(direct_input.as_tensor())
            .expect("direct gradient"),
        "interleaved f32 gradient",
    )
}

#[test]
fn selected_zero_extent_diagonal_preserves_shape_and_backward_edge() -> Result<()> {
    let input = Var::from_vec(Vec::<f32>::new(), (2, 0, 0), &Device::Cpu)?;
    let output = einsum!("batch i i -> batch i", input.as_tensor())?;
    assert_eq!(output.dims(), &[2, 0]);
    let gradients = output.sum_all()?.backward()?;
    assert_eq!(
        gradients
            .get(input.as_tensor())
            .expect("selected empty gather keeps the input edge")
            .dims(),
        &[2, 0, 0]
    );
    Ok(())
}
