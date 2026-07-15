use candle_core::{DType, Device, Result, Tensor, Var};
use candle_einops::einsum;

fn flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

fn assert_close(actual: &Tensor, expected: &Tensor, context: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{context}: shape");
    let actual = flat_f32(actual)?;
    let expected = flat_f32(expected)?;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5 * (1. + expected.abs()),
            "{context}[{index}]: actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

#[test]
fn unary_ellipsis_handles_zero_one_many_reduction_scalars_and_zero_dims() -> Result<()> {
    let matrix = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    assert_close(
        &einsum!("row .. column -> column .. row", &matrix)?,
        &matrix.transpose(0, 1)?,
        "zero capture",
    )?;

    let rank_three = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((2, 2, 3))?;
    assert_close(
        &einsum!("row .. column -> column .. row", &rank_three)?,
        &rank_three.permute((2, 1, 0))?,
        "one capture",
    )?;
    assert_close(
        &einsum!(".. column -> column ..", &rank_three)?,
        &rank_three.permute((2, 0, 1))?,
        "many captures",
    )?;
    assert_close(
        &einsum!(".. feature -> feature", &rank_three)?,
        &rank_three.sum((0, 1))?,
        "omitted ellipsis reduction",
    )?;

    let scalar = Tensor::new(7f32, &Device::Cpu)?;
    assert_eq!(einsum!(".. -> ..", &scalar)?.to_scalar::<f32>()?, 7.);
    let empty = Tensor::zeros((0, 2, 3), DType::F32, &Device::Cpu)?;
    assert_eq!(
        einsum!(".. feature -> .. feature", &empty)?.dims(),
        &[0, 2, 3]
    );
    Ok(())
}

#[test]
fn binary_ellipsis_right_aligns_and_broadcasts_different_capture_ranks() -> Result<()> {
    let left = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((2, 1, 2, 3))?;
    let right = Tensor::arange(0f32, 24f32, &Device::Cpu)?.reshape((4, 3, 2))?;
    let actual = einsum!(
        ".. row inner, .. inner column -> row .. column",
        &left,
        &right
    )?;
    let expected = left
        .broadcast_matmul(&right.unsqueeze(0)?)?
        .permute((2, 0, 1, 3))?;
    assert_close(&actual, &expected, "right-aligned binary ellipsis")?;

    let empty = Tensor::zeros((0, 1, 2, 3), DType::F32, &Device::Cpu)?;
    assert_eq!(
        einsum!(
            ".. row inner, .. inner column -> row .. column",
            &empty,
            &right
        )?
        .dims(),
        &[2, 0, 4, 2]
    );
    Ok(())
}

#[test]
fn omitted_binary_ellipsis_reduces_and_retained_ellipsis_has_gradients() -> Result<()> {
    let left = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let right = Tensor::ones((1, 3), DType::F32, &Device::Cpu)?;
    assert_eq!(
        einsum!(".. feature, .. feature ->", &left, &right)?.to_scalar::<f32>()?,
        15.
    );

    let macro_left = Var::from_vec((0..12).map(|v| v as f32).collect(), (2, 2, 3), &Device::Cpu)?;
    let macro_right = Var::from_vec(
        (0..12).map(|v| v as f32 / 3.).collect(),
        (2, 3, 2),
        &Device::Cpu,
    )?;
    let direct_left = Var::from_vec((0..12).map(|v| v as f32).collect(), (2, 2, 3), &Device::Cpu)?;
    let direct_right = Var::from_vec(
        (0..12).map(|v| v as f32 / 3.).collect(),
        (2, 3, 2),
        &Device::Cpu,
    )?;
    let macro_output = einsum!(
        ".. row inner, .. inner column -> .. row column",
        macro_left.as_tensor(),
        macro_right.as_tensor()
    )?;
    let direct_output = direct_left.matmul(&direct_right)?;
    assert_close(&macro_output, &direct_output, "gradient forward")?;
    let macro_grads = macro_output.sum_all()?.backward()?;
    let direct_grads = direct_output.sum_all()?.backward()?;
    assert_close(
        macro_grads
            .get(macro_left.as_tensor())
            .expect("macro left gradient"),
        direct_grads
            .get(direct_left.as_tensor())
            .expect("direct left gradient"),
        "left gradient",
    )?;
    assert_close(
        macro_grads
            .get(macro_right.as_tensor())
            .expect("macro right gradient"),
        direct_grads
            .get(direct_right.as_tensor())
            .expect("direct right gradient"),
        "right gradient",
    )
}

#[test]
fn ellipsis_rank_and_broadcast_errors_are_contextual() -> Result<()> {
    let vector = Tensor::zeros(3, DType::F32, &Device::Cpu)?;
    let error = einsum!(".. row inner -> .. row", &vector).expect_err("rank must fail");
    assert!(error.to_string().contains("einsum operand 0 has rank 1"));
    assert!(error.to_string().contains("ellipsis"));

    let left = Tensor::zeros((2, 3, 1), DType::F32, &Device::Cpu)?;
    let right = Tensor::zeros((4, 1), DType::F32, &Device::Cpu)?;
    let error = einsum!(".. feature, .. feature -> .. feature", &left, &right)
        .expect_err("right-aligned broadcast mismatch must fail");
    assert!(
        error
            .to_string()
            .contains("einsum label `..[1]` cannot broadcast")
    );
    Ok(())
}
