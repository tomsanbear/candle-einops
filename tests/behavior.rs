use candle_core::{Device, Result, Tensor};
use candle_einops::einops;

#[test]
fn rearranges_axes_against_explicit_values() -> Result<()> {
    let input = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &Device::Cpu)?;

    let output = einops!("rows columns -> columns rows", &input)?;

    assert_eq!(output.dims(), &[3, 2]);
    assert_eq!(output.to_vec2::<u32>()?, &[[1, 4], [2, 5], [3, 6]]);
    Ok(())
}

#[test]
fn composes_and_decomposes_axes_against_explicit_values() -> Result<()> {
    let input = Tensor::arange(0i64, 12, &Device::Cpu)?.reshape((2, 6))?;

    let decomposed = einops!("batch (row column:2) -> batch row column", &input)?;
    assert_eq!(decomposed.dims(), &[2, 3, 2]);
    assert_eq!(
        decomposed.to_vec3::<i64>()?,
        &[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]],]
    );

    let composed = einops!("batch row column -> (row batch) column", &decomposed)?;
    assert_eq!(composed.dims(), &[6, 2]);
    assert_eq!(
        composed.to_vec2::<i64>()?,
        &[[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]]
    );
    Ok(())
}

#[test]
fn rearranges_ellipsis_against_explicit_values() -> Result<()> {
    let input = Tensor::arange(0u32, 12, &Device::Cpu)?.reshape((2, 2, 3))?;

    let output = einops!("batch .. channels -> channels batch (..)", &input)?;

    assert_eq!(output.dims(), &[3, 2, 2]);
    assert_eq!(
        output.to_vec3::<u32>()?,
        &[[[0, 3], [6, 9]], [[1, 4], [7, 10]], [[2, 5], [8, 11]],]
    );
    Ok(())
}

#[test]
fn repeats_named_and_literal_axes_against_explicit_values() -> Result<()> {
    let input = Tensor::new(&[[1i64, 2], [3, 4]], &Device::Cpu)?;

    let output = einops!("row column -> row copy:2 column 1", &input)?;

    assert_eq!(output.dims(), &[2, 2, 2, 1]);
    assert_eq!(
        output.flatten_all()?.to_vec1::<i64>()?,
        &[1, 2, 1, 2, 3, 4, 3, 4]
    );
    Ok(())
}

#[test]
fn squeezes_singleton_axes_without_changing_values() -> Result<()> {
    let input = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.]]], &Device::Cpu)?;

    let output = einops!("1 batch value -> batch value", &input)?;

    assert_eq!(output.dims(), &[2, 3]);
    assert_eq!(output.to_vec2::<f32>()?, &[[1., 2., 3.], [4., 5., 6.]]);
    Ok(())
}

#[test]
fn reduces_each_supported_non_product_operation_against_values() -> Result<()> {
    let input = Tensor::new(
        &[
            [[1f32, 8., 3.], [4., 2., 6.]],
            [[7., 0., 9.], [5., 11., 10.]],
        ],
        &Device::Cpu,
    )?;

    let minimum = einops!("batch min(row) column -> batch column", &input)?;
    assert_eq!(minimum.to_vec2::<f32>()?, &[[1., 2., 3.], [5., 0., 9.]]);

    let maximum = einops!("max(batch) row column -> row column", &input)?;
    assert_eq!(maximum.to_vec2::<f32>()?, &[[7., 8., 9.], [5., 11., 10.]]);

    let sum = einops!("batch sum(row column) -> batch", &input)?;
    assert_eq!(sum.to_vec1::<f32>()?, &[24., 42.]);

    let mean = einops!("mean(batch row) column -> column", &input)?;
    assert_eq!(mean.to_vec1::<f32>()?, &[4.25, 5.25, 7.]);
    Ok(())
}

#[test]
fn reduces_all_ellipsis_axes_to_a_scalar() -> Result<()> {
    let input = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;

    let output = einops!("sum(..) -> ", &input)?;

    assert!(output.dims().is_empty());
    assert_eq!(output.to_scalar::<f32>()?, 10.);
    Ok(())
}

#[test]
fn accepts_owned_input_against_an_independent_oracle() -> Result<()> {
    let input = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &Device::Cpu)?;

    let output = einops!("row column -> column row", input)?;

    assert_eq!(output.to_vec2::<u32>()?, &[[1, 4], [2, 5], [3, 6]]);
    Ok(())
}

#[test]
fn uses_runtime_axis_lengths_against_explicit_values() -> Result<()> {
    let input = Tensor::arange(0u32, 8, &Device::Cpu)?.reshape((2, 4))?;
    let columns = 2;

    let output = einops!("batch (row {columns}) -> row batch {columns}", &input)?;

    assert_eq!(output.dims(), &[2, 2, 2]);
    assert_eq!(
        output.to_vec3::<u32>()?,
        &[[[0, 1], [4, 5]], [[2, 3], [6, 7]]]
    );
    Ok(())
}
