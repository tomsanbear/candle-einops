use candle_core::{Device, Result, Tensor};
use candle_einops::einops;

#[test]
fn standalone_braced_repeat_is_preserved_during_composition() -> Result<()> {
    let input = Tensor::arange(0u32, 6, &Device::Cpu)?.reshape((2, 3))?;
    let copies = 2;

    let output = einops!("row column -> {copies} (row column)", &input);

    assert_eq!(output.dims(), &[2, 6]);
    assert_eq!(
        output.to_vec2::<u32>()?,
        &[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    );
    Ok(())
}

#[test]
fn standalone_braced_input_axis_is_preserved_during_composition() -> Result<()> {
    let input = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape((6, 4))?;
    let columns = 3;

    let output = einops!("(row {columns}) value -> {columns} (row value)", &input);

    assert_eq!(output.dims(), &[3, 8]);
    assert_eq!(
        output.to_vec2::<u32>()?,
        &[
            [0, 1, 2, 3, 12, 13, 14, 15],
            [4, 5, 6, 7, 16, 17, 18, 19],
            [8, 9, 10, 11, 20, 21, 22, 23],
        ]
    );
    Ok(())
}

#[test]
fn reduction_on_an_inferred_axis_is_applied() -> Result<()> {
    let input = Tensor::arange(1f32, 13., &Device::Cpu)?.reshape((2, 6))?;

    let output = einops!("batch (sum(row) column:2) -> batch column", &input);

    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(output.to_vec2::<f32>()?, &[[9., 12.], [27., 30.]]);
    Ok(())
}

#[test]
fn reduction_on_an_inferred_axis_after_ellipsis_is_applied() -> Result<()> {
    let input = Tensor::arange(1f32, 25., &Device::Cpu)?.reshape((2, 2, 6))?;

    let output = einops!(".. (sum(row) column:2) -> .. column", &input);

    assert_eq!(output.dims(), &[2, 2, 2]);
    assert_eq!(
        output.to_vec3::<f32>()?,
        &[[[9., 12.], [27., 30.]], [[45., 48.], [63., 66.]]]
    );
    Ok(())
}
