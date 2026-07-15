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
