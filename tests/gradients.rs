use candle_core::{Device, Result, Shape, Tensor, Var};
use candle_einops::einops;

fn flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-5 * (1. + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}[{index}]: actual={actual}, expected={expected}"
        );
    }
}

fn compare_forward_and_gradient(
    data: &[f32],
    shape: &[usize],
    macro_operation: impl FnOnce(&Tensor) -> Result<Tensor>,
    candle_operation: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<()> {
    let device = Device::Cpu;
    let macro_input = Var::from_vec(data.to_vec(), Shape::from_dims(shape), &device)?;
    let candle_input = Var::from_vec(data.to_vec(), Shape::from_dims(shape), &device)?;

    let macro_output = macro_operation(macro_input.as_tensor())?;
    let candle_output = candle_operation(candle_input.as_tensor())?;
    assert_eq!(macro_output.dims(), candle_output.dims());
    assert_close(
        &flat_f32(&macro_output)?,
        &flat_f32(&candle_output)?,
        "forward",
    );

    let weight_values = (1..=macro_output.elem_count())
        .map(|value| value as f32 / 7.)
        .collect::<Vec<_>>();
    let weights = Tensor::from_vec(
        weight_values,
        Shape::from_dims(macro_output.dims()),
        &device,
    )?;
    let macro_loss = macro_output.mul(&weights)?.sum_all()?;
    let candle_loss = candle_output.mul(&weights)?.sum_all()?;
    let macro_gradients = macro_loss.backward()?;
    let candle_gradients = candle_loss.backward()?;
    let Some(macro_gradient) = macro_gradients.get(macro_input.as_tensor()) else {
        candle_core::bail!("einops path did not produce an input gradient")
    };
    let Some(candle_gradient) = candle_gradients.get(candle_input.as_tensor()) else {
        candle_core::bail!("Candle oracle did not produce an input gradient")
    };
    assert_eq!(macro_gradient.dims(), candle_gradient.dims());
    assert_close(
        &flat_f32(macro_gradient)?,
        &flat_f32(candle_gradient)?,
        "gradient",
    );
    Ok(())
}

#[test]
fn transpose_gradient_matches_candle() -> Result<()> {
    let data = (1..=24).map(|value| value as f32).collect::<Vec<_>>();
    compare_forward_and_gradient(
        &data,
        &[2, 3, 4],
        |input| einops!("a b c -> c a b", input),
        |input| input.permute((2, 0, 1)),
    )
}

#[test]
fn composition_and_decomposition_gradients_match_candle() -> Result<()> {
    let data = (1..=24).map(|value| value as f32).collect::<Vec<_>>();
    compare_forward_and_gradient(
        &data,
        &[2, 3, 4],
        |input| einops!("a b c -> (b a) c", input),
        |input| input.permute((1, 0, 2))?.reshape((6, 4)),
    )?;

    compare_forward_and_gradient(
        &data,
        &[6, 4],
        |input| einops!("(a b:2) c -> b a c", input),
        |input| input.reshape((3, 2, 4))?.permute((1, 0, 2)),
    )
}

#[test]
fn repeat_gradient_matches_candle() -> Result<()> {
    let data = (1..=6).map(|value| value as f32).collect::<Vec<_>>();
    compare_forward_and_gradient(
        &data,
        &[2, 3],
        |input| einops!("a b -> copies:3 b a", input),
        |input| input.permute((1, 0))?.unsqueeze(0)?.repeat((3, 1, 1)),
    )
}

#[test]
fn sum_and_mean_gradients_match_candle() -> Result<()> {
    let data = (1..=24).map(|value| value as f32).collect::<Vec<_>>();
    compare_forward_and_gradient(
        &data,
        &[2, 3, 4],
        |input| einops!("a sum(b) c -> c a", input),
        |input| input.sum(1)?.permute((1, 0)),
    )?;

    compare_forward_and_gradient(
        &data,
        &[2, 3, 4],
        |input| einops!("a mean(b) c -> c a", input),
        |input| input.mean(1)?.permute((1, 0)),
    )
}

#[test]
fn product_gradients_match_direct_candle_for_zero_cases() -> Result<()> {
    for (name, data) in [
        ("no zero", [2., 3., 4.]),
        ("one zero", [0., 3., 4.]),
        ("multiple zeros", [0., 0., 4.]),
    ] {
        compare_forward_and_gradient(
            &data,
            &[1, 3],
            |input| einops!("row prod(column) -> row", input),
            |input| {
                input
                    .narrow(1, 0, 1)?
                    .squeeze(1)?
                    .mul(&input.narrow(1, 1, 1)?.squeeze(1)?)?
                    .mul(&input.narrow(1, 2, 1)?.squeeze(1)?)
            },
        )
        .map_err(|error| error.with_path(name))?;
    }
    Ok(())
}

#[test]
fn unique_minimum_and_maximum_gradients_match_candle() -> Result<()> {
    let data = [3., 1., 5., 7., 9., 2.];
    compare_forward_and_gradient(
        &data,
        &[2, 3],
        |input| einops!("row min(column) -> row", input),
        |input| input.min(1),
    )?;
    compare_forward_and_gradient(
        &data,
        &[2, 3],
        |input| einops!("row max(column) -> row", input),
        |input| input.max(1),
    )
}
