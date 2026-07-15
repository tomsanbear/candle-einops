use candle_core::{Device, Result, Tensor};
use candle_einops::{Backend, Operation, einops};

#[test]
fn macro_propagates_invalid_runtime_shape() -> Result<()> {
    let input = Tensor::arange(0u32, 5, &Device::Cpu)?;

    let error = einops!("(rows:2 columns) -> rows columns", &input).unwrap_err();

    assert!(error.to_string().to_lowercase().contains("shape"));
    Ok(())
}

#[test]
fn backend_propagates_invalid_reshape() -> Result<()> {
    let input = Tensor::arange(0u32, 6, &Device::Cpu)?;

    assert!(Backend::reshape(&input, &[4, 2]).is_err());
    Ok(())
}

#[test]
fn backend_propagates_invalid_axis() -> Result<()> {
    let input = Tensor::reshape(&Tensor::arange(0u32, 6, &Device::Cpu)?, (2, 3))?;
    let mut reductions = [(2, Operation::Sum)];

    assert!(Backend::transpose(&input, &[0, 0]).is_err());
    assert!(Backend::reduce_axes(&input, &mut reductions).is_err());
    Ok(())
}

#[test]
fn backend_rejects_invalid_added_axis_metadata() -> Result<()> {
    let input = Tensor::arange(0u32, 3, &Device::Cpu)?;

    assert!(Backend::add_axes(&input, 1, &[(0, 2)]).is_err());
    assert!(Backend::add_axes(&input, 2, &[(2, 2)]).is_err());
    Ok(())
}
