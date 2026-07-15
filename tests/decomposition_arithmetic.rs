use std::panic::{AssertUnwindSafe, catch_unwind};

use candle_core::{Device, Result, Tensor};
use candle_einops::einops;

#[test]
fn zero_decomposition_factor_returns_an_error() -> Result<()> {
    let input = Tensor::arange(0u32, 6, &Device::Cpu)?;

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        einops!("(axis factor:0) -> axis factor", &input)
    }));

    assert!(outcome.is_ok(), "zero factor must not panic");
    let error = outcome.unwrap().unwrap_err();
    assert!(error.to_string().contains("must be non-zero"));
    Ok(())
}

#[test]
fn non_divisible_decomposition_returns_a_specific_error() -> Result<()> {
    let input = Tensor::arange(0u32, 5, &Device::Cpu)?;

    let error = einops!("(axis factor:2) -> axis factor", &input).unwrap_err();

    assert!(error.to_string().contains("not divisible"));
    Ok(())
}

#[test]
fn overflowing_runtime_factor_product_returns_an_error() -> Result<()> {
    let input = Tensor::new(&[1u32], &Device::Cpu)?;
    let large = usize::MAX;
    let two = 2usize;

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        einops!("(axis {large} {two}) -> axis {large} {two}", &input)
    }));

    assert!(outcome.is_ok(), "factor overflow must not panic");
    let error = outcome.unwrap().unwrap_err();
    assert!(error.to_string().contains("factor product overflows usize"));
    Ok(())
}

#[test]
fn valid_inferred_decomposition_is_unchanged() -> Result<()> {
    let input = Tensor::arange(0u32, 12, &Device::Cpu)?;

    let output = einops!("(row column:3) -> row column", &input)?;

    assert_eq!(output.dims(), &[4, 3]);
    assert_eq!(
        output.to_vec2::<u32>()?,
        &[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    );
    Ok(())
}
