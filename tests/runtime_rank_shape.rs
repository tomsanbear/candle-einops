use std::panic::{AssertUnwindSafe, catch_unwind};

use candle_core::{Device, Result, Tensor};
use candle_einops::einops;

#[test]
fn ellipsis_with_too_few_runtime_axes_returns_an_error() -> Result<()> {
    let input = Tensor::arange(0u32, 4, &Device::Cpu)?;

    let outcome = catch_unwind(AssertUnwindSafe(|| einops!("a b .. -> .. b a", &input)));

    assert!(outcome.is_ok(), "rank-deficient ellipsis must not panic");
    let error = outcome.unwrap().unwrap_err();
    assert!(error.to_string().contains("requires at least 2 axes"));
    Ok(())
}

#[test]
fn generated_shape_access_beyond_rank_returns_an_error() -> Result<()> {
    let input = Tensor::arange(0u32, 4, &Device::Cpu)?;

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        einops!("(row column:2) channel -> row column channel", &input)
    }));

    assert!(outcome.is_ok(), "out-of-range shape access must not panic");
    let error = outcome.unwrap().unwrap_err();
    assert!(error.to_string().contains("shape index 1 out of range"));
    Ok(())
}
