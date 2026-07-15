use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use candle_core::{Device, Result, Tensor};
use candle_einops::einsum;

#[test]
fn unary_explicit_equations_match_independent_oracles() -> Result<()> {
    let matrix = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;

    let transposed = einsum!("rows columns -> columns rows", &matrix)?;
    assert_eq!(
        transposed.to_vec2::<f32>()?,
        [[1., 4.], [2., 5.], [3., 6.]]
    );

    let reduced = einsum!("rows columns -> rows", &matrix)?;
    assert_eq!(reduced.to_vec1::<f32>()?, [6., 15.]);

    let identity = einsum!("rows columns -> rows columns", &matrix)?;
    assert_eq!(identity.to_vec2::<f32>()?, matrix.to_vec2::<f32>()?);

    let scalar = Tensor::new(7f32, &Device::Cpu)?;
    let scalar_identity = einsum!(" -> ", &scalar)?;
    assert_eq!(scalar_identity.to_scalar::<f32>()?, 7.);

    Ok(())
}

#[test]
fn unary_scalar_reduction_and_zero_axes_are_supported() -> Result<()> {
    let vector = Tensor::new(&[1f32, 2., 3., 4.], &Device::Cpu)?;
    let scalar = einsum!("feature ->", &vector)?;
    assert_eq!(scalar.to_scalar::<f32>()?, 10.);

    let empty = Tensor::zeros((0, 2), candle_core::DType::F32, &Device::Cpu)?;
    let transposed = einsum!("empty feature -> feature empty", &empty)?;
    assert_eq!(transposed.dims(), &[2, 0]);

    let empty = Tensor::zeros((2, 0), candle_core::DType::F32, &Device::Cpu)?;
    let reduced = einsum!("row empty -> row", &empty)?;
    assert_eq!(reduced.to_vec1::<f32>()?, [0., 0.]);

    Ok(())
}

#[test]
fn operand_expression_is_evaluated_once() -> Result<()> {
    let calls = Cell::new(0);
    let matrix = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
    let output = einsum!("row column -> column row", {
        calls.set(calls.get() + 1);
        &matrix
    })?;

    assert_eq!(calls.get(), 1);
    assert_eq!(output.to_vec2::<f32>()?, [[1., 3.], [2., 4.]]);
    Ok(())
}

#[test]
fn rank_mismatch_returns_contextual_error_without_unwinding() -> Result<()> {
    let vector = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        einsum!("rows columns -> rows", &vector)
    }));

    let error = outcome
        .expect("rank validation must not unwind")
        .expect_err("rank mismatch must return an error");
    assert!(
        error
            .to_string()
            .contains("einsum operand 0 has rank 1, expected 2"),
        "unexpected error: {error}"
    );
    Ok(())
}
