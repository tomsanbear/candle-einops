use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_einops::einsum;

fn flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

fn assert_close(actual: &Tensor, expected: &Tensor, context: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{context}: shape");
    let actual = flat_f32(actual)?;
    let expected = flat_f32(expected)?;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 1e-5 * (1. + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}[{index}]: actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

#[test]
fn binary_values_match_independent_oracles() -> Result<()> {
    let left = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    let right = Tensor::new(&[4f32, 5., 6.], &Device::Cpu)?;
    assert_eq!(
        einsum!("feature, feature ->", &left, &right)?.to_scalar::<f32>()?,
        32.
    );

    let row = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
    let column = Tensor::new(&[10f32, 20., 30.], &Device::Cpu)?;
    assert_eq!(
        einsum!("row, column -> row column", &row, &column)?.to_vec2::<f32>()?,
        [[10., 20., 30.], [20., 40., 60.]]
    );

    let matrix = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let broadcast = Tensor::new(&[[10f32, 20., 30.]], &Device::Cpu)?;
    assert_eq!(
        einsum!(
            "batch feature, batch feature -> batch feature",
            &matrix,
            &broadcast
        )?
        .to_vec2::<f32>()?,
        [[10., 40., 90.], [40., 100., 180.]]
    );

    assert_eq!(
        einsum!("row feature, feature -> row", &matrix, &left)?.to_vec1::<f32>()?,
        [14., 32.]
    );

    let rhs = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
    assert_eq!(
        einsum!("row inner, inner column -> row column", &matrix, &rhs)?.to_vec2::<f32>()?,
        [[22., 28.], [49., 64.]]
    );

    let batched_left = Tensor::new(
        &[
            [[1f32, 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
        ],
        &Device::Cpu,
    )?;
    let batched_right = rhs.unsqueeze(0)?;
    assert_eq!(
        einsum!(
            "batch row inner, batch inner column -> batch row column",
            &batched_left,
            &batched_right
        )?
        .to_vec3::<f32>()?,
        [[[22., 28.], [49., 64.]], [[76., 100.], [103., 136.]]]
    );

    let with_private_left = Tensor::new(
        &[
            [[[1f32, 2., 3.], [4., 5., 6.]]],
            [[[7., 8., 9.], [10., 11., 12.]]],
        ],
        &Device::Cpu,
    )?;
    let with_private_right = Tensor::ones((2, 3, 4, 2), DType::F32, &Device::Cpu)?;
    let actual = einsum!(
        "private batch row inner, batch inner column extra -> column batch row",
        &with_private_left,
        &with_private_right
    )?;
    let expected = with_private_left
        .sum(0)?
        .broadcast_matmul(&with_private_right.sum(3)?)?
        .permute((2, 0, 1))?;
    assert_close(&actual, &expected, "pre-reduction and output permutation")?;
    Ok(())
}

#[test]
fn canonical_batched_gemm_materializes_broadcasts_before_matmul() -> Result<()> {
    let left_singleton = Tensor::ones((1, 32, 32), DType::F32, &Device::Cpu)?;
    let left_full = Tensor::ones((32, 32, 32), DType::F32, &Device::Cpu)?;
    let right_singleton = Tensor::ones((1, 32, 32), DType::F32, &Device::Cpu)?;
    let right_full = Tensor::ones((32, 32, 32), DType::F32, &Device::Cpu)?;

    let left_broadcast = einsum!(
        "batch row inner, batch inner column -> batch row column",
        &left_singleton,
        &right_full
    )?;
    let left_expected = left_singleton.broadcast_matmul(&right_full)?;
    assert_close(
        &left_broadcast,
        &left_expected,
        "canonical left batch broadcast",
    )?;

    let right_broadcast = einsum!(
        "batch row inner, batch inner column -> batch row column",
        &left_full,
        &right_singleton
    )?;
    let right_expected = left_full.broadcast_matmul(&right_singleton)?;
    assert_close(
        &right_broadcast,
        &right_expected,
        "canonical right batch broadcast",
    )
}

#[test]
fn canonical_gemm_keeps_arbitrary_exact_batches_and_output_views() -> Result<()> {
    let left = Tensor::arange(0f32, (2 * 2 * 4 * 3) as f32, &Device::Cpu)?
        .reshape((2, 2, 4, 3))?
        .transpose(2, 3)?;
    let right = Tensor::arange(0f32, (2 * 2 * 5 * 4) as f32, &Device::Cpu)?
        .reshape((2, 2, 5, 4))?
        .transpose(2, 3)?;
    assert!(!left.is_contiguous());
    assert!(!right.is_contiguous());

    let actual = einsum!(
        "outer batch row inner, outer batch inner column -> column batch row outer",
        &left,
        &right
    )?;
    let expected = left.matmul(&right)?.permute((3, 1, 2, 0))?;
    assert_close(&actual, &expected, "arbitrary exact batch direct GEMM")
}

#[test]
fn binary_scalars_broadcast_contract_and_zero_dimensions() -> Result<()> {
    let scalar = Tensor::new(3f32, &Device::Cpu)?;
    let other = Tensor::new(4f32, &Device::Cpu)?;
    assert_eq!(einsum!(", ->", &scalar, &other)?.to_scalar::<f32>()?, 12.);

    let vector = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    assert_eq!(
        einsum!(", feature -> feature", &scalar, &vector)?.to_vec1::<f32>()?,
        [3., 6., 9.]
    );

    let singleton = Tensor::new(&[2f32], &Device::Cpu)?;
    assert_eq!(
        einsum!("feature, feature ->", &singleton, &vector)?.to_scalar::<f32>()?,
        12.
    );

    let empty_rows = Tensor::zeros(0, DType::F32, &Device::Cpu)?;
    let pair = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
    assert_eq!(
        einsum!("row, column -> row column", &empty_rows, &pair)?.dims(),
        &[0, 2]
    );

    let empty_left = Tensor::zeros((2, 0), DType::F32, &Device::Cpu)?;
    let empty_right = Tensor::zeros((0, 3), DType::F32, &Device::Cpu)?;
    assert_eq!(
        einsum!(
            "row inner, inner column -> row column",
            &empty_left,
            &empty_right
        )?
        .to_vec2::<f32>()?,
        [[0., 0., 0.], [0., 0., 0.]]
    );
    Ok(())
}

#[test]
fn operands_are_evaluated_once_from_left_to_right() -> Result<()> {
    let order = RefCell::new(Vec::new());
    let left = Tensor::new(&[2f32, 3.], &Device::Cpu)?;
    let right = Tensor::new(&[5f32, 7.], &Device::Cpu)?;
    let output = einsum!(
        "feature, feature ->",
        {
            order.borrow_mut().push("left");
            &left
        },
        {
            order.borrow_mut().push("right");
            &right
        }
    )?;
    assert_eq!(order.into_inner(), ["left", "right"]);
    assert_eq!(output.to_scalar::<f32>()?, 31.);
    Ok(())
}

fn compare_binary_gradients(
    left_data: &[f32],
    left_shape: &[usize],
    right_data: &[f32],
    right_shape: &[usize],
    macro_operation: impl FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
    candle_operation: impl FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
) -> Result<()> {
    let device = Device::Cpu;
    let macro_left = Var::from_vec(left_data.to_vec(), Shape::from_dims(left_shape), &device)?;
    let macro_right = Var::from_vec(right_data.to_vec(), Shape::from_dims(right_shape), &device)?;
    let candle_left = Var::from_vec(left_data.to_vec(), Shape::from_dims(left_shape), &device)?;
    let candle_right = Var::from_vec(right_data.to_vec(), Shape::from_dims(right_shape), &device)?;

    let macro_output = macro_operation(macro_left.as_tensor(), macro_right.as_tensor())?;
    let candle_output = candle_operation(candle_left.as_tensor(), candle_right.as_tensor())?;
    assert_close(&macro_output, &candle_output, "forward")?;

    let macro_gradients = macro_output.sum_all()?.backward()?;
    let candle_gradients = candle_output.sum_all()?.backward()?;
    assert_close(
        macro_gradients
            .get(macro_left.as_tensor())
            .expect("macro left gradient"),
        candle_gradients
            .get(candle_left.as_tensor())
            .expect("Candle left gradient"),
        "left gradient",
    )?;
    assert_close(
        macro_gradients
            .get(macro_right.as_tensor())
            .expect("macro right gradient"),
        candle_gradients
            .get(candle_right.as_tensor())
            .expect("Candle right gradient"),
        "right gradient",
    )
}

#[test]
fn binary_cpu_gradients_match_direct_candle() -> Result<()> {
    compare_binary_gradients(
        &[1., 2., 3.],
        &[3],
        &[4., 5., 6.],
        &[3],
        |left, right| einsum!("feature, feature ->", left, right),
        |left, right| left.mul(right)?.sum_all(),
    )?;
    compare_binary_gradients(
        &[1., 2., 3., 4., 5., 6.],
        &[2, 3],
        &[1., 2., 3., 4., 5., 6.],
        &[3, 2],
        |left, right| einsum!("row inner, inner column -> row column", left, right),
        Tensor::matmul,
    )?;
    compare_binary_gradients(
        &[1., 2., 3., 4., 5., 6.],
        &[2, 3],
        &[10., 20., 30.],
        &[1, 3],
        |left, right| einsum!("batch feature, batch feature -> batch feature", left, right),
        Tensor::broadcast_mul,
    )?;
    compare_binary_gradients(
        &(0..16).map(|value| value as f32 / 16.).collect::<Vec<_>>(),
        &[1, 4, 4],
        &(0..48).map(|value| value as f32 / 48.).collect::<Vec<_>>(),
        &[3, 4, 4],
        |left, right| {
            einsum!(
                "batch row inner, batch inner column -> batch row column",
                left,
                right
            )
        },
        Tensor::broadcast_matmul,
    )?;
    compare_binary_gradients(
        &(0..48).map(|value| value as f32 / 48.).collect::<Vec<_>>(),
        &[3, 4, 4],
        &(0..16).map(|value| value as f32 / 16.).collect::<Vec<_>>(),
        &[1, 4, 4],
        |left, right| {
            einsum!(
                "batch row inner, batch inner column -> batch row column",
                left,
                right
            )
        },
        Tensor::broadcast_matmul,
    )
}

#[test]
fn binary_runtime_errors_are_contextual_and_do_not_unwind() -> Result<()> {
    let matrix = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    let vector = Tensor::zeros(3, DType::F32, &Device::Cpu)?;
    let wrong_rank = catch_unwind(AssertUnwindSafe(|| {
        einsum!("row inner, inner -> row", &vector, &vector)
    }));
    let error = wrong_rank
        .expect("rank validation must not unwind")
        .expect_err("rank mismatch must fail");
    assert!(error.to_string().contains("einsum operand 0 has rank"));

    let incompatible = Tensor::zeros(4, DType::F32, &Device::Cpu)?;
    let error = einsum!("row inner, inner -> row", &matrix, &incompatible)
        .expect_err("shared-label broadcast mismatch must fail");
    assert!(
        error
            .to_string()
            .contains("einsum label `inner` cannot broadcast")
    );

    let different_dtype = Tensor::zeros(3, DType::F64, &Device::Cpu)?;
    let error = einsum!("inner, inner ->", &vector, &different_dtype)
        .expect_err("dtype mismatch must fail");
    assert!(
        error
            .to_string()
            .contains("einsum operands have different dtypes")
    );

    if let Ok(other_device) = Device::new_metal(0) {
        let other = Tensor::zeros(3, DType::F32, &other_device)?;
        let error =
            einsum!("inner, inner ->", &vector, &other).expect_err("device mismatch must fail");
        assert!(
            error
                .to_string()
                .contains("einsum operands are on different devices")
        );
    }
    Ok(())
}
