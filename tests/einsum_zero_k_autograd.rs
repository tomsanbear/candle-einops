use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_einops::einsum;

fn assert_graph_preserving_zero(
    left_shape: &[usize],
    right_shape: &[usize],
    output_shape: &[usize],
    contract: impl FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
) -> Result<()> {
    let device = Device::Cpu;
    let left = Var::zeros(Shape::from_dims(left_shape), DType::F32, &device)?;
    let right = Var::zeros(Shape::from_dims(right_shape), DType::F32, &device)?;

    let output = contract(left.as_tensor(), right.as_tensor())?;
    assert_eq!(output.dims(), output_shape);
    assert!(
        output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&value| value == 0.)
    );

    let gradients = output.sum_all()?.backward()?;
    for (name, variable, shape) in [("left", &left, left_shape), ("right", &right, right_shape)] {
        let gradient = gradients
            .get(variable.as_tensor())
            .unwrap_or_else(|| panic!("missing {name} gradient for zero-length contraction"));
        assert_eq!(gradient.dims(), shape, "{name} gradient shape");
        assert!(
            gradient
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|&value| value == 0.),
            "{name} gradient must be exactly zero"
        );
    }
    Ok(())
}

#[test]
fn zero_k_contractions_retain_both_autograd_edges() -> Result<()> {
    assert_graph_preserving_zero(&[2, 0], &[0, 3], &[2, 3], |left, right| {
        einsum!("row inner, inner column -> row column", left, right)
    })?;
    assert_graph_preserving_zero(&[1, 2, 0], &[4, 0, 3], &[4, 2, 3], |left, right| {
        einsum!(
            "batch row inner, batch inner column -> batch row column",
            left,
            right
        )
    })?;
    assert_graph_preserving_zero(&[0], &[0], &[], |left, right| {
        einsum!("inner, inner ->", left, right)
    })
}

#[test]
fn empty_batch_and_free_axes_retain_both_autograd_edges() -> Result<()> {
    assert_graph_preserving_zero(&[0, 2, 3], &[0, 3, 4], &[0, 2, 4], |left, right| {
        einsum!(
            "batch row inner, batch inner column -> batch row column",
            left,
            right
        )
    })?;
    assert_graph_preserving_zero(&[0, 3], &[3, 4], &[0, 4], |left, right| {
        einsum!("row inner, inner column -> row column", left, right)
    })?;
    assert_graph_preserving_zero(&[2, 3], &[3, 0], &[2, 0], |left, right| {
        einsum!("row inner, inner column -> row column", left, right)
    })
}

#[test]
fn zero_shortcut_does_not_precede_dimension_or_dtype_validation() -> Result<()> {
    let device = Device::Cpu;
    let left = Tensor::zeros((2, 4, 0), DType::F32, &device)?;
    let incompatible_batch = Tensor::zeros((3, 0, 5), DType::F32, &device)?;
    let error = einsum!(
        "batch row inner, batch inner column -> batch row column",
        &left,
        &incompatible_batch
    )
    .expect_err("incompatible batch dimensions must fail before zero construction");
    assert!(
        error
            .to_string()
            .contains("cannot broadcast extents 2 and 3")
    );

    let left = Tensor::zeros((2, 0), DType::F32, &device)?;
    let incompatible_k = Tensor::zeros((2, 3), DType::F32, &device)?;
    let error = einsum!(
        "row inner, inner column -> row column",
        &left,
        &incompatible_k
    )
    .expect_err("incompatible contracted dimensions must fail before zero construction");
    assert!(
        error
            .to_string()
            .contains("cannot broadcast extents 0 and 2")
    );

    let different_dtype = Tensor::zeros((0, 3), DType::F64, &device)?;
    let error = einsum!(
        "row inner, inner column -> row column",
        &left,
        &different_dtype
    )
    .expect_err("dtype validation must precede zero construction");
    assert!(error.to_string().contains("different dtypes"));
    Ok(())
}

#[test]
fn zero_k_forward_preserves_candle_dtype_support() -> Result<()> {
    let device = Device::Cpu;
    for dtype in [DType::BF16, DType::U8, DType::U32, DType::I64] {
        let left = Tensor::zeros((2, 0), dtype, &device)?;
        let right = Tensor::zeros((0, 3), dtype, &device)?;
        let direct = left.matmul(&right);
        let einsum = einsum!("row inner, inner column -> row column", &left, &right);
        match (direct, einsum) {
            (Ok(direct), Ok(einsum)) => {
                assert_eq!(einsum.dtype(), dtype);
                assert_eq!(einsum.dims(), [2, 3]);
                assert_eq!(einsum.to_dtype(DType::F32)?.to_vec2::<f32>()?, [[0.; 3]; 2]);
                assert_eq!(direct.to_dtype(DType::F32)?.to_vec2::<f32>()?, [[0.; 3]; 2]);
            }
            (Err(_), Err(_)) => {}
            (direct, einsum) => panic!(
                "zero-K support diverged for {dtype:?}: direct={}, einsum={}",
                direct.is_ok(),
                einsum.is_ok()
            ),
        }
    }
    Ok(())
}
