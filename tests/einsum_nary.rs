use std::cell::RefCell;

use candle_core::{DType, Device, Result, Tensor};
use candle_einops::einsum;

#[test]
fn three_and_four_operand_chains_match_checked_values() -> Result<()> {
    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let b = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
    let c = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
    let d = Tensor::new(&[[2f32], [3.]], &Device::Cpu)?;

    assert_eq!(
        einsum!("a b, b c, c d -> a d", &a, &b, &c)?.to_vec2::<f32>()?,
        [[106., 156.], [241., 354.]]
    );
    assert_eq!(
        einsum!("a b, b c, c d, d e -> a e", &a, &b, &c, &d)?.to_vec2::<f32>()?,
        [[680.], [1544.]]
    );
    Ok(())
}

#[test]
fn nary_preserves_live_labels_and_reduces_only_safe_labels() -> Result<()> {
    let left = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    let middle = Tensor::ones((3, 4), DType::F32, &Device::Cpu)?;
    let right = Tensor::ones((3, 5), DType::F32, &Device::Cpu)?;
    let live = einsum!(
        "a shared, shared c, shared d -> a c d",
        &left,
        &middle,
        &right
    )?;
    assert_eq!(live.dims(), &[2, 4, 5]);
    assert!(
        live.flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&v| v == 3.)
    );

    let private = Tensor::ones((2, 2, 3), DType::F32, &Device::Cpu)?;
    let tail = Tensor::ones((4, 5), DType::F32, &Device::Cpu)?;
    let reduced = einsum!("private a b, b c, c d -> a d", &private, &middle, &tail)?;
    assert_eq!(reduced.dims(), &[2, 5]);
    assert!(
        reduced
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&v| v == 24.)
    );

    let singleton_left = Tensor::new(&[2f32], &Device::Cpu)?;
    let broadcast = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    let singleton_right = Tensor::new(&[5f32], &Device::Cpu)?;
    assert_eq!(
        einsum!(
            "feature, feature, feature -> feature",
            &singleton_left,
            &broadcast,
            &singleton_right
        )?
        .to_vec1::<f32>()?,
        [10., 20., 30.]
    );
    Ok(())
}

#[test]
fn nary_combines_ellipsis_diagonals_scalars_and_zero_dimensions() -> Result<()> {
    let diagonal = Tensor::arange(0f32, 18f32, &Device::Cpu)?.reshape((2, 3, 3))?;
    let matrix = Tensor::ones((1, 3, 2), DType::F32, &Device::Cpu)?;
    let vector = Tensor::ones(2, DType::F32, &Device::Cpu)?;
    assert_eq!(
        einsum!(".. i i, .. i j, j -> ..", &diagonal, &matrix, &vector)?.to_vec1::<f32>()?,
        [24., 78.]
    );

    let ellipsis_left = Tensor::ones((2, 1, 2, 3), DType::F32, &Device::Cpu)?;
    let ellipsis_middle = Tensor::ones((4, 3, 2), DType::F32, &Device::Cpu)?;
    let ellipsis_right = Tensor::ones((2, 5), DType::F32, &Device::Cpu)?;
    let ellipsis = einsum!(
        ".. a b, .. b c, c d -> a .. d",
        &ellipsis_left,
        &ellipsis_middle,
        &ellipsis_right
    )?;
    assert_eq!(ellipsis.dims(), &[2, 2, 4, 5]);
    assert!(
        ellipsis
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|&v| v == 6.)
    );

    let scalar = Tensor::new(3f32, &Device::Cpu)?;
    let x = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    let y = Tensor::new(&[4f32, 5., 6.], &Device::Cpu)?;
    assert_eq!(
        einsum!(", feature, feature ->", &scalar, &x, &y)?.to_scalar::<f32>()?,
        96.
    );

    let empty_left = Tensor::zeros((2, 0), DType::F32, &Device::Cpu)?;
    let empty_middle = Tensor::zeros((0, 3), DType::F32, &Device::Cpu)?;
    let tail = Tensor::ones(3, DType::F32, &Device::Cpu)?;
    assert_eq!(
        einsum!("a b, b c, c -> a", &empty_left, &empty_middle, &tail)?.to_vec1::<f32>()?,
        [0., 0.]
    );
    Ok(())
}

#[test]
fn nary_validates_every_operand_before_planning() -> Result<()> {
    let matrix = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    let vector = Tensor::ones(3, DType::F32, &Device::Cpu)?;
    let wrong_rank = Tensor::ones((3, 1), DType::F32, &Device::Cpu)?;
    let error = einsum!(
        "row feature, feature, feature -> row",
        &matrix,
        &vector,
        &wrong_rank
    )
    .expect_err("third-operand rank mismatch must fail");
    assert!(error.to_string().contains("einsum operand 2 has rank"));

    let wrong_dtype = Tensor::ones(3, DType::F64, &Device::Cpu)?;
    let error = einsum!(
        "row feature, feature, feature -> row",
        &matrix,
        &vector,
        &wrong_dtype
    )
    .expect_err("third-operand dtype mismatch must fail");
    assert!(error.to_string().contains("different dtypes"));

    let incompatible = Tensor::ones(4, DType::F32, &Device::Cpu)?;
    let error = einsum!(
        "row feature, feature, feature -> row",
        &matrix,
        &vector,
        &incompatible
    )
    .expect_err("global broadcast mismatch must fail");
    assert!(
        error
            .to_string()
            .contains("cannot broadcast extents 3 and 4")
    );
    Ok(())
}

#[test]
fn nary_operand_expressions_are_evaluated_once_from_left_to_right() -> Result<()> {
    let order = RefCell::new(Vec::new());
    let operands = [
        Tensor::new(2f32, &Device::Cpu)?,
        Tensor::new(3f32, &Device::Cpu)?,
        Tensor::new(5f32, &Device::Cpu)?,
        Tensor::new(7f32, &Device::Cpu)?,
    ];
    let output = einsum!(
        ", , , ->",
        {
            order.borrow_mut().push(0);
            &operands[0]
        },
        {
            order.borrow_mut().push(1);
            &operands[1]
        },
        {
            order.borrow_mut().push(2);
            &operands[2]
        },
        {
            order.borrow_mut().push(3);
            &operands[3]
        }
    )?;
    assert_eq!(order.into_inner(), [0, 1, 2, 3]);
    assert_eq!(output.to_scalar::<f32>()?, 210.);
    Ok(())
}

#[test]
fn planner_boundaries_preserve_six_operand_and_non_f32_execution() -> Result<()> {
    let scalars = [2f32, 3., 5., 7., 11., 13.]
        .into_iter()
        .map(|value| Tensor::new(value, &Device::Cpu))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        einsum!(
            ", , , , , ->",
            &scalars[0],
            &scalars[1],
            &scalars[2],
            &scalars[3],
            &scalars[4],
            &scalars[5]
        )?
        .to_scalar::<f32>()?,
        30_030.
    );

    let left = Tensor::new(&[1f64, 2., 3.], &Device::Cpu)?;
    let middle = Tensor::new(&[4f64, 5., 6.], &Device::Cpu)?;
    let right = Tensor::new(&[7f64, 8., 9.], &Device::Cpu)?;
    assert_eq!(
        einsum!("i, i, i -> i", &left, &middle, &right)?.to_vec1::<f64>()?,
        [28., 80., 162.]
    );
    Ok(())
}
