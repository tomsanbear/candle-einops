use candle_core::{DType, Device, Result, Tensor, Var};
use candle_einops::{Backend, Operation, einops};

fn assert_close(left: &Tensor, right: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(left.dims(), right.dims());
    let left = left.flatten_all()?.to_vec1::<f32>()?;
    let right = right.flatten_all()?.to_vec1::<f32>()?;
    for (index, (&left, &right)) in left.iter().zip(&right).enumerate() {
        if left.is_nan() && right.is_nan() {
            continue;
        }
        assert!(
            (left - right).abs() <= tolerance * right.abs().max(1.),
            "value {index} differs: {left} vs {right}"
        );
    }
    Ok(())
}

#[test]
fn fused_sum_and_mean_match_direct_candle_on_contiguous_and_strided_layouts() -> Result<()> {
    let contiguous =
        Tensor::arange(0f32, 2. * 3. * 4. * 5., &Device::Cpu)?.reshape(&[2, 3, 4, 5])?;
    let sum = einops!(
        "batch channel sum(height width) -> batch channel",
        &contiguous
    )?;
    assert_close(&sum, &contiguous.sum(&[2, 3][..])?, 1e-6)?;
    let mean = einops!(
        "batch channel mean(height width) -> batch channel",
        &contiguous
    )?;
    assert_close(&mean, &contiguous.mean(&[2, 3][..])?, 1e-6)?;

    let strided = contiguous.permute([0, 2, 1, 3])?;
    let sum = einops!(
        "batch sum(height) channel sum(width) -> batch channel",
        &strided
    )?;
    assert_close(&sum, &strided.sum(&[1, 3][..])?, 1e-6)?;
    let mean = einops!(
        "batch mean(height) channel mean(width) -> batch channel",
        &strided
    )?;
    assert_close(&mean, &strided.mean(&[1, 3][..])?, 1e-6)?;
    Ok(())
}

#[test]
fn fused_reductions_preserve_gradients_ellipsis_and_boundary_shapes() -> Result<()> {
    let values = (0..24).map(|value| value as f32 / 7.).collect::<Vec<_>>();
    for mean in [false, true] {
        let library_var = Var::from_vec(values.clone(), (2, 3, 4), &Device::Cpu)?;
        let direct_var = Var::from_vec(values.clone(), (2, 3, 4), &Device::Cpu)?;
        let weights = Tensor::new(&[1f32, -2.], &Device::Cpu)?;
        let library = if mean {
            einops!("batch mean(row column) -> batch", library_var.as_tensor())?
        } else {
            einops!("batch sum(row column) -> batch", library_var.as_tensor())?
        };
        let direct = if mean {
            direct_var.as_tensor().mean(&[1, 2][..])?
        } else {
            direct_var.as_tensor().sum(&[1, 2][..])?
        };
        let library_gradients = library.mul(&weights)?.sum_all()?.backward()?;
        let direct_gradients = direct.mul(&weights)?.sum_all()?.backward()?;
        assert_close(
            library_gradients
                .get(&library_var)
                .expect("library gradient"),
            direct_gradients.get(&direct_var).expect("direct gradient"),
            1e-6,
        )?;
    }

    let input = Tensor::arange(0f32, 24., &Device::Cpu)?.reshape(&[2, 3, 4])?;
    let ellipsis = einops!("batch sum(..) -> batch", &input)?;
    assert_close(&ellipsis, &input.sum(&[1, 2][..])?, 1e-6)?;

    let empty = Tensor::zeros((2, 0, 1), DType::F32, &Device::Cpu)?;
    assert_close(
        &einops!("batch sum(empty singleton) -> batch", &empty)?,
        &empty.sum(&[1, 2][..])?,
        1e-6,
    )?;
    assert_close(
        &einops!("batch mean(empty singleton) -> batch", &empty)?,
        &empty.mean(&[1, 2][..])?,
        1e-6,
    )?;
    Ok(())
}

#[test]
fn mixed_order_and_dtype_support_remain_unchanged() -> Result<()> {
    let input = Tensor::arange(0f32, 24., &Device::Cpu)?.reshape(&[2, 3, 4])?;
    let mut reductions = [
        (0, Operation::Sum),
        (1, Operation::Max),
        (2, Operation::Sum),
    ];
    let mixed = (&input).reduce_axes(&mut reductions)?;
    let direct = input.sum(2)?.max(1)?.sum(0)?;
    assert_close(&mixed, &direct, 1e-6)?;

    let integers = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape(&[2, 3, 4])?;
    let sum = einops!("batch sum(row column) -> batch", &integers)?;
    assert_eq!(
        sum.to_vec1::<u32>()?,
        integers.sum(&[1, 2][..])?.to_vec1::<u32>()?
    );
    let library_mean = einops!("batch mean(row column) -> batch", &integers);
    let direct_mean = integers.mean(&[1, 2][..]);
    assert_eq!(library_mean.is_ok(), direct_mean.is_ok());
    Ok(())
}
