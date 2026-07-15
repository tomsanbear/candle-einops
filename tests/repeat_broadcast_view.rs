use std::panic::{AssertUnwindSafe, catch_unwind};

use candle_core::{Device, Result, Tensor, Var};
use candle_einops::{Backend, einops};

#[test]
fn middle_repeat_is_a_zero_stride_view_with_exact_values() -> Result<()> {
    let input = Tensor::new(&[[1u32, 2], [3, 4]], &Device::Cpu)?;
    let output = einops!("row column -> row copy:4 column", &input)?;

    assert_eq!(output.dims(), &[2, 4, 2]);
    assert_eq!(
        output.to_vec3::<u32>()?,
        [
            [[1, 2], [1, 2], [1, 2], [1, 2]],
            [[3, 4], [3, 4], [3, 4], [3, 4]],
        ]
    );
    assert!(!output.is_contiguous());
    let (_storage, layout) = output.storage_and_layout();
    assert_eq!(layout.stride()[1], 0);
    Ok(())
}

#[test]
fn unsorted_multiple_axes_preserve_a_noncontiguous_input_layout() -> Result<()> {
    let input = Tensor::arange(0u32, 6, &Device::Cpu)?
        .reshape(&[2, 3])?
        .permute([1, 0])?;
    assert!(!input.is_contiguous());

    let output = Backend::add_axes(&input, 5, &[(4, 2), (0, 3), (2, 1)])?;
    let expected = input
        .unsqueeze(0)?
        .unsqueeze(2)?
        .unsqueeze(4)?
        .broadcast_as(&[3, 3, 1, 2, 2])?;
    assert_eq!(output.dims(), &[3, 3, 1, 2, 2]);
    assert_eq!(
        output.flatten_all()?.to_vec1::<u32>()?,
        expected.flatten_all()?.to_vec1::<u32>()?
    );
    let (_storage, layout) = output.storage_and_layout();
    assert_eq!(layout.stride()[0], 0);
    assert_eq!(layout.stride()[1], 1);
    assert_eq!(layout.stride()[3], 3);
    assert_eq!(layout.stride()[4], 0);
    Ok(())
}

#[test]
fn zero_repeat_keeps_the_input_in_the_backward_graph() -> Result<()> {
    let input = Var::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &Device::Cpu)?;
    let output = einops!(
        "row column -> leading:1 row empty:0 column trailing:2",
        input.as_tensor()
    )?;
    assert_eq!(output.dims(), &[1, 2, 0, 3, 2]);
    assert_eq!(output.elem_count(), 0);

    let gradients = output.sum_all()?.backward()?;
    let gradient = gradients
        .get(&input)
        .expect("zero-length broadcast must preserve the input graph");
    assert_eq!(gradient.to_vec2::<f32>()?, [[0., 0., 0.], [0., 0., 0.]]);
    Ok(())
}

#[test]
fn repeat_then_composition_materializes_the_same_logical_order() -> Result<()> {
    let input = Tensor::new(&[[1u8, 2], [3, 4]], &Device::Cpu)?;
    let output = einops!("row column -> row (copy:3 column)", &input)?;
    assert_eq!(output.dims(), &[2, 6]);
    assert_eq!(
        output.to_vec2::<u8>()?,
        [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
    );
    Ok(())
}

#[test]
fn singleton_dtype_and_invalid_metadata_behavior_remain_stable() -> Result<()> {
    let input = Tensor::new(&[[1i64, 2], [3, 4]], &Device::Cpu)?;
    let singleton = einops!("row column -> leading:1 row column trailing:1", &input)?;
    assert_eq!(singleton.dims(), &[1, 2, 2, 1]);
    assert_eq!(singleton.flatten_all()?.to_vec1::<i64>()?, [1, 2, 3, 4]);

    for result in [
        catch_unwind(AssertUnwindSafe(|| {
            Backend::add_axes(&input, 3, &[(1, 2), (1, 3)])
        })),
        catch_unwind(AssertUnwindSafe(|| Backend::add_axes(&input, 2, &[(0, 2)]))),
        catch_unwind(AssertUnwindSafe(|| Backend::add_axes(&input, 3, &[(3, 2)]))),
    ] {
        assert!(result.is_ok(), "invalid repeat metadata must not unwind");
        assert!(result.expect("checked above").is_err());
    }
    Ok(())
}
